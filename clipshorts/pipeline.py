"""Stage pipeline: download -> probe -> layout -> transcribe -> captions -> render.

Every stage writes its artifacts under data/clips/<id>/ and records a params_hash in
stages.json. A stage is skipped when its hash matches and its artifacts still exist.
The hash chains over the upstream stage hash, so changing an early input invalidates
everything after it, while a late change (e.g. a layout override) only re-runs the tail.
"""

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .db import DB
from .events import EventBus
from .media import MediaError, Tools
from .models import Clip

log = logging.getLogger(__name__)

STAGE_NAMES = ["download", "probe", "layout", "transcribe", "captions", "render"]
STATE_FILE = "stages.json"


class StageError(RuntimeError):
    """Expected stage failure; message is shown to the user in clip.error."""


@dataclass
class StageContext:
    clip: Clip
    dir: Path
    cfg: Config
    db: DB
    tools: Tools
    progress: Callable[[float], None]  # fraction 0..1 within the current stage


@dataclass
class Stage:
    name: str
    artifacts: list[str]
    params: Callable[[Clip, Config], dict[str, Any]]
    run: Callable[[StageContext], None] | None = None  # None = not implemented yet


# -- hashing / state -------------------------------------------------------
def params_hash(stage: str, params: dict[str, Any], upstream: str = "") -> str:
    blob = json.dumps({"stage": stage, "params": params, "up": upstream}, sort_keys=True,
                      default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_state(clip_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((clip_dir / STATE_FILE).read_text())
    except (OSError, ValueError):
        return {}


def save_state(clip_dir: Path, state: dict[str, Any]) -> None:
    tmp = clip_dir / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(clip_dir / STATE_FILE)


def is_fresh(stage: Stage, clip_dir: Path, state: dict[str, Any], h: str) -> bool:
    entry = state.get(stage.name)
    return bool(
        entry
        and entry.get("hash") == h
        and all((clip_dir / a).exists() for a in stage.artifacts)
    )


# -- stage implementations (M1) ----------------------------------------------
def _download(ctx: StageContext) -> None:
    dest = ctx.dir / "source.mp4"
    info = ctx.tools.download(
        ctx.clip.source_url, dest, fmt=ctx.cfg.ytdlp_format, on_progress=ctx.progress
    )
    (ctx.dir / "meta.json").write_text(
        json.dumps({"twitch": info}, indent=2, ensure_ascii=False)
    )


def _probe(ctx: StageContext) -> None:
    src = ctx.dir / "source.mp4"
    probe = ctx.tools.probe(src, ctx.cfg.ffprobe_bin)
    meta_path = ctx.dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        meta = {}
    meta["ffprobe"] = probe
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    tw = meta.get("twitch", {})
    ctx.db.update_clip(
        ctx.clip.id,
        title=tw.get("title") or ctx.clip.title,
        channel=(tw.get("channel") or ctx.clip.channel),
        game=tw.get("game") or ctx.clip.game,
        duration_s=probe.get("duration") or tw.get("duration"),
    )


STAGES: list[Stage] = [
    Stage("download", ["source.mp4"],
          lambda c, cfg: {"slug": c.slug, "format": cfg.ytdlp_format}, _download),
    Stage("probe", ["meta.json"], lambda c, cfg: {}, _probe),
    Stage("layout", ["layout.json"],
          lambda c, cfg: {"override": c.layout_override}),
    Stage("transcribe", ["transcript.json"],
          lambda c, cfg: {"language": c.language, "allowed": cfg.language_list,
                          "model": cfg.whisper_model, "compute": cfg.whisper_compute_type}),
    Stage("captions", ["captions.ass"], lambda c, cfg: {"language": c.language}),
    Stage("render", ["output.mp4"],
          lambda c, cfg: {"layout": c.layout_override, "fps_cap": cfg.fps_cap}),
]


@dataclass
class PipelineResult:
    status: str  # done | failed | waiting
    stage: str | None = None
    error: str | None = None
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def run_pipeline(
    clip_id: int,
    cfg: Config,
    db: DB,
    bus: EventBus,
    tools: Tools,
    stages: list[Stage] | None = None,
) -> PipelineResult:
    """Run all stages for a clip, skipping cached ones. Blocking; run in a worker thread."""
    stages = STAGES if stages is None else stages
    clip = db.get_clip(clip_id)
    if clip is None:
        return PipelineResult("failed", error="clip not found")
    clip_dir = cfg.clip_dir(clip_id)
    clip_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(clip_dir)
    result = PipelineResult("done")
    total = len(stages)

    def emit(**fields) -> None:
        updated = db.update_clip(clip_id, **fields)
        if updated:
            bus.publish("clip", updated.model_dump())

    emit(status="running", error=None)
    upstream = ""
    for idx, stage in enumerate(stages):
        clip = db.get_clip(clip_id) or clip  # may have been updated by previous stage
        h = params_hash(stage.name, stage.params(clip, cfg), upstream)
        upstream = h
        if is_fresh(stage, clip_dir, state, h):
            result.skipped.append(stage.name)
            emit(stage=stage.name, progress=(idx + 1) / total)
            continue
        if stage.run is None:
            result.status, result.stage = "waiting", stage.name
            emit(status="waiting", stage=stage.name, progress=idx / total)
            return result

        emit(stage=stage.name, progress=idx / total)
        last = [0.0]

        def on_progress(frac: float, idx=idx, name=stage.name, last=last) -> None:
            frac = max(0.0, min(1.0, frac))
            if frac - last[0] < 0.02 and frac < 1.0:
                return
            last[0] = frac
            # progress is a transient: published, persisted only on stage boundaries
            bus.publish("clip", {"id": clip_id, "stage": name,
                                 "progress": (idx + frac) / total, "status": "running"})

        ctx = StageContext(clip, clip_dir, cfg, db, tools, on_progress)
        t0 = time.monotonic()
        try:
            stage.run(ctx)
        except (StageError, MediaError) as e:
            return _fail(result, stage.name, str(e), emit)
        except Exception as e:  # noqa: BLE001 - surface anything to the user
            log.exception("clip %s stage %s crashed", clip_id, stage.name)
            return _fail(result, stage.name, f"{type(e).__name__}: {e}", emit)
        missing = [a for a in stage.artifacts if not (clip_dir / a).exists()]
        if missing:
            return _fail(result, stage.name, f"missing artifact(s): {', '.join(missing)}", emit)
        state[stage.name] = {"hash": h, "seconds": round(time.monotonic() - t0, 3)}
        save_state(clip_dir, state)
        result.ran.append(stage.name)
        emit(stage=stage.name, progress=(idx + 1) / total)

    emit(status="done", stage=stages[-1].name if stages else None, progress=1.0)
    return result


def _fail(result: PipelineResult, stage: str, msg: str, emit) -> PipelineResult:
    err = f"{stage}: {msg}"
    result.status, result.stage, result.error = "failed", stage, err
    emit(status="failed", stage=stage, error=err)
    return result
