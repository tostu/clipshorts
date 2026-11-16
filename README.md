# clipshorts

Paste a Twitch clip URL, get a vertical 1080x1920 short with auto-detected layout and burned-in German/English subtitles. Everything runs server-side (built for a low-power Intel N100 box, behind Tailscale/VPN); the browser is only a review, override and download UI.

For private use. Respect Twitch's ToS and the streamers' rights when re-uploading clips.

## Pipeline

`download -> probe -> layout -> transcribe -> captions -> render`

| Stage | What it does |
|---|---|
| download | yt-dlp fetches the clip (`source.mp4`) |
| probe | ffprobe + Twitch metadata (`meta.json`) |
| layout | YuNet face detection on 8 sampled frames, rule-based classifier (`layout.json`) |
| transcribe | faster-whisper `small` int8, language auto-detected and clamped to `LANGUAGES` (`transcript.json`) |
| captions | own ASS writer, max 3 words per caption, word highlight (`captions.ass`) |
| render | one ffmpeg pass: layout filter graph + `ass` burn-in + `loudnorm` (`output.mp4`) |

Every stage is cached by a params hash. A layout override re-runs only `layout` and `render`; transcription and captions stay cached. Jobs run one at a time at low priority (`NICE_LEVEL`).

## Quick start (Docker)

```bash
cp .env.example .env          # optional: AUTH_TOKEN, BIND_ADDR, PUID/PGID, TZ
mkdir -p data
docker compose up -d --build
```

Open `http://127.0.0.1:8000` (or `BIND_ADDR:PORT`). The first transcription downloads the Whisper model into `data/.cache/huggingface`.

**Network exposure.** There is no login by default. Set `BIND_ADDR` to the host's Tailscale IP (`tailscale ip -4`) so the port is not reachable from outside the tailnet. Optional second layer: set `AUTH_TOKEN`; the API then needs `Authorization: Bearer <token>` or `?token=<token>` (EventSource and media URLs cannot send headers). `/`, `/static` and `/healthz` stay public.

**Prebuilt image.** A GitHub Actions workflow publishes `ghcr.io/<owner>/<repo>` (linux/amd64) on pushes to `main` and `v*` tags, tagged `latest`, `sha-<short>` and the semver. To deploy without building, edit the image name in `docker-compose.prod.yml` (or set `CLIPSHORTS_IMAGE`), then:

```bash
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
```

## Run locally (no Docker)

Needs Python 3.12+, `ffmpeg` and `ffprobe`. **ffmpeg must be built with libass** (the `ass` filter) and libx264; many minimal builds (e.g. some Homebrew builds) lack libass, in which case captions cannot be burned in. Check with `ffmpeg -filters | grep -w ass`.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
DATA_DIR=./data .venv/bin/uvicorn clipshorts.main:app --port 8000
```

Keep it at one worker: the job worker and event bus hold in-process state.

## Configuration

Environment variables or `.env` (see `.env.example`):

| Var | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `./data` (`/data` in the image) | Clips, artifacts and the SQLite DB |
| `AUTH_TOKEN` | empty | Optional bearer token; empty = no auth |
| `LANGUAGES` | `de,en` | Caption languages; auto-detect is clamped to this set |
| `WHISPER_MODEL` | `large-v3-turbo` | faster-whisper model (`base`, `small`, `medium`, ...) |
| `WHISPER_COMPUTE_TYPE` | `int8` | CTranslate2 compute type |
| `YUNET_MODEL` | vendored | Path to a YuNet ONNX model |
| `LAYOUT_OVERRIDES` | `{}` | JSON of classifier thresholds (see `LayoutParams` in `clipshorts/layout.py`) |
| `FPS_CAP` | `30` | Output fps cap |
| `VIDEO_ENCODER` | `libx264` | `libx264` or `h264_qsv` (untested, see below) |
| `X264_PRESET` / `X264_CRF` | `veryfast` / `20` | x264 quality settings |
| `RENDER_SPLIT_RATIO` | `0.38` | Stacked layout: facecam share of output height |
| `RENDER_TIMEOUT_FACTOR` / `RENDER_TIMEOUT_MIN_S` | `8` / `120` | Render timeout = factor x clip duration, at least the minimum |
| `FIXED_HASHTAGS` / `MAX_HASHTAGS` | `twitchclips,twitch,gaming,fyp,deutsch,twitchde` / `10` | Hashtags are creator, `<creator>clips`, game + this set (no `#`), capped at the max |
| `NICE_LEVEL` | `10` | Worker niceness |
| `LOG_LEVEL` | `INFO` | Log level |

Compose-only: `BIND_ADDR` (default `127.0.0.1`), `PORT` (`8000`), `PUID`/`PGID` (`1000`, must own `./data`), `TZ`.

## API

```
POST   /api/clips                 {url, language?: auto|de|en, layout?: auto|fullcam|stacked|face|content}
GET    /api/clips[?status=]       list
GET    /api/clips/{id}            status, stage, progress, layout, language, error, output url
DELETE /api/clips/{id}            remove clip + files (409 while queued/running)
POST   /api/clips/{id}/rerender   {layout?}  re-run; cached stages are skipped
GET    /api/clips/{id}/output     MP4, Range supported; ?download=true forces attachment
GET    /api/events                SSE: clip state changes
GET    /healthz                   {"ok": true}

GET    /api/clips/{id}/metadata            title + hashtags (defaults until edited by hand)
PUT    /api/clips/{id}/metadata            {title?, caption?, hashtags?} manual edit
GET    /api/clips/{id}/export              per-platform copy-paste text + download URL
GET    /api/clips/{id}/publications        where the short was published
POST   /api/clips/{id}/publications        {platform, url?, published_at?, views_24h?, rating?, note?}
PATCH  /api/publications/{id}              {url?, views_24h?, rating?: good|ok|flop, note?}
DELETE /api/publications/{id}
GET    /api/report/ratings                 channel x layout x rating counts
```

A clip's post text is its Twitch title plus hashtags `#<creator> #<creator>clips #<game> #twitchclips #twitch #gaming #fyp #deutsch #twitchde` (creator = the streamer from the Twitch metadata, not whoever clipped it; set `FIXED_HASHTAGS` to change the fixed part). No LLM involved. The UI shows this under "Post text and publishing" on every finished clip. Export is manual: download the MP4 and copy the caption. TikTok/YouTube/Instagram upload is not built.

Posting a slug that already exists returns the existing clip (and re-queues it if it had failed).

## Data layout

```
data/
  clipshorts.db                 SQLite (WAL): clips, jobs, renders, settings, clip_metadata, publications
  clips/<id>/
    source.mp4                  downloaded clip
    meta.json                   ffprobe + Twitch metadata
    frames/                     sampled frames
    layout.json                 detected layout, confidence, rects
    transcript.json             words with timestamps, language
    captions.ass                generated captions
    output.mp4                  final 1080x1920 video
```

## Restart and recovery

Jobs live in SQLite. On startup any clip left `running` goes back to `queued` and the worker resumes it; stages whose inputs and params are unchanged are skipped, so a killed container does not re-download or re-transcribe finished work.

## Layouts

| Layout | Used when | Render |
|---|---|---|
| `fullcam` | big face fills the frame | 9:16 crop around the median face position |
| `stacked` | small corner facecam over content | facecam on top (~38%), content below |
| `face` | corner facecam, content barely changes | whole facecam fitted to the width over a blurred copy of itself |
| `content` | no (or too few) faces | whole frame fitted over a blurred background |

The layout is auto-detected; low confidence is shown in the UI. Pick another layout in the dropdown to re-render with it (`auto` returns to detection).

## Known limits

- Layout thresholds are starting values, tuned on very few real clips; expect some misclassification (use the dropdown).
- `fullcam` uses a static crop (no face tracking). The facecam rect for `stacked`/`face` is the face box plus a margin, not the real border.
- A corner facecam (~460 px wide) upscaled to 1080 px stays soft; that is a source limit.
- German speech with English gaming slang can be transcribed phonetically; there is no caption editor yet.
- Twitch changes often: yt-dlp is pinned (`pyproject.toml`); bump it and rebuild when downloads fail.
- No channel sync, ranking, upload or accounts (see `PLAN.md` for later phases).

## Tests

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
```

Pure logic runs without media. The real-ffmpeg render tests use `ffmpeg` from `PATH`, or the binary named in `CLIPSHORTS_TEST_FFMPEG` (useful when the system ffmpeg lacks libass); the caption burn-in tests skip when the chosen binary has no `ass` filter. The real Whisper test skips when the model is not cached.

## Performance

`scripts/benchmark.py` times YuNet on a 640x360 frame, Whisper on audio, and one ffmpeg 9:16 render of a synthetic 1080p clip:

```bash
.venv/bin/python scripts/benchmark.py --yunet clipshorts/models/face_detection_yunet_2023mar.onnx \
    --audio path/to/real_speech.wav --seconds 30
```

Without `--audio` it uses a tone and the Whisper number is only indicative. The Whisper model must be cached locally unless `--download` is passed.

| Machine | YuNet (ms/frame) | Whisper small int8 (30 s audio) | Render (30 s clip, libx264 veryfast) |
|---|---|---|---|
| Apple M4, 10 cores, 24 GB, macOS (Darwin 27.2, arm64), measured 2026-11-16 | 3.7 | 1.2 s on a 30 s tone with no speech (rtf 0.04; not representative, no real speech was used) | 3.3 s (x0.11 realtime) |
| Intel N100 | not measured yet | not measured yet | not measured yet |

The M4 row is a development laptop, not the target host. **To fill the N100 row:** run the command above on the N100 (inside the container: `docker compose exec clipshorts python scripts/benchmark.py ...` after copying the script in, or on the host with the venv), use a real 30 s speech clip for `--audio`, and add an end-to-end timing for a real 30 s clip (URL to `output.mp4`). The provisional target is at most ~3 min for a 30 s clip.

## Quick Sync (optional, untested)

To try hardware encoding on the N100: mount the iGPU into the container (uncomment `devices: - /dev/dri:/dev/dri` in the compose file) and set `VIDEO_ENCODER=h264_qsv`. The Debian ffmpeg in the image may also need VAAPI/QSV driver packages. This path has not been tested; `libx264` is the default and the only verified encoder.
