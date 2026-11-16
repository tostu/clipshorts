"""FastAPI app: lifespan wires DB + event bus + worker."""

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from .api import router
from .config import Config, get_config
from .db import DB
from .events import EventBus
from .media import Tools, default_tools
from .worker import Worker


class AuthMiddleware:
    """Optional bearer token (header `Authorization: Bearer <t>` or `?token=<t>` for
    EventSource, which cannot set headers). /healthz is always open.

    Pure ASGI (not BaseHTTPMiddleware) so SSE streaming is untouched."""

    def __init__(self, app, token: str = ""):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.token or scope["path"] == "/healthz":
            return await self.app(scope, receive, send)
        request = Request(scope)
        auth = request.headers.get("authorization", "")
        candidates = [request.query_params.get("token")]
        if auth.lower().startswith("bearer "):
            candidates.append(auth[7:].strip())
        if any(c and hmac.compare_digest(c.encode(), self.token.encode()) for c in candidates):
            return await self.app(scope, receive, send)
        resp = PlainTextResponse(
            "unauthorized", status_code=401, headers={"WWW-Authenticate": "Bearer"}
        )
        return await resp(scope, receive, send)


def create_app(cfg: Config | None = None, tools: Tools | None = None) -> FastAPI:
    cfg = cfg or get_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=cfg.log_level.upper(),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        cfg.clips_dir.mkdir(parents=True, exist_ok=True)
        db = DB(cfg.db_path)
        bus = EventBus()
        app.state.cfg, app.state.db, app.state.bus = cfg, db, bus
        worker = Worker(cfg, db, bus, tools or default_tools(cfg))
        app.state.worker = worker
        if cfg.worker_enabled:
            await worker.start()
        else:
            bus.bind_loop(asyncio.get_running_loop())
            db.recover()
        try:
            yield
        finally:
            await worker.shutdown()
            db.close()

    app = FastAPI(title="clipshorts", lifespan=lifespan)
    app.add_middleware(AuthMiddleware, token=cfg.auth_token)
    app.include_router(router)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app


app = create_app()
