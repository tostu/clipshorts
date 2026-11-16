"""In-process event bus feeding SSE. Publishing is safe from any thread."""

import asyncio
import contextlib
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subs: set[asyncio.Queue[dict[str, Any]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subs.discard(q)

    def _deliver(self, event: dict[str, Any]) -> None:
        for q in list(self._subs):
            with contextlib.suppress(asyncio.QueueFull):  # slow consumer: drop
                q.put_nowait(event)

    def publish(self, type_: str, data: dict[str, Any] | None = None) -> None:
        event = {"type": type_, "data": data or {}}
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None or self._loop is None:
            self._deliver(event)
        elif not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._deliver, event)
