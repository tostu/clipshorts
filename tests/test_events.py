import asyncio
import threading

from clipshorts.api import sse_stream
from clipshorts.events import EventBus


async def test_publish_subscribe_unsubscribe():
    bus = EventBus()
    q1, q2 = bus.subscribe(), bus.subscribe()
    bus.publish("clip", {"id": 1})
    assert q1.get_nowait() == {"type": "clip", "data": {"id": 1}}
    assert q2.get_nowait()["data"] == {"id": 1}
    bus.unsubscribe(q2)
    bus.publish("clip", {"id": 2})
    assert q1.get_nowait()["data"] == {"id": 2}
    assert q2.empty()


async def test_slow_consumer_does_not_block():
    bus = EventBus()
    q = bus.subscribe()
    for i in range(500):
        bus.publish("clip", {"i": i})  # never raises
    assert q.qsize() == 100


async def test_publish_from_thread():
    bus = EventBus()
    bus.bind_loop(asyncio.get_running_loop())
    q = bus.subscribe()
    t = threading.Thread(target=lambda: bus.publish("clip", {"id": 9}))
    t.start()
    t.join()
    ev = await asyncio.wait_for(q.get(), 2)
    assert ev["data"] == {"id": 9}


async def test_sse_stream_frames_and_cleanup():
    bus = EventBus()
    state = {"n": 0}

    async def disconnected() -> bool:
        state["n"] += 1
        return state["n"] > 3

    out = []
    gen = sse_stream(bus, disconnected, heartbeat=0.05)
    out.append(await gen.__anext__())  # ": connected"
    bus.publish("clip", {"id": 5})
    async for frame in gen:
        out.append(frame)
    assert out[0].startswith(":")
    assert 'event: clip\ndata: {"id": 5}\n\n' in out
    assert any(f.startswith(": keepalive") for f in out)
    assert not bus._subs
