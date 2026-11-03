from __future__ import annotations

import contextlib
import threading
import time

import pytest
import uvicorn


@pytest.fixture
def anyio_backend():
    return "asyncio"


@contextlib.contextmanager
def serve(app):
    """Run an ASGI app on a real socket in a background thread; yields its base URL.

    httpx's ASGITransport buffers whole responses, so tests that check streaming
    timing (TTFT) need a real server.
    """
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("server did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
