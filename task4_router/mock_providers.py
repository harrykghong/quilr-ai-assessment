"""Two mock OpenAI-compatible providers ("primary", "backup") with switchable failure modes.

    POST /{provider}/v1/chat/completions
    POST /{provider}/mode   {"mode": "ok" | "429" | "slow" | "500" | "400"}

Modes: ok -> 200 with usage; 429 -> rate limited; slow -> sleeps slow_seconds
(default 5) before answering; 500 -> error body with a fake stack trace, so tests
can prove the gateway doesn't leak it; 400 -> a client error (must NOT fail over).

Run: python -m task4_router.mock_providers   (listens on :9003)
"""

from __future__ import annotations

import asyncio
import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

FAKE_TRACE = (
    'Traceback (most recent call last):\n  File "/srv/internal/inference/pool.py", line 88, in dispatch\n'
    "    conn = self._pool.acquire(host='10.0.3.17')\nRuntimeError: GPU pool exhausted"
)


def create_mock_providers(slow_seconds: float = 5.0) -> Starlette:
    async def completions(request: Request):
        name = request.path_params["provider"]
        state = request.app.state
        state.calls[name] = state.calls.get(name, 0) + 1
        body = await request.json()
        mode = state.modes.get(name, "ok")
        if mode == "429":
            return JSONResponse({"error": {"message": "Rate limit reached for requests"}}, status_code=429, headers={"retry-after": "20"})
        if mode == "500":
            return PlainTextResponse(FAKE_TRACE, status_code=500)
        if mode == "400":
            return JSONResponse({"error": {"message": "context_length_exceeded"}}, status_code=400)
        if mode == "slow":
            await asyncio.sleep(state.slow_seconds)
        prompt_chars = sum(len(m.get("content", "")) for m in body.get("messages", []))
        prompt_tokens, completion_tokens = max(1, prompt_chars // 4), 12
        return JSONResponse(
            {
                "id": f"cmpl-{name}",
                "object": "chat.completion",
                "model": body.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": f"Hello from {name}."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
            }
        )

    async def set_mode(request: Request):
        name = request.path_params["provider"]
        request.app.state.modes[name] = (await request.json())["mode"]
        return JSONResponse({"provider": name, "mode": request.app.state.modes[name]})

    app = Starlette(
        routes=[
            Route("/{provider}/v1/chat/completions", completions, methods=["POST"]),
            Route("/{provider}/mode", set_mode, methods=["POST"]),
        ]
    )
    app.state.modes = {"primary": "ok", "backup": "ok"}
    app.state.calls = {}
    app.state.slow_seconds = slow_seconds
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_mock_providers(), host="127.0.0.1", port=int(os.environ.get("PORT", "9003")))
