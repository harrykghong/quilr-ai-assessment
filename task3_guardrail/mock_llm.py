"""Mock OpenAI-compatible LLM provider that streams a reply in small, uneven chunks.

POST /v1/chat/completions
  * If the last user message starts with "echo:", the rest of it is the reply
    (tests use this to control exactly what gets streamed). Otherwise a canned
    reply containing an email, an SSN and a card number is used.
  * Extra request fields (mock only): mock_chunk_size (int, default random 1-6),
    mock_delay_ms (default 20), mock_fail_after (int: drop the connection after
    N chunks, to simulate an upstream crash mid-stream).

Run: python -m task3_guardrail.mock_llm   (listens on :9002)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

CANNED_REPLY = (
    "Sure! I found the customer. Their email is jane.doe@example.com and the SSN on file is "
    "123-45-6789. The card ending the transaction was 4111 1111 1111 1111, charged on order 20261103. "
    "Let me know if you need anything else."
)


def split_uneven(text: str, chunk_size: int | None, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    chunks, i = [], 0
    while i < len(text):
        n = chunk_size or rng.randint(1, 6)
        chunks.append(text[i : i + n])
        i += n
    return chunks


def _chunk_event(completion_id: str, model: str, delta: dict, finish_reason: str | None = None) -> str:
    event = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(event)}\n\n"


async def chat_completions(request: Request):
    body = await request.json()
    last_user = next((m["content"] for m in reversed(body.get("messages", [])) if m.get("role") == "user"), "")
    reply = last_user[len("echo:") :] if last_user.startswith("echo:") else CANNED_REPLY
    model = body.get("model", "mock-gpt")
    completion_id = "chatcmpl-mock"

    if not body.get("stream"):
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            }
        )

    chunks = split_uneven(reply, body.get("mock_chunk_size"))
    delay = body.get("mock_delay_ms", 20) / 1000
    fail_after = body.get("mock_fail_after")

    async def events():
        yield _chunk_event(completion_id, model, {"role": "assistant", "content": ""})
        for n, piece in enumerate(chunks):
            if fail_after is not None and n >= fail_after:
                raise ConnectionError("mock upstream crashed mid-stream")
            await asyncio.sleep(delay)
            yield _chunk_event(completion_id, model, {"content": piece})
        yield _chunk_event(completion_id, model, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def create_mock_llm() -> Starlette:
    return Starlette(routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"])])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_mock_llm(), host="127.0.0.1", port=int(os.environ.get("PORT", "9002")))
