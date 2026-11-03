"""Task 3: LLM gateway endpoint with a streaming PII guardrail.

    client ──▶ gateway POST /v1/chat/completions ──▶ LLM provider (OpenAI-compatible SSE)
           ◀── redacted SSE, event by event ◀──

Run:
    python -m task3_guardrail.mock_llm        # :9002
    python -m task3_guardrail.gateway         # :8082, UPSTREAM_URL defaults to the mock

Two layers of "split across chunks" are handled:
  1. Transport: one SSE event can arrive across several TCP reads. httpx's
     aiter_lines() reassembles complete lines before we parse them.
  2. Content: one email can be split across several *events*. Each choice index
     gets a StreamRedactor, which holds back only the trailing word-like run
     (see redactor.py).
Nothing accumulates the full response; per-stream memory is bounded by max_hold.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from task3_guardrail.redactor import StreamRedactor, redact

log = logging.getLogger("task3.gateway")


def gateway_error(status: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"type": "gateway_error", "code": code, "message": message, "request_id": request_id}},
        status_code=status,
    )


def _sse(obj: Any) -> str:
    return f"data: {json.dumps(obj)}\n\n"


async def redact_sse_stream(upstream: httpx.Response, request_id: str) -> AsyncIterator[str]:
    redactors: dict[int, StreamRedactor] = {}

    def flush_all() -> list[dict]:
        tails = []
        for index, r in redactors.items():
            tail = r.flush()
            if tail:
                tails.append({"index": index, "delta": {"content": tail}, "finish_reason": None})
        return tails

    last_event: dict = {}
    try:
        async for line in upstream.aiter_lines():
            if not line.startswith("data:"):
                continue  # blank separators, comments (": ping"), event:/id: fields
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                tails = flush_all()
                if tails:
                    yield _sse({**last_event, "choices": tails})
                yield "data: [DONE]\n\n"
                return

            event = json.loads(data)
            last_event = {k: v for k, v in event.items() if k != "choices"}
            out_choices = []
            for choice in event.get("choices", []):
                index = choice.get("index", 0)
                redactor = redactors.setdefault(index, StreamRedactor())
                if not isinstance(choice.get("delta"), dict):
                    choice["delta"] = {}
                delta = choice["delta"]  # mutate in place: the finish event's delta is {}
                had_content = bool(delta.get("content"))
                if had_content:
                    delta["content"] = redactor.feed(delta["content"])
                if choice.get("finish_reason"):
                    # This choice is done: release whatever it was holding.
                    delta["content"] = delta.get("content", "") + redactor.flush()
                # Skip deltas that became empty only because their text is being held.
                if had_content and not delta.get("content") and len(delta) == 1 and not choice.get("finish_reason"):
                    continue
                out_choices.append(choice)
            if out_choices or not event.get("choices"):
                yield _sse({**event, "choices": out_choices})
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        # Upstream died mid-stream. The held tail may be half of an email, and
        # "flushing" it would leak the partial value, so drop it.
        for r in redactors.values():
            r.discard()
        log.error("request_id=%s upstream stream failed: %r", request_id, exc)
        yield _sse({"error": {"type": "gateway_error", "code": "upstream_stream_interrupted",
                              "message": "The upstream model stream was interrupted.", "request_id": request_id}})
        return
    # Upstream closed without [DONE]: still release held text (it's complete, just unterminated).
    tails = flush_all()
    if tails:
        yield _sse({**last_event, "choices": tails})


def create_app(upstream_url: str, upstream_api_key: str | None = None, client: httpx.AsyncClient | None = None) -> Starlette:
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=60.0))

    async def chat_completions(request: Request) -> Response:
        request_id = uuid.uuid4().hex[:16]
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return gateway_error(400, "invalid_json", "Request body must be JSON.", request_id)

        headers = {"content-type": "application/json"}
        if upstream_api_key:
            headers["authorization"] = f"Bearer {upstream_api_key}"
        upstream_req = http.build_request("POST", upstream_url, json=body, headers=headers)
        try:
            upstream = await http.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            log.error("request_id=%s upstream unreachable: %r", request_id, exc)
            return gateway_error(502, "upstream_unavailable", "The model provider is unavailable.", request_id)

        if upstream.status_code != 200:
            await upstream.aread()
            await upstream.aclose()
            log.error("request_id=%s upstream status=%s body=%r", request_id, upstream.status_code, upstream.text[:500])
            return gateway_error(502, "upstream_error", "The model provider returned an error.", request_id)

        if not body.get("stream"):
            await upstream.aread()
            await upstream.aclose()
            data = upstream.json()
            for choice in data.get("choices", []):
                msg = choice.get("message") or {}
                if isinstance(msg.get("content"), str):
                    msg["content"] = redact(msg["content"])
            return JSONResponse(data)

        return StreamingResponse(
            redact_sse_stream(upstream, request_id),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            background=BackgroundTask(upstream.aclose),
        )

    return Starlette(routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"])])


def main() -> None:
    import uvicorn

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    upstream = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:9002/v1/chat/completions")
    app = create_app(upstream, os.environ.get("UPSTREAM_API_KEY"))
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8082")))


if __name__ == "__main__":
    main()
