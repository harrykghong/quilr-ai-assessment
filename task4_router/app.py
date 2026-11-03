"""Task 4: LLM gateway endpoint = auth -> token rate limit -> primary/backup routing.

Run:
    python -m task4_router.mock_providers     # :9003 (primary + backup, switchable modes)
    python -m task4_router.app                # :8084, SQLite file ./gateway.db

Every error the client sees has the same shape and never includes upstream
bodies, stack traces, hostnames or exception names:

    {"error": {"type": "gateway_error", "code": "<stable code>",
               "message": "<generic text>", "request_id": "<id>"}}

The request_id ties the response to the full detail in the gateway's stderr logs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from task4_router.rate_limiter import SlidingWindowRateLimiter, estimate_tokens
from task4_router.router import ModelRouter, Provider, UpstreamError

log = logging.getLogger("task4.app")

ERRORS = {
    # code: (http status, client-facing message)
    "unauthorized": (401, "Missing or invalid API key."),
    "invalid_request": (400, "The request body is not a valid chat completion request."),
    "rate_limited": (429, "Token rate limit exceeded for this API key."),
    "request_too_large": (413, "The request exceeds the per-minute token limit on its own."),
    "upstream_rejected_request": (400, "The model provider rejected the request."),
    "upstream_unavailable": (503, "All model providers are currently unavailable. Please retry."),
    "internal_error": (500, "Internal gateway error."),
}


def gateway_error(code: str, request_id: str, headers: dict[str, str] | None = None) -> JSONResponse:
    status, message = ERRORS[code]
    return JSONResponse(
        {"error": {"type": "gateway_error", "code": code, "message": message, "request_id": request_id}},
        status_code=status,
        headers=headers,
    )


def create_app(
    db_path: str,
    primary: Provider,
    backup: Provider,
    tenant_keys: set[str],
    limit_tokens: int = 50_000,
    primary_timeout_s: float = 3.0,
    client: httpx.AsyncClient | None = None,
    limiter: SlidingWindowRateLimiter | None = None,
) -> Starlette:
    limiter = limiter or SlidingWindowRateLimiter(db_path, limit_tokens=limit_tokens)
    router = ModelRouter(primary, backup, client or httpx.AsyncClient(), primary_timeout_s=primary_timeout_s)

    async def chat_completions(request: Request) -> Response:
        request_id = uuid.uuid4().hex[:16]
        request.state.request_id = request_id

        scheme, _, api_key = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or api_key not in tenant_keys:
            return gateway_error("unauthorized", request_id)

        try:
            body = await request.json()
        except json.JSONDecodeError:
            return gateway_error("invalid_request", request_id)
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or body.get("stream"):
            # Streaming is Task 3's path; this router handles complete responses.
            return gateway_error("invalid_request", request_id)

        # SQLite I/O runs off the event loop; atomicity comes from SQLite's write lock.
        estimate = estimate_tokens(body)
        decision = await asyncio.to_thread(limiter.acquire, api_key, estimate)
        rate_headers = {"x-ratelimit-limit-tokens": str(decision.limit), "x-ratelimit-remaining-tokens": str(decision.remaining)}
        if not decision.allowed:
            if decision.retry_after_s is None:
                return gateway_error("request_too_large", request_id, rate_headers)
            retry = max(1, int(-(-decision.retry_after_s // 1)))  # round up to whole seconds
            return gateway_error("rate_limited", request_id, {**rate_headers, "retry-after": str(retry)})

        try:
            routed = await router.complete(body, request_id)
        except UpstreamError as exc:
            log.error("request_id=%s upstream failure kind=%s detail=%s", request_id, exc.kind, exc.detail)
            # Nothing was generated, so the reservation is released.
            await asyncio.to_thread(limiter.reconcile, decision.reservation_id, 0)
            return gateway_error("upstream_rejected_request" if exc.kind == "client_error" else "upstream_unavailable", request_id)

        actual = (routed.body.get("usage") or {}).get("total_tokens")
        if isinstance(actual, int):
            await asyncio.to_thread(limiter.reconcile, decision.reservation_id, actual)

        headers = {**rate_headers, "x-gateway-provider": routed.provider, "x-request-id": request_id}
        if routed.failover_reason:
            headers["x-gateway-failover"] = "true"
        return JSONResponse(routed.body, headers=headers)

    async def unhandled(request: Request, exc: Exception) -> Response:
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:16])
        log.exception("request_id=%s unhandled error", request_id, exc_info=exc)
        return gateway_error("internal_error", request_id)

    return Starlette(
        routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"])],
        exception_handlers={Exception: unhandled},
    )


def main() -> None:
    import uvicorn

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mock = os.environ.get("MOCK_PROVIDERS_URL", "http://127.0.0.1:9003")
    app = create_app(
        db_path=os.environ.get("GATEWAY_DB", "gateway.db"),
        primary=Provider("primary", f"{mock}/primary/v1/chat/completions", "primary-model"),
        backup=Provider("backup", f"{mock}/backup/v1/chat/completions", "backup-model"),
        tenant_keys=set(os.environ.get("TENANT_KEYS", "tenant-a-key,tenant-b-key").split(",")),
    )
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8084")))


if __name__ == "__main__":
    main()
