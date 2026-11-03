"""Task 2: MCP security gateway, an HTTP/JSON-RPC reverse proxy with tool-level authz.

    agent ──HTTP──▶ gateway (:8080/mcp) ──HTTP──▶ downstream MCP server

Run:
    python -m task2_gateway.mock_downstream            # :9001
    python -m task2_gateway.gateway                    # :8080, forwards to :9001
    python -m task2_gateway.make_token viewer          # prints a bearer token

Request pipeline (each step either rejects or hands off to the next):
    1. authenticate   Bearer JWT (HS256) -> role claim in {admin, viewer}   else HTTP 401
    2. parse          single JSON-RPC 2.0 object, no duplicate keys         else -32700 / -32600
    3. authorize      tools/call + params.name starts with admin_ + role != admin
                      -> -32001 "Unauthorized Tool Call", downstream never called
    4. forward        original body bytes, streamed back (JSON or SSE)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import httpx
import jwt
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

log = logging.getLogger("task2.gateway")

UNAUTHORIZED_TOOL_CALL = -32001
UNAUTHENTICATED = -32000
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

ROLES = {"admin", "viewer"}
ADMIN_TOOL_PREFIX = "admin_"
MAX_BODY_BYTES = 1_000_000

# Request headers worth passing downstream. Authorization is NOT forwarded: the
# gateway is the trust boundary and the user's token stays with it.
FORWARD_REQUEST_HEADERS = {"content-type", "accept", "mcp-session-id", "mcp-protocol-version", "last-event-id"}
FORWARD_RESPONSE_HEADERS = {"content-type", "mcp-session-id", "cache-control"}


class AuthError(Exception):
    pass


class DuplicateKeyError(ValueError):
    pass


def resolve_role(authorization: str | None, secret: str) -> str:
    """Bearer <JWT> -> role. Signature, exp, and the role claim are all required."""
    if not authorization:
        raise AuthError("missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("expected 'Authorization: Bearer <token>'")
    try:
        # algorithms is pinned: never let the token's header choose (alg=none attacks).
        claims = jwt.decode(token.strip(), secret, algorithms=["HS256"], options={"require": ["exp", "role"]})
    except jwt.PyJWTError as exc:
        raise AuthError(f"invalid token: {exc}") from None
    role = claims["role"]
    if role not in ROLES:
        raise AuthError(f"unknown role {role!r}")
    return role


def is_admin_tool(name: str) -> bool:
    # Case-insensitive on purpose (fail closed): if any downstream ever normalizes
    # tool names, "Admin_Reset_Key" must not slip past an exact-case check.
    return name.strip().lower().startswith(ADMIN_TOOL_PREFIX)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # {"name": "get_x", "name": "admin_reset_key"}: Python keeps the last value,
    # other parsers keep the first. If the gateway and downstream disagree, the
    # gateway authorizes one tool while the downstream runs another. Rejecting
    # duplicates removes that parser-differential bypass.
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise DuplicateKeyError("duplicate keys in JSON object")
    return dict(pairs)


def jsonrpc_error(req_id: Any, code: int, message: str, http_status: int = 200, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}},
        status_code=http_status,
        headers=headers,
    )


def create_app(downstream_url: str, jwt_secret: str, client: httpx.AsyncClient | None = None) -> Starlette:
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=60.0))

    async def mcp_endpoint(request: Request) -> Response:
        # 1. authenticate
        try:
            role = resolve_role(request.headers.get("authorization"), jwt_secret)
        except AuthError as exc:
            log.warning("auth rejected: %s", exc)
            return jsonrpc_error(
                None, UNAUTHENTICATED, "Authentication required", 401, {"WWW-Authenticate": 'Bearer realm="mcp-gateway"'}
            )

        # GET (SSE notification stream) / DELETE (end session) carry no JSON-RPC
        # body to inspect; they pass through once the caller is authenticated.
        if request.method != "POST":
            return await forward(request, body=None)

        # 2. parse
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return jsonrpc_error(None, INVALID_REQUEST, "Request body too large", 413)
        try:
            msg = json.loads(body, object_pairs_hook=_reject_duplicate_keys)
        except DuplicateKeyError:
            return jsonrpc_error(None, INVALID_REQUEST, "Invalid Request: duplicate JSON keys", 400)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return jsonrpc_error(None, PARSE_ERROR, "Parse error", 400)

        if isinstance(msg, list):
            # JSON-RPC batches were removed from MCP in 2025-06-18. Allowing them
            # would mean authorizing each element; a gateway that only checked
            # msg[0] would let an admin_ call ride along in msg[1].
            return jsonrpc_error(None, INVALID_REQUEST, "Invalid Request: batch requests are not supported", 400)
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            req_id = msg.get("id") if isinstance(msg, dict) else None
            return jsonrpc_error(req_id, INVALID_REQUEST, "Invalid Request", 400)

        req_id = msg.get("id")
        method = msg["method"]

        # 3. authorize
        if method == "tools/call":
            params = msg.get("params")
            name = params.get("name") if isinstance(params, dict) else None
            if not isinstance(name, str):
                return jsonrpc_error(req_id, INVALID_PARAMS, "Invalid params: tools/call requires params.name (string)")
            if is_admin_tool(name) and role != "admin":
                log.warning("DENY role=%s method=tools/call tool=%s", role, name)
                return jsonrpc_error(req_id, UNAUTHORIZED_TOOL_CALL, "Unauthorized Tool Call")
            log.info("ALLOW role=%s method=tools/call tool=%s", role, name)
        else:
            # tools/list, initialize, notifications, ... are forwarded transparently.
            log.info("FORWARD role=%s method=%s", role, method)

        # 4. forward
        return await forward(request, body=body, req_id=req_id)

    async def forward(request: Request, body: bytes | None, req_id: Any = None) -> Response:
        headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_REQUEST_HEADERS}
        upstream_req = http.build_request(request.method, downstream_url, headers=headers, content=body)
        try:
            upstream = await http.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            log.error("downstream unreachable: %r", exc)
            return jsonrpc_error(req_id, INTERNAL_ERROR, "Downstream MCP server unavailable", 502)
        # Stream the body through instead of buffering it, so SSE responses
        # (Streamable HTTP) reach the client as the downstream produces them.
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers={k: v for k, v in upstream.headers.items() if k.lower() in FORWARD_RESPONSE_HEADERS},
            background=BackgroundTask(upstream.aclose),
        )

    return Starlette(routes=[Route("/mcp", mcp_endpoint, methods=["GET", "POST", "DELETE"])])


DEV_SECRET = "dev-only-secret-change-me-in-production!!"


def jwt_secret_from_env() -> str:
    return os.environ.get("GATEWAY_JWT_SECRET", DEV_SECRET)


def main() -> None:
    import uvicorn

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if jwt_secret_from_env() == DEV_SECRET:
        log.warning("GATEWAY_JWT_SECRET not set; using the built-in dev secret")
    downstream = os.environ.get("DOWNSTREAM_URL", "http://127.0.0.1:9001/mcp")
    app = create_app(downstream, jwt_secret_from_env())
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8080")))


if __name__ == "__main__":
    main()
