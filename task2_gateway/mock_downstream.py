"""Mock downstream MCP server for Task 2: plain JSON-RPC over HTTP POST.

It records every message it receives in app.state.received, so tests can prove
the gateway never forwarded a blocked call.

Run: python -m task2_gateway.mock_downstream   (listens on :9001)
"""

from __future__ import annotations

import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

TOOLS = [
    {"name": "get_customer_record", "description": "Read a customer record", "inputSchema": {"type": "object"}},
    {"name": "admin_reset_key", "description": "Rotate a tenant API key (admin only)", "inputSchema": {"type": "object"}},
]


def create_mock_downstream() -> Starlette:
    async def mcp(request: Request) -> Response:
        msg = await request.json()
        request.app.state.received.append({"message": msg, "headers": dict(request.headers)})
        method, req_id = msg.get("method"), msg.get("id")
        if req_id is None:  # notification
            return Response(status_code=202)
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-downstream", "version": "0"},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            name = msg["params"]["name"]
            result = {"content": [{"type": "text", "text": f"executed {name}"}], "isError": False}
        else:
            return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}})
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": result})

    app = Starlette(routes=[Route("/mcp", mcp, methods=["POST"])])
    app.state.received = []
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_mock_downstream(), host="127.0.0.1", port=int(os.environ.get("PORT", "9001")))
