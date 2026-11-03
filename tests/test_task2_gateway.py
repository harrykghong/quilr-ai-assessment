from __future__ import annotations

import json
import time

import httpx
import jwt
import pytest

from task2_gateway.gateway import create_app
from task2_gateway.make_token import make_token
from task2_gateway.mock_downstream import create_mock_downstream

pytestmark = pytest.mark.anyio

SECRET = "test-secret-at-least-32-bytes-long!!"


@pytest.fixture
def downstream():
    return create_mock_downstream()


@pytest.fixture
async def gw(downstream):
    downstream_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=downstream))
    app = create_app("http://downstream/mcp", SECRET, client=downstream_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        yield c
    await downstream_client.aclose()


def auth(role: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {make_token(role, SECRET)}"}


def rpc(method: str, params: dict | None = None, req_id: int = 1) -> dict:
    msg = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


async def test_tools_list_passes_through_for_viewer(gw, downstream):
    r = await gw.post("/mcp", json=rpc("tools/list"), headers=auth("viewer"))
    assert r.status_code == 200
    names = [t["name"] for t in r.json()["result"]["tools"]]
    assert names == ["get_customer_record", "admin_reset_key"]
    assert downstream.state.received[0]["message"] == rpc("tools/list")


async def test_viewer_admin_tool_call_is_blocked_with_32001_and_never_forwarded(gw, downstream):
    r = await gw.post("/mcp", json=rpc("tools/call", {"name": "admin_reset_key", "arguments": {}}, req_id=7), headers=auth("viewer"))
    assert r.status_code == 200
    assert r.json() == {"jsonrpc": "2.0", "id": 7, "error": {"code": -32001, "message": "Unauthorized Tool Call"}}
    assert downstream.state.received == []


@pytest.mark.parametrize("name", ["Admin_reset_key", "ADMIN_RESET_KEY", " admin_reset_key"])
async def test_admin_prefix_check_is_case_and_whitespace_insensitive(gw, downstream, name):
    r = await gw.post("/mcp", json=rpc("tools/call", {"name": name}), headers=auth("viewer"))
    assert r.json()["error"]["code"] == -32001
    assert downstream.state.received == []


async def test_admin_can_call_admin_tool(gw, downstream):
    r = await gw.post("/mcp", json=rpc("tools/call", {"name": "admin_reset_key", "arguments": {}}), headers=auth("admin"))
    assert r.json()["result"]["content"][0]["text"] == "executed admin_reset_key"
    assert len(downstream.state.received) == 1


async def test_viewer_can_call_non_admin_tool(gw, downstream):
    r = await gw.post("/mcp", json=rpc("tools/call", {"name": "get_customer_record"}), headers=auth("viewer"))
    assert r.json()["result"]["isError"] is False


async def test_user_token_is_not_forwarded_downstream(gw, downstream):
    await gw.post("/mcp", json=rpc("tools/list"), headers=auth("admin"))
    assert "authorization" not in downstream.state.received[0]["headers"]


async def test_original_body_bytes_are_forwarded_verbatim(gw, downstream):
    raw = b'{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{"cursor":null}}'
    await gw.post("/mcp", content=raw, headers={**auth("viewer"), "content-type": "application/json"})
    assert downstream.state.received[0]["message"] == json.loads(raw)


async def test_notifications_are_forwarded(gw, downstream):
    r = await gw.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=auth("viewer"))
    assert r.status_code == 202
    assert len(downstream.state.received) == 1


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Basic Zm9vOmJhcg=="},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer not-a-jwt"},
        {"Authorization": f"Bearer {make_token("admin", "a-different-secret-also-32-bytes-long")}"},
        {"Authorization": f"Bearer {make_token('admin', SECRET, ttl_seconds=-10)}"},  # expired
        {"Authorization": f"Bearer {make_token('superuser', SECRET)}"},  # unknown role
        {"Authorization": "Bearer " + jwt.encode({"role": "admin", "exp": int(time.time()) + 60}, None, algorithm="none")},
    ],
)
async def test_bad_credentials_get_401_and_nothing_is_forwarded(gw, downstream, headers):
    r = await gw.post("/mcp", json=rpc("tools/list"), headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Bearer")
    assert r.json()["error"]["code"] == -32000
    assert downstream.state.received == []


async def test_batch_request_rejected_so_admin_call_cannot_hide_in_it(gw, downstream):
    batch = [rpc("tools/list", req_id=1), rpc("tools/call", {"name": "admin_reset_key"}, req_id=2)]
    r = await gw.post("/mcp", json=batch, headers=auth("viewer"))
    assert r.status_code == 400 and r.json()["error"]["code"] == -32600
    assert downstream.state.received == []


async def test_duplicate_name_key_rejected(gw, downstream):
    # Parser-differential smuggling: one parser sees get_customer_record, another sees admin_reset_key.
    raw = b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key","name":"get_customer_record"}}'
    r = await gw.post("/mcp", content=raw, headers={**auth("viewer"), "content-type": "application/json"})
    assert r.json()["error"]["code"] == -32600
    assert downstream.state.received == []


@pytest.mark.parametrize(
    "body,code",
    [
        (b"{not json", -32700),
        (b'"just a string"', -32600),
        (b'{"id":1,"method":"tools/list"}', -32600),  # missing jsonrpc
        (b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":123}}', -32602),
        (b'{"jsonrpc":"2.0","id":1,"method":"tools/call"}', -32602),
    ],
)
async def test_malformed_jsonrpc(gw, downstream, body, code):
    r = await gw.post("/mcp", content=body, headers={**auth("admin"), "content-type": "application/json"})
    assert r.json()["error"]["code"] == code
    assert downstream.state.received == []


async def test_downstream_down_returns_sanitized_502():
    app = create_app("http://127.0.0.1:1/mcp", SECRET)  # nothing listens on port 1
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        r = await c.post("/mcp", json=rpc("tools/list", req_id=5), headers=auth("viewer"))
    assert r.status_code == 502
    assert r.json() == {"jsonrpc": "2.0", "id": 5, "error": {"code": -32603, "message": "Downstream MCP server unavailable"}}
