"""End-to-end tests for the Task 1 MCP server.

The server runs as a real subprocess and we talk raw JSON-RPC over its stdin and
stdout, the way an MCP host would. Testing against the real pipes is the only way
to prove stdout carries nothing but protocol frames.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INVALID_PARAMS = -32602


class StdioClient:
    def __init__(self, argv: list[str]):
        self.proc = subprocess.Popen(
            argv,
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []
        self._responses: queue.Queue[dict] = queue.Queue()
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self._next_id = 0

    def _pump_stdout(self) -> None:
        for line in self.proc.stdout:
            self.stdout_lines.append(line)
            try:
                self._responses.put(json.loads(line))
            except json.JSONDecodeError:
                pass  # recorded in stdout_lines; the purity assertion will catch it

    def _pump_stderr(self) -> None:
        for line in self.proc.stderr:
            self.stderr_lines.append(line)

    def send(self, msg: dict) -> None:
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        msg = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        while True:
            resp = self._responses.get(timeout=15)
            if resp.get("id") == self._next_id:
                return resp

    def initialize(self) -> dict:
        resp = self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        )
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return resp

    def call(self, name: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> None:
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _assert_stdout_is_pure_jsonrpc(client: StdioClient) -> None:
    assert client.stdout_lines, "expected at least one frame on stdout"
    for line in client.stdout_lines:
        frame = json.loads(line)  # raises if anything non-JSON leaked onto stdout
        assert frame.get("jsonrpc") == "2.0", line


@pytest.fixture(scope="module")
def client():
    c = StdioClient([sys.executable, "-m", "task1_mcp_server.server"])
    init = c.initialize()
    assert "result" in init, init
    yield c
    c.close()


def test_tools_list_exposes_both_tools_with_schemas(client):
    tools = {t["name"]: t for t in client.request("tools/list")["result"]["tools"]}
    assert set(tools) == {"get_customer_record", "trigger_refund"}
    refund_schema = tools["trigger_refund"]["inputSchema"]
    assert refund_schema["properties"]["reason"]["minLength"] == 10
    assert refund_schema["properties"]["amount"]["exclusiveMinimum"] == 0
    assert refund_schema["additionalProperties"] is False


def test_get_customer_record_happy_path(client):
    resp = client.call("get_customer_record", {"customer_id": "CUST-00042"})
    result = resp["result"]
    assert result.get("isError") in (None, False)
    assert result["structuredContent"]["name"] == "Alan Turing"


def test_unknown_customer_is_tool_error_not_protocol_error(client):
    resp = client.call("get_customer_record", {"customer_id": "CUST-99999"})
    assert "error" not in resp
    assert resp["result"]["isError"] is True


def test_trigger_refund_happy_path(client):
    resp = client.call(
        "trigger_refund",
        {"customer_id": "CUST-00001", "amount": 25.5, "reason": "Double charged in October"},
    )
    refund = resp["result"]["structuredContent"]
    assert refund["status"] == "queued" and refund["amount"] == 25.5


@pytest.mark.parametrize(
    "tool,arguments,bad_field",
    [
        ("get_customer_record", {"customer_id": "CUST-1234"}, "customer_id"),  # 4 digits
        ("get_customer_record", {"customer_id": "cust-12345"}, "customer_id"),  # lowercase
        ("get_customer_record", {"customer_id": "CUST-12345 "}, "customer_id"),  # trailing space
        ("get_customer_record", {"customer_id": "CUST-ABCDE"}, "customer_id"),
        ("get_customer_record", {"customer_id": 12345}, "customer_id"),  # wrong type
        ("get_customer_record", {}, "customer_id"),  # missing
        ("get_customer_record", {"customer_id": "CUST-00001", "admin": True}, "admin"),  # extra field
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 0, "reason": "valid reason here"}, "amount"),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": -5.0, "reason": "valid reason here"}, "amount"),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": "10.0", "reason": "valid reason here"}, "amount"),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": True, "reason": "valid reason here"}, "amount"),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 1e9, "reason": "valid reason here"}, "amount"),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 10.0, "reason": "too short"}, "reason"),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 10.0, "reason": "          "}, "reason"),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 10.0}, "reason"),
    ],
)
def test_invalid_arguments_return_jsonrpc_invalid_params(client, tool, arguments, bad_field):
    resp = client.call(tool, arguments)
    assert "result" not in resp, resp
    assert resp["error"]["code"] == INVALID_PARAMS
    assert any(d["field"] == bad_field for d in resp["error"]["data"]), resp["error"]


def test_nan_amount_rejected(client):
    # Python's json emits NaN as a bare token. It isn't valid JSON, but a sloppy
    # client could send it, and it must not slip past "amount > 0".
    client.proc.stdin.write(
        '{"jsonrpc":"2.0","id":9001,"method":"tools/call","params":{"name":"trigger_refund",'
        '"arguments":{"customer_id":"CUST-00001","amount":NaN,"reason":"valid reason here"}}}\n'
    )
    client.proc.stdin.flush()
    while True:
        resp = client._responses.get(timeout=15)
        if resp.get("id") == 9001:
            break
    assert "result" not in resp or resp["result"].get("isError"), resp


def test_unknown_tool_is_invalid_params(client):
    resp = client.call("delete_everything", {})
    assert resp["error"]["code"] == INVALID_PARAMS


def test_unknown_method_is_method_not_found(client):
    assert client.request("does/not/exist")["error"]["code"] == -32601


def test_stdout_is_pure_jsonrpc_and_logs_go_to_stderr(client):
    client.call("get_customer_record", {"customer_id": "CUST-00001"})
    _assert_stdout_is_pure_jsonrpc(client)
    assert any("calling tool" in line for line in client.stderr_lines)


def test_stray_print_in_a_tool_cannot_corrupt_stdout():
    # Patch a tool to print() mid-request, the classic way stdio MCP servers break.
    # The print must end up on stderr and the response must still parse.
    script = (
        "import task1_mcp_server.server as s\n"
        "model, impl, desc = s.TOOLS['get_customer_record']\n"
        "def noisy(args):\n"
        "    print('STRAY DEBUG PRINT')\n"
        "    return impl(args)\n"
        "s.TOOLS['get_customer_record'] = (model, noisy, desc)\n"
        "s.main()\n"
    )
    c = StdioClient([sys.executable, "-c", script])
    try:
        c.initialize()
        resp = c.call("get_customer_record", {"customer_id": "CUST-00001"})
        assert resp["result"]["structuredContent"]["customer_id"] == "CUST-00001"
    finally:
        c.close()
    _assert_stdout_is_pure_jsonrpc(c)
    assert any("STRAY DEBUG PRINT" in line for line in c.stderr_lines)
