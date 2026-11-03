"""Task 1: MCP server with strict input validation over stdio.

Run:  python -m task1_mcp_server.server

stdout carries only JSON-RPC frames. Everything else (logs, debug output, even a
stray print()) goes to stderr:
  * logging is configured with stream=sys.stderr before anything else runs;
  * run() hands the original stdout buffer to the transport as a private "wire",
    then rebinds sys.stdout to sys.stderr, so a stray print() lands on stderr.
    (Relying on the SDK's own fd-1 diversion alone wasn't enough: a print()
    sitting in Python's block buffer got flushed to the real stdout at exit.
    The stray-print test caught it on Windows.)
  tests/test_task1_mcp_server.py checks both properties end to end.
"""

from __future__ import annotations

import io
import logging
import sys
import uuid
from typing import Annotated, Any

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("task1.mcp_server")


# ---------------------------------------------------------------------------
# Input schemas. strict=True means no type coercion ("12.5" is not a float,
# true is not a number); extra="forbid" rejects unknown argument names.
# ---------------------------------------------------------------------------

CustomerId = Annotated[
    str,
    Field(
        pattern=r"^CUST-\d{5}$",
        description="Customer id formatted as CUST-XXXXX (5 digits), e.g. CUST-00042",
    ),
]


class GetCustomerRecordInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    customer_id: CustomerId


class TriggerRefundInput(BaseModel):
    # str_strip_whitespace: "          " (10 spaces) must not satisfy min_length=10.
    model_config = ConfigDict(strict=True, extra="forbid", str_strip_whitespace=True)

    customer_id: CustomerId
    # allow_inf_nan=False: JSON can't carry NaN/Infinity, but Python's json module
    # accepts them, so reject them explicitly. Upper bound keeps a typo from
    # becoming a $1e12 refund.
    amount: float = Field(gt=0, le=10_000, allow_inf_nan=False, description="Refund amount in USD, > 0")
    reason: str = Field(min_length=10, max_length=500, description="Why the refund is issued (10-500 chars)")


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-00001": {"name": "Ada Lovelace", "tier": "enterprise", "balance": 1250.00},
    "CUST-00042": {"name": "Alan Turing", "tier": "pro", "balance": 89.99},
    "CUST-12345": {"name": "Grace Hopper", "tier": "free", "balance": 0.00},
}


def get_customer_record(args: GetCustomerRecordInput) -> types.CallToolResult:
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        # Well-formed input but no such customer: a tool *execution* error
        # (isError=true), not a protocol error. The model can read it and recover.
        return _tool_error(f"No customer found with id {args.customer_id}")
    data = {"customer_id": args.customer_id, **record}
    return types.CallToolResult(content=[types.TextContent(text=_fmt(data))], structuredContent=data)


def trigger_refund(args: TriggerRefundInput) -> types.CallToolResult:
    if args.customer_id not in CUSTOMERS:
        return _tool_error(f"No customer found with id {args.customer_id}")
    refund = {
        "refund_id": f"RF-{uuid.uuid4().hex[:10].upper()}",
        "customer_id": args.customer_id,
        "amount": round(args.amount, 2),
        "reason": args.reason,
        "status": "queued",
    }
    log.info("refund queued refund_id=%s customer_id=%s amount=%.2f", refund["refund_id"], args.customer_id, args.amount)
    return types.CallToolResult(content=[types.TextContent(text=_fmt(refund))], structuredContent=refund)


# name -> (input model, implementation, description)
TOOLS: dict[str, tuple[type[BaseModel], Any, str]] = {
    "get_customer_record": (
        GetCustomerRecordInput,
        get_customer_record,
        "Look up a customer record by customer id (CUST-XXXXX).",
    ),
    "trigger_refund": (
        TriggerRefundInput,
        trigger_refund,
        "Queue a refund for a customer. amount must be > 0, reason at least 10 characters.",
    ),
}


# ---------------------------------------------------------------------------
# MCP handlers
# ---------------------------------------------------------------------------


async def handle_list_tools(ctx: Any, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            types.Tool(name=name, description=desc, inputSchema=model.model_json_schema())
            for name, (model, _, desc) in TOOLS.items()
        ]
    )


async def handle_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
    entry = TOOLS.get(params.name)
    if entry is None:
        # Spec: unknown tool is a protocol error, -32602 Invalid params.
        raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown tool: {params.name}")

    model, impl, _ = entry
    try:
        args = model.model_validate(params.arguments or {})
    except ValidationError as exc:
        # -32602 with field-level details. The offending *values* are not echoed
        # back (they might be sensitive); only where and why validation failed.
        details = [
            {"field": ".".join(str(p) for p in err["loc"]) or "(root)", "error": err["msg"], "type": err["type"]}
            for err in exc.errors()
        ]
        log.warning("rejected %s call: %s", params.name, details)
        raise MCPError(code=types.INVALID_PARAMS, message=f"Invalid arguments for tool {params.name}", data=details)

    log.info("calling tool %s", params.name)
    return impl(args)


def _tool_error(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(text=message)], isError=True)


def _fmt(data: dict[str, Any]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in data.items())


server = Server("quilr-customer-support", on_list_tools=handle_list_tools, on_call_tool=handle_call_tool)


async def run() -> None:
    # Take ownership of the real stdout for JSON-RPC frames, then point sys.stdout
    # at stderr so any print() anywhere in the process can't reach the wire.
    # newline="\n": no \r\n translation on Windows; frames are \n-delimited.
    wire = anyio.wrap_file(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n"))
    sys.stdout = sys.stderr
    async with stdio_server(stdout=wire) as (read_stream, write_stream):
        log.info("MCP server ready on stdio")
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    anyio.run(run)


if __name__ == "__main__":
    main()
