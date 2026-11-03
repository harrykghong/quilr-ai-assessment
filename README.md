# QuilrAI AI Field Developer assessment

Python 3.11+ solutions to the four tasks: an MCP server, an MCP security gateway, a streaming PII guardrail, and a rate-limiting model router. Everything runs locally against mock providers, so no API keys are needed. 101 tests cover the tricky parts.

| Task | Code | Tests |
|---|---|---|
| 1. MCP server, strict validation, stdio | [`task1_mcp_server/`](task1_mcp_server/server.py) | [`tests/test_task1_mcp_server.py`](tests/test_task1_mcp_server.py) |
| 2. MCP security gateway (tool authz) | [`task2_gateway/`](task2_gateway/gateway.py) | [`tests/test_task2_gateway.py`](tests/test_task2_gateway.py) |
| 3. LLM gateway streaming PII guardrail | [`task3_guardrail/`](task3_guardrail/redactor.py) | [`tests/test_task3_guardrail.py`](tests/test_task3_guardrail.py) |
| 4. Token rate limiter + model failover (SQLite) | [`task4_router/`](task4_router/router.py) | [`tests/test_task4_router.py`](tests/test_task4_router.py) |

## Setup

```bash
# with uv (recommended)
uv sync
uv run pytest            # 101 passed, ~20 s (one test deliberately waits out the real 3 s timeout)

# or with pip
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

Stack: official `mcp` SDK (v2), Pydantic v2, Starlette + uvicorn, httpx, PyJWT, stdlib `sqlite3`. I used the same stack across tasks so the gateways read alike.

---

## Task 1: MCP server with strict validation

```bash
python -m task1_mcp_server.server                   # speaks MCP over stdin/stdout
npx @modelcontextprotocol/inspector python -m task1_mcp_server.server   # optional: poke it in the Inspector
```

Tools: `get_customer_record(customer_id)` and `trigger_refund(customer_id, amount, reason)`. Mock customers are `CUST-00001`, `CUST-00042` and `CUST-12345`.

**Validation (Pydantic, `strict=True`, `extra="forbid"`)**
- `customer_id` must match `^CUST-\d{5}$`. I read "XXXXX" as five digits; the pattern is a one-line change if it's meant to be alphanumeric.
- `amount` must be a JSON number with `0 < amount <= 10000`. Strict mode rejects `"10.0"` (a string) and `true` (a bool). NaN and Infinity are rejected too. The upper bound is my addition, as a guard against fat-finger refunds.
- `reason` must be 10 to 500 characters after stripping whitespace, so ten spaces doesn't pass.
- Unknown argument names are rejected, which closes off things like `{"customer_id": ..., "admin": true}`.

**Error mapping**

| Situation | Response |
|---|---|
| Arguments fail validation | JSON-RPC error `-32602`. `data` lists each bad field and why (values are not echoed back) |
| Unknown tool | JSON-RPC error `-32602` (per the MCP spec) |
| Unknown method | JSON-RPC error `-32601` (from the SDK) |
| Valid input, customer doesn't exist | Normal result with `isError: true`. This is a business-logic failure the model can read and act on |

> Spec note: the assessment asks for JSON-RPC error codes on invalid input, so that's what this does. The newest MCP spec revision (2025-11-25) recommends reporting input-validation failures as `isError: true` tool results instead, so the model can self-correct. In production I'd follow the spec version the clients negotiate. The switch is confined to `handle_call_tool`.

**stdout purity.** In a stdio MCP server, any byte on stdout that isn't a JSON-RPC frame breaks the client. Three layers protect it:
1. `logging` is configured with `stream=sys.stderr` at import.
2. `run()` hands the original stdout buffer to the SDK transport as a private "wire", then rebinds `sys.stdout = sys.stderr`. A stray `print()` anywhere in the process goes to stderr.
3. The tests run the server as a real subprocess over real pipes and check that every stdout line parses as a JSON-RPC 2.0 frame. One test monkey-patches a tool to `print()` mid-request. It still gets a valid response, and the print appears on stderr.

A finding from layer 3: I first relied on the SDK's built-in fd-1 diversion alone. On Windows, a `print()` held in Python's block buffer got flushed to the real stdout at interpreter exit, after the SDK had already restored fd 1. The stray-print test caught this, and owning the wire explicitly fixed it.

---

## Task 2: MCP security gateway

```bash
python -m task2_gateway.mock_downstream        # :9001, mock MCP server (has admin_reset_key)
python -m task2_gateway.gateway                # :8080, set GATEWAY_JWT_SECRET in real use
VIEWER=$(python -m task2_gateway.make_token viewer)
curl -s localhost:8080/mcp -H "Authorization: Bearer $VIEWER" -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{}}}'
# {"jsonrpc":"2.0","id":1,"error":{"code":-32001,"message":"Unauthorized Tool Call"}}
```

**Pipeline:** authenticate, then parse, then authorize, then forward. Each step either rejects the request or passes it on.

1. **Authenticate.** `Authorization: Bearer <JWT>`, HS256, with the allowed algorithm pinned so `alg: none` tokens are refused. `exp` and `role` are required, and `role` must be `admin` or `viewer`. Any failure gets HTTP 401 with `WWW-Authenticate` and a JSON-RPC error body, and nothing is forwarded. I chose JWT so the role travels inside a signed token; a static token-to-role table would also satisfy the spec.
2. **Parse.** The body must be a single JSON-RPC 2.0 object. Invalid JSON gets `-32700`; a malformed envelope gets `-32600`.
3. **Authorize.** For `tools/call`, if `params.name` starts with `admin_` and the role isn't `admin`, the gateway returns `-32001 Unauthorized Tool Call` and never calls the downstream (tests assert the mock received nothing). Every other method, including `tools/list`, `initialize` and notifications, is forwarded transparently.
4. **Forward.** The original body bytes go downstream unchanged. The response is streamed back (`aiter_raw`), so a Streamable-HTTP SSE response passes through as it's produced, not buffered. If the downstream is unreachable, the client gets a sanitized `502`.

**Hardening beyond the spec:**
- **Batches are rejected (`-32600`).** MCP removed JSON-RPC batching in 2025-06-18. A gateway that only inspected `batch[0]` would let `[tools/list, tools/call admin_reset_key]` through.
- **Duplicate JSON keys are rejected.** With `{"name":"get_x","name":"admin_reset_key"}`, Python keeps the last value and some parsers keep the first. If the gateway and the downstream disagree, the gateway authorizes one tool and the downstream runs another. Refusing duplicates closes that parser-differential gap.
- **The admin-prefix check ignores case and surrounding whitespace** (fail closed): `Admin_Reset_Key` is treated as an admin tool in case a downstream normalizes names.
- **The user's token is not forwarded downstream.** The gateway is the trust boundary.
- **Audit log lines** (`ALLOW`/`DENY role=... tool=...`) go to stderr.

**HTTP status for -32001:** I return 200 with the JSON-RPC error body, the usual JSON-RPC-over-HTTP convention, so MCP clients surface it as a normal tool-call error rather than a transport failure. Authentication failures are 401 because the caller has to fix its credentials.

**Production next steps:** verify JWTs against the IdP's JWKS (RS256) instead of a shared secret; put the policy in config (tool allow/deny lists per role, not just a prefix); optionally filter `admin_*` tools out of `tools/list` for viewers (the spec here says pass-through, so I didn't); add a request-size limit per tenant and per-tool rate limits.

---

## Task 3: Streaming PII guardrail

```bash
python -m task3_guardrail.mock_llm             # :9002, OpenAI-compatible SSE, uneven 1-6 char chunks
python -m task3_guardrail.gateway              # :8082
curl -sN localhost:8082/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

The mock's canned reply contains an email, an SSN and a card number. They come out as `[REDACTED]` even though the mock splits them across several SSE events.

**The core problem:** `"...email is jane.do"` + `"e@example.com"`. Redacting each chunk on its own misses the email. Buffering the whole response finds it, but then time-to-first-token equals total generation time.

**The approach (`redactor.py`): hold back only what could still become a match.** Every pattern we redact is a contiguous run of *run characters*: ASCII letters and digits, `. _ % + - @`, and a space directly after a digit (card groups like `4111 1111`). Anything else (a space between words, `, ; : ! ? ( ) " /`, non-ASCII text) can never be part of a match. So each time a delta arrives:

```
buffer = held_tail + delta
         [ ...can no longer change... ][break char][ trailing run ]
           └─ redact + emit now ───────────────────┘ └─ hold ─────┘
```

Everything up to the last break character is final: no future text can extend a match backwards across a character that can't be in a match. That part is redacted and sent immediately. Only the trailing run, usually the word in progress, is held until the next delta. In ordinary prose that's a few characters of delay, not the whole response.

- **Correctness check:** for every corpus string, the tests try every two-way split point, 1-character chunks, and 300 random chunkings. Each time the streamed output must equal `redact(full_text)` exactly, so there are no leaks and no extra redactions.
- **Bounded memory:** a run longer than `max_hold` (256 chars, for example a base64 blob) is force-emitted except its last 256 characters, and the cut backs off to the start of any match it would split. 256 is above the longest email we'd expect (254 by RFC 5321).
- **Patterns:** email; SSN in `ddd-dd-dddd` form; cards of 13-19 digits with optional single space or dash separators, kept only if they pass the **Luhn** checksum, so order numbers aren't redacted. Precompiled, one combined alternation, single pass.
- **Two layers of chunking:** (1) one SSE event can be split across TCP reads, and httpx's `aiter_lines()` reassembles lines before parsing; (2) a PII value can be split across events, which is what `StreamRedactor` handles. There's one redactor per `choices[].index`.
- **Event handling:** role and finish events pass through right away. Deltas that became empty only because their text is held are skipped. On `finish_reason` or `[DONE]`, the held tail is flushed.
- **Upstream crashes mid-stream:** the held tail is **discarded**, not flushed. It may be half an email (`jane.doe@exa`), and flushing would leak the partial value. The client gets an SSE error event with a `request_id`.
- **TTFT test:** a real-socket test streams about 2 s of upstream output (200 chunks × 10 ms). The first content must reach the client in under 0.5 s.

**Tradeoffs and limits:** regex PII detection is heuristic. It won't catch names, addresses or obfuscated formats ("john at example dot com"); production would layer an NER/ML detector or a DLP service on top, with the same hold-back framing. SSNs without dashes aren't matched, because nine bare digits produce too many false positives. Tool-call argument deltas aren't scanned in this version. The one-word hold adds at most one chunk of latency to each word.

---

## Task 4: Token rate limiter + model failover

```bash
python -m task4_router.mock_providers          # :9003, /primary and /backup with switchable modes
python -m task4_router.app                     # :8084, SQLite file ./gateway.db, keys tenant-a-key / tenant-b-key
curl -s localhost:8084/v1/chat/completions -H 'Authorization: Bearer tenant-a-key' \
  -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"hi"}]}'
curl -s -XPOST localhost:9003/primary/mode -d '{"mode":"slow"}'   # or "429", "500", "400", "ok"
# repeat the completion: ~3.0 s later it is answered by the backup (x-gateway-provider: backup)
```

**Rate limiter (`rate_limiter.py`): a sliding log in SQLite.**
- Each admitted request writes one row `(tenant, ts_ms, tokens)`. Usage is `SUM(tokens)` over rows inside the last 60 s. The window is exact, so there's no fixed-bucket edge burst (with fixed 1-minute buckets, a tenant could spend 50k at 12:00:59 and another 50k at 12:01:00).
- `acquire()` is a single `BEGIN IMMEDIATE` transaction: delete rows that have left the window (eviction), sum what's left, then insert the reservation or deny. `BEGIN IMMEDIATE` takes SQLite's write lock up front, so check-then-insert is atomic across coroutines, threads and separate gateway processes sharing the file. In a test, 40 threads race for 5k tokens each against a 50k limit, and exactly 10 win.
- **Reserve, then reconcile:** the real token count is only known after the model answers, so the gateway reserves an estimate (prompt characters / 4 + `max_tokens`), then replaces it with the provider's `usage.total_tokens`. If every provider fails, the reservation is released (set to 0).
- On denial the client gets `429` with `Retry-After`, computed by walking the log oldest-first until enough tokens would have expired. A single request bigger than the whole limit gets `413`, because waiting will never make it fit.
- Tenant API keys are stored as SHA-256 hashes. WAL mode is on so readers don't block the writer. SQLite calls run in `asyncio.to_thread` so they don't block the event loop.
- Tests use an injected clock to cover the exact boundary (an entry 59.999 s old still counts; at 60.000 s it's evicted), sliding versus resetting, persistence across restarts, and tenant isolation.

**Failover (`router.py`)**
- **Fails over** on 429, on the primary not finishing within **3000 ms**, on 5xx, and on connection errors. **Doesn't fail over** on other 4xx (bad request, context too long), because the backup would reject them too.
- **The timeout race:** the whole primary call (connect, send, read body) runs inside one `asyncio.timeout(3.0)`. That's a total deadline. httpx's per-operation timeouts aren't: a provider that sends a byte every 2.9 s never trips a 3 s read timeout. When the deadline fires, the in-flight request task is cancelled and httpx aborts the connection. If the primary's answer arrives a moment later, nothing is waiting for it, so exactly one provider's result reaches the client and nothing is returned twice. A test runs the real 3.0 s value and checks the request completes via the backup in 3.0 to 4.0 s.
- **Tradeoff:** failover is sequential, so the worst case is 3 s plus the backup's latency. Hedged requests (start the backup at around p95 latency and take whichever finishes first) cut tail latency, but they can double provider cost and complicate token accounting, since you'd have to charge for both calls.

**Error sanitization:** every client-facing error has the same shape:

```json
{"error": {"type": "gateway_error", "code": "upstream_unavailable",
           "message": "All model providers are currently unavailable. Please retry.",
           "request_id": "3f9c..."}}
```

Codes: `unauthorized` 401, `invalid_request` 400, `rate_limited` 429, `request_too_large` 413, `upstream_rejected_request` 400, `upstream_unavailable` 503, `internal_error` 500. Upstream bodies, stack traces, hostnames and exception names only go to the gateway's stderr log, keyed by `request_id`. A catch-all exception handler covers bugs too. Tests make the mock return a fake Python traceback with an internal IP and assert none of it reaches the client.

**Production next steps:** a circuit breaker so the gateway stops trying a primary that's been failing (and honors its `Retry-After`); a real tokenizer (for example tiktoken) for the estimate; Redis or Postgres for the window if the gateway scales beyond hosts that can share one SQLite file; streaming support, where the 3 s budget would apply to time-to-first-byte.

---

## Repo layout

```
task1_mcp_server/server.py        MCP server (stdio)
task2_gateway/gateway.py          MCP security gateway      mock_downstream.py, make_token.py
task3_guardrail/redactor.py       streaming redactor        gateway.py, mock_llm.py
task4_router/rate_limiter.py      SQLite sliding window     router.py, app.py, mock_providers.py
tests/                            pytest suite (conftest.serve runs apps on real sockets for timing tests)
```
