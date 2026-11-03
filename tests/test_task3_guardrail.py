from __future__ import annotations

import json
import random
import time

import httpx
import pytest

from task3_guardrail.gateway import create_app
from task3_guardrail.mock_llm import CANNED_REPLY, create_mock_llm
from task3_guardrail.redactor import REDACTED, StreamRedactor, luhn_valid, redact
from tests.conftest import serve

PII_SAMPLES = ["jane.doe@example.com", "123-45-6789", "4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111"]

CORPUS = [
    CANNED_REPLY,
    "Email first.last+tag@sub.example.co.uk, then SSN 078-05-1120; card 5500 0000 0000 0004!",
    "a@b.io",
    "no pii here, just text: order 20261103 shipped (ticket #42).",
    "Two emails a.b@c.com,d@e.org and a fake card 4111 1111 1111 1112 that fails Luhn.",
    "ssn:123-45-6789 card:4242424242424242 mail:x@y.dev",
]


def stream_through(text: str, chunks: list[str]) -> tuple[str, list[str]]:
    r = StreamRedactor()
    pieces = [r.feed(c) for c in chunks] + [r.flush()]
    return "".join(pieces), pieces


# --- pattern-level -------------------------------------------------------------


@pytest.mark.parametrize("pii", PII_SAMPLES)
def test_each_pattern_is_redacted(pii):
    assert redact(f"value: {pii}.") == f"value: {REDACTED}."


def test_luhn_filters_non_card_digit_runs():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert redact("order 4111 1111 1111 1112 shipped") == "order 4111 1111 1111 1112 shipped"
    assert redact("tracking 20261103") == "tracking 20261103"  # too short to be a card


# --- streaming correctness -----------------------------------------------------


@pytest.mark.parametrize("text", CORPUS)
def test_every_two_way_split_matches_whole_text_redaction(text):
    # For every possible split point, streaming must produce exactly what
    # redacting the full text at once produces: no leaks, no over-redaction.
    expected = redact(text)
    for i in range(len(text) + 1):
        out, _ = stream_through(text, [text[:i], text[i:]])
        assert out == expected, f"split at {i}: {text[:i]!r} | {text[i:]!r}"


@pytest.mark.parametrize("text", CORPUS)
def test_one_character_chunks(text):
    out, _ = stream_through(text, list(text))
    assert out == redact(text)


def test_random_chunkings():
    rng = random.Random(1234)
    for _ in range(300):
        text = rng.choice(CORPUS)
        cuts = sorted(rng.sample(range(1, len(text)), k=min(len(text) - 1, rng.randint(1, 12))))
        chunks = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)])]
        out, _ = stream_through(text, chunks)
        assert out == redact(text), chunks


def test_no_emitted_piece_contains_raw_pii():
    text = CORPUS[1]
    _, pieces = stream_through(text, list(text))
    for pii in ["first.last+tag@sub.example.co.uk", "078-05-1120", "5500 0000 0000 0004"]:
        assert pii not in "".join(pieces)


# --- latency / memory ----------------------------------------------------------


def test_holds_back_only_the_current_word():
    # Emits as soon as a word is finished; never sits on more than one word.
    r = StreamRedactor()
    words = ("the quick brown fox jumps over the lazy dog " * 20).split(" ")
    emitted_total = 0
    for w in words:
        emitted_total += len(r.feed(w + " "))
        assert r.pending == ""  # every chunk ends in a space, so nothing is held
    assert emitted_total > 0
    r2 = StreamRedactor()
    assert r2.feed("hello wor") == "hello "
    assert r2.pending == "wor"


def test_memory_bounded_on_long_unbroken_runs():
    r = StreamRedactor(max_hold=256)
    blob = "A" * 5000
    out = []
    for ch in blob:
        out.append(r.feed(ch))
        assert len(r.pending) <= 256
    out.append(r.flush())
    assert "".join(out) == blob


def test_forced_cut_never_splits_a_match():
    # 46-char run, max_hold=32 -> naive cut at index 14, inside the email (0..16).
    # The cut must back off to the start of the match instead of emitting "jane@exampl".
    r = StreamRedactor(max_hold=32)
    first = r.feed("jane@example.com" + "_" * 30)
    assert "jane" not in first
    rest = r.feed(" done") + r.flush()
    assert (first + rest) == REDACTED + "_" * 30 + " done"


# --- gateway end to end ----------------------------------------------------------


def _collect_content(sse_text: str) -> tuple[str, list[dict]]:
    content, events = [], []
    for line in sse_text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            ev = json.loads(line[6:])
            events.append(ev)
            for ch in ev.get("choices", []):
                content.append((ch.get("delta") or {}).get("content") or "")
    return "".join(content), events


def _payload(text: str | None = None, **mock) -> dict:
    msg = f"echo:{text}" if text is not None else "hi"
    return {"model": "mock-gpt", "stream": True, "messages": [{"role": "user", "content": msg}], **mock}


def test_gateway_streams_redacted_sse_end_to_end():
    with serve(create_mock_llm()) as llm, serve(create_app(f"{llm}/v1/chat/completions")) as gw:
        r = httpx.post(f"{gw}/v1/chat/completions", json=_payload(mock_delay_ms=1), timeout=10)
    assert r.headers["content-type"].startswith("text/event-stream")
    content, events = _collect_content(r.text)
    assert content == redact(CANNED_REPLY)
    assert content.count(REDACTED) == 3
    assert "jane.doe" not in r.text and "6789" not in r.text and "1111" not in r.text
    assert r.text.rstrip().endswith("data: [DONE]")
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_gateway_ttft_is_not_blocked_by_full_response():
    # 200 chunks x 10 ms = ~2 s of upstream generation. If the gateway buffered the
    # whole response, the first content would show up after ~2 s. It must show up
    # within the first few chunks.
    text = "hello world " * 200
    with serve(create_mock_llm()) as llm, serve(create_app(f"{llm}/v1/chat/completions")) as gw:
        start = time.monotonic()
        first_content_at = None
        with httpx.stream("POST", f"{gw}/v1/chat/completions", json=_payload(text, mock_chunk_size=12, mock_delay_ms=10), timeout=30) as r:
            for line in r.iter_lines():
                if line.startswith("data: {") and '"content": "h' in line:
                    first_content_at = time.monotonic() - start
                    break
    assert first_content_at is not None
    assert first_content_at < 0.5, f"first token after {first_content_at:.2f}s"


def test_gateway_mid_stream_crash_drops_held_partial_and_emits_error():
    # The upstream dies while "jane.doe@exa" is still being held: the partial
    # value must not be flushed to the client.
    text = "contact jane.doe@example.com now"
    with serve(create_mock_llm()) as llm, serve(create_app(f"{llm}/v1/chat/completions")) as gw:
        r = httpx.post(f"{gw}/v1/chat/completions", json=_payload(text, mock_chunk_size=4, mock_delay_ms=1, mock_fail_after=5), timeout=10)
    content, events = _collect_content(r.text)
    assert "jane" not in r.text
    assert content == "contact "
    assert events[-1]["error"]["code"] == "upstream_stream_interrupted"


def test_gateway_non_streaming_is_redacted():
    with serve(create_mock_llm()) as llm, serve(create_app(f"{llm}/v1/chat/completions")) as gw:
        r = httpx.post(f"{gw}/v1/chat/completions", json={**_payload(), "stream": False}, timeout=10)
    assert r.json()["choices"][0]["message"]["content"] == redact(CANNED_REPLY)


def test_gateway_upstream_down_is_sanitized():
    with serve(create_app("http://127.0.0.1:1/v1/chat/completions")) as gw:
        r = httpx.post(f"{gw}/v1/chat/completions", json=_payload(), timeout=10)
    assert r.status_code == 502
    err = r.json()["error"]
    assert err["code"] == "upstream_unavailable" and "request_id" in err
    assert "127.0.0.1" not in r.text and "ConnectError" not in r.text
