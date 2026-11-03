"""Streaming PII redaction that still catches patterns split across chunks.

The problem: the LLM streams "...contact john.do" + "e@example.com today". Redacting
each chunk on its own misses the email, and buffering the whole response kills
time-to-first-token. We need the smallest hold-back that is still correct.

The rule: every pattern we redact (email, SSN, card number) is a contiguous run of
"run characters" (letters, digits, . _ % + - @, plus a space when it follows a
digit, as in "4111 1111 ..."). A character outside that set (ordinary spaces
between words, punctuation like , ; : ! ? ( ) " / and non-ASCII text) can never
be inside a match. So at any moment:

    buffer = [ ...safe to emit... ][break char][ trailing run (held) ]

Text up to and including the last break character can be redacted and emitted
right away. Only the trailing run, usually the word being typed, is held until the
next chunk shows whether it grows into a match. In normal prose that's one word of
hold-back, a few characters, never the whole response.

Memory stays bounded: if the trailing run exceeds max_hold characters (a base64
blob, a long token with no spaces), we force-emit everything except the last
max_hold characters, never cutting through a match. max_hold (256) is above the
longest email we'd expect (254 chars by RFC 5321), so real PII doesn't get split.
"""

from __future__ import annotations

import re
import string

REDACTED = "[REDACTED]"

# Characters that may appear inside a pattern match. Must be a superset of every
# character the patterns below can match (except the digit-separator space,
# handled in _safe_cut).
RUN_CHARS = frozenset(string.ascii_letters + string.digits + "._%+-@")
DIGITS = frozenset(string.digits)

PII_PATTERN = re.compile(
    r"(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})"
    r"|(?P<ssn>(?<!\d)\d{3}-\d{2}-\d{4}(?!\d))"
    # 13-19 digits, optionally grouped by single spaces or dashes; Luhn-checked below.
    r"|(?P<card>(?<!\d)\d(?:[ -]?\d){12,18}(?!\d))"
)


def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _replace(m: re.Match[str]) -> str:
    # The Luhn check keeps order numbers and timestamps that happen to be 13-19
    # digits from being redacted. Real card numbers always pass Luhn.
    if m.lastgroup == "card" and not luhn_valid(m.group()):
        return m.group()
    return REDACTED


def redact(text: str) -> str:
    """Redact a complete piece of text (no streaming)."""
    return PII_PATTERN.sub(_replace, text)


def _safe_cut(buf: str) -> int:
    """Index where the trailing run starts; buf[:index] can no longer change."""
    i = len(buf)
    while i > 0:
        c = buf[i - 1]
        if c in RUN_CHARS or (c == " " and i >= 2 and buf[i - 2] in DIGITS):
            i -= 1
        else:
            break
    return i


class StreamRedactor:
    """Feed text deltas in order; get back redacted text that is safe to send now.

        r = StreamRedactor()
        for delta in stream:
            out = r.feed(delta)    # may be "" while a word is still being held
            send(out)
        send(r.flush())            # end of stream: release the held tail
    """

    def __init__(self, max_hold: int = 256):
        self.max_hold = max_hold
        self._pending = ""

    @property
    def pending(self) -> str:
        return self._pending

    def feed(self, delta: str) -> str:
        buf = self._pending + delta
        cut = _safe_cut(buf)
        if len(buf) - cut > self.max_hold:
            cut = self._forced_cut(buf)
        self._pending = buf[cut:]
        return redact(buf[:cut])

    def flush(self) -> str:
        out, self._pending = redact(self._pending), ""
        return out

    def discard(self) -> None:
        """Drop the held tail without emitting it (e.g. upstream died mid-stream)."""
        self._pending = ""

    def _forced_cut(self, buf: str) -> int:
        cut = len(buf) - self.max_hold
        for m in PII_PATTERN.finditer(buf):
            if m.start() < cut < m.end():
                return m.start()  # don't slice a match in half
            if m.start() >= cut:
                break
        return cut
