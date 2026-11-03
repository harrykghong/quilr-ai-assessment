"""Token-aware sliding-window rate limiter backed by an on-disk SQLite database.

Model: a sliding *log*. Every admitted request writes one row (tenant, ts_ms,
tokens). A tenant's usage is the sum of tokens over rows with ts in
(now - 60s, now]. That's an exact window with no fixed-bucket edge burst: with
fixed 1-minute buckets a tenant could spend 50k at 12:00:59 and another 50k at
12:01:00.

acquire() is one SQLite write transaction (BEGIN IMMEDIATE):
    evict rows older than the window -> sum what's left -> admit (insert) or deny
BEGIN IMMEDIATE takes SQLite's write lock up front, so check-then-insert is
atomic across coroutines, threads, *and* gateway processes sharing the same
file. Two concurrent requests can't both see "45k used" and both insert 5k.

Tokens are reserved up front from an estimate (prompt chars/4 + max_tokens)
because the real count only exists after the model answers. reconcile()
replaces the estimate with the provider's reported usage.

Tenant API keys are stored as SHA-256 hashes, never in plaintext.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant  TEXT    NOT NULL,
    ts_ms   INTEGER NOT NULL,
    tokens  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_token_usage_tenant_ts ON token_usage (tenant, ts_ms);
"""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    used: int  # tokens in the window *before* this request
    limit: int
    retry_after_s: float | None = None  # when denied: earliest time the request could fit
    reservation_id: int | None = None

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


def tenant_id(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


class SlidingWindowRateLimiter:
    def __init__(
        self,
        db_path: str,
        limit_tokens: int = 50_000,
        window_ms: int = 60_000,
        clock: Callable[[], float] = time.time,
    ):
        self.db_path = db_path
        self.limit = limit_tokens
        self.window_ms = window_ms
        self.clock = clock
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # One short-lived connection per operation: safe to call from any thread
        # (the app runs these in asyncio.to_thread), and SQLite connections are cheap.
        # isolation_level=None -> we issue BEGIN/COMMIT ourselves.
        conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _now_ms(self) -> int:
        return int(self.clock() * 1000)

    def acquire(self, api_key: str, tokens: int) -> Decision:
        tenant = tenant_id(api_key)
        now = self._now_ms()
        cutoff = now - self.window_ms
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Eviction: rows at or before the cutoff have left the window.
            conn.execute("DELETE FROM token_usage WHERE tenant = ? AND ts_ms <= ?", (tenant, cutoff))
            (used,) = conn.execute("SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ?", (tenant,)).fetchone()

            if tokens > self.limit:
                conn.execute("COMMIT")
                return Decision(False, used, self.limit, retry_after_s=None)  # can never fit

            if used + tokens > self.limit:
                retry_after = self._retry_after_s(conn, tenant, need=used + tokens - self.limit, now=now)
                conn.execute("COMMIT")
                return Decision(False, used, self.limit, retry_after_s=retry_after)

            cur = conn.execute("INSERT INTO token_usage (tenant, ts_ms, tokens) VALUES (?, ?, ?)", (tenant, now, tokens))
            conn.execute("COMMIT")
            return Decision(True, used, self.limit, reservation_id=cur.lastrowid)
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _retry_after_s(self, conn: sqlite3.Connection, tenant: str, need: int, now: int) -> float:
        # Walk the log oldest-first until enough tokens would have expired.
        freed = 0
        for ts_ms, tokens in conn.execute(
            "SELECT ts_ms, tokens FROM token_usage WHERE tenant = ? ORDER BY ts_ms", (tenant,)
        ):
            freed += tokens
            if freed >= need:
                return max(0.0, (ts_ms + self.window_ms - now) / 1000)
        return self.window_ms / 1000

    def reconcile(self, reservation_id: int, actual_tokens: int) -> None:
        """Replace an up-front estimate with the provider-reported usage."""
        conn = self._connect()
        try:
            conn.execute("UPDATE token_usage SET tokens = ? WHERE id = ?", (max(0, actual_tokens), reservation_id))
        finally:
            conn.close()

    def usage(self, api_key: str) -> int:
        conn = self._connect()
        try:
            (used,) = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ? AND ts_ms > ?",
                (tenant_id(api_key), self._now_ms() - self.window_ms),
            ).fetchone()
            return used
        finally:
            conn.close()


def estimate_tokens(body: dict, default_max_tokens: int = 256) -> int:
    """Rough pre-flight estimate: ~4 chars per token for the prompt, plus the
    completion budget the caller asked for (the worst case it can consume)."""
    chars = 0
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):  # multi-part content
            chars += sum(len(p.get("text", "")) for p in content if isinstance(p, dict))
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or default_max_tokens
    return -(-chars // 4) + int(max_tokens)
