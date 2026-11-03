"""Primary -> backup model failover with a hard 3000 ms budget on the primary.

Failover triggers: HTTP 429, primary not done within primary_timeout_s, 5xx,
or a connection error. A different 4xx (bad request, context too long) is the
caller's problem; the backup would reject it too, so we don't retry it.

The timeout race: the whole primary call (connect + send + read body) runs
inside one asyncio.timeout(3.0). That's a total deadline, unlike httpx's
per-operation timeouts, where a provider that dribbles a byte every 2.9 s
never trips a 3 s read timeout. When the deadline fires, the in-flight
request is cancelled: httpx aborts the connection, and if the primary's
response arrives a moment later, nobody is waiting for it, so it can't be
returned twice or mixed with the backup's answer. Exactly one provider's
result reaches the client.

Trade-off: failover is sequential, so the worst case is 3 s + backup latency.
Hedging (fire the backup at, say, p95 latency and take whichever finishes
first) lowers tail latency but can double cost and token usage. A rate-
limiting gateway would have to charge for both.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger("task4.router")


@dataclass(frozen=True)
class Provider:
    name: str
    url: str
    model: str
    api_key: str | None = None


@dataclass
class RoutedResponse:
    provider: str
    body: dict[str, Any]
    failover_reason: str | None = None


class UpstreamError(Exception):
    """Internal error with full detail for logs. The app maps it to a sanitized payload."""

    def __init__(self, kind: str, detail: str):
        super().__init__(detail)
        self.kind = kind  # "client_error" | "unavailable"
        self.detail = detail


class _Failover(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class ModelRouter:
    def __init__(
        self,
        primary: Provider,
        backup: Provider,
        client: httpx.AsyncClient,
        primary_timeout_s: float = 3.0,
        backup_timeout_s: float = 15.0,
    ):
        self.primary = primary
        self.backup = backup
        self.http = client
        self.primary_timeout_s = primary_timeout_s
        self.backup_timeout_s = backup_timeout_s

    async def complete(self, body: dict[str, Any], request_id: str) -> RoutedResponse:
        try:
            return RoutedResponse(self.primary.name, await self._attempt(self.primary, body, self.primary_timeout_s))
        except _Failover as f:
            log.warning("request_id=%s primary=%s failed over: %s", request_id, self.primary.name, f.reason)
            reason = f.reason

        try:
            data = await self._attempt(self.backup, body, self.backup_timeout_s)
        except _Failover as f:
            raise UpstreamError("unavailable", f"primary: {reason}; backup: {f.reason}") from None
        return RoutedResponse(self.backup.name, data, failover_reason=reason)

    async def _attempt(self, provider: Provider, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        """Return the provider's JSON body, raise _Failover (try the next provider)
        or UpstreamError("client_error") (don't)."""
        headers = {"authorization": f"Bearer {provider.api_key}"} if provider.api_key else {}
        payload = {**body, "model": provider.model}
        try:
            async with asyncio.timeout(timeout_s):
                resp = await self.http.post(provider.url, json=payload, headers=headers)
        except TimeoutError:
            raise _Failover(f"timeout after {int(timeout_s * 1000)}ms") from None
        except httpx.HTTPError as exc:
            raise _Failover(f"transport error {type(exc).__name__}: {exc}") from None

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                raise _Failover("invalid JSON in 200 response") from None
        if resp.status_code == 429 or resp.status_code >= 500:
            raise _Failover(f"HTTP {resp.status_code}: {resp.text[:300]!r}")
        raise UpstreamError("client_error", f"{provider.name} HTTP {resp.status_code}: {resp.text[:300]!r}")
