from __future__ import annotations

import threading
import time

import httpx
import pytest

from task4_router.app import create_app
from task4_router.mock_providers import create_mock_providers
from task4_router.rate_limiter import SlidingWindowRateLimiter, estimate_tokens
from task4_router.router import Provider

KEY_A, KEY_B = "tenant-a-key", "tenant-b-key"


class FakeClock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def limiter(tmp_path, clock):
    return SlidingWindowRateLimiter(str(tmp_path / "rl.db"), limit_tokens=50_000, clock=clock)


# --- sliding window ------------------------------------------------------------


def test_admits_up_to_limit_then_denies(limiter):
    assert limiter.acquire(KEY_A, 30_000).allowed
    assert limiter.acquire(KEY_A, 20_000).allowed  # exactly 50k
    d = limiter.acquire(KEY_A, 1)
    assert not d.allowed and d.used == 50_000


def test_eviction_exactly_at_window_boundary(limiter, clock):
    limiter.acquire(KEY_A, 50_000)
    clock.advance(59.999)
    assert not limiter.acquire(KEY_A, 1).allowed
    clock.advance(0.001)  # entry is now exactly 60 s old -> outside the window
    assert limiter.acquire(KEY_A, 50_000).allowed


def test_window_slides_rather_than_resetting(limiter, clock):
    limiter.acquire(KEY_A, 30_000)  # t=0
    clock.advance(30)
    limiter.acquire(KEY_A, 20_000)  # t=30
    clock.advance(31)  # t=61: the first entry has expired, the second hasn't
    assert limiter.usage(KEY_A) == 20_000
    assert limiter.acquire(KEY_A, 30_000).allowed
    assert not limiter.acquire(KEY_A, 1).allowed


def test_evicted_rows_are_deleted_from_disk(limiter, clock, tmp_path):
    import sqlite3

    limiter.acquire(KEY_A, 100)
    clock.advance(61)
    limiter.acquire(KEY_A, 100)
    with sqlite3.connect(tmp_path / "rl.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0] == 1


def test_retry_after_points_at_when_enough_tokens_expire(limiter, clock):
    limiter.acquire(KEY_A, 30_000)  # t=0
    clock.advance(30)
    limiter.acquire(KEY_A, 20_000)  # t=30
    clock.advance(10)  # t=40, window full
    d = limiter.acquire(KEY_A, 10_000)
    assert not d.allowed
    assert d.retry_after_s == pytest.approx(20.0)  # the 30k entry from t=0 expires at t=60


def test_request_larger_than_limit_is_never_admitted(limiter):
    d = limiter.acquire(KEY_A, 50_001)
    assert not d.allowed and d.retry_after_s is None


def test_tenants_are_isolated(limiter):
    limiter.acquire(KEY_A, 50_000)
    assert limiter.acquire(KEY_B, 50_000).allowed


def test_reconcile_replaces_estimate_with_actual(limiter):
    d = limiter.acquire(KEY_A, 40_000)
    limiter.reconcile(d.reservation_id, 1_000)
    assert limiter.usage(KEY_A) == 1_000
    assert limiter.acquire(KEY_A, 49_000).allowed


def test_state_survives_restart_on_disk(tmp_path, clock):
    path = str(tmp_path / "rl.db")
    SlidingWindowRateLimiter(path, clock=clock).acquire(KEY_A, 45_000)
    restarted = SlidingWindowRateLimiter(path, clock=clock)
    assert not restarted.acquire(KEY_A, 10_000).allowed


def test_api_keys_are_not_stored_in_plaintext(limiter, tmp_path):
    limiter.acquire(KEY_A, 10)
    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir())
    assert KEY_A.encode() not in raw


def test_concurrent_acquires_never_overshoot(tmp_path):
    # 40 threads race for 5k tokens each against a 50k limit, with separate
    # connections and a real clock. Exactly 10 must win; check-then-insert is atomic.
    limiter = SlidingWindowRateLimiter(str(tmp_path / "race.db"))
    barrier = threading.Barrier(40)
    results: list[bool] = []

    def worker():
        barrier.wait()
        results.append(limiter.acquire(KEY_A, 5_000).allowed)

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 10
    assert limiter.usage(KEY_A) == 50_000


def test_estimate_tokens():
    assert estimate_tokens({"messages": [{"role": "user", "content": "x" * 400}], "max_tokens": 50}) == 150
    assert estimate_tokens({"messages": [{"role": "user", "content": "abc"}]}) == 1 + 256


# --- routing & failover ----------------------------------------------------------

@pytest.fixture
def providers():
    return create_mock_providers(slow_seconds=5.0)


def _make_gateway(tmp_path, providers, *, primary_timeout_s=0.3, limit_tokens=50_000, limiter=None):
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=providers), base_url="http://mock")
    app = create_app(
        db_path=str(tmp_path / "gw.db"),
        primary=Provider("primary", "http://mock/primary/v1/chat/completions", "primary-model"),
        backup=Provider("backup", "http://mock/backup/v1/chat/completions", "backup-model"),
        tenant_keys={KEY_A, KEY_B},
        limit_tokens=limit_tokens,
        primary_timeout_s=primary_timeout_s,
        client=upstream,
        limiter=limiter,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://gw")


BODY = {"model": "auto", "messages": [{"role": "user", "content": "hello there"}], "max_tokens": 100}
AUTH = {"Authorization": f"Bearer {KEY_A}"}


def _assert_sanitized(resp: httpx.Response, code: str) -> None:
    err = resp.json()["error"]
    assert set(err) == {"type", "code", "message", "request_id"}
    assert err["type"] == "gateway_error" and err["code"] == code
    for leak in ["Traceback", "pool.py", "10.0.3.17", "GPU", "context_length", "Rate limit reached", "mock", "ConnectError"]:
        assert leak not in resp.text, leak


@pytest.mark.anyio
async def test_primary_ok(tmp_path, providers):
    async with _make_gateway(tmp_path, providers) as gw:
        r = await gw.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "primary"
    assert r.json()["model"] == "primary-model"
    assert providers.state.calls == {"primary": 1}


@pytest.mark.anyio
async def test_429_fails_over_to_backup(tmp_path, providers):
    providers.state.modes["primary"] = "429"
    async with _make_gateway(tmp_path, providers) as gw:
        r = await gw.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "backup" and r.headers["x-gateway-failover"] == "true"
    assert providers.state.calls == {"primary": 1, "backup": 1}


@pytest.mark.anyio
async def test_timeout_fails_over_and_bounds_latency(tmp_path, providers):
    providers.state.modes["primary"] = "slow"  # would take 5 s
    async with _make_gateway(tmp_path, providers, primary_timeout_s=0.3) as gw:
        start = time.monotonic()
        r = await gw.post("/v1/chat/completions", json=BODY, headers=AUTH)
        elapsed = time.monotonic() - start
    assert r.headers["x-gateway-provider"] == "backup"
    assert 0.3 <= elapsed < 1.5


@pytest.mark.anyio
async def test_default_primary_timeout_is_3000ms(tmp_path, providers):
    providers.state.modes["primary"] = "slow"
    async with _make_gateway(tmp_path, providers, primary_timeout_s=3.0) as gw:
        start = time.monotonic()
        r = await gw.post("/v1/chat/completions", json=BODY, headers=AUTH)
        elapsed = time.monotonic() - start
    assert r.headers["x-gateway-provider"] == "backup"
    assert 3.0 <= elapsed < 4.0


@pytest.mark.anyio
async def test_5xx_with_stack_trace_fails_over_without_leaking(tmp_path, providers):
    providers.state.modes["primary"] = "500"
    async with _make_gateway(tmp_path, providers) as gw:
        r = await gw.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.headers["x-gateway-provider"] == "backup"
    assert "Traceback" not in r.text


@pytest.mark.anyio
async def test_both_providers_down_returns_sanitized_503_and_releases_tokens(tmp_path, providers):
    providers.state.modes.update(primary="500", backup="429")
    async with _make_gateway(tmp_path, providers) as gw:
        r = await gw.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 503
    _assert_sanitized(r, "upstream_unavailable")
    assert SlidingWindowRateLimiter(str(tmp_path / "gw.db")).usage(KEY_A) == 0


@pytest.mark.anyio
async def test_client_error_does_not_fail_over(tmp_path, providers):
    providers.state.modes["primary"] = "400"
    async with _make_gateway(tmp_path, providers) as gw:
        r = await gw.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 400
    _assert_sanitized(r, "upstream_rejected_request")
    assert providers.state.calls == {"primary": 1}


@pytest.mark.anyio
async def test_unreachable_primary_fails_over(tmp_path, providers):
    # The primary is a real socket nobody listens on (connection refused); the
    # backup is the in-process mock. Route each host to the right transport.
    real, mock = httpx.AsyncHTTPTransport(), httpx.ASGITransport(app=providers)

    class Split(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            transport = real if request.url.host == "127.0.0.1" else mock
            return await transport.handle_async_request(request)

    app = create_app(
        db_path=str(tmp_path / "gw.db"),
        primary=Provider("primary", "http://127.0.0.1:1/v1/chat/completions", "p"),
        backup=Provider("backup", "http://mock/backup/v1/chat/completions", "b"),
        tenant_keys={KEY_A},
        client=httpx.AsyncClient(transport=Split()),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw:
        r = await gw.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 200 and r.headers["x-gateway-provider"] == "backup"


@pytest.mark.anyio
async def test_rate_limit_returns_429_with_retry_after(tmp_path, providers):
    async with _make_gateway(tmp_path, providers, limit_tokens=300) as gw:
        first = await gw.post("/v1/chat/completions", json=BODY, headers=AUTH)  # estimate ~103, actual ~14
        assert first.status_code == 200
        big = {**BODY, "max_tokens": 290}
        r = await gw.post("/v1/chat/completions", json=big, headers=AUTH)
    assert r.status_code == 429
    _assert_sanitized(r, "rate_limited")
    assert int(r.headers["retry-after"]) >= 1


@pytest.mark.anyio
async def test_missing_or_unknown_key_is_401(tmp_path, providers):
    async with _make_gateway(tmp_path, providers) as gw:
        r1 = await gw.post("/v1/chat/completions", json=BODY)
        r2 = await gw.post("/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer nope"})
    for r in (r1, r2):
        assert r.status_code == 401
        _assert_sanitized(r, "unauthorized")
    assert providers.state.calls == {}


@pytest.mark.anyio
async def test_unexpected_exception_is_sanitized(tmp_path, providers):
    class Broken(SlidingWindowRateLimiter):
        def acquire(self, api_key, tokens):
            raise RuntimeError("sqlite3.OperationalError at /var/lib/gateway/secret.db")

    broken = Broken(str(tmp_path / "b.db"))
    async with _make_gateway(tmp_path, providers, limiter=broken) as gw:
        r = await gw.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 500
    _assert_sanitized(r, "internal_error")
    assert "secret.db" not in r.text and "sqlite3" not in r.text
