"""Mint a demo bearer token: python -m task2_gateway.make_token <admin|viewer> [ttl_seconds]"""

from __future__ import annotations

import sys
import time

import jwt

from task2_gateway.gateway import jwt_secret_from_env


def make_token(role: str, secret: str, ttl_seconds: int = 3600, sub: str = "demo-user") -> str:
    now = int(time.time())
    return jwt.encode({"sub": sub, "role": role, "iat": now, "exp": now + ttl_seconds}, secret, algorithm="HS256")


if __name__ == "__main__":
    role = sys.argv[1] if len(sys.argv) > 1 else "viewer"
    ttl = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
    print(make_token(role, jwt_secret_from_env(), ttl))
