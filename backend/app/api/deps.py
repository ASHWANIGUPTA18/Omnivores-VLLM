"""Shared API dependencies: authentication and rate limiting.

Authentication is opt-in via ``API_KEY``. Empty means open, which is right for
local development and wrong for anything exposed - ``Settings.production_warnings``
says so at startup.

The rate limiter is a per-connection token bucket held in memory. It bounds a
runaway client on a single instance; it is not a distributed limiter and does
not claim to be. Behind more than one worker, put a real one at the proxy.
"""

from __future__ import annotations

import time
from collections import deque

from fastapi import Header, Query

from app.core.config import settings
from app.core.exceptions import AuthenticationError, RateLimitExceeded


def _token_matches(provided: str | None) -> bool:
    if not settings.auth_enabled:
        return True
    if not provided:
        return False
    # Constant-time compare so the key cannot be recovered by timing.
    import hmac

    return hmac.compare_digest(provided, settings.api_key)


async def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency for REST routes."""
    if not settings.auth_enabled:
        return

    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()

    if not _token_matches(token):
        raise AuthenticationError(
            "missing or invalid API key",
            user_message="A valid API key is required.",
        )


def authorize_websocket(token: str | None) -> None:
    """WebSocket auth.

    Browsers cannot set headers on a WebSocket handshake, so the key arrives as
    a query parameter. That puts it in server access logs, which is why the
    deployment guide recommends terminating auth at the proxy instead when the
    service is public.
    """
    if not settings.auth_enabled:
        return
    if not _token_matches(token):
        raise AuthenticationError("missing or invalid API key")


async def websocket_token(token: str | None = Query(default=None)) -> str | None:
    return token


class RateLimiter:
    """Sliding-window limiter keyed by an arbitrary string."""

    def __init__(self, limit: int | None = None, window_seconds: float = 60.0) -> None:
        self.limit = limit if limit is not None else settings.rate_limit_messages_per_minute
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> None:
        """Record a hit. Raises ``RateLimitExceeded`` when over the limit."""
        if self.limit <= 0:
            return

        now = time.monotonic()
        bucket = self._hits.setdefault(key, deque())
        cutoff = now - self.window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= self.limit:
            retry_after = max(0.0, self.window - (now - bucket[0]))
            raise RateLimitExceeded(
                f"{key} exceeded {self.limit} messages per {self.window:.0f}s",
                user_message=(
                    f"You are sending messages too quickly. Try again in "
                    f"{retry_after:.0f}s."
                ),
                details={"retry_after_seconds": round(retry_after, 1)},
            )
        bucket.append(now)

    def forget(self, key: str) -> None:
        self._hits.pop(key, None)


message_rate_limiter = RateLimiter()
