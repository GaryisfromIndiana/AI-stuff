"""Web middleware for rate limiting and security."""

from web.middleware.rate_limit import get_rate_limiter, rate_limit

__all__ = ["get_rate_limiter", "rate_limit"]
