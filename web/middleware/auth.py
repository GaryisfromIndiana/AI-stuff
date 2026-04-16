"""Authentication middleware for Empire web UI and API."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable
from functools import wraps

from flask import jsonify, redirect, request, session

logger = logging.getLogger(__name__)


def _get_auth_config() -> tuple[str, str, str]:
    """Get auth credentials from settings."""
    from config.settings import get_settings
    s = get_settings()
    return s.auth_username, s.auth_password, s.api_key


def _is_auth_enabled() -> bool:
    """Check if authentication is enabled (password is set)."""
    _, password, _ = _get_auth_config()
    return bool(password)


def _check_password(password: str) -> bool:
    """Verify password against configured password."""
    _, correct_password, _ = _get_auth_config()
    if not correct_password:
        return True  # Auth disabled
    return secrets.compare_digest(password, correct_password)


def _check_api_key(key: str) -> bool:
    """Verify API key."""
    _, _, correct_key = _get_auth_config()
    if not correct_key:
        return True  # API key auth disabled
    return secrets.compare_digest(key, correct_key)


def require_login(f: Callable) -> Callable:
    """Decorator for web routes — redirects to login if not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not _is_auth_enabled():
            return f(*args, **kwargs)
        if session.get("authenticated"):
            return f(*args, **kwargs)
        return redirect("/login")
    return decorated


def require_api_auth(f: Callable) -> Callable:
    """Decorator for API routes — returns 401 if not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not _is_auth_enabled():
            return f(*args, **kwargs)

        # Check API key in header
        api_key = request.headers.get("X-API-Key", "")
        if api_key and _check_api_key(api_key):
            return f(*args, **kwargs)

        # Check session (for browser-based API calls)
        if session.get("authenticated"):
            return f(*args, **kwargs)

        return jsonify({"error": "Unauthorized", "message": "Provide X-API-Key header or login via /login"}), 401
    return decorated
