"""Cryptographic utilities — hashing, ID generation, token management."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
import uuid
from typing import Any


def generate_id(prefix: str = "", length: int = 16) -> str:
    """Generate a random ID.

    Args:
        prefix: Optional prefix (e.g., 'emp_', 'lt_').
        length: Length of the random part.

    Returns:
        Random ID string.
    """
    random_part = uuid.uuid4().hex[:length]
    return f"{prefix}{random_part}" if prefix else random_part


def generate_short_id(length: int = 8) -> str:
    """Generate a short random ID."""
    return secrets.token_hex(length // 2)


def generate_secret_key(length: int = 32) -> str:
    """Generate a cryptographically secure secret key."""
    return secrets.token_urlsafe(length)


def hash_content(content: str, algorithm: str = "sha256") -> str:
    """Hash content using specified algorithm.

    Args:
        content: Content to hash.
        algorithm: Hash algorithm (md5, sha256, sha512).

    Returns:
        Hex digest.
    """
    hasher = hashlib.new(algorithm)
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()


def hash_content_short(content: str, length: int = 12) -> str:
    """Generate a short hash of content."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:length]


def hmac_sign(message: str, key: str, algorithm: str = "sha256") -> str:
    """Create HMAC signature.

    Args:
        message: Message to sign.
        key: Secret key.
        algorithm: Hash algorithm.

    Returns:
        Hex digest of HMAC.
    """
    return hmac.new(
        key.encode("utf-8"),
        message.encode("utf-8"),
        getattr(hashlib, algorithm),
    ).hexdigest()


def hmac_verify(message: str, key: str, signature: str, algorithm: str = "sha256") -> bool:
    """Verify an HMAC signature.

    Args:
        message: Original message.
        key: Secret key.
        signature: Signature to verify.
        algorithm: Hash algorithm.

    Returns:
        True if signature matches.
    """
    expected = hmac_sign(message, key, algorithm)
    return hmac.compare_digest(expected, signature)


def _get_token_key() -> str:
    """Get the signing key for tokens (uses Flask secret key or fallback)."""
    try:
        from config.settings import get_settings
        return get_settings().flask_secret_key
    except Exception:
        return "empire-fallback-key"


def generate_token(payload: str = "", ttl_seconds: int = 3600) -> str:
    """Generate an HMAC-signed time-limited token.

    Args:
        payload: Optional payload to include.
        ttl_seconds: Token time-to-live.

    Returns:
        Token string.
    """
    timestamp = str(int(time.time()) + ttl_seconds)
    random_part = secrets.token_hex(16)
    data = f"{timestamp}:{payload}:{random_part}"
    sig = hmac_sign(data, _get_token_key())
    return f"{data}:{sig}"


def validate_token(token: str) -> tuple[bool, str]:
    """Validate an HMAC-signed time-limited token.

    Args:
        token: Token to validate.

    Returns:
        Tuple of (valid, payload).
    """
    try:
        parts = token.rsplit(":", 1)
        if len(parts) != 2:
            return False, ""

        data, sig = parts
        if not hmac_verify(data, _get_token_key(), sig):
            return False, ""

        data_parts = data.split(":", 2)
        if len(data_parts) < 3:
            return False, ""

        expiry = int(data_parts[0])
        payload = data_parts[1]

        if time.time() > expiry:
            return False, ""

        return True, payload
    except (ValueError, IndexError):
        return False, ""


def mask_api_key(key: str) -> str:
    """Mask an API key for display (show first 4 and last 4 chars).

    Args:
        key: API key.

    Returns:
        Masked key.
    """
    if not key or len(key) < 10:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def generate_embedding_id(text: str) -> str:
    """Generate a deterministic ID for an embedding based on content."""
    return f"emb_{hash_content_short(text, 16)}"
