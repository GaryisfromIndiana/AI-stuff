"""Input validation utilities for Empire."""

from __future__ import annotations

import re
from typing import Any


class ValidationError(Exception):
    """Raised when validation fails."""

    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"{field}: {message}")


class Validator:
    """Fluent validator for input data."""

    def __init__(self, data: dict):
        self._data = data
        self._errors: list[ValidationError] = []

    def require(self, field: str, message: str = "is required") -> Validator:
        """Require a field to be present and non-empty."""
        value = self._data.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            self._errors.append(ValidationError(field, message, value))
        return self

    def require_type(self, field: str, expected_type: type, message: str = "") -> Validator:
        """Require a field to be a specific type."""
        value = self._data.get(field)
        if value is not None and not isinstance(value, expected_type):
            msg = message or f"must be {expected_type.__name__}"
            self._errors.append(ValidationError(field, msg, value))
        return self

    def min_length(self, field: str, length: int, message: str = "") -> Validator:
        """Require minimum string length."""
        value = self._data.get(field, "")
        if isinstance(value, str) and len(value) < length:
            msg = message or f"must be at least {length} characters"
            self._errors.append(ValidationError(field, msg, value))
        return self

    def max_length(self, field: str, length: int, message: str = "") -> Validator:
        """Require maximum string length."""
        value = self._data.get(field, "")
        if isinstance(value, str) and len(value) > length:
            msg = message or f"must be at most {length} characters"
            self._errors.append(ValidationError(field, msg, value))
        return self

    def in_range(self, field: str, min_val: float, max_val: float, message: str = "") -> Validator:
        """Require a numeric value within range."""
        value = self._data.get(field)
        if value is not None:
            try:
                num = float(value)
                if num < min_val or num > max_val:
                    msg = message or f"must be between {min_val} and {max_val}"
                    self._errors.append(ValidationError(field, msg, value))
            except (TypeError, ValueError):
                self._errors.append(ValidationError(field, "must be numeric", value))
        return self

    def one_of(self, field: str, choices: list, message: str = "") -> Validator:
        """Require value to be one of the given choices."""
        value = self._data.get(field)
        if value is not None and value not in choices:
            msg = message or f"must be one of: {', '.join(str(c) for c in choices)}"
            self._errors.append(ValidationError(field, msg, value))
        return self

    def matches(self, field: str, pattern: str, message: str = "") -> Validator:
        """Require value to match a regex pattern."""
        value = self._data.get(field, "")
        if isinstance(value, str) and not re.match(pattern, value):
            msg = message or "does not match expected pattern"
            self._errors.append(ValidationError(field, msg, value))
        return self

    def custom(self, field: str, check_fn: Any, message: str = "") -> Validator:
        """Apply a custom validation function."""
        value = self._data.get(field)
        try:
            if not check_fn(value):
                msg = message or "failed custom validation"
                self._errors.append(ValidationError(field, msg, value))
        except Exception as e:
            self._errors.append(ValidationError(field, str(e), value))
        return self

    @property
    def is_valid(self) -> bool:
        return len(self._errors) == 0

    @property
    def errors(self) -> list[ValidationError]:
        return self._errors

    @property
    def error_messages(self) -> list[str]:
        return [str(e) for e in self._errors]

    def raise_if_invalid(self) -> None:
        """Raise the first validation error if any."""
        if self._errors:
            raise self._errors[0]

    def to_dict(self) -> dict:
        """Get validation result as dict."""
        return {
            "valid": self.is_valid,
            "errors": [{"field": e.field, "message": e.message} for e in self._errors],
        }


def validate_directive(data: dict) -> Validator:
    """Validate directive creation data."""
    return (
        Validator(data)
        .require("title")
        .min_length("title", 3)
        .max_length("title", 256)
        .require("description")
        .min_length("description", 10)
        .in_range("priority", 1, 10)
        .one_of("source", ["human", "evolution", "autonomous"])
    )


def validate_lieutenant(data: dict) -> Validator:
    """Validate lieutenant creation data."""
    return (
        Validator(data)
        .require("name")
        .min_length("name", 2)
        .max_length("name", 128)
        .max_length("domain", 64)
    )


def validate_empire(data: dict) -> Validator:
    """Validate empire creation data."""
    return (
        Validator(data)
        .require("name")
        .min_length("name", 2)
        .max_length("name", 128)
        .max_length("domain", 64)
    )


def validate_task(data: dict) -> Validator:
    """Validate task data."""
    return (
        Validator(data)
        .require("title")
        .min_length("title", 2)
        .max_length("title", 256)
        .in_range("priority", 1, 10)
        .one_of("task_type", ["general", "research", "analysis", "code", "creative", "extraction", "classification", "planning"])
    )


def sanitize_string(value: str, max_length: int = 10000) -> str:
    """Sanitize a string input."""
    if not isinstance(value, str):
        return ""
    # Remove null bytes
    value = value.replace("\x00", "")
    # Normalize whitespace
    value = " ".join(value.split())
    # Truncate
    return value[:max_length]


def sanitize_dict(data: dict, max_depth: int = 5, current_depth: int = 0) -> dict:
    """Recursively sanitize a dict."""
    if current_depth >= max_depth:
        return {}

    sanitized: dict = {}
    for key, value in data.items():
        if isinstance(key, str):
            key = sanitize_string(key, 256)
        if isinstance(value, str):
            sanitized[key] = sanitize_string(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value, max_depth, current_depth + 1)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_string(v) if isinstance(v, str)
                else sanitize_dict(v, max_depth, current_depth + 1) if isinstance(v, dict)
                else v
                for v in value[:1000]
            ]
        else:
            sanitized[key] = value

    return sanitized
