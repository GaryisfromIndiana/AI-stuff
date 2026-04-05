"""Empire error hierarchy — four categories, each with a clear contract.

TransientError  → retry with backoff (network blip, rate limit, temp DB lock)
ConfigError     → disable job, log error, wait for human (bad key, missing table)
DataError       → skip this item, continue the batch (bad JSON, malformed entity)
FatalError      → halt the job immediately (DB gone, budget blown, disk full)

Every bare ``except Exception`` in the codebase should be replaced with one of
these.  If you genuinely don't know which category an error falls into, let it
propagate — an unhandled crash is better than a silent swallow.
"""

from __future__ import annotations


class EmpireError(Exception):
    """Base for all classified Empire errors."""

    retryable: bool = False

    def __init__(self, message: str = "", *, cause: BaseException | None = None, context: dict | None = None):
        self.context = context or {}
        if cause is not None:
            self.__cause__ = cause
        super().__init__(message)


class TransientError(EmpireError):
    """Retry with backoff.  Network timeouts, rate limits, temporary DB locks."""

    retryable = True


class ConfigError(EmpireError):
    """Stop the job and alert.  Bad API key, missing model, wrong table name.

    Will never self-heal — a human or deploy must fix it.
    """

    retryable = False


class DataError(EmpireError):
    """Skip this item, continue the batch.  Malformed JSON, unparseable entity,
    missing required field.

    The batch can continue; just this one item is bad.
    """

    retryable = False


class FatalError(EmpireError):
    """Halt everything.  Database unreachable, budget exceeded, disk full.

    No point retrying or continuing — the whole system is compromised.
    """

    retryable = False
