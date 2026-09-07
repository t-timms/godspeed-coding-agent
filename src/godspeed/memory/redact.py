"""Shared secrets redaction for all memory writes.

Fail-closed: if ``redact_secrets`` cannot be imported, the entire string
is replaced with ``[REDACTED]`` — raw secrets **never** persist.

All memory modules (user_memory, session, store) import ``redact_or_fail``
from here as the single redaction entry-point.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_REDACTED = "[REDACTED]"


def redact_or_fail(text: str) -> str:
    """Redact secrets from *text*.  Fail-closed on any error.

    Returns:
        The redacted string.  On ``ImportError`` (secrets module missing)
        or any other exception the *entire* input is replaced with
        ``[REDACTED]`` so that no plaintext secret can leak through.
    """
    if not text:
        return text
    try:
        from godspeed.security.secrets import redact_secrets

        return redact_secrets(text)
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
        # Fail-closed: never return raw text when redaction is unavailable.
        logger.error(
            "redact_or_fail: secrets module unavailable or failed; "
            "entire content redacted (%d chars dropped)",
            len(text),
        )
        return _REDACTED
