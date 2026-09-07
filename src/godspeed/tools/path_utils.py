"""Shared path utilities for file tools."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_tool_path(file_path: str, cwd: Path) -> Path:
    """Resolve a file path relative to the project root.

    Includes symlink resolution to prevent symlink traversal attacks.
    Raises:
        ValueError: If the resolved path is outside the project directory.
    """
    # NT device / long-prefix paths (\\.\PhysicalDrive0, \\?\C:\...) are
    # never legitimate tool targets — reject before any resolution.
    if str(file_path).startswith(("\\\\.", "\\\\?\\")):
        raise ValueError(f"Access denied: NT device paths are not permitted: '{file_path}'")

    # On non-Windows platforms, defensively reject Windows drive-letter paths
    # that can never be inside the project directory.
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\\/]", str(file_path)):
        raise ValueError(
            f"Access denied: path '{file_path}' is a Windows absolute path "
            f"which is outside the project directory '{cwd.resolve()}'"
        )

    path = Path(file_path).expanduser()
    resolved = path.resolve() if path.is_absolute() else (cwd / path).resolve()
    cwd_resolved = cwd.resolve()

    try:
        resolved.relative_to(cwd_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Access denied: path '{file_path}' resolves to '{resolved}' "
            f"which is outside the project directory '{cwd_resolved}'"
        ) from exc

    # Additional symlink protection: resolve symlinks and verify the real path
    # is still within the project directory
    try:
        real_path = Path(os.path.realpath(str(resolved)))
        real_path.relative_to(cwd_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Access denied: path '{file_path}' (real path: '{real_path}') "
            f"resolves via symlinks to outside the project directory '{cwd_resolved}'"
        ) from exc
    except OSError as exc:
        logger.warning("Could not resolve real path for %s: %s", resolved, exc)

    return resolved
