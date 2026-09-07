"""Before-edit file checkpoints — snapshot originals so edits are reversible.

Every file-mutating tool (file_edit, file_write, diff_apply) snapshots the
on-disk original into ``.godspeed/checkpoints/files/<session>/`` before it
writes. ``restore_latest`` copies the newest snapshot back over the target,
giving per-file undo without touching git state.
"""

from __future__ import annotations

import logging
import re
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_SESSION_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_session_id(session_id: str) -> str:
    """Sanitize a session id for use as a directory name."""
    cleaned = _SESSION_SAFE_RE.sub("_", session_id).strip("._") or "session"
    return cleaned[:80]


def checkpoints_dir(cwd: Path, session_id: str) -> Path:
    """Directory holding this session's before-edit snapshots."""
    return Path(cwd) / ".godspeed" / "checkpoints" / "files" / _safe_session_id(session_id)


def snapshot_file(path: Path, cwd: Path, session_id: str) -> Path | None:
    """Copy the current on-disk *path* into the session checkpoint dir.

    Returns the snapshot path, or None when the file does not exist or the
    copy fails (checkpointing is best-effort — it must never block a
    legitimate edit).
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        target_dir = checkpoints_dir(cwd, session_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        # Zero-padded sequence keeps lexicographic order == chronological
        # order, so restore_latest always picks the true newest snapshot.
        counter = 0
        dest = target_dir / f"{stamp}_{counter:03d}_{path.name}"
        while dest.exists():
            counter += 1
            dest = target_dir / f"{stamp}_{counter:03d}_{path.name}"
        shutil.copy2(path, dest)
        logger.info("Before-edit snapshot saved file=%s snapshot=%s", path, dest)
        return dest
    except OSError as exc:
        logger.warning("Before-edit snapshot failed file=%s error=%s", path, exc)
        return None


def list_checkpoints(path: Path, cwd: Path, session_id: str) -> list[Path]:
    """All snapshots taken of *path* in this session, oldest first."""
    target_dir = checkpoints_dir(cwd, session_id)
    if not target_dir.is_dir():
        return []
    return sorted(target_dir.glob(f"*_{Path(path).name}"))


def restore_latest(path: Path, cwd: Path, session_id: str) -> Path | None:
    """Restore the newest snapshot of *path* over the current file.

    Returns the snapshot path that was restored, or None when no snapshot
    exists or the restore fails.
    """
    snapshots = list_checkpoints(path, cwd, session_id)
    if not snapshots:
        return None
    newest = snapshots[-1]
    try:
        shutil.copy2(newest, Path(path))
        logger.info("Restored before-edit snapshot file=%s snapshot=%s", path, newest)
        return newest
    except OSError as exc:
        logger.warning("Snapshot restore failed file=%s error=%s", path, exc)
        return None
