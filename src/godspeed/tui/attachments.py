"""Image attachment queue for the Godspeed TUI.

Handles attaching images to the next user message, either from a
deterministic text directive (``:img <path>`` / ``@image=<path>``) or from
best-effort clipboard paste. The queue is intentionally a small, pure helper
so it can be unit-tested without any interactive mocking (mirrors
``message_queue.py``).

Attachments are one-shot: they are consumed by the next message that is sent
to the model and are not persisted beyond that.
"""

from __future__ import annotations

import base64
import logging
import re
import struct
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Size / format limits
# ---------------------------------------------------------------------------

# Hard cap on the raw image file size we will base64-encode into a message.
# Base64 inflates the payload by ~33%, so a 10MB file becomes ~13.3MB of
# message content — well beyond what most vision models accept. Mirrors the
# size-limit convention used by file_write.py (10MB) and image_read.py (20MB).
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024

# Supported image extensions -> MIME type. Mirrors tools/image_read.py.
SUPPORTED_FORMATS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# ---------------------------------------------------------------------------
# Text directives
# ---------------------------------------------------------------------------

# Match `:img <path>` and `@image=<path>` at the start of (or anywhere in) the
# input. Captures the path (non-whitespace).
_IMG_DIRECTIVE_RE = re.compile(r"(?::img\s+|@image=)(\S+)")

# Collapse multiple spaces left by stripping directives.
_MULTI_SPACE_RE = re.compile(r" {2,}")

# ---------------------------------------------------------------------------
# Vision capability heuristic
# ---------------------------------------------------------------------------

# Model families known to accept image_url content blocks. This is a
# best-effort heuristic used only to decide whether to warn the user; the
# attachment is always collected regardless.
_VISION_MODEL_MARKERS = (
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.5",
    "gpt-5",
    "claude-3",
    "claude-3.5",
    "claude-3.7",
    "claude-sonnet-4",
    "claude-opus-4",
    "gemini",
    "gemma-3",
    "llama-3.2",
    "llama3.2",
    "llama-4",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen-vl",
    "pixtral",
    "moondream",
    "llava",
    "internvl",
    "phi-3-vision",
    "phi-4-multimodal",
    "mistral-small-3.2",
    "glm-4v",
    "glm-4.5v",
    "idefics",
    "fuyu",
    "cogvlm",
    "minicpm-v",
    "grok-2-vision",
    "grok-4",
    "o1",
    "o3",
    "o4",
    "o5",
)


def model_supports_vision(model: str) -> bool:
    """Return True if *model* is (heuristically) vision-capable.

    Uses a substring match against known vision-capable model families.
    Unknown models are treated as vision-capable (optimistic) so that we do
    not silently drop attachments for models we have not seen before.
    """
    lowered = model.lower()
    return any(marker in lowered for marker in _VISION_MODEL_MARKERS)


# ---------------------------------------------------------------------------
# Attachment model
# ---------------------------------------------------------------------------


class AttachmentError(ValueError):
    """Raised when an image attachment cannot be validated or encoded."""


@dataclass(frozen=True)
class Attachment:
    """A validated, base64-encoded image ready to attach to a message."""

    path: Path
    mime_type: str
    size_bytes: int
    data_uri: str
    width: int | None = None
    height: int | None = None

    @property
    def size_kb(self) -> float:
        return self.size_bytes / 1024

    @property
    def dimensions(self) -> str:
        if self.width is not None and self.height is not None:
            return f"{self.width}x{self.height}"
        return "unknown"


# ---------------------------------------------------------------------------
# Dimension sniffing (no PIL dependency)
# ---------------------------------------------------------------------------


def _sniff_dimensions(path: Path, mime_type: str) -> tuple[int | None, int | None]:
    """Best-effort read of image dimensions from file headers.

    Returns ``(width, height)`` or ``(None, None)`` when the format is not
    parseable from its header. Never raises.
    """
    try:
        with path.open("rb") as fh:
            header = fh.read(64)
    except OSError:
        return None, None

    try:
        if mime_type == "image/png":
            # PNG: 8-byte signature, then IHDR chunk: width(4) height(4)
            if header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
                return struct.unpack(">II", header[16:24])
        elif mime_type == "image/jpeg":
            # JPEG: scan for SOF0/SOF2 markers carrying dimensions.
            return _sniff_jpeg(path)
        elif mime_type == "image/gif":
            # GIF: "GIF87a"/"GIF89a" then width(2) height(2) little-endian.
            if header[:6] in (b"GIF87a", b"GIF89a"):
                return struct.unpack("<HH", header[6:10])
        elif mime_type == "image/webp":
            # WebP: "RIFF"...."WEBP" then VP8/VP8L/VP8X chunk.
            return _sniff_webp(path)
    except (struct.error, IndexError):
        logger.debug("Could not parse dimensions for %s", path)

    return None, None


def _sniff_jpeg(path: Path) -> tuple[int | None, int | None]:
    """Scan JPEG markers for a SOF segment carrying dimensions."""
    try:
        with path.open("rb") as fh:
            data = fh.read(64 * 1024)
    except OSError:
        return None, None

    idx = 2  # skip SOI
    while idx + 9 < len(data):
        if data[idx] != 0xFF:
            idx += 1
            continue
        marker = data[idx + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            idx += 2
            continue
        if idx + 4 > len(data):
            break
        length = struct.unpack(">H", data[idx + 2 : idx + 4])[0]
        # SOF0..SOF15 (except DHT 0xC4, JPG 0xC8, DAC 0xCC)
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if idx + 9 <= len(data):
                height, width = struct.unpack(">HH", data[idx + 5 : idx + 9])
                return width, height
            break
        idx += 2 + length
    return None, None


def _sniff_webp(path: Path) -> tuple[int | None, int | None]:
    """Parse WebP VP8/VP8L/VP8X chunks for dimensions."""
    try:
        with path.open("rb") as fh:
            data = fh.read(64)
    except OSError:
        return None, None

    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None, None
    chunk = data[12:16]
    if chunk == b"VP8X":
        # VP8X: 1-byte flags, 3-byte canvas width-1, 3-byte canvas height-1
        if len(data) >= 30:
            w = int.from_bytes(data[24:27], "little") + 1
            h = int.from_bytes(data[27:30], "little") + 1
            return w, h
    elif chunk == b"VP8 " and len(data) >= 30:
        # VP8 lossy: frame tag at offset 23
        w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return w, h
    elif chunk == b"VP8L" and len(data) >= 25:
        # VP8L lossless: 4-byte signature + 14-bit width/height
        bits = int.from_bytes(data[21:25], "little")
        w = (bits & 0x3FFF) + 1
        h = ((bits >> 14) & 0x3FFF) + 1
        return w, h
    return None, None


# ---------------------------------------------------------------------------
# Attachment queue
# ---------------------------------------------------------------------------


class PendingAttachments:
    """FIFO queue of validated image attachments for the next message.

    Attachments are one-shot: ``drain()`` returns and clears them so they are
    consumed by exactly one message send.
    """

    def __init__(self) -> None:
        self._items: deque[Attachment] = deque()

    def attach(self, path: str, cwd: Path) -> Attachment:
        """Validate *path* and enqueue it as a pending attachment.

        Args:
            path: Image file path (relative to *cwd* or absolute).
            cwd: Project working directory for path resolution.

        Returns:
            The validated :class:`Attachment`.

        Raises:
            AttachmentError: If the path does not exist, is not a file, has an
                unsupported extension, or exceeds the size cap.
        """
        attachment = build_attachment(path, cwd)
        self._items.append(attachment)
        return attachment

    def add(self, attachment: Attachment) -> None:
        """Enqueue an already-built attachment (e.g. from clipboard paste)."""
        self._items.append(attachment)

    def drain(self) -> list[Attachment]:
        """Remove and return all pending attachments, clearing the queue."""
        items = list(self._items)
        self._items.clear()
        return items

    def peek(self) -> Attachment | None:
        """Return the front attachment without removing it, or None."""
        if not self._items:
            return None
        return self._items[0]

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __iter__(self) -> Iterator[Attachment]:
        return iter(self._items)

    def __repr__(self) -> str:
        return f"PendingAttachments({list(self._items)!r})"


# ---------------------------------------------------------------------------
# Validation / encoding
# ---------------------------------------------------------------------------


def build_attachment(path: str, cwd: Path) -> Attachment:
    """Validate *path* and build a base64-encoded :class:`Attachment`.

    Raises:
        AttachmentError: With a clear, user-facing message on any failure.
    """
    try:
        resolved = resolve_tool_path(path, cwd)
    except ValueError as exc:
        raise AttachmentError(str(exc)) from exc

    if not resolved.exists():
        raise AttachmentError(f"Image not found: {path}")
    if not resolved.is_file():
        raise AttachmentError(f"Not a file (is a directory?): {path}")

    suffix = resolved.suffix.lower()
    mime_type = SUPPORTED_FORMATS.get(suffix)
    if mime_type is None:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise AttachmentError(
            f"Unsupported image format '{suffix or '(none)'}'. Supported formats: {supported}"
        )

    try:
        size_bytes = resolved.stat().st_size
    except OSError as exc:
        raise AttachmentError(f"Failed to stat image: {exc}") from exc

    if size_bytes > MAX_ATTACHMENT_BYTES:
        size_mb = size_bytes / (1024 * 1024)
        max_mb = MAX_ATTACHMENT_BYTES // (1024 * 1024)
        raise AttachmentError(f"Image too large: {size_mb:.1f}MB (max {max_mb}MB)")

    try:
        raw_bytes = resolved.read_bytes()
    except OSError as exc:
        raise AttachmentError(f"Failed to read image: {exc}") from exc

    b64_data = base64.b64encode(raw_bytes).decode("ascii")
    data_uri = f"data:{mime_type};base64,{b64_data}"
    width, height = _sniff_dimensions(resolved, mime_type)

    return Attachment(
        path=resolved,
        mime_type=mime_type,
        size_bytes=size_bytes,
        data_uri=data_uri,
        width=width,
        height=height,
    )


def parse_attachment_directives(text: str) -> tuple[str, list[str]]:
    """Extract ``:img <path>`` / ``@image=<path>`` directives from *text*.

    Returns:
        Tuple of ``(cleaned_text, paths)`` where *cleaned_text* has the
        directives stripped and *paths* is the list of image paths in order.
    """
    paths = [m.group(1) for m in _IMG_DIRECTIVE_RE.finditer(text)]
    cleaned = _IMG_DIRECTIVE_RE.sub("", text).strip()
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    return cleaned, paths


def interpret_clipboard(text: str, cwd: Path) -> tuple[Attachment | None, str | None]:
    """Interpret clipboard text as an image attachment (best-effort paste).

    prompt_toolkit's clipboard abstraction exposes only text, so image paste
    is detected by checking whether the clipboard content is a path to an
    existing supported image file.

    Returns:
        Tuple of ``(attachment, notice)``:

        - ``(Attachment, None)``: the clipboard was an image path that was
          successfully attached.
        - ``(None, str)``: the clipboard looked like an image path but could
          not be attached (e.g. file missing, unsupported format, too large).
          *notice* is a user-facing error message.
        - ``(None, None)``: the clipboard is not image-related; treat it as
          plain text (fall back to normal paste).
    """
    stripped = text.strip()
    if not stripped:
        return None, None

    # Only treat the clipboard as an image if it is a bare path (no spaces /
    # shell metacharacters) that looks like an image file.
    if " " in stripped or any(c in stripped for c in "\"'`$;&|<>"):
        return None, None

    suffix = Path(stripped).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        return None, None

    try:
        attachment = build_attachment(stripped, cwd)
    except AttachmentError as exc:
        return None, str(exc)

    return attachment, None
