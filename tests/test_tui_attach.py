"""Tests for src/godspeed/tui/attachments.py — image attachment queue and TUI wiring."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.tui.app import TUIApp
from godspeed.tui.attachments import (
    MAX_ATTACHMENT_BYTES,
    Attachment,
    AttachmentError,
    PendingAttachments,
    build_attachment,
    interpret_clipboard,
    model_supports_vision,
    parse_attachment_directives,
)

# 1x1 transparent PNG (valid header for dimension sniffing).
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def make_png(path: Path, size: int | None = None) -> Path:
    """Write a PNG file, optionally padded to *size* bytes."""
    data = PNG_1X1 if size is None else PNG_1X1 + b"\x00" * (size - len(PNG_1X1))
    path.write_bytes(data)
    return path


def make_app(tmp_path: Path, model: str = "gpt-4o") -> TUIApp:
    """Build a TUIApp with mocked dependencies for integration tests."""
    llm_client = MagicMock()
    llm_client.model = model
    llm_client.total_input_tokens = 0
    llm_client.total_output_tokens = 0
    llm_client.total_cost_usd = 0.0
    llm_client.max_cost_usd = 0.0
    tool_registry = MagicMock()
    tool_context = MagicMock()
    tool_context.cwd = tmp_path
    conversation = MagicMock()
    conversation.token_count = 0
    conversation.max_tokens = 100_000
    permission_engine = MagicMock()
    permission_engine.deny_rules = []
    permission_engine.ask_rules = [MagicMock()]
    permission_engine.session_grants = []
    permission_engine.plan_mode = False
    permission_engine._mode = "normal"
    return TUIApp(
        llm_client=llm_client,
        tool_registry=tool_registry,
        tool_context=tool_context,
        conversation=conversation,
        permission_engine=permission_engine,
        audit_trail=None,
        session_id="test-session",
    )


def printed_text(mock_print: MagicMock) -> str:
    """Join all text passed to a mocked console.print, positional or keyword."""
    parts: list[str] = []
    for call in mock_print.call_args_list:
        if call.args:
            parts.append(str(call.args[0]))
        if "text" in call.kwargs:
            parts.append(str(call.kwargs["text"]))
    return " ".join(parts)


class TestBuildAttachment:
    """:img path validation — exists / extension / size cap / errors."""

    def test_valid_png(self, tmp_path: Path) -> None:
        img = make_png(tmp_path / "photo.png")
        attachment = build_attachment("photo.png", tmp_path)
        assert isinstance(attachment, Attachment)
        assert attachment.path == img
        assert attachment.mime_type == "image/png"
        assert attachment.size_bytes == len(PNG_1X1)
        assert attachment.data_uri.startswith("data:image/png;base64,")
        assert attachment.width == 1
        assert attachment.height == 1

    def test_valid_jpg(self, tmp_path: Path) -> None:
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
        attachment = build_attachment("photo.jpg", tmp_path)
        assert attachment.mime_type == "image/jpeg"
        assert attachment.data_uri.startswith("data:image/jpeg;base64,")

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(AttachmentError, match="Image not found"):
            build_attachment("nope.png", tmp_path)

    def test_directory(self, tmp_path: Path) -> None:
        (tmp_path / "dir.png").mkdir()
        with pytest.raises(AttachmentError, match="Not a file"):
            build_attachment("dir.png", tmp_path)

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("hello")
        with pytest.raises(AttachmentError, match="Unsupported image format"):
            build_attachment("notes.txt", tmp_path)

    def test_too_large(self, tmp_path: Path) -> None:
        make_png(tmp_path / "big.png", size=MAX_ATTACHMENT_BYTES + 1)
        with pytest.raises(AttachmentError, match="Image too large"):
            build_attachment("big.png", tmp_path)

    def test_path_traversal_rejected(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside.png"
        make_png(outside)
        with pytest.raises(AttachmentError, match="Access denied"):
            build_attachment(str(outside), tmp_path)

    def test_absolute_path_inside_cwd(self, tmp_path: Path) -> None:
        img = make_png(tmp_path / "abs.png")
        attachment = build_attachment(str(img), tmp_path)
        assert attachment.path == img


class TestParseAttachmentDirectives:
    def test_img_directive(self) -> None:
        cleaned, paths = parse_attachment_directives(":img photo.png describe this")
        assert cleaned == "describe this"
        assert paths == ["photo.png"]

    def test_image_eq_directive(self) -> None:
        cleaned, paths = parse_attachment_directives("@image=photo.png describe this")
        assert cleaned == "describe this"
        assert paths == ["photo.png"]

    def test_multiple_directives(self) -> None:
        cleaned, paths = parse_attachment_directives(":img a.png @image=b.jpg look at both")
        assert cleaned == "look at both"
        assert paths == ["a.png", "b.jpg"]

    def test_directive_in_middle(self) -> None:
        cleaned, paths = parse_attachment_directives("what is :img a.png about")
        assert cleaned == "what is about"
        assert paths == ["a.png"]

    def test_no_directive(self) -> None:
        cleaned, paths = parse_attachment_directives("plain message")
        assert cleaned == "plain message"
        assert paths == []


class TestPendingAttachmentsQueue:
    def test_attach_and_drain_one_shot(self, tmp_path: Path) -> None:
        make_png(tmp_path / "a.png")
        queue = PendingAttachments()
        queue.attach("a.png", tmp_path)
        assert len(queue) == 1
        assert bool(queue)

        drained = queue.drain()
        assert len(drained) == 1
        assert drained[0].path.name == "a.png"
        # One-shot: drained queue is empty.
        assert len(queue) == 0
        assert not queue

    def test_add_and_peek(self, tmp_path: Path) -> None:
        make_png(tmp_path / "a.png")
        attachment = build_attachment("a.png", tmp_path)
        queue = PendingAttachments()
        queue.add(attachment)
        assert queue.peek() is attachment
        assert len(queue) == 1

    def test_drain_empty(self) -> None:
        queue = PendingAttachments()
        assert queue.drain() == []
        assert queue.peek() is None

    def test_attach_error_does_not_enqueue(self, tmp_path: Path) -> None:
        queue = PendingAttachments()
        with pytest.raises(AttachmentError):
            queue.attach("missing.png", tmp_path)
        assert len(queue) == 0


class TestModelSupportsVision:
    def test_vision_models(self) -> None:
        assert model_supports_vision("gpt-4o")
        assert model_supports_vision("anthropic/claude-3-5-sonnet")
        assert model_supports_vision("gemini-2.0-flash")
        assert model_supports_vision("ollama/llama3.2-vision")

    def test_non_vision_models(self) -> None:
        assert not model_supports_vision("deepseek-chat")
        assert not model_supports_vision("gpt-4-turbo-preview")
        assert not model_supports_vision("llama-3.1-8b")


class TestInterpretClipboard:
    def test_image_path_attaches(self, tmp_path: Path) -> None:
        make_png(tmp_path / "clip.png")
        attachment, notice = interpret_clipboard("clip.png", tmp_path)
        assert notice is None
        assert attachment is not None
        assert attachment.path.name == "clip.png"

    def test_plain_text_falls_through(self, tmp_path: Path) -> None:
        attachment, notice = interpret_clipboard("hello world", tmp_path)
        assert attachment is None
        assert notice is None

    def test_missing_image_path_returns_notice(self, tmp_path: Path) -> None:
        attachment, notice = interpret_clipboard("missing.png", tmp_path)
        assert attachment is None
        assert notice is not None
        assert "Image not found" in notice

    def test_non_image_extension_falls_through(self, tmp_path: Path) -> None:
        attachment, notice = interpret_clipboard("notes.txt", tmp_path)
        assert attachment is None
        assert notice is None

    def test_empty_clipboard_falls_through(self, tmp_path: Path) -> None:
        attachment, notice = interpret_clipboard("", tmp_path)
        assert attachment is None
        assert notice is None


class TestTuiWiring:
    @pytest.mark.asyncio
    async def test_pending_attachment_sent_with_next_message(self, tmp_path: Path) -> None:
        app = make_app(tmp_path)
        make_png(tmp_path / "shot.png")
        app._pending_attachments.attach("shot.png", tmp_path)

        with (
            patch("godspeed.tui.app.agent_loop", new=AsyncMock()) as mock_loop,
            patch("godspeed.tui.output._configured_statusline_template", return_value=None),
        ):
            await app._process_user_input(
                "describe this",
                running_loop=MagicMock(),
                sigint_installed=False,
            )

        # Multimodal message with image block added to conversation.
        call = app._conversation.add_user_message.call_args
        assert call is not None
        content = call.args[0]
        assert isinstance(content, list)
        assert {"type": "text", "text": "describe this"} in content
        image_blocks = [b for b in content if b["type"] == "image_url"]
        assert len(image_blocks) == 1
        assert image_blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")

        # Loop must not re-add the user message.
        mock_loop.assert_awaited_once()
        assert mock_loop.await_args.kwargs["skip_user_message"] is True

        # One-shot: queue cleared after send.
        assert len(app._pending_attachments) == 0

    @pytest.mark.asyncio
    async def test_vision_unsupported_sends_text_only(self, tmp_path: Path) -> None:
        app = make_app(tmp_path, model="deepseek-chat")
        make_png(tmp_path / "shot.png")
        app._pending_attachments.attach("shot.png", tmp_path)

        with (
            patch("godspeed.tui.app.agent_loop", new=AsyncMock()) as mock_loop,
            patch("godspeed.tui.output._configured_statusline_template", return_value=None),
            patch("godspeed.tui.app._output.console.print") as mock_print,
        ):
            await app._process_user_input(
                "describe this",
                running_loop=MagicMock(),
                sigint_installed=False,
            )

        # Warning notice shown.
        warning_text = printed_text(mock_print)
        assert "may not support vision" in warning_text

        # Message sent as text only — no image blocks added to conversation.
        app._conversation.add_user_message.assert_not_called()
        mock_loop.assert_awaited_once()
        assert mock_loop.await_args.kwargs["skip_user_message"] is False

        # One-shot: queue cleared even when dropped.
        assert len(app._pending_attachments) == 0

    @pytest.mark.asyncio
    async def test_img_directive_attaches_and_strips(self, tmp_path: Path) -> None:
        app = make_app(tmp_path)
        make_png(tmp_path / "shot.png")

        with (
            patch("godspeed.tui.app.agent_loop", new=AsyncMock()) as mock_loop,
            patch("godspeed.tui.output._configured_statusline_template", return_value=None),
            patch("godspeed.tui.app._output.console.print"),
        ):
            await app._process_user_input(
                ":img shot.png describe this",
                running_loop=MagicMock(),
                sigint_installed=False,
            )

        call = app._conversation.add_user_message.call_args
        assert call is not None
        content = call.args[0]
        assert {"type": "text", "text": "describe this"} in content
        assert any(b["type"] == "image_url" for b in content)
        assert len(app._pending_attachments) == 0

    @pytest.mark.asyncio
    async def test_invalid_img_directive_errors_without_crash(self, tmp_path: Path) -> None:
        app = make_app(tmp_path)

        with (
            patch("godspeed.tui.app.agent_loop", new=AsyncMock()) as mock_loop,
            patch("godspeed.tui.output._configured_statusline_template", return_value=None),
            patch("godspeed.tui.app._output.console.print") as mock_print,
        ):
            await app._process_user_input(
                ":img missing.png describe this",
                running_loop=MagicMock(),
                sigint_installed=False,
            )

        error_text = printed_text(mock_print)
        assert "Image not found" in error_text
        # Message still sent as text.
        mock_loop.assert_awaited_once()
        assert mock_loop.await_args.kwargs["skip_user_message"] is False

    def test_paste_attaches_image_path(self, tmp_path: Path) -> None:
        app = make_app(tmp_path)
        make_png(tmp_path / "clip.png")
        event = MagicMock()
        event.clipboard.get_data.return_value.text = "clip.png"

        with patch("godspeed.tui.app._output.console.print") as mock_print:
            app._on_paste(event)

        assert len(app._pending_attachments) == 1
        notice = printed_text(mock_print)
        assert "attached: clip.png" in notice
        assert "1x1" in notice
        # Image path was not inserted as text.
        event.current_buffer.insert_text.assert_not_called()

    def test_paste_plain_text_inserts(self, tmp_path: Path) -> None:
        app = make_app(tmp_path)
        event = MagicMock()
        event.clipboard.get_data.return_value.text = "hello world"

        app._on_paste(event)

        assert len(app._pending_attachments) == 0
        event.current_buffer.insert_text.assert_called_once_with("hello world")

    def test_paste_clipboard_error_graceful(self, tmp_path: Path) -> None:
        app = make_app(tmp_path)
        event = MagicMock()
        event.clipboard.get_data.side_effect = RuntimeError("no clipboard")

        with patch("godspeed.tui.app._output.console.print") as mock_print:
            app._on_paste(event)

        assert len(app._pending_attachments) == 0
        notice = printed_text(mock_print)
        assert "image paste unsupported in this terminal" in notice
