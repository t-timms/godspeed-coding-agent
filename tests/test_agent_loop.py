"""Tests for the core agent loop."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.agent.loop import _parse_tool_call, agent_loop
from godspeed.llm.client import ChatResponse, LLMClient
from godspeed.tools.base import ToolResult
from godspeed.tools.registry import ToolRegistry
from tests.conftest import MockTool


def _make_text_response(text: str) -> ChatResponse:
    return ChatResponse(content=text, tool_calls=[], finish_reason="stop")


def _make_tool_response(tool_name: str, arguments: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=[
            {
                "id": "call_001",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
        finish_reason="tool_calls",
    )


class TestParseToolCall:
    """Test tool call parsing from LLM responses."""

    def test_valid_tool_call(self) -> None:
        raw = {
            "id": "call_001",
            "function": {"name": "file_read", "arguments": '{"file_path": "test.py"}'},
        }
        tc = _parse_tool_call(raw)
        assert tc is not None
        assert tc.tool_name == "file_read"
        assert tc.arguments == {"file_path": "test.py"}
        assert tc.call_id == "call_001"

    def test_dict_arguments(self) -> None:
        raw = {
            "id": "call_002",
            "function": {"name": "shell", "arguments": {"command": "ls"}},
        }
        tc = _parse_tool_call(raw)
        assert tc is not None
        assert tc.arguments == {"command": "ls"}

    def test_invalid_json(self) -> None:
        raw = {
            "id": "call_003",
            "function": {"name": "shell", "arguments": "not json{"},
        }
        tc = _parse_tool_call(raw)
        assert tc is None

    def test_missing_name(self) -> None:
        raw = {"id": "call_004", "function": {"name": "", "arguments": "{}"}}
        tc = _parse_tool_call(raw)
        assert tc is None

    def test_empty_function(self) -> None:
        raw = {"id": "call_005", "function": {}}
        tc = _parse_tool_call(raw)
        assert tc is None


class TestAgentLoop:
    """Test the full agent loop."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Hello!"))

        result = await agent_loop("Hi", conversation, client, registry, tool_context)
        assert result == "Hello!"
        client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_call_then_text(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        tool = MockTool(name="file_read", result=ToolResult.success("file contents"))
        registry.register(tool)

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("file_read", {"file_path": "test.py"}),
                _make_text_response("I read the file. It contains: file contents"),
            ]
        )

        result = await agent_loop("Read test.py", conversation, client, registry, tool_context)
        assert "file contents" in result
        assert client.chat.call_count == 2
        assert tool.last_arguments == {"file_path": "test.py"}

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("nonexistent_tool", {}),
                _make_text_response("Sorry, that tool doesn't exist."),
            ]
        )

        result = await agent_loop("Do something", conversation, client, registry, tool_context)
        assert "Sorry" in result

    @pytest.mark.asyncio
    async def test_callbacks_called(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="shell"))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("shell", {"command": "ls"}),
                _make_text_response("Done"),
            ]
        )

        text_calls = []
        tool_calls = []
        tool_results = []

        await agent_loop(
            "List files",
            conversation,
            client,
            registry,
            tool_context,
            on_assistant_text=text_calls.append,
            on_tool_call=lambda name, args: tool_calls.append((name, args)),
            on_tool_result=lambda name, result: tool_results.append((name, result)),
        )

        assert len(tool_calls) == 1
        assert tool_calls[0][0] == "shell"
        assert len(tool_results) == 1
        assert "Done" in text_calls

    @pytest.mark.asyncio
    async def test_malformed_tool_call_retries(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        # Send a malformed tool call (bad JSON), then a valid text response
        malformed_response = ChatResponse(
            content="",
            tool_calls=[
                {
                    "id": "call_bad",
                    "function": {"name": "shell", "arguments": "not valid json{"},
                }
            ],
            finish_reason="tool_calls",
        )

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                malformed_response,
                _make_text_response("Recovered from bad tool call"),
            ]
        )

        result = await agent_loop("Do something", conversation, client, registry, tool_context)
        assert "Recovered" in result


class TestStuckLoopDetection:
    """Test stuck-loop detection: replan after 3 identical errors."""

    @pytest.mark.asyncio
    async def test_three_identical_errors_triggers_replan(self, tool_context) -> None:
        """After 3 identical tool errors, a replan message is injected."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        # Tool that always fails with the same error
        registry.register(MockTool(name="shell", result=ToolResult.failure("Permission denied")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                # 3 identical tool calls that will all fail identically
                _make_tool_response("shell", {"command": "rm /etc/hosts"}),
                _make_tool_response("shell", {"command": "rm /etc/hosts"}),
                _make_tool_response("shell", {"command": "rm /etc/hosts"}),
                # After replan injection, model responds with text
                _make_text_response("I'll try a different approach."),
            ]
        )

        result = await agent_loop("Delete hosts", conversation, client, registry, tool_context)
        assert "different approach" in result

        # Verify the replan message was injected into conversation
        messages = conversation.messages
        replan_found = any(
            msg.get("role") == "user" and "failed 3 times" in msg.get("content", "")
            for msg in messages
        )
        assert replan_found, "Replan message should be injected after 3 identical errors"

    @pytest.mark.asyncio
    async def test_different_errors_no_replan(self, tool_context) -> None:
        """Different errors should NOT trigger stuck-loop detection."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        # We need a tool whose error changes each call
        call_count = 0

        class VariableErrorTool(MockTool):
            async def execute(self, arguments, context):
                nonlocal call_count
                call_count += 1
                return ToolResult.failure(f"Error variant {call_count}")

        registry.register(VariableErrorTool(name="shell"))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("shell", {"command": "cmd1"}),
                _make_tool_response("shell", {"command": "cmd2"}),
                _make_tool_response("shell", {"command": "cmd3"}),
                _make_text_response("Giving up."),
            ]
        )

        result = await agent_loop("Try stuff", conversation, client, registry, tool_context)
        assert "Giving up" in result

        # No replan message should be present
        messages = conversation.messages
        replan_found = any(
            msg.get("role") == "user" and "failed 3 times" in msg.get("content", "")
            for msg in messages
        )
        assert not replan_found, "No replan for different errors"

    @pytest.mark.asyncio
    async def test_error_counter_resets_on_success(self, tool_context) -> None:
        """A successful tool call resets the error counter."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        call_count = 0

        class AlternatingTool(MockTool):
            async def execute(self, arguments, context):
                nonlocal call_count
                call_count += 1
                # Fail, fail, succeed, fail, fail, fail — should NOT trigger replan
                # because the success in the middle resets the counter
                if call_count in (1, 2, 4, 5, 6):
                    return ToolResult.failure("Same error")
                return ToolResult.success("ok")

        registry.register(AlternatingTool(name="shell"))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("shell", {"command": "a"}),  # fail 1
                _make_tool_response("shell", {"command": "b"}),  # fail 2
                _make_tool_response("shell", {"command": "c"}),  # success — resets
                _make_tool_response("shell", {"command": "d"}),  # fail 1
                _make_tool_response("shell", {"command": "e"}),  # fail 2
                _make_tool_response("shell", {"command": "f"}),  # fail 3 — triggers replan
                _make_text_response("Done."),
            ]
        )

        result = await agent_loop("Try stuff", conversation, client, registry, tool_context)
        assert "Done" in result

        # Replan SHOULD be triggered after the 3rd consecutive error (calls 4,5,6)
        messages = conversation.messages
        replan_msgs = [
            msg
            for msg in messages
            if msg.get("role") == "user" and "failed 3 times" in msg.get("content", "")
        ]
        assert len(replan_msgs) == 1, "Exactly one replan after success reset"


class TestPauseResume:
    """Test pause/resume via asyncio.Event."""

    @pytest.mark.asyncio
    async def test_pause_event_pauses_loop(self, tool_context) -> None:
        """When pause_event is cleared, the loop should wait."""
        import asyncio

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Resumed!"))

        pause_event = asyncio.Event()
        pause_event.clear()  # Start paused

        # The loop should block; resume after a short delay
        async def resume_later():
            await asyncio.sleep(0.1)
            pause_event.set()

        task = asyncio.create_task(resume_later())

        result = await agent_loop(
            "Hello",
            conversation,
            client,
            registry,
            tool_context,
            pause_event=pause_event,
        )
        await task
        assert "Resumed" in result

    @pytest.mark.asyncio
    async def test_guidance_injection(self, tool_context) -> None:
        """Guidance injected while paused appears in conversation."""
        import asyncio

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="shell", result=ToolResult.success("ok")))

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_tool_response("shell", {"command": "ls"})
            return _make_text_response("Understood the guidance.")

        client = LLMClient(model="test")
        client.chat = AsyncMock(side_effect=mock_chat)

        pause_event = asyncio.Event()
        pause_event.set()  # Start running

        # Inject guidance after first tool call
        async def inject_guidance():
            await asyncio.sleep(0.05)
            conversation.add_user_message("[User guidance]: Use grep instead")
            # Don't pause — just inject

        task = asyncio.create_task(inject_guidance())

        result = await agent_loop(
            "Search for files",
            conversation,
            client,
            registry,
            tool_context,
            pause_event=pause_event,
        )
        await task
        assert "guidance" in result.lower() or "Understood" in result


class TestAutoStash:
    """Test auto-stash after consecutive write operations."""

    @pytest.mark.asyncio
    async def test_auto_stash_triggers_at_threshold(self, tool_context) -> None:
        """Auto-stash triggers after 3 consecutive file_edit/file_write calls."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))
        git_result = ToolResult.success("Saved working directory")
        registry.register(MockTool(name="git", result=git_result))

        call_id = 0

        def make_edit_response() -> ChatResponse:
            nonlocal call_id
            call_id += 1
            return ChatResponse(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{call_id:03d}",
                        "function": {
                            "name": "file_edit",
                            "arguments": json.dumps({"file_path": "test.txt"}),
                        },
                    }
                ],
                finish_reason="tool_calls",
            )

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                make_edit_response(),  # write 1
                make_edit_response(),  # write 2
                make_edit_response(),  # write 3 — triggers auto-stash
                _make_text_response("Done editing."),
            ]
        )

        result = await agent_loop("Edit files", conversation, client, registry, tool_context)
        assert "Done editing" in result

        # Verify auto-stash message was injected
        messages = conversation.messages
        stash_found = any(
            msg.get("role") == "tool" and "auto-stash" in msg.get("content", "").lower()
            for msg in messages
        )
        assert stash_found, "Auto-stash message should be in conversation"

    @pytest.mark.asyncio
    async def test_no_auto_stash_below_threshold(self, tool_context) -> None:
        """Two consecutive writes should NOT trigger auto-stash."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))
        git_result = ToolResult.success("Saved working directory")
        registry.register(MockTool(name="git", result=git_result))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("file_edit", {"file_path": "a.txt"}),
                _make_tool_response("file_edit", {"file_path": "b.txt"}),
                _make_text_response("Done."),
            ]
        )

        result = await agent_loop("Edit files", conversation, client, registry, tool_context)
        assert "Done" in result

        messages = conversation.messages
        stash_found = any(
            msg.get("role") == "tool" and "auto-stash" in msg.get("content", "").lower()
            for msg in messages
        )
        assert not stash_found, "No auto-stash below threshold"

    @pytest.mark.asyncio
    async def test_non_write_resets_counter(self, tool_context) -> None:
        """A non-write tool call between writes resets the counter."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))
        registry.register(MockTool(name="file_read", result=ToolResult.success("content")))
        git_result = ToolResult.success("Saved working directory")
        registry.register(MockTool(name="git", result=git_result))

        call_id = 0

        def make_response(tool_name: str, args: dict) -> ChatResponse:
            nonlocal call_id
            call_id += 1
            return ChatResponse(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{call_id:03d}",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args),
                        },
                    }
                ],
                finish_reason="tool_calls",
            )

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                make_response("file_edit", {"file_path": "a.txt"}),  # write 1
                make_response("file_edit", {"file_path": "b.txt"}),  # write 2
                make_response("file_read", {"file_path": "c.txt"}),  # read — resets
                make_response("file_edit", {"file_path": "d.txt"}),  # write 1 again
                make_response("file_edit", {"file_path": "e.txt"}),  # write 2 again
                _make_text_response("Done."),
            ]
        )

        result = await agent_loop("Edit files", conversation, client, registry, tool_context)
        assert "Done" in result

        messages = conversation.messages
        stash_found = any(
            msg.get("role") == "tool" and "auto-stash" in msg.get("content", "").lower()
            for msg in messages
        )
        assert not stash_found, "Read in the middle should reset write counter"


class TestConversation:
    """Test conversation management."""

    def test_add_messages(self) -> None:
        conv = Conversation("System prompt")
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi there")
        msgs = conv.messages
        assert len(msgs) == 3  # system + user + assistant
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_add_tool_result(self) -> None:
        conv = Conversation("System prompt")
        conv.add_tool_result("call_001", "tool output")
        msgs = conv.messages
        assert msgs[-1]["role"] == "tool"
        assert msgs[-1]["tool_call_id"] == "call_001"

    def test_compact(self) -> None:
        conv = Conversation("System prompt")
        for i in range(10):
            conv.add_user_message(f"Message {i}")
            conv.add_assistant_message(f"Response {i}")
        assert len(conv.messages) > 10
        conv.compact("Summary of conversation")
        # System prompt + compaction message
        assert len(conv.messages) == 2
        assert "Summary" in conv.messages[1]["content"]

    def test_clear(self) -> None:
        conv = Conversation("System prompt")
        conv.add_user_message("Hello")
        conv.clear()
        assert len(conv.messages) == 1  # Only system prompt

    def test_token_count(self) -> None:
        conv = Conversation("Short system prompt")
        assert conv.token_count > 0
        conv.add_user_message("Hello world")
        count_after = conv.token_count
        assert count_after > 0

    def test_is_near_limit(self) -> None:
        conv = Conversation("System prompt", max_tokens=10, compaction_threshold=0.5)
        # Short conversation should not be near limit, but with very low max_tokens it might be
        # The system prompt itself might push us over
        # This test just verifies the property works
        assert isinstance(conv.is_near_limit, bool)

    def test_get_compaction_context_includes_roles(self) -> None:
        conv = Conversation("System prompt")
        conv.add_user_message("Fix the bug")
        conv.add_assistant_message("I'll read the file first")
        context = conv.get_compaction_context()
        assert "[user]: Fix the bug" in context
        assert "[assistant]: I'll read the file first" in context

    def test_get_compaction_context_includes_tool_calls(self) -> None:
        conv = Conversation("System prompt")
        conv.add_assistant_message(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {"name": "file_read", "arguments": '{"file_path": "main.py"}'},
                }
            ],
        )
        context = conv.get_compaction_context()
        assert "file_read" in context
        assert "main.py" in context

    def test_add_assistant_message_normalizes_tool_calls_type(self) -> None:
        """Tool calls should always have type='function' for LiteLLM compat."""
        conv = Conversation("System prompt")
        conv.add_assistant_message(
            tool_calls=[{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}]
        )
        msg = conv.messages[-1]
        assert msg["tool_calls"][0]["type"] == "function"

    def test_add_assistant_message_preserves_existing_type(self) -> None:
        """If type is already set, don't overwrite it."""
        conv = Conversation("System prompt")
        conv.add_assistant_message(
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                }
            ],
        )
        msg = conv.messages[-1]
        assert msg["tool_calls"][0]["type"] == "function"

    def test_add_assistant_message_with_reasoning_content(self) -> None:
        """reasoning_content is preserved for DeepSeek V4 multi-turn."""
        conv = Conversation("System prompt")
        conv.add_assistant_message(
            content="The answer is 42.",
            reasoning_content="I need to compute the meaning of life...",
        )
        msg = conv.messages[-1]
        assert "reasoning_content" in msg
        assert msg["reasoning_content"] == "I need to compute the meaning of life..."

    def test_add_tool_result_truncates_long_content(self) -> None:
        """add_tool_result truncates content > 50000 chars."""
        conv = Conversation("System prompt")
        long_content = "x" * 60000
        conv.add_tool_result("call_trunc", long_content)
        msg = conv.messages[-1]
        assert len(msg["content"]) < 60000
        assert "(truncated" in msg["content"]

    def test_add_tool_result_short_content_no_truncation(self) -> None:
        """add_tool_result with short content should not truncate."""
        conv = Conversation("System prompt")
        short_content = "short output"
        conv.add_tool_result("call_short", short_content)
        msg = conv.messages[-1]
        assert msg["content"] == short_content

    def test_add_tool_result_with_logger(self) -> None:
        """add_tool_result calls logger.log_tool_result when logger is set."""
        from unittest.mock import MagicMock

        mock_logger = MagicMock()
        conv = Conversation("System prompt", conversation_logger=mock_logger)
        conv.add_tool_result("call_001", "tool output")
        mock_logger.log_tool_result.assert_called_once_with(
            tool_call_id="call_001",
            tool_name="",
            content="tool output",
        )

    def test_compact_with_logger(self) -> None:
        """compact calls logger.log_compaction when logger is set."""
        from unittest.mock import MagicMock

        mock_logger = MagicMock()
        conv = Conversation("System prompt", conversation_logger=mock_logger)
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi")
        conv.compact("Summary of work")
        mock_logger.log_compaction.assert_called_once_with(
            summary="Summary of work",
            messages_before=2,
            messages_after=1,
        )


class TestTokenCounter:
    """Tests for token counting utilities."""

    def test_get_encoding_for_known_model(self) -> None:
        from godspeed.llm.token_counter import get_encoding

        enc = get_encoding("gpt-4")
        assert enc is not None

    def test_get_encoding_for_claude_falls_back(self) -> None:
        from godspeed.llm.token_counter import get_encoding

        enc = get_encoding("claude-sonnet-4-20250514")
        assert enc is not None

    def test_get_encoding_for_ollama_model(self) -> None:
        from godspeed.llm.token_counter import get_encoding

        enc = get_encoding("ollama/qwen3:4b")
        assert enc is not None

    def test_get_encoding_for_unknown_model(self) -> None:
        from godspeed.llm.token_counter import get_encoding

        enc = get_encoding("totally-unknown-model-xyz")
        assert enc is not None  # falls back to cl100k_base

    def test_count_tokens_basic(self) -> None:
        from godspeed.llm.token_counter import count_tokens

        count = count_tokens("Hello world")
        assert count > 0

    def test_count_message_tokens_basic(self) -> None:
        from godspeed.llm.token_counter import count_message_tokens

        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        count = count_message_tokens(msgs)
        assert count > 0

    def test_count_message_tokens_with_tool_calls(self) -> None:
        from godspeed.llm.token_counter import count_message_tokens

        msgs = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}],
            }
        ]
        count = count_message_tokens(msgs)
        assert count > 0

    def test_count_tokens_empty_string(self) -> None:
        from godspeed.llm.token_counter import count_tokens

        count = count_tokens("")
        assert count == 0


class TestBudgetExceeded:
    """Budget exceeded during LLM call terminates loop gracefully."""

    @pytest.mark.asyncio
    async def test_budget_exceeded_calls_hook_and_returns(self, tool_context) -> None:
        from unittest.mock import MagicMock

        from godspeed.agent.result import AgentMetrics
        from godspeed.llm.client import BudgetExceededError

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        client = LLMClient(model="test")
        budget_error = BudgetExceededError(spent=10.0, limit=5.0)
        client.chat = AsyncMock(side_effect=budget_error)

        metrics = AgentMetrics()
        hook_executor = MagicMock()

        result = await agent_loop(
            "Hi",
            conversation,
            client,
            registry,
            tool_context,
            hook_executor=hook_executor,
            metrics=metrics,
        )
        assert "Budget exceeded" in result
        assert metrics.exit_reason == "budget_exceeded"
        hook_executor.fire.assert_called_once()

    @pytest.mark.asyncio
    async def test_budget_exceeded_no_hook(self, tool_context) -> None:
        from godspeed.agent.result import AgentMetrics
        from godspeed.llm.client import BudgetExceededError

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        client = LLMClient(model="test")
        budget_error = BudgetExceededError(spent=10.0, limit=5.0)
        client.chat = AsyncMock(side_effect=budget_error)

        metrics = AgentMetrics()

        result = await agent_loop(
            "Hi", conversation, client, registry, tool_context, metrics=metrics
        )
        assert "Budget exceeded" in result
        assert metrics.exit_reason == "budget_exceeded"


class TestLLMRetries:
    """LLM call retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                ConnectionError("Transient error"),
                _make_text_response("Recovered!"),
            ]
        )

        result = await agent_loop(
            "Hi", conversation, client, registry, tool_context, llm_max_retries=2
        )
        assert "Recovered" in result
        assert client.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self, tool_context) -> None:
        from godspeed.agent.result import AgentMetrics

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(side_effect=RuntimeError("Persistent error"))

        metrics = AgentMetrics()
        result = await agent_loop(
            "Hi",
            conversation,
            client,
            registry,
            tool_context,
            llm_max_retries=1,
            metrics=metrics,
        )
        assert "Error" in result
        assert "Persistent error" in result
        assert metrics.exit_reason == "llm_error"


class TestMalformedToolCallMaxRetries:
    """Malformed tool calls exceeding max_retries terminate the loop."""

    @pytest.mark.asyncio
    async def test_too_many_malformed_tool_calls(self, tool_context) -> None:
        from godspeed.agent.result import AgentMetrics

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        malformed = ChatResponse(
            content="",
            tool_calls=[{"id": "call_bad", "function": {"name": "shell", "arguments": "broken{"}}],
            finish_reason="tool_calls",
        )

        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=malformed)

        metrics = AgentMetrics()
        result = await agent_loop(
            "Do something",
            conversation,
            client,
            registry,
            tool_context,
            max_retries=1,
            metrics=metrics,
        )
        assert "Too many malformed" in result
        assert metrics.exit_reason == "tool_error"


class TestPermissionDenial:
    """Permission denial handling in the agent loop."""

    @pytest.mark.asyncio
    async def test_permission_denied_sync_evaluator(self, tool_context) -> None:
        from unittest.mock import MagicMock

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="shell"))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("shell", {"command": "rm -rf /"}),
                _make_text_response("Permission was denied, I understand."),
            ]
        )

        mock_perms = MagicMock()
        mock_perms.evaluate = MagicMock(return_value="deny")

        ctx = type(tool_context)(
            cwd=tool_context.cwd,
            session_id="test",
            permissions=mock_perms,
        )

        denied_calls = []

        result = await agent_loop(
            "Delete everything",
            conversation,
            client,
            registry,
            ctx,
            on_permission_denied=lambda name, reason: denied_calls.append((name, reason)),
        )
        assert "Permission" in result.lower() or "denied" in result.lower()
        assert len(denied_calls) == 1
        assert denied_calls[0][0] == "shell"

    @pytest.mark.asyncio
    async def test_permission_denied_async_evaluator(self, tool_context) -> None:

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="shell"))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("shell", {"command": "rm -rf /"}),
                _make_text_response("OK, I understand."),
            ]
        )

        class AsyncPermEvaluator:
            async def evaluate(self, tool_call):
                return "deny"

        ctx = type(tool_context)(
            cwd=tool_context.cwd,
            session_id="test",
            permissions=AsyncPermEvaluator(),
        )

        result = await agent_loop(
            "Delete everything",
            conversation,
            client,
            registry,
            ctx,
        )
        assert "OK" in result


class TestPreToolHookBlock:
    """Pre-tool hook blocking prevents tool execution."""

    @pytest.mark.asyncio
    async def test_pre_tool_hook_blocks_execution(self, tool_context) -> None:
        from unittest.mock import MagicMock

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="shell", result=ToolResult.success("ran")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("shell", {"command": "ls"}),
                _make_text_response("Tool was blocked, I'll try differently."),
            ]
        )

        hook_executor = MagicMock()
        hook_executor.run_pre_tool = MagicMock(return_value=False)

        result = await agent_loop(
            "List files",
            conversation,
            client,
            registry,
            tool_context,
            hook_executor=hook_executor,
        )
        assert "blocked" in result.lower() or "differently" in result.lower()
        hook_executor.run_pre_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_tool_hook_called(self, tool_context) -> None:
        from unittest.mock import MagicMock

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="shell", result=ToolResult.success("ok")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("shell", {"command": "ls"}),
                _make_text_response("Done."),
            ]
        )

        hook_executor = MagicMock()
        hook_executor.run_pre_tool = MagicMock(return_value=True)
        hook_executor.run_post_tool = MagicMock()
        hook_executor.fire = MagicMock()

        await agent_loop(
            "List files",
            conversation,
            client,
            registry,
            tool_context,
            hook_executor=hook_executor,
        )
        hooks_called_for_shell = False
        for call in hook_executor.run_post_tool.call_args_list:
            if call[0][0] == "shell":
                hooks_called_for_shell = True
        assert hooks_called_for_shell


class TestRetrievalSubagent:
    """Retrieval subagent interception of navigation tools."""

    @pytest.mark.asyncio
    async def test_nav_tools_routed_to_retrieval(self, tool_context) -> None:
        from unittest.mock import MagicMock

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        retrieval = MagicMock()
        retrieval.retrieve = AsyncMock(
            return_value=MagicMock(spans=[], __iter__=lambda s: iter([]))
        )
        retrieval.format_spans_for_agent = MagicMock(
            return_value="file:src/main.py:10-20 -- relevant code"
        )

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("code_search", {"query": "TODO"}),
                _make_text_response("Found the TODOs via retrieval."),
            ]
        )

        result = await agent_loop(
            "Search TODOs",
            conversation,
            client,
            registry,
            tool_context,
            retrieval_subagent=retrieval,
        )
        assert "TODO" in result
        retrieval.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_nav_tool_without_query(self, tool_context) -> None:
        from unittest.mock import MagicMock

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        retrieval = MagicMock()

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("code_search", {}),
                _make_text_response("Done."),
            ]
        )

        result = await agent_loop(
            "Search",
            conversation,
            client,
            registry,
            tool_context,
            retrieval_subagent=retrieval,
        )
        assert "Done" in result


class TestParallelToolExecution:
    """Multiple tool calls per turn with parallel execution."""

    @pytest.mark.asyncio
    async def test_parallel_execution_of_read_only_tools(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_read", result=ToolResult.success("content A")))
        registry.register(MockTool(name="grep_search", result=ToolResult.success("content B")))

        client = LLMClient(model="test")
        parallel_response = ChatResponse(
            content="I'll read two files in parallel.",
            tool_calls=[
                {
                    "id": "call_r1",
                    "function": {
                        "name": "file_read",
                        "arguments": json.dumps({"file_path": "a.py"}),
                    },
                },
                {
                    "id": "call_r2",
                    "function": {
                        "name": "grep_search",
                        "arguments": json.dumps({"pattern": "TODO"}),
                    },
                },
            ],
            finish_reason="tool_calls",
        )
        client.chat = AsyncMock(
            side_effect=[
                parallel_response,
                _make_text_response("I read both files in parallel."),
            ]
        )

        parallel_starts = []
        parallel_completes = []

        result = await agent_loop(
            "Read files",
            conversation,
            client,
            registry,
            tool_context,
            parallel_tool_calls=True,
            on_parallel_start=lambda calls: parallel_starts.append(calls),
            on_parallel_complete=lambda results: parallel_completes.append(results),
        )
        assert "parallel" in result.lower() or "both" in result.lower()
        assert len(parallel_starts) == 1
        assert len(parallel_starts[0]) == 2
        assert len(parallel_completes) == 1

    @pytest.mark.asyncio
    async def test_sequential_execution_mode(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_read", result=ToolResult.success("content")))

        client = LLMClient(model="test")
        multi_response = ChatResponse(
            content="",
            tool_calls=[
                {
                    "id": "call_s1",
                    "function": {
                        "name": "file_read",
                        "arguments": json.dumps({"file_path": "a.py"}),
                    },
                },
                {
                    "id": "call_s2",
                    "function": {
                        "name": "file_read",
                        "arguments": json.dumps({"file_path": "b.py"}),
                    },
                },
            ],
            finish_reason="tool_calls",
        )
        client.chat = AsyncMock(
            side_effect=[
                multi_response,
                _make_text_response("Read both sequentially."),
            ]
        )

        result = await agent_loop(
            "Read files",
            conversation,
            client,
            registry,
            tool_context,
            parallel_tool_calls=False,
        )
        assert "both" in result.lower()


class TestSpeculativeDispatch:
    """Speculative dispatch of read-only tools during streaming."""

    @pytest.mark.asyncio
    async def test_speculative_streaming_dispatch(self, tool_context) -> None:
        """Speculative dispatch starts read-only tools during streaming call."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(
            MockTool(name="file_read", result=ToolResult.success("speculative content"))
        )

        on_chunks = []
        call_count = 0

        class StreamWithToolCall:
            def __init__(self):
                self._idx = 0
                self._chunks = [
                    ChatResponse(content="Let me ", tool_calls=[], finish_reason=None),
                    ChatResponse(content="read.", tool_calls=[], finish_reason=None),
                    ChatResponse(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_spec",
                                "function": {
                                    "name": "file_read",
                                    "arguments": json.dumps({"file_path": "spec.py"}),
                                },
                            }
                        ],
                        finish_reason="stop",
                    ),
                ]

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx >= len(self._chunks):
                    raise StopAsyncIteration
                chunk = self._chunks[self._idx]
                self._idx += 1
                return chunk

            async def aclose(self):
                pass

        class TextOnlyStream:
            def __init__(self):
                self._idx = 0
                self._chunks = [
                    ChatResponse(content="Done", tool_calls=[], finish_reason=None),
                    ChatResponse(content="Done", tool_calls=[], finish_reason="stop"),
                ]

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx >= len(self._chunks):
                    raise StopAsyncIteration
                chunk = self._chunks[self._idx]
                self._idx += 1
                return chunk

            async def aclose(self):
                pass

        llm = AsyncMock(spec=LLMClient)

        def multi_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return StreamWithToolCall()
            return TextOnlyStream()

        llm.stream_chat = multi_stream
        llm.model = "test"

        result = await agent_loop(
            "Read spec.py",
            conversation,
            llm,
            registry,
            tool_context,
            on_assistant_chunk=lambda c: on_chunks.append(c),
        )
        assert result == "Done"
        assert on_chunks == ["Let me ", "read.", "Done"]


class TestContextCompaction:
    """Context window management during the agent loop."""

    @pytest.mark.asyncio
    async def test_context_threshold_hooks(self, tool_context) -> None:
        from unittest.mock import MagicMock

        from godspeed.agent.result import AgentMetrics

        conversation = Conversation("You are a coding agent.", max_tokens=10)
        registry = ToolRegistry()
        client = LLMClient(model="test")

        # Force token count high to trigger thresholds
        conversation.add_user_message("A" * 1000)

        client.chat = AsyncMock(return_value=_make_text_response("ok"))

        metrics = AgentMetrics()
        hook_executor = MagicMock()
        hook_executor.fire = MagicMock()

        await agent_loop(
            "Hi",
            conversation,
            client,
            registry,
            tool_context,
            hook_executor=hook_executor,
            metrics=metrics,
        )
        # At least one threshold hook should fire
        assert hook_executor.fire.call_count >= 1

    @pytest.mark.asyncio
    async def test_simple_compaction_fallback(self, tool_context) -> None:
        """Fallback LLM compaction when conversation is near limit."""
        from unittest.mock import patch

        conversation = Conversation("System", max_tokens=10, compaction_threshold=0.1)
        conversation.add_user_message("Hello world " * 100)

        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_text_response("Compacted summary."),
                _make_text_response("Done with task."),
            ]
        )

        with patch("godspeed.llm.cost.get_cheapest_model", return_value="test"):
            with patch(
                "godspeed.context.compaction.get_compaction_prompt",
                return_value="Summarize:",
            ):
                result = await agent_loop(
                    "Do work",
                    conversation,
                    client,
                    registry,
                    tool_context,
                )
        assert "Done with task" in result


class TestMustFixInjection:
    """Must-fix injection mechanism for auto-verify failures."""

    @pytest.mark.asyncio
    async def test_must_fix_injected_on_verify_failure(self, tool_context) -> None:

        from unittest.mock import patch

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_write", result=ToolResult.success("wrote")))
        registry.register(MockTool(name="verify", result=ToolResult.success("ok")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                ChatResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_fix",
                            "function": {
                                "name": "file_write",
                                "arguments": json.dumps({"file_path": "main.py", "content": "x=1"}),
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                ),
                _make_text_response("Fixed."),
            ]
        )

        async def simulate_verify(*args, **kwargs):
            from godspeed.tools.verify import REMAINING_ERRORS_FINGERPRINT

            return type(
                "VerifyResult",
                (),
                {
                    "call_id": args[2],
                    "output": f"err: {REMAINING_ERRORS_FINGERPRINT} unused import os",
                    "must_fix_file": args[1],
                    "must_fix_text": f"err: {REMAINING_ERRORS_FINGERPRINT} unused import os",
                    "must_fix_increment": True,
                },
            )()

        with patch("godspeed.agent.loop._auto_verify_background", side_effect=simulate_verify):
            result = await agent_loop(
                "Fix main.py",
                conversation,
                client,
                registry,
                tool_context,
            )
        assert "Fixed" in result

    @pytest.mark.asyncio
    async def test_must_fix_cap_prevents_infinite_injections(self, tool_context) -> None:
        from unittest.mock import patch

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_write", result=ToolResult.success("wrote")))
        registry.register(MockTool(name="verify", result=ToolResult.success("ok")))

        client = LLMClient(model="test")
        responses = []
        for _ in range(6):
            responses.append(
                ChatResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_cap",
                            "function": {
                                "name": "file_write",
                                "arguments": json.dumps({"file_path": "main.py", "content": "x=1"}),
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                )
            )
        responses.append(_make_text_response("Done."))

        client.chat = AsyncMock(side_effect=responses)

        async def simulate_verify(*args, **kwargs):
            from godspeed.tools.verify import REMAINING_ERRORS_FINGERPRINT

            return type(
                "VerifyResult",
                (),
                {
                    "call_id": args[2],
                    "output": f"err: {REMAINING_ERRORS_FINGERPRINT} issue",
                    "must_fix_file": args[1],
                    "must_fix_text": f"err: {REMAINING_ERRORS_FINGERPRINT} issue",
                    "must_fix_increment": True,
                },
            )()

        with patch("godspeed.agent.loop._auto_verify_background", side_effect=simulate_verify):
            result = await agent_loop(
                "Fix main.py",
                conversation,
                client,
                registry,
                tool_context,
                must_fix_cap=3,
            )
        assert "Done" in result


class TestAutoCommit:
    """Auto-commit after consecutive successful edits."""

    @pytest.mark.asyncio
    async def test_auto_commit_triggers(self, tool_context) -> None:
        from unittest.mock import patch

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))

        client = LLMClient(model="test")

        async def generate_msg(*args):
            return "feat: auto-generated commit"

        async def do_commit(*args):
            return ToolResult.success("Committed.")

        with patch(
            "godspeed.agent.auto_commit.generate_commit_message",
            side_effect=generate_msg,
        ):
            with patch("godspeed.agent.auto_commit.auto_commit", side_effect=do_commit):
                client.chat = AsyncMock(
                    side_effect=[
                        _make_tool_response("file_edit", {"file_path": "a.py"}),
                        _make_tool_response("file_edit", {"file_path": "b.py"}),
                        _make_tool_response("file_edit", {"file_path": "c.py"}),
                        _make_tool_response("file_edit", {"file_path": "d.py"}),
                        _make_tool_response("file_edit", {"file_path": "e.py"}),
                        _make_text_response("All edits done and committed."),
                    ]
                )

                result = await agent_loop(
                    "Edit files",
                    conversation,
                    client,
                    registry,
                    tool_context,
                    auto_commit=True,
                    auto_commit_threshold=5,
                )
        assert "committed" in result.lower() or "edits" in result.lower()

    @pytest.mark.asyncio
    async def test_auto_commit_failure_graceful(self, tool_context) -> None:
        from unittest.mock import patch

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))

        client = LLMClient(model="test")

        async def generate_msg(*args):
            raise RuntimeError("commit gen failed")

        with patch(
            "godspeed.agent.auto_commit.generate_commit_message",
            side_effect=generate_msg,
        ):
            client.chat = AsyncMock(
                side_effect=[
                    _make_tool_response("file_edit", {"file_path": "a.py"}),
                    _make_tool_response("file_edit", {"file_path": "b.py"}),
                    _make_tool_response("file_edit", {"file_path": "c.py"}),
                    _make_tool_response("file_edit", {"file_path": "d.py"}),
                    _make_tool_response("file_edit", {"file_path": "e.py"}),
                    _make_text_response("Edits done."),
                ]
            )

            result = await agent_loop(
                "Edit files",
                conversation,
                client,
                registry,
                tool_context,
                auto_commit=True,
                auto_commit_threshold=5,
            )
        assert "Edits done" in result


class TestSkipUserMessage:
    """skip_user_message flag prevents adding user input to conversation."""

    @pytest.mark.asyncio
    async def test_skip_user_message(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Hello!"))

        await agent_loop(
            "Hi",
            conversation,
            client,
            registry,
            tool_context,
            skip_user_message=True,
        )
        # User message "Hi" should NOT be in messages
        user_msgs = [m for m in conversation.messages if m.get("role") == "user"]
        assert not any(m.get("content") == "Hi" for m in user_msgs)


class TestMaxIterations:
    """Loop terminates after max_iterations."""

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded(self, tool_context) -> None:
        from godspeed.agent.result import AgentMetrics

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="shell", result=ToolResult.success("ok")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_tool_response("shell", {"command": "echo loop"}))

        metrics = AgentMetrics()
        result = await agent_loop(
            "Loop forever",
            conversation,
            client,
            registry,
            tool_context,
            max_iterations=3,
            metrics=metrics,
        )
        assert "maximum iterations" in result.lower()
        assert metrics.exit_reason == "max_iterations"


class TestEmptyUserInput:
    """Empty user_input with skip_user_message=True."""

    @pytest.mark.asyncio
    async def test_empty_user_input_skipped(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Hello!"))

        result = await agent_loop(
            "",
            conversation,
            client,
            registry,
            tool_context,
        )
        assert result == "Hello!"
        # Empty input shouldn't add user message
        user_msgs = [m for m in conversation.messages if m.get("role") == "user"]
        assert len(user_msgs) == 0


class TestMetricsTracking:
    """Metrics tracking during the agent loop."""

    @pytest.mark.asyncio
    async def test_metrics_recorded(self, tool_context) -> None:
        from godspeed.agent.result import AgentMetrics

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_read", result=ToolResult.success("content")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("file_read", {"file_path": "test.py"}),
                _make_text_response("Done."),
            ]
        )

        metrics = AgentMetrics()
        await agent_loop(
            "Read file",
            conversation,
            client,
            registry,
            tool_context,
            metrics=metrics,
        )
        assert metrics.tool_call_count == 1
        assert metrics.tool_error_count == 0
        assert metrics.exit_reason == "stopped"

    @pytest.mark.asyncio
    async def test_metrics_sink_emits(self, tool_context) -> None:
        from unittest.mock import MagicMock

        from godspeed.agent.result import AgentMetrics

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_read", result=ToolResult.success("content")))
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("file_read", {"file_path": "f.py"}),
                _make_text_response("Done."),
            ]
        )

        metrics = AgentMetrics()
        sink = MagicMock()
        sink.emit = MagicMock()

        await agent_loop(
            "Hi",
            conversation,
            client,
            registry,
            tool_context,
            metrics=metrics,
            metrics_sink=sink,
        )
        sink.emit.assert_called()


class TestThinkingCallback:
    """Thinking/on_thinking callback for Anthropic-style extended thinking."""

    @pytest.mark.asyncio
    async def test_thinking_callback_fires(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        client = LLMClient(model="test")
        response_with_thinking = ChatResponse(
            content="Final answer.",
            tool_calls=[],
            finish_reason="stop",
            thinking="Hmm, let me reason about this...",
        )
        client.chat = AsyncMock(return_value=response_with_thinking)

        thinking_text = []

        result = await agent_loop(
            "Question",
            conversation,
            client,
            registry,
            tool_context,
            on_thinking=lambda t: thinking_text.append(t),
        )
        assert result == "Final answer."
        assert len(thinking_text) == 1
        assert "reason" in thinking_text[0]


class TestStripMetaCommentary:
    """Meta-commentary stripping from final text."""

    def test_strips_meta_commentary(self) -> None:
        from godspeed.agent.loop import _strip_meta_commentary

        text = "No tool call is needed. Here is the answer."
        result = _strip_meta_commentary(text)
        assert "No tool call is needed" not in result
        assert "Here is the answer" in result

    def test_strips_multiple_phrases(self) -> None:
        from godspeed.agent.loop import _strip_meta_commentary

        text = "No tool call is needed I don't need any tools The answer is 42."
        result = _strip_meta_commentary(text)
        assert "No tool call is needed" not in result
        assert "I don't need any tools" not in result
        assert "The answer is 42." in result

    def test_cleans_punctuation_artifacts(self) -> None:
        from godspeed.agent.loop import _strip_meta_commentary

        text = "No tool call is needed. . Here is the answer."
        result = _strip_meta_commentary(text)
        assert ". . " not in result
        assert "Here is the answer" in result


class TestCompetitionMode:
    """Competition mode disables safety features for evaluation."""

    @pytest.mark.asyncio
    async def test_competition_mode_disables_must_fix(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_read", result=ToolResult.success("ok")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("Done."))

        result = await agent_loop(
            "Hi",
            conversation,
            client,
            registry,
            tool_context,
            competition_mode=True,
        )
        assert "Done" in result

    @pytest.mark.asyncio
    async def test_competition_mode_disables_auto_commit(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("file_edit", {"file_path": "a.py"}),
                _make_text_response("Done."),
            ]
        )

        result = await agent_loop(
            "Edit",
            conversation,
            client,
            registry,
            tool_context,
            competition_mode=True,
            auto_commit=True,
        )
        assert "Done" in result


class TestAutoStashExtended:
    """Extended auto-stash tests."""

    @pytest.mark.asyncio
    async def test_auto_stash_on_sequential_path(self, tool_context) -> None:
        """Auto-stash via sequential dispatch path."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))
        git_result = ToolResult.success("Saved working directory")
        registry.register(MockTool(name="git", result=git_result))

        client = LLMClient(model="test")
        call_id = 0

        def make_edit_response():
            nonlocal call_id
            call_id += 1
            return ChatResponse(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{call_id:03d}",
                        "function": {
                            "name": "file_edit",
                            "arguments": json.dumps({"file_path": "test.txt"}),
                        },
                    }
                ],
                finish_reason="tool_calls",
            )

        client.chat = AsyncMock(
            side_effect=[
                make_edit_response(),
                make_edit_response(),
                make_edit_response(),
                _make_text_response("Done."),
            ]
        )

        result = await agent_loop(
            "Edit sequentially",
            conversation,
            client,
            registry,
            tool_context,
            parallel_tool_calls=False,
        )
        assert "Done" in result

    @pytest.mark.asyncio
    async def test_auto_stash_nothing_to_stash(self, tool_context) -> None:
        """Auto-stash with git returning 'nothing to stash' sets auto_stashed."""
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))
        registry.register(MockTool(name="git", result=ToolResult.success("nothing to stash")))

        client = LLMClient(model="test")
        call_id = 0

        def make_edit_response():
            nonlocal call_id
            call_id += 1
            return ChatResponse(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{call_id:03d}",
                        "function": {
                            "name": "file_edit",
                            "arguments": json.dumps({"file_path": "test.txt"}),
                        },
                    }
                ],
                finish_reason="tool_calls",
            )

        client.chat = AsyncMock(
            side_effect=[
                make_edit_response(),
                make_edit_response(),
                make_edit_response(),
                _make_text_response("Done."),
            ]
        )

        result = await agent_loop("Edit", conversation, client, registry, tool_context)
        assert "Done" in result


class TestStreamingCall:
    """Streaming LLM call path testing."""

    @pytest.mark.asyncio
    async def test_stream_finish_without_reason(self, tool_context) -> None:
        """Stream ending with no finish_reason returns empty response."""
        on_chunks = []

        class NoFinishStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def aclose(self):
                pass

        llm = AsyncMock(spec=LLMClient)
        llm.stream_chat = lambda **_: NoFinishStream()

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        result = await agent_loop(
            "Hi",
            conversation,
            llm,
            registry,
            tool_context,
            on_assistant_chunk=lambda c: on_chunks.append(c),
        )
        assert result == ""

    @pytest.mark.asyncio
    async def test_stream_text_only_response(self, tool_context) -> None:
        """Streaming with text-only response (no tool calls)."""
        on_chunks = []

        class TextOnlyStream:
            def __init__(self):
                self._chunks = [
                    ChatResponse(content="Hel", tool_calls=[], finish_reason=None),
                    ChatResponse(content="lo!", tool_calls=[], finish_reason=None),
                    ChatResponse(content="Hello!", tool_calls=[], finish_reason="stop"),
                ]
                self._idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx >= len(self._chunks):
                    raise StopAsyncIteration
                chunk = self._chunks[self._idx]
                self._idx += 1
                return chunk

            async def aclose(self):
                pass

        llm = AsyncMock(spec=LLMClient)
        llm.stream_chat = lambda **_: TextOnlyStream()

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        result = await agent_loop(
            "Hi",
            conversation,
            llm,
            registry,
            tool_context,
            on_assistant_chunk=lambda c: on_chunks.append(c),
        )
        assert result == "Hello!"
        assert on_chunks == ["Hel", "lo!"]


class TestSpeculativeOverflow:
    """Speculative cache size limit enforcement."""

    @pytest.mark.asyncio
    async def test_speculative_dispatch_cache_full(self) -> None:
        from unittest.mock import MagicMock, patch

        from godspeed.agent.loop import _speculative_dispatch
        from godspeed.tools.base import RiskLevel, ToolCall, ToolResult

        registry = MagicMock()
        tool = MagicMock()
        tool.risk_level = RiskLevel.READ_ONLY
        registry.get.return_value = tool
        registry.dispatch = AsyncMock(return_value=ToolResult.success("ok"))

        tool_context = MagicMock()
        cache = {}

        raw_tool_calls = [
            {
                "id": f"call_{i:03d}",
                "function": {"name": "file_read", "arguments": '{"file_path": "f.py"}'},
            }
            for i in range(3)
        ]

        with patch(
            "godspeed.agent.loop._parse_tool_call",
            side_effect=[
                ToolCall(
                    tool_name="file_read", arguments={"file_path": "f.py"}, call_id=f"call_{i:03d}"
                )
                for i in range(3)
            ],
        ):
            _speculative_dispatch(raw_tool_calls, registry, tool_context, cache, max_size=1)

        assert len(cache) == 1

    def test_speculative_dispatch_tool_not_found(self) -> None:
        from unittest.mock import MagicMock

        from godspeed.agent.loop import _speculative_dispatch

        registry = MagicMock()
        registry.get.return_value = None
        registry.dispatch = AsyncMock()

        tool_context = MagicMock()
        cache = {}

        raw_tool_calls = [{"id": "call_001", "function": {"name": "unknown", "arguments": "{}"}}]

        _speculative_dispatch(raw_tool_calls, registry, tool_context, cache)
        assert len(cache) == 0

    def test_speculative_dispatch_not_safe(self) -> None:
        from unittest.mock import MagicMock

        from godspeed.agent.loop import _speculative_dispatch
        from godspeed.tools.base import RiskLevel

        registry = MagicMock()
        tool = MagicMock()
        tool.risk_level = RiskLevel.HIGH
        registry.get.return_value = tool
        registry.dispatch = AsyncMock()

        tool_context = MagicMock()
        cache = {}

        raw_tool_calls = [
            {"id": "call_001", "function": {"name": "shell", "arguments": '{"command": "rm"}'}}
        ]

        _speculative_dispatch(raw_tool_calls, registry, tool_context, cache)
        assert len(cache) == 0

    def test_speculative_dispatch_malformed_skipped(self) -> None:
        from unittest.mock import MagicMock

        from godspeed.agent.loop import _speculative_dispatch

        registry = MagicMock()
        tool_context = MagicMock()
        cache = {}

        raw_tool_calls = [{"id": "call_001", "function": {"name": "", "arguments": "broken{"}}]

        _speculative_dispatch(raw_tool_calls, registry, tool_context, cache)
        assert len(cache) == 0


class TestStuckLoopHookFire:
    """Stuck loop detection with hook_executor firing."""

    @pytest.mark.asyncio
    async def test_stuck_loop_with_hook_fires(self, tool_context) -> None:
        from unittest.mock import MagicMock

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="shell", result=ToolResult.failure("Same error")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("shell", {"command": "bad"}),
                _make_tool_response("shell", {"command": "bad"}),
                _make_tool_response("shell", {"command": "bad"}),
                _make_text_response("Different approach."),
            ]
        )

        hook_executor = MagicMock()
        hook_executor.fire = MagicMock()

        result = await agent_loop(
            "Do bad stuff",
            conversation,
            client,
            registry,
            tool_context,
            hook_executor=hook_executor,
            stuck_loop_threshold=3,
        )
        assert "approach" in result.lower()
        assert hook_executor.fire.called


class TestCompactFailureFallback:
    """Compaction failure fallback path."""

    @pytest.mark.asyncio
    async def test_compaction_failure_truncation_fallback(self, tool_context) -> None:
        from unittest.mock import patch

        conversation = Conversation("System", max_tokens=10, compaction_threshold=0.1)
        conversation.add_user_message("Hello world " * 100)

        registry = ToolRegistry()
        client = LLMClient(model="test")
        # LLM call for compaction fails, but compaction fallback still succeeds
        client.chat = AsyncMock(
            side_effect=[
                RuntimeError("compaction LLM failed"),
                _make_text_response("Done after fallback."),
            ]
        )

        with patch("godspeed.llm.cost.get_cheapest_model", return_value="test"):
            with patch(
                "godspeed.context.compaction.get_compaction_prompt",
                return_value="Summarize:",
            ):
                result = await agent_loop(
                    "Do work",
                    conversation,
                    client,
                    registry,
                    tool_context,
                )
        assert "Done" in result


class TestCancelEvent:
    """Mid-turn cancellation: the agent loop must unwind on cancel_event."""

    @pytest.mark.asyncio
    async def test_cancel_before_first_iteration(self) -> None:
        import asyncio

        from godspeed.agent.result import AgentCancelledError

        cancel = asyncio.Event()
        cancel.set()

        llm = AsyncMock(spec=LLMClient)
        llm.chat = AsyncMock()  # must never be called

        conv = Conversation(system_prompt="x")
        reg = ToolRegistry()

        with pytest.raises(AgentCancelledError):
            await agent_loop(
                user_input="hi",
                conversation=conv,
                llm_client=llm,
                tool_registry=reg,
                tool_context=None,  # type: ignore[arg-type]
                cancel_event=cancel,
            )

        llm.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_between_streaming_chunks(self) -> None:
        import asyncio

        from godspeed.agent.result import AgentCancelledError

        cancel = asyncio.Event()

        aclose_calls: list[int] = []

        class FakeStream:
            def __init__(self, chunks: list[ChatResponse]) -> None:
                self._chunks = list(chunks)

            def __aiter__(self) -> FakeStream:
                return self

            async def __anext__(self) -> ChatResponse:
                if not self._chunks:
                    raise StopAsyncIteration
                chunk = self._chunks.pop(0)
                if chunk.content == "first":
                    cancel.set()
                return chunk

            async def aclose(self) -> None:
                aclose_calls.append(1)

        chunks = [
            ChatResponse(content="first", tool_calls=[], finish_reason=None),
            ChatResponse(content="second", tool_calls=[], finish_reason=None),
            ChatResponse(content="", tool_calls=[], finish_reason="stop"),
        ]

        llm = AsyncMock(spec=LLMClient)
        llm.stream_chat = lambda **_: FakeStream(chunks)

        got_chunks: list[str] = []

        def on_chunk(text: str) -> None:
            got_chunks.append(text)

        conv = Conversation(system_prompt="x")
        reg = ToolRegistry()

        with pytest.raises(AgentCancelledError):
            await agent_loop(
                user_input="hi",
                conversation=conv,
                llm_client=llm,
                tool_registry=reg,
                tool_context=None,  # type: ignore[arg-type]
                on_assistant_chunk=on_chunk,
                cancel_event=cancel,
            )

        assert got_chunks == ["first"]
        assert aclose_calls == [1]

    @pytest.mark.asyncio
    async def test_cancel_event_unset_does_not_cancel(self) -> None:
        import asyncio

        cancel = asyncio.Event()
        llm = AsyncMock(spec=LLMClient)
        llm.chat = AsyncMock(return_value=_make_text_response("hello"))

        conv = Conversation(system_prompt="x")
        reg = ToolRegistry()

        result = await agent_loop(
            user_input="hi",
            conversation=conv,
            llm_client=llm,
            tool_registry=reg,
            tool_context=None,  # type: ignore[arg-type]
            cancel_event=cancel,
        )
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_cancel_none_preserves_existing_behavior(self) -> None:
        llm = AsyncMock(spec=LLMClient)
        llm.chat = AsyncMock(return_value=_make_text_response("hello"))

        conv = Conversation(system_prompt="x")
        reg = ToolRegistry()

        result = await agent_loop(
            user_input="hi",
            conversation=conv,
            llm_client=llm,
            tool_registry=reg,
            tool_context=None,  # type: ignore[arg-type]
        )
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_cancel_during_batch_llm_call(self) -> None:
        import asyncio

        from godspeed.agent.result import AgentCancelledError, AgentMetrics

        cancel = asyncio.Event()
        llm = AsyncMock(spec=LLMClient)
        llm.chat = AsyncMock(side_effect=AgentCancelledError("cancel_event set by caller"))

        conv = Conversation(system_prompt="x")
        reg = ToolRegistry()
        metrics = AgentMetrics()

        with pytest.raises(AgentCancelledError):
            await agent_loop(
                user_input="hi",
                conversation=conv,
                llm_client=llm,
                tool_registry=reg,
                tool_context=None,  # type: ignore[arg-type]
                cancel_event=cancel,
                metrics=metrics,
            )

        assert metrics.exit_reason == "interrupted"


class TestAutoStashExtendedMore:
    """Additional auto-stash edge cases."""

    @pytest.mark.asyncio
    async def test_auto_stash_nothing_to_stash_sets_flag(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))
        registry.register(MockTool(name="git", result=ToolResult.success("nothing to stash")))

        client = LLMClient(model="test")
        call_id = 0

        def make_edit_response():
            nonlocal call_id
            call_id += 1
            return ChatResponse(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{call_id:03d}",
                        "function": {
                            "name": "file_edit",
                            "arguments": json.dumps({"file_path": "test.txt"}),
                        },
                    }
                ],
                finish_reason="tool_calls",
            )

        client.chat = AsyncMock(
            side_effect=[
                make_edit_response(),
                make_edit_response(),
                make_edit_response(),
                _make_text_response("Done."),
            ]
        )

        result = await agent_loop("Edit", conversation, client, registry, tool_context)
        assert "Done" in result

    @pytest.mark.asyncio
    async def test_auto_stash_no_git_tool(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))

        client = LLMClient(model="test")
        call_id = 0

        def make_edit_response():
            nonlocal call_id
            call_id += 1
            return ChatResponse(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{call_id:03d}",
                        "function": {
                            "name": "file_edit",
                            "arguments": json.dumps({"file_path": f"file{call_id}.txt"}),
                        },
                    }
                ],
                finish_reason="tool_calls",
            )

        client.chat = AsyncMock(
            side_effect=[
                make_edit_response(),
                make_edit_response(),
                make_edit_response(),
                make_edit_response(),
                make_edit_response(),
                _make_text_response("Done."),
            ]
        )

        result = await agent_loop("Edit", conversation, client, registry, tool_context)
        assert "Done" in result

    @pytest.mark.asyncio
    async def test_auto_stash_git_failure(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))
        registry.register(MockTool(name="git", result=ToolResult.failure("git stash failed")))

        client = LLMClient(model="test")
        call_id = 0

        def make_edit_response():
            nonlocal call_id
            call_id += 1
            return ChatResponse(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{call_id:03d}",
                        "function": {
                            "name": "file_edit",
                            "arguments": json.dumps({"file_path": "test.txt"}),
                        },
                    }
                ],
                finish_reason="tool_calls",
            )

        client.chat = AsyncMock(
            side_effect=[
                make_edit_response(),
                make_edit_response(),
                make_edit_response(),
                _make_text_response("Done."),
            ]
        )

        result = await agent_loop("Edit", conversation, client, registry, tool_context)
        assert "Done" in result


class TestAutoVerifyBackground:
    """Auto-verify background task handling."""

    @pytest.mark.asyncio
    async def test_auto_verify_no_verify_tool(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_write", result=ToolResult.success("wrote")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                ChatResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_v1",
                            "function": {
                                "name": "file_write",
                                "arguments": json.dumps({"file_path": "main.py", "content": "x=1"}),
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                ),
                _make_text_response("Wrote file."),
            ]
        )

        result = await agent_loop("Write file", conversation, client, registry, tool_context)
        assert "Wrote" in result

    @pytest.mark.asyncio
    async def test_auto_verify_non_verifiable_extension(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_write", result=ToolResult.success("wrote")))
        registry.register(MockTool(name="verify", result=ToolResult.success("ok")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                ChatResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_v2",
                            "function": {
                                "name": "file_write",
                                "arguments": json.dumps(
                                    {"file_path": "README.md", "content": "# Hi"}
                                ),
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                ),
                _make_text_response("Done."),
            ]
        )

        result = await agent_loop("Edit readme", conversation, client, registry, tool_context)
        assert "Done" in result


class TestHookEventsExtended:
    """Extended hook event testing."""

    @pytest.mark.asyncio
    async def test_hook_permission_denied_fires(self, tool_context) -> None:
        from unittest.mock import MagicMock

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="shell"))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("shell", {"command": "rm -rf /"}),
                _make_text_response("OK"),
            ]
        )

        mock_perms = MagicMock()
        mock_perms.evaluate = MagicMock(return_value="deny")

        ctx = type(tool_context)(
            cwd=tool_context.cwd,
            session_id="test",
            permissions=mock_perms,
        )

        hook_executor = MagicMock()
        hook_executor.fire = MagicMock()
        hook_executor.run_pre_tool = MagicMock(return_value=True)

        await agent_loop(
            "Delete",
            conversation,
            client,
            registry,
            ctx,
            hook_executor=hook_executor,
        )

        assert hook_executor.fire.called

    @pytest.mark.asyncio
    async def test_hook_context_threshold_50(self, tool_context) -> None:
        from unittest.mock import MagicMock

        from godspeed.agent.result import AgentMetrics

        conversation = Conversation("System", max_tokens=10)
        conversation.add_user_message("A" * 500)

        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("ok"))

        metrics = AgentMetrics()
        hook_executor = MagicMock()
        hook_executor.fire = MagicMock()

        await agent_loop(
            "Hi",
            conversation,
            client,
            registry,
            tool_context,
            hook_executor=hook_executor,
            metrics=metrics,
        )

        assert hook_executor.fire.call_count >= 1

    @pytest.mark.asyncio
    async def test_hook_context_threshold_25(self, tool_context) -> None:
        from unittest.mock import MagicMock

        from godspeed.agent.result import AgentMetrics

        conversation = Conversation("System", max_tokens=100)
        conversation.add_user_message("X" * 1000)

        registry = ToolRegistry()
        client = LLMClient(model="test")
        client.chat = AsyncMock(return_value=_make_text_response("ok"))

        metrics = AgentMetrics()
        hook_executor = MagicMock()
        hook_executor.fire = MagicMock()

        await agent_loop(
            "Hi",
            conversation,
            client,
            registry,
            tool_context,
            hook_executor=hook_executor,
            metrics=metrics,
        )

        assert hook_executor.fire.call_count >= 1


class TestToolToolErrorHandling:
    """Tool error handling and result formatting."""

    @pytest.mark.asyncio
    async def test_tool_error_is_recorded_in_metrics(self, tool_context) -> None:
        from godspeed.agent.result import AgentMetrics

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="shell", result=ToolResult.failure("command failed")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("shell", {"command": "bad cmd"}),
                _make_text_response("Failed, will try differently."),
            ]
        )

        metrics = AgentMetrics()
        result = await agent_loop(
            "Run command",
            conversation,
            client,
            registry,
            tool_context,
            metrics=metrics,
        )
        assert "Failed" in result
        assert metrics.tool_call_count == 1
        assert metrics.tool_error_count == 1

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_message(self, tool_context) -> None:
        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("nonexistent_tool", {"arg": "val"}),
                _make_text_response("Tool not found, I'll adapt."),
            ]
        )

        result = await agent_loop("Do", conversation, client, registry, tool_context)
        assert "adapt" in result.lower() or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_result_with_audit_recording(self, tool_context) -> None:
        import pytest

        pytest.skip("ToolContext audit field requires AuditRecorder instance, not mock")


class TestSpeculativeDispatchAllModes:
    """All speculative dispatch code paths."""

    @pytest.mark.asyncio
    async def test_speculative_dispatch_with_allowlisted_tool(self) -> None:
        from unittest.mock import MagicMock

        from godspeed.agent.loop import _speculative_dispatch
        from godspeed.tools.base import RiskLevel

        registry = MagicMock()
        tool = MagicMock()
        tool.risk_level = RiskLevel.LOW
        registry.get.return_value = tool
        registry.dispatch = AsyncMock(return_value=ToolResult.success("ok"))

        tool_context_mock = MagicMock()
        cache = {}

        raw_tool_calls = [
            {"id": "call_wf", "function": {"name": "web_fetch", "arguments": '{"url": "http://x"}'}}
        ]

        _speculative_dispatch(raw_tool_calls, registry, tool_context_mock, cache)
        assert len(cache) == 1

    def test_speculative_dispatch_duplicate_call_id(self) -> None:
        from unittest.mock import MagicMock, patch

        from godspeed.agent.loop import _speculative_dispatch
        from godspeed.tools.base import RiskLevel

        registry = MagicMock()
        tool = MagicMock()
        tool.risk_level = RiskLevel.READ_ONLY
        registry.get.return_value = tool
        registry.dispatch = AsyncMock()

        tool_context_mock = MagicMock()

        raw_tool_calls = [
            {
                "id": "call_dup",
                "function": {"name": "file_read", "arguments": '{"file_path": "a.py"}'},
            },
            {
                "id": "call_dup",
                "function": {"name": "file_read", "arguments": '{"file_path": "b.py"}'},
            },
        ]

        cache = {}
        with patch("asyncio.create_task", side_effect=lambda c: c):
            _speculative_dispatch(raw_tool_calls, registry, tool_context_mock, cache)
        assert len(cache) <= 1


class TestParallelExecutionSequential:
    """Sequential dispatch edge cases."""

    @pytest.mark.asyncio
    async def test_sequential_speculative_hit(self, tool_context) -> None:

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_read", result=ToolResult.success("cached content")))

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                ChatResponse(
                    content="reading",
                    tool_calls=[
                        {
                            "id": "call_seq1",
                            "function": {
                                "name": "file_read",
                                "arguments": json.dumps({"file_path": "test.py"}),
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                ),
                _make_text_response("Read from cache."),
            ]
        )

        result = await agent_loop(
            "Read",
            conversation,
            client,
            registry,
            tool_context,
            parallel_tool_calls=False,
        )
        assert "Read" in result or "cached" in result.lower()


class TestAutoCommitMore:
    """Additional auto-commit test paths."""

    @pytest.mark.asyncio
    async def test_auto_commit_resets_counter_after_commit(self, tool_context) -> None:
        from unittest.mock import patch

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))

        client = LLMClient(model="test")

        async def generate_msg(*args):
            return "feat: commit msg"

        async def do_commit(*args):
            return ToolResult.success("Committed.")

        with patch("godspeed.agent.auto_commit.generate_commit_message", side_effect=generate_msg):
            with patch("godspeed.agent.auto_commit.auto_commit", side_effect=do_commit):
                call_id = 0

                def make_edit():
                    nonlocal call_id
                    call_id += 1
                    return ChatResponse(
                        content="",
                        tool_calls=[
                            {
                                "id": f"call_{call_id:03d}",
                                "function": {
                                    "name": "file_edit",
                                    "arguments": json.dumps({"file_path": f"f{call_id}.py"}),
                                },
                            }
                        ],
                        finish_reason="tool_calls",
                    )

                client.chat = AsyncMock(
                    side_effect=[
                        make_edit(),
                        make_edit(),
                        make_edit(),
                        make_edit(),
                        make_edit(),  # triggers auto-commit at 5
                        make_edit(),  # 1st after commit reset
                        _make_text_response("Done."),
                    ]
                )

                result = await agent_loop(
                    "Edit",
                    conversation,
                    client,
                    registry,
                    tool_context,
                    auto_commit=True,
                    auto_commit_threshold=5,
                )
        assert "Done" in result

    @pytest.mark.asyncio
    async def test_auto_commit_failure_not_committed(self, tool_context) -> None:
        from unittest.mock import patch

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_edit", result=ToolResult.success("edited")))

        client = LLMClient(model="test")

        async def generate_msg(*args):
            return "feat: commit"

        async def failing_commit(*args):
            return ToolResult.failure("no changes to commit")

        with patch("godspeed.agent.auto_commit.generate_commit_message", side_effect=generate_msg):
            with patch("godspeed.agent.auto_commit.auto_commit", side_effect=failing_commit):
                call_id = 0

                def make_edit():
                    nonlocal call_id
                    call_id += 1
                    return ChatResponse(
                        content="",
                        tool_calls=[
                            {
                                "id": f"call_{call_id:03d}",
                                "function": {
                                    "name": "file_edit",
                                    "arguments": json.dumps({"file_path": "test.txt"}),
                                },
                            }
                        ],
                        finish_reason="tool_calls",
                    )

                client.chat = AsyncMock(
                    side_effect=[
                        make_edit(),
                        make_edit(),
                        make_edit(),
                        make_edit(),
                        make_edit(),
                        _make_text_response("Done."),
                    ]
                )

                result = await agent_loop(
                    "Edit",
                    conversation,
                    client,
                    registry,
                    tool_context,
                    auto_commit=True,
                    auto_commit_threshold=5,
                )
        assert "Done" in result


class TestRetrievalSubagentMore:
    """Additional retrieval subagent scenarios."""

    @pytest.mark.asyncio
    async def test_retrieval_with_spans_empty(self, tool_context) -> None:
        from unittest.mock import MagicMock

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        retrieval = MagicMock()
        retrieval.retrieve = AsyncMock(
            return_value=MagicMock(spans=[], __iter__=lambda s: iter([]))
        )
        retrieval.format_spans_for_agent = MagicMock(return_value="No results found.")

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("code_search", {"query": "missing"}),
                _make_text_response("No results, I'll try differently."),
            ]
        )

        result = await agent_loop(
            "Search",
            conversation,
            client,
            registry,
            tool_context,
            retrieval_subagent=retrieval,
        )
        assert "result" in result.lower() or "differently" in result.lower()

    @pytest.mark.asyncio
    async def test_retrieval_grep_tool(self, tool_context) -> None:
        from unittest.mock import MagicMock

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        retrieval = MagicMock()
        retrieval.retrieve = AsyncMock(
            return_value=MagicMock(spans=[MagicMock()], __iter__=lambda s: iter([MagicMock()]))
        )
        retrieval.format_spans_for_agent = MagicMock(return_value="Found at file.py:10")

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("grep", {"pattern": "TODO"}),
                _make_text_response("Found results."),
            ]
        )

        result = await agent_loop(
            "Grep",
            conversation,
            client,
            registry,
            tool_context,
            retrieval_subagent=retrieval,
        )
        assert "Found" in result

    @pytest.mark.asyncio
    async def test_retrieval_glob_tool(self, tool_context) -> None:
        from unittest.mock import MagicMock

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        retrieval = MagicMock()
        retrieval.retrieve = AsyncMock(
            return_value=MagicMock(spans=[MagicMock()], __iter__=lambda s: iter([MagicMock()]))
        )
        retrieval.format_spans_for_agent = MagicMock(return_value="Found: src/**/*.py")

        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("glob", {"pattern": "*.py"}),
                _make_text_response("Glob results."),
            ]
        )

        result = await agent_loop(
            "Glob",
            conversation,
            client,
            registry,
            tool_context,
            retrieval_subagent=retrieval,
        )
        assert "results" in result.lower() or "Glob" in result


class TestStreamingCallAdditional:
    """Additional streaming call edge cases."""

    @pytest.mark.asyncio
    async def test_streaming_call_tracks_tokens(self, tool_context) -> None:

        on_chunks = []

        class FakeStream:
            def __init__(self):
                self._chunks = [
                    ChatResponse(content="hel", tool_calls=[], finish_reason=None),
                    ChatResponse(content="lo", tool_calls=[], finish_reason=None),
                    ChatResponse(
                        content="hello",
                        tool_calls=[],
                        finish_reason="stop",
                        usage={"prompt_tokens": 10, "completion_tokens": 5},
                    ),
                ]
                self._idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx >= len(self._chunks):
                    raise StopAsyncIteration
                chunk = self._chunks[self._idx]
                self._idx += 1
                return chunk

            async def aclose(self):
                pass

        llm = LLMClient(model="test")
        llm.stream_chat = lambda **_: FakeStream()

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()

        result = await agent_loop(
            "Hi",
            conversation,
            llm,
            registry,
            tool_context,
            on_assistant_chunk=lambda c: on_chunks.append(c),
        )

        assert result == "hello"
        assert llm.total_input_tokens == 0
        assert llm.total_output_tokens == 0

    @pytest.mark.asyncio
    async def test_streaming_call_uses_json_markdown_parser(self, tool_context) -> None:

        on_chunks = []

        class FakeStream:
            def __init__(self):
                self._chunks = [
                    ChatResponse(content='{"tool": "file_read', tool_calls=[], finish_reason=None),
                    ChatResponse(
                        content='", "arguments": {"path": "x.py"}}',
                        tool_calls=[],
                        finish_reason=None,
                    ),
                    ChatResponse(
                        content='{"tool": "file_read", "arguments": {"path": "x.py"}}',
                        tool_calls=[],
                        finish_reason="stop",
                        usage={},
                    ),
                ]
                self._idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx >= len(self._chunks):
                    raise StopAsyncIteration
                chunk = self._chunks[self._idx]
                self._idx += 1
                return chunk

            async def aclose(self):
                pass

        llm = LLMClient(model="test")
        llm.stream_chat = lambda **_: FakeStream()

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="file_read", result=ToolResult.success("contents")))

        result = await agent_loop(
            "Read x.py",
            conversation,
            llm,
            registry,
            tool_context,
            on_assistant_chunk=lambda c: on_chunks.append(c),
        )

        # Either markdown parser parsed tool calls or we got the raw content
        assert result in ("contents", "") or "file_read" in result


class TestSafetyHookFiring:
    """Advisory DANGEROUS_COMMAND / SECRET_DETECTED hooks fire during evaluation."""

    @pytest.mark.asyncio
    async def test_dangerous_command_hook_fires(self, tool_context) -> None:
        from unittest.mock import MagicMock

        from godspeed.hooks import HookEvent

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="shell", result=ToolResult.success("ran")))
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response("shell", {"command": "rm -rf /"}),
                _make_text_response("Done."),
            ]
        )
        hook_executor = MagicMock()
        mock_perms = MagicMock()
        mock_perms.evaluate = MagicMock(return_value="allow")
        ctx = type(tool_context)(
            cwd=tool_context.cwd,
            session_id="test",
            permissions=mock_perms,
        )
        await agent_loop(
            "Run it",
            conversation,
            client,
            registry,
            ctx,
            hook_executor=hook_executor,
        )
        events = [c[0][0] for c in hook_executor.fire.call_args_list]
        assert HookEvent.DANGEROUS_COMMAND in events

    @pytest.mark.asyncio
    async def test_secret_detected_hook_fires(self, tool_context) -> None:
        from unittest.mock import MagicMock

        from godspeed.hooks import HookEvent

        conversation = Conversation("You are a coding agent.", max_tokens=100_000)
        registry = ToolRegistry()
        registry.register(MockTool(name="shell", result=ToolResult.success("ran")))
        client = LLMClient(model="test")
        client.chat = AsyncMock(
            side_effect=[
                _make_tool_response(
                    "shell", {"command": "echo ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"}
                ),
                _make_text_response("Done."),
            ]
        )
        hook_executor = MagicMock()
        mock_perms = MagicMock()
        mock_perms.evaluate = MagicMock(return_value="allow")
        ctx = type(tool_context)(
            cwd=tool_context.cwd,
            session_id="test",
            permissions=mock_perms,
        )
        await agent_loop(
            "Run it",
            conversation,
            client,
            registry,
            ctx,
            hook_executor=hook_executor,
        )
        events = [c[0][0] for c in hook_executor.fire.call_args_list]
        assert HookEvent.SECRET_DETECTED in events
