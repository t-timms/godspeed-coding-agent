"""Tests for extended thinking support (Unit 1)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.llm.client import ChatResponse, LLMClient

# ---------------------------------------------------------------------------
# Config: thinking_budget field
# ---------------------------------------------------------------------------


def test_thinking_budget_defaults_zero():
    """thinking_budget defaults to 0 (disabled)."""
    from godspeed.config import GodspeedSettings

    with patch("godspeed.config.DEFAULT_GLOBAL_DIR", MagicMock(exists=lambda: False)):
        settings = GodspeedSettings(model="test")
    assert settings.thinking_budget == 0


def test_thinking_budget_config_value():
    """thinking_budget can be set via constructor."""
    from godspeed.config import GodspeedSettings

    with patch("godspeed.config.DEFAULT_GLOBAL_DIR", MagicMock(exists=lambda: False)):
        settings = GodspeedSettings(model="test", thinking_budget=10000)
    assert settings.thinking_budget == 10000


# ---------------------------------------------------------------------------
# LLMClient: thinking parameter passed to Anthropic models
# ---------------------------------------------------------------------------


def test_llm_client_stores_thinking_budget():
    """LLMClient stores thinking_budget from constructor."""
    client = LLMClient(model="claude-sonnet-4-20250514", thinking_budget=8000)
    assert client.thinking_budget == 8000


def test_is_anthropic_model_true():
    """Claude models are correctly identified as Anthropic."""
    client = LLMClient(model="claude-sonnet-4-20250514")
    assert client._is_anthropic_model() is True
    assert client._is_anthropic_model("anthropic/claude-3.5-sonnet") is True


def test_is_anthropic_model_false():
    """Non-Claude models are not Anthropic."""
    client = LLMClient(model="gpt-4o")
    assert client._is_anthropic_model() is False
    assert client._is_anthropic_model("ollama/qwen3:4b") is False


@pytest.mark.asyncio
async def test_thinking_param_added_for_claude():
    """When thinking_budget > 0 and model is Claude, thinking param is added."""
    client = LLMClient(model="claude-sonnet-4-20250514", thinking_budget=10000)

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Hello",
                tool_calls=None,
                thinking=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch("godspeed.llm.client._get_litellm") as mock_litellm:
        mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_response)
        await client._call("claude-sonnet-4-20250514", [{"role": "user", "content": "hi"}], None)

        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert "thinking" in call_kwargs
        assert call_kwargs["thinking"]["type"] == "enabled"
        assert call_kwargs["thinking"]["budget_tokens"] == 10000


@pytest.mark.asyncio
async def test_thinking_param_skipped_for_non_claude():
    """When model is not Claude, thinking param is not added."""
    client = LLMClient(model="gpt-4o", thinking_budget=10000)

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Hello",
                tool_calls=None,
                thinking=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch("godspeed.llm.client._get_litellm") as mock_litellm:
        mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_response)
        await client._call("gpt-4o", [{"role": "user", "content": "hi"}], None)

        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert "thinking" not in call_kwargs


@pytest.mark.asyncio
async def test_thinking_param_skipped_when_zero():
    """When thinking_budget is 0, no thinking param even for Claude."""
    client = LLMClient(model="claude-sonnet-4-20250514", thinking_budget=0)

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Hello",
                tool_calls=None,
                thinking=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch("godspeed.llm.client._get_litellm") as mock_litellm:
        mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_response)
        await client._call("claude-sonnet-4-20250514", [{"role": "user", "content": "hi"}], None)

        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert "thinking" not in call_kwargs


# ---------------------------------------------------------------------------
# ChatResponse: thinking field
# ---------------------------------------------------------------------------


def test_chat_response_thinking_field():
    """ChatResponse includes thinking field."""
    resp = ChatResponse(content="hello", thinking="I need to think about this...")
    assert resp.thinking == "I need to think about this..."


def test_chat_response_thinking_default_empty():
    """ChatResponse thinking defaults to empty string."""
    resp = ChatResponse(content="hello")
    assert resp.thinking == ""


# ---------------------------------------------------------------------------
# Agent loop: on_thinking callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_thinking_callback_called(tmp_path):
    """Agent loop calls on_thinking when response has thinking content."""
    from godspeed.agent.conversation import Conversation
    from godspeed.tools.base import ToolContext
    from godspeed.tools.registry import ToolRegistry

    # Mock LLM to return a response with thinking
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat = AsyncMock(
        return_value=ChatResponse(
            content="The answer is 42.",
            thinking="Let me reason step by step...",
            finish_reason="stop",
        )
    )
    mock_llm.stream_chat = AsyncMock()

    conversation = Conversation(system_prompt="test", model="test", max_tokens=10000)
    registry = ToolRegistry()
    context = ToolContext(cwd=tmp_path, session_id="test")

    thinking_texts: list[str] = []

    def capture_thinking(text: str) -> None:
        thinking_texts.append(text)

    from godspeed.agent.loop import agent_loop

    await agent_loop(
        user_input="What is the meaning of life?",
        conversation=conversation,
        llm_client=mock_llm,
        tool_registry=registry,
        tool_context=context,
        on_thinking=capture_thinking,
    )

    assert len(thinking_texts) == 1
    assert "step by step" in thinking_texts[0]


@pytest.mark.asyncio
async def test_on_thinking_not_called_when_empty(tmp_path):
    """Agent loop does not call on_thinking when thinking is empty."""
    from godspeed.agent.conversation import Conversation
    from godspeed.tools.base import ToolContext
    from godspeed.tools.registry import ToolRegistry

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat = AsyncMock(
        return_value=ChatResponse(
            content="Hello!",
            thinking="",
            finish_reason="stop",
        )
    )

    conversation = Conversation(system_prompt="test", model="test", max_tokens=10000)
    registry = ToolRegistry()
    context = ToolContext(cwd=tmp_path, session_id="test")

    thinking_texts: list[str] = []

    from godspeed.agent.loop import agent_loop

    await agent_loop(
        user_input="Hi",
        conversation=conversation,
        llm_client=mock_llm,
        tool_registry=registry,
        tool_context=context,
        on_thinking=lambda t: thinking_texts.append(t),
    )

    assert len(thinking_texts) == 0


# ---------------------------------------------------------------------------
# TUI: /think command
# ---------------------------------------------------------------------------


def test_think_command_toggle_on(tmp_path):
    """'/think' toggles thinking ON with default 10k budget."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.thinking_budget = 0
    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/think")
    assert result.handled
    assert llm_client.thinking_budget == 10_000


def test_think_command_toggle_off(tmp_path):
    """'/think' toggles thinking OFF when already on."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.thinking_budget = 10_000
    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/think")
    assert result.handled
    assert llm_client.thinking_budget == 0


def test_think_command_set_budget(tmp_path):
    """'/think 20000' sets a custom budget."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.thinking_budget = 0
    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/think 20000")
    assert result.handled
    assert llm_client.thinking_budget == 20_000


def test_think_command_reject_small_budget(tmp_path):
    """'/think 500' rejects budgets under 1000."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.thinking_budget = 0
    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/think 500")
    assert result.handled
    assert llm_client.thinking_budget == 0  # unchanged


def test_think_command_off_keyword(tmp_path):
    """'/think off' explicitly disables."""
    from godspeed.tui.commands import Commands

    llm_client = MagicMock()
    llm_client.thinking_budget = 10_000
    commands = Commands(
        conversation=MagicMock(),
        llm_client=llm_client,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="test",
        cwd=tmp_path,
    )
    result = commands.dispatch("/think off")
    assert result.handled
    assert llm_client.thinking_budget == 0


# ---------------------------------------------------------------------------
# TUI: format_thinking
# ---------------------------------------------------------------------------


def test_format_thinking_nonempty(capsys):
    """format_thinking displays non-empty text."""
    from godspeed.tui.output import format_thinking

    # Just verify it doesn't raise
    format_thinking("I'm thinking about this problem...")


def test_format_thinking_empty():
    """format_thinking does nothing for empty text."""
    from godspeed.tui.output import format_thinking

    format_thinking("")  # Should not raise
    format_thinking("   ")  # Whitespace only


# ---------------------------------------------------------------------------
# Qwen3.5+ chat-template thinking control (enable_thinking / reasoning_effort)
# ---------------------------------------------------------------------------

QWEN38 = "openai/qwen3.8-27b"


@pytest.mark.parametrize(
    "model",
    [
        "qwen3.8-27b",
        "openai/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp",
        "llamacpp/qwen3.8-27b",
        "qwen3.5-35b-a3b",
        "openai/qwen3.6-27b",
    ],
)
def test_qwen35_plus_models_are_thinking_capable(model: str) -> None:
    client = LLMClient(model=model)
    assert client._is_qwen_template_model(model) is True
    assert client._supports_thinking(model) is True


@pytest.mark.parametrize(
    "model", ["openai/qwen2.5-coder-14b", "gpt-4o", "claude-sonnet-4-20250514", "ollama/llama3.1"]
)
def test_other_models_are_not_qwen_template_models(model: str) -> None:
    client = LLMClient(model=model)
    assert client._is_qwen_template_model(model) is False


def test_legacy_qwen3_dash_prefix_still_thinking_capable_but_not_template() -> None:
    client = LLMClient(model="ollama/qwen3:4b")
    assert client._supports_thinking("qwen3-coder-30b") is True
    assert client._is_qwen_template_model("qwen3-coder-30b") is False


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        ("none", {"enable_thinking": False}),
        ("off", {"enable_thinking": False}),
        ("NONE", {"enable_thinking": False}),
        ("low", {"enable_thinking": True, "reasoning_effort": "low"}),
        # "minimal" is not a template value; "low" does not actually shorten on
        # Qwen3.8, so the alias maps to the shortest EFFECTIVE tier (medium).
        ("minimal", {"enable_thinking": True, "reasoning_effort": "medium"}),
        ("medium", {"enable_thinking": True, "reasoning_effort": "medium"}),
        ("high", {"enable_thinking": True, "reasoning_effort": "xhigh"}),
        ("xhigh", {"enable_thinking": True, "reasoning_effort": "xhigh"}),
    ],
)
def test_qwen_effort_maps_to_chat_template_kwargs(effort: str, expected: dict) -> None:
    client = LLMClient(model=QWEN38, reasoning_effort=effort)
    kwargs: dict = {}
    client._apply_thinking_params(kwargs, QWEN38)
    assert kwargs == {"extra_body": {"chat_template_kwargs": expected}}
    # A raw top-level reasoning_effort must never be forwarded to the template.
    assert "reasoning_effort" not in kwargs


def test_qwen_unrecognised_effort_is_dropped_not_forwarded() -> None:
    """The Qwen3.8 template raises on unknown efforts, so never send them."""
    client = LLMClient(model=QWEN38, reasoning_effort="extreme")
    kwargs: dict = {}
    client._apply_thinking_params(kwargs, QWEN38)
    assert kwargs == {}


def test_qwen_default_leaves_template_default_untouched() -> None:
    client = LLMClient(model=QWEN38)
    kwargs: dict = {}
    client._apply_thinking_params(kwargs, QWEN38)
    assert kwargs == {}


@pytest.mark.parametrize(
    ("budget", "effort"),
    [
        # A derived tier never picks "low" (it does not shorten reasoning on Qwen3.8).
        (1, "medium"),
        (500, "medium"),
        (2048, "medium"),
        (2049, "medium"),
        (8192, "medium"),
        (8193, "xhigh"),
        (10_000, "xhigh"),
    ],
)
def test_qwen_budget_maps_to_effort_tier_and_keeps_legacy_keys(budget: int, effort: str) -> None:
    client = LLMClient(model=QWEN38, thinking_budget=budget)
    kwargs: dict = {}
    client._apply_thinking_params(kwargs, QWEN38)
    body = kwargs["extra_body"]
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": effort}
    assert body["thinking"] is True
    assert body["thinking_budget"] == budget


def test_qwen_explicit_effort_beats_budget_tier() -> None:
    client = LLMClient(model=QWEN38, thinking_budget=10_000, reasoning_effort="low")
    kwargs: dict = {}
    client._apply_thinking_params(kwargs, QWEN38)
    assert kwargs["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "low"


def test_qwen_effort_none_suppresses_legacy_thinking_keys() -> None:
    """effort=none + a budget must not send thinking=True beside enable_thinking=False."""
    client = LLMClient(model=QWEN38, thinking_budget=10_000, reasoning_effort="none")
    kwargs: dict = {}
    client._apply_thinking_params(kwargs, QWEN38)
    assert kwargs == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def test_non_qwen_reasoning_effort_passthrough_unchanged() -> None:
    """Regression guard: OpenAI-style models still get the raw top-level value."""
    client = LLMClient(model="o3-mini", reasoning_effort="high")
    kwargs: dict = {}
    client._apply_thinking_params(kwargs, "o3-mini")
    assert kwargs == {"reasoning_effort": "high"}


def test_legacy_qwen3_reasoning_effort_goes_into_extra_body() -> None:
    client = LLMClient(model="qwen3-coder", thinking_budget=1000, reasoning_effort="high")
    kwargs: dict = {}
    client._apply_thinking_params(kwargs, "qwen3-coder")
    assert kwargs["extra_body"] == {
        "thinking": True,
        "thinking_budget": 1000,
        "reasoning_effort": "high",
    }


@pytest.mark.asyncio
async def test_call_sends_chat_template_kwargs_for_qwen38() -> None:
    client = LLMClient(model=QWEN38, reasoning_effort="medium")
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content="ok", tool_calls=None, thinking=None),
            finish_reason="stop",
        )
    ]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch("godspeed.llm.client._get_litellm") as mock_litellm:
        mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_response)
        await client._call(QWEN38, [{"role": "user", "content": "hi"}], None)

        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert call_kwargs["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": True,
            "reasoning_effort": "medium",
        }
        assert "reasoning_effort" not in call_kwargs


def test_effort_command_accepts_none(tmp_path) -> None:
    """`/effort none` is accepted and stored so Qwen3.5+ thinking can be disabled."""
    from godspeed.tui.commands import Commands

    client = LLMClient(model=QWEN38)
    commands = Commands.__new__(Commands)
    commands._llm_client = client
    result = commands._cmd_effort("none")
    assert result.handled is True
    assert client.reasoning_effort == "none"
