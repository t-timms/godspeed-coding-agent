"""Tests for LLM client — ChatResponse, ModelRouter, LLMClient, BudgetExceededError."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.llm.client import (
    BudgetExceededError,
    ChatResponse,
    LLMClient,
    ModelRouter,
)

# ---------------------------------------------------------------------------
# Test: ChatResponse
# ---------------------------------------------------------------------------


class TestChatResponse:
    def test_has_tool_calls_true(self) -> None:
        r = ChatResponse(tool_calls=[{"id": "1", "function": {"name": "x"}}])
        assert r.has_tool_calls is True

    def test_has_tool_calls_false(self) -> None:
        r = ChatResponse(content="hello")
        assert r.has_tool_calls is False

    def test_defaults(self) -> None:
        r = ChatResponse()
        assert r.content == ""
        assert r.tool_calls == []
        assert r.thinking == ""


# ---------------------------------------------------------------------------
# Test: BudgetExceededError
# ---------------------------------------------------------------------------


class TestBudgetExceededError:
    def test_attributes(self) -> None:
        err = BudgetExceededError(spent=1.50, limit=1.00)
        assert err.spent == 1.50
        assert err.limit == 1.00
        assert "$1.5" in str(err)


# ---------------------------------------------------------------------------
# Test: ModelRouter
# ---------------------------------------------------------------------------


class TestModelRouter:
    def test_no_routing(self) -> None:
        router = ModelRouter()
        assert router.route("ollama/qwen3:4b") == "ollama/qwen3:4b"
        assert router.has_routing is False

    def test_with_routing(self) -> None:
        router = ModelRouter({"plan": "claude-sonnet", "edit": "gpt-4o"})
        assert router.route("default", "plan") == "claude-sonnet"
        assert router.route("default", "edit") == "gpt-4o"
        assert router.route("default", "chat") == "default"
        assert router.has_routing is True
        assert router.routes == {"plan": "claude-sonnet", "edit": "gpt-4o"}

    def test_unknown_task_type_uses_default(self) -> None:
        router = ModelRouter({"plan": "claude"})
        assert router.route("fallback", "unknown") == "fallback"

    def test_none_task_type(self) -> None:
        router = ModelRouter({"plan": "claude"})
        assert router.route("default", None) == "default"


# ---------------------------------------------------------------------------
# Test: LLMClient helpers (no LLM calls)
# ---------------------------------------------------------------------------


class TestLLMClientHelpers:
    def test_supports_tool_calling_ollama_qwen(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        assert client._supports_tool_calling() is True

    def test_supports_tool_calling_ollama_unknown(self) -> None:
        client = LLMClient(model="ollama/phi4:latest")
        assert client._supports_tool_calling() is False

    def test_supports_tool_calling_api_model(self) -> None:
        client = LLMClient(model="claude-sonnet-4-20250514")
        assert client._supports_tool_calling() is True

    def test_effective_model_upgrades_ollama(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        assert client._effective_model() == "ollama_chat/qwen3:4b"

    def test_effective_model_no_upgrade_unsupported(self) -> None:
        client = LLMClient(model="ollama/phi4:latest")
        assert client._effective_model() == "ollama/phi4:latest"

    def test_effective_model_api_unchanged(self) -> None:
        client = LLMClient(model="claude-sonnet-4-20250514")
        assert client._effective_model() == "claude-sonnet-4-20250514"

    def test_is_connection_error(self) -> None:
        assert LLMClient._is_connection_error(ConnectionError("Connection refused"))
        assert not LLMClient._is_connection_error(ValueError("bad input"))

    def test_is_anthropic_model(self) -> None:
        client = LLMClient(model="claude-sonnet-4-20250514")
        assert client._is_anthropic_model() is True
        assert client._is_anthropic_model("anthropic/claude-haiku") is True
        client2 = LLMClient(model="gpt-4o")
        assert client2._is_anthropic_model() is False

    def test_check_budget_no_limit(self) -> None:
        client = LLMClient(model="test", max_cost_usd=0.0)
        client.total_cost_usd = 100.0
        client._check_budget()  # Should not raise

    def test_check_budget_exceeded(self) -> None:
        client = LLMClient(model="test", max_cost_usd=1.0)
        client.total_cost_usd = 1.50
        with pytest.raises(BudgetExceededError):
            client._check_budget()

    def test_check_budget_within_limit(self) -> None:
        client = LLMClient(model="test", max_cost_usd=2.0)
        client.total_cost_usd = 1.50
        client._check_budget()  # Should not raise

    def test_build_failure_error_ollama(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        err = client._build_failure_error(ConnectionError("Connection refused"))
        assert "ollama serve" in str(err).lower()

    def test_build_failure_error_generic(self) -> None:
        client = LLMClient(model="gpt-4o")
        err = client._build_failure_error(RuntimeError("timeout"))
        assert "All models failed" in str(err)

    def test_apply_prompt_caching_claude(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Query"},
            {"role": "assistant", "content": "Reply"},
        ]
        result = LLMClient._apply_prompt_caching("claude-sonnet", messages)
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_apply_prompt_caching_ollama_noop(self) -> None:
        messages = [{"role": "system", "content": "You are helpful."}]
        result = LLMClient._apply_prompt_caching("ollama/qwen3", messages)
        assert result == messages


# ---------------------------------------------------------------------------
# Test: LLMClient.chat (mocked LiteLLM)
# ---------------------------------------------------------------------------


def _mock_response(
    content: str = "Hello",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> MagicMock:
    """Build a mock litellm response."""
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        thinking=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class TestLLMClientChat:
    @pytest.mark.asyncio
    async def test_basic_chat(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        mock_resp = _mock_response(content="Hi there")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client.chat([{"role": "user", "content": "hello"}])

        assert response.content == "Hi there"
        assert response.finish_reason == "stop"
        assert client.total_input_tokens == 10
        assert client.total_output_tokens == 20

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self) -> None:
        tc = SimpleNamespace(
            id="tc_1",
            function=SimpleNamespace(name="file_read", arguments='{"path": "foo.py"}'),
        )
        mock_resp = _mock_response(content="", tool_calls=[tc])

        client = LLMClient(model="ollama/qwen3:4b")
        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client.chat([{"role": "user", "content": "read foo"}])

        assert response.has_tool_calls
        assert response.tool_calls[0]["function"]["name"] == "file_read"

    @pytest.mark.asyncio
    async def test_chat_with_thinking(self) -> None:
        msg = SimpleNamespace(
            content="Result",
            tool_calls=[],
            thinking="I need to think about this...",
        )
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
        mock_resp = SimpleNamespace(choices=[choice], usage=usage)

        client = LLMClient(model="claude-sonnet-4-20250514", thinking_budget=1000)
        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client.chat([{"role": "user", "content": "think"}])

        assert response.thinking == "I need to think about this..."
        # Verify thinking param was passed
        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert call_kwargs["thinking"]["budget_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_chat_content_blocks(self) -> None:
        """Handle responses where content is a list of blocks."""
        msg = SimpleNamespace(
            content=[
                {"type": "text", "text": "Part 1. "},
                {"type": "text", "text": "Part 2."},
            ],
            tool_calls=[],
            thinking=None,
        )
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=10)
        mock_resp = SimpleNamespace(choices=[choice], usage=usage)

        client = LLMClient(model="claude-sonnet-4-20250514")
        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "Part 1. Part 2."

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self) -> None:
        client = LLMClient(
            model="ollama/qwen3:4b",
            fallback_models=["ollama/gemma3:4b"],
        )

        call_count = 0

        async def _failing_then_success(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Primary + retry both fail
                raise RuntimeError("model down")
            return _mock_response(content="from fallback")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(side_effect=_failing_then_success)
            response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "from fallback"
        assert call_count == 3  # primary + retry + fallback

    @pytest.mark.asyncio
    async def test_all_models_fail_raises(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(side_effect=RuntimeError("down"))
            with pytest.raises(RuntimeError, match="All models failed"):
                await client.chat([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_budget_exceeded_during_chat(self) -> None:
        client = LLMClient(model="claude-sonnet-4-20250514", max_cost_usd=0.001)
        mock_resp = _mock_response(input_tokens=100_000, output_tokens=50_000)

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            # BudgetExceededError is caught by fallback loop, surfaced as RuntimeError
            with pytest.raises(RuntimeError, match="Budget exceeded"):
                await client.chat([{"role": "user", "content": "expensive"}])

    @pytest.mark.asyncio
    async def test_model_routing(self) -> None:
        router = ModelRouter({"plan": "claude-sonnet-4-20250514"})
        client = LLMClient(model="ollama/qwen3:4b", router=router)
        mock_resp = _mock_response()

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            await client.chat(
                [{"role": "user", "content": "plan this"}],
                task_type="plan",
            )

        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        # Model should be restored after call
        assert client.model == "ollama/qwen3:4b"

    @pytest.mark.asyncio
    async def test_connection_refused_skips_retry(self) -> None:
        """Connection-refused errors skip the retry and go straight to fallback."""
        client = LLMClient(
            model="ollama/qwen3:4b",
            fallback_models=["ollama/gemma3:4b"],
        )

        call_count = 0

        async def _conn_refused_then_ok(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs["model"].startswith("ollama_chat/qwen"):
                raise ConnectionError("Connection refused")
            return _mock_response(content="ok")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(side_effect=_conn_refused_then_ok)
            response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "ok"
        # Should be: primary (fail, conn refused) + fallback (ok) = 2 calls
        assert call_count == 2


# ---------------------------------------------------------------------------
# Test: Retry / fallback / error-classification helpers
# ---------------------------------------------------------------------------


class TestRetryHelpers:
    """Unit tests for static/class retry methods on LLMClient."""

    def test_is_rate_limit_429_pair(self) -> None:
        assert LLMClient._is_rate_limit_error(RuntimeError("429 too many requests"))
        assert LLMClient._is_rate_limit_error(RuntimeError("429 rate limit hit"))
        assert LLMClient._is_rate_limit_error(RuntimeError("429 quota exceeded"))
        assert LLMClient._is_rate_limit_error(RuntimeError("429 throttled"))

    def test_is_rate_limit_marker(self) -> None:
        assert LLMClient._is_rate_limit_error(RuntimeError("rate_limit_exceeded"))
        assert LLMClient._is_rate_limit_error(RuntimeError("rate limit"))
        assert LLMClient._is_rate_limit_error(RuntimeError("ratelimiterror"))
        assert LLMClient._is_rate_limit_error(RuntimeError("quota exceeded"))
        assert LLMClient._is_rate_limit_error(RuntimeError("too many requests"))

    def test_is_rate_limit_bare_429_no_match(self) -> None:
        """A bare '429' without a rate-limit keyword nearby should not match."""
        assert not LLMClient._is_rate_limit_error(RuntimeError("port 429 is in use"))
        assert not LLMClient._is_rate_limit_error(RuntimeError("error code 429"))

    def test_is_rate_limit_no_match(self) -> None:
        assert not LLMClient._is_rate_limit_error(RuntimeError("internal server error"))
        assert not LLMClient._is_rate_limit_error(ValueError("bad input"))

    def test_parse_retry_after_none(self) -> None:
        assert LLMClient._parse_retry_after("rate limit hit") is None
        assert LLMClient._parse_retry_after("") is None

    def test_parse_retry_after_variants(self) -> None:
        assert LLMClient._parse_retry_after("retry-after: 5") == 5.0
        assert LLMClient._parse_retry_after("Retry-After: 30") == 30.0
        assert LLMClient._parse_retry_after("retryafter: 15") == 15.0
        assert LLMClient._parse_retry_after("RetryAfter: 20") == 20.0

    def test_parse_retry_after_clamps_to_max(self) -> None:
        from godspeed.llm.client import RATE_LIMIT_MAX_DELAY

        result = LLMClient._parse_retry_after(f"retry-after: {RATE_LIMIT_MAX_DELAY + 100}")
        assert result == RATE_LIMIT_MAX_DELAY

    def test_backoff_delay_with_retry_after(self) -> None:
        with patch("random.uniform", return_value=0.1):
            delay = LLMClient._backoff_delay(0, retry_after=5.0)
        # 5.0 * (1.0 + 0.1) = 5.5
        assert abs(delay - 5.5) < 0.01

    def test_backoff_delay_exponential(self) -> None:
        with patch("random.uniform", return_value=0.0):  # no jitter
            d0 = LLMClient._backoff_delay(0, retry_after=None)
            d1 = LLMClient._backoff_delay(1, retry_after=None)
            d2 = LLMClient._backoff_delay(2, retry_after=None)
        assert abs(d0 - 1.0) < 0.01  # base * 2^0 = 1
        assert abs(d1 - 2.0) < 0.01  # base * 2^1 = 2
        assert abs(d2 - 4.0) < 0.01  # base * 2^2 = 4

    def test_backoff_delay_caps_at_max(self) -> None:
        from godspeed.llm.client import RATE_LIMIT_MAX_DELAY

        with patch("random.uniform", return_value=1.25):  # maximum +jitter
            delay = LLMClient._backoff_delay(100, retry_after=None)
        assert delay <= RATE_LIMIT_MAX_DELAY


class TestLLMClientRateLimit:
    """Tests for _retry_on_rate_limit and rate-limit integration in chat()."""

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_recovers(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=[
                    RuntimeError("429 rate limit hit"),  # first call fails
                    _mock_response(content="recovered"),  # retry succeeds
                ]
            )
            with patch("asyncio.sleep", AsyncMock()):
                response = await client._retry_on_rate_limit(
                    "ollama_chat/qwen3:4b",
                    [{"role": "user", "content": "hi"}],
                    None,
                    RuntimeError("429 rate limit hit"),
                )

        assert response is not None
        assert response.content == "recovered"

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_exhausted(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=RuntimeError("429 rate limit hit"),
            )
            with patch("asyncio.sleep", AsyncMock()):
                response = await client._retry_on_rate_limit(
                    "ollama_chat/qwen3:4b",
                    [{"role": "user", "content": "hi"}],
                    None,
                    RuntimeError("429 rate limit hit"),
                )

        assert response is None

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_morphs_error(self) -> None:
        """Non-rate-limit error during retry should propagate immediately."""
        client = LLMClient(model="ollama/qwen3:4b")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=RuntimeError("connection refused"),
            )
            with patch("asyncio.sleep", AsyncMock()):
                with pytest.raises(RuntimeError, match="connection refused"):
                    await client._retry_on_rate_limit(
                        "ollama_chat/qwen3:4b",
                        [{"role": "user", "content": "hi"}],
                        None,
                        RuntimeError("429 rate limit hit"),
                    )

    @pytest.mark.asyncio
    async def test_rate_limit_falls_to_fallback(self) -> None:
        """Rate-limit exhausts on primary, moves to fallback model."""
        client = LLMClient(
            model="ollama/qwen3:4b",
            fallback_models=["ollama/gemma3:4b"],
        )

        call_count = 0

        async def _rate_limit_then_ok(**kwargs):
            nonlocal call_count
            call_count += 1
            if "qwen" in kwargs["model"]:
                raise RuntimeError("429 rate limit hit")
            return _mock_response(content="from fallback")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(side_effect=_rate_limit_then_ok)
            with patch("asyncio.sleep", AsyncMock()):
                response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "from fallback"
        # primary + 4 rate-limit retries + fallback
        assert call_count >= 5

    @pytest.mark.asyncio
    async def test_chat_rate_limit_then_recovers_on_primary(self) -> None:
        """Rate-limit triggers retry, then recovers on same model."""
        client = LLMClient(model="ollama/qwen3:4b")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=[
                    RuntimeError("429 rate limit hit"),
                    _mock_response(content="ok after backoff"),
                ]
            )
            with patch("asyncio.sleep", AsyncMock()):
                response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "ok after backoff"

    @pytest.mark.asyncio
    async def test_generic_error_triggers_primary_retry(self) -> None:
        """Non-rate-limit, non-connection error retries primary once."""
        client = LLMClient(
            model="ollama/qwen3:4b",
            fallback_models=["ollama/gemma3:4b"],
        )

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=[
                    RuntimeError("server error"),  # primary fails
                    _mock_response(content="retry ok"),  # primary retry succeeds
                ]
            )
            with patch("asyncio.sleep", AsyncMock()):
                response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "retry ok"

    @pytest.mark.asyncio
    async def test_generic_error_primary_retry_then_fallback(self) -> None:
        """Primary retry fails, falls through to fallback model."""
        client = LLMClient(
            model="ollama/qwen3:4b",
            fallback_models=["ollama/gemma3:4b"],
        )

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=[
                    RuntimeError("server error"),  # primary
                    RuntimeError("server error again"),  # primary retry
                    _mock_response(content="fallback ok"),  # fallback
                ]
            )
            with patch("asyncio.sleep", AsyncMock()):
                response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "fallback ok"


class TestLLMClientCall:
    """Tests for the internal _call method."""

    @pytest.mark.asyncio
    async def test_call_basic(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        mock_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hello", tool_calls=[], thinking=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
        )

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client._call(
                "ollama_chat/qwen3:4b",
                [{"role": "user", "content": "hi"}],
                None,
            )

        assert response.content == "hello"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_call_empty_choices(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        mock_resp = SimpleNamespace(choices=[], usage=None)

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client._call(
                "ollama_chat/qwen3:4b",
                [{"role": "user", "content": "hi"}],
                None,
            )

        assert response.content == ""
        assert response.tool_calls == []

    @pytest.mark.asyncio
    async def test_call_with_tools(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        tc = SimpleNamespace(
            id="tc1",
            function=SimpleNamespace(name="file_read", arguments='{"path": "x"}'),
        )
        mock_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="", tool_calls=[tc], thinking=None),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
        )

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client._call(
                "ollama_chat/qwen3:4b",
                [{"role": "user", "content": "read x"}],
                [{"type": "function", "function": {"name": "file_read"}}],
            )

        assert response.has_tool_calls
        assert response.tool_calls[0]["function"]["name"] == "file_read"


class TestLLMClientConnectionError:
    """Edge-case tests for connection error detection and messaging."""

    def test_is_connection_error_variants(self) -> None:
        assert LLMClient._is_connection_error(ConnectionError("Connection refused"))
        assert LLMClient._is_connection_error(OSError("Cannot connect"))
        assert LLMClient._is_connection_error(RuntimeError("Connect call failed"))
        assert not LLMClient._is_connection_error(RuntimeError("timeout"))
        assert not LLMClient._is_connection_error(ValueError("something else"))

    def test_build_failure_error_llamacpp(self) -> None:
        client = LLMClient(model="llamacpp/qwen2.5-coder")
        err = client._build_failure_error(ConnectionError("Connection refused"))
        assert "llama.cpp" in str(err)
        assert "not running" in str(err)

    def test_build_failure_error_llamacpp_openai(self) -> None:
        client = LLMClient(model="openai/qwen2.5-coder")
        err = client._build_failure_error(ConnectionError("Connection refused"))
        assert "llama.cpp" in str(err)

    @pytest.mark.asyncio
    async def test_connection_error_no_retry(self) -> None:
        """Connection errors should not trigger a retry — go straight to fallback."""
        client = LLMClient(
            model="ollama/qwen3:4b",
            fallback_models=["ollama/gemma3:4b"],
        )

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=[
                    ConnectionError("Connection refused"),  # primary, no retry
                    _mock_response(content="fallback ok"),  # fallback
                ]
            )
            response = await client.chat([{"role": "user", "content": "test"}])

        assert response.content == "fallback ok"


class TestLLMClientStreaming:
    """Tests for the streaming chat path."""

    @pytest.mark.asyncio
    async def test_stream_chat_basic(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        messages = [{"role": "user", "content": "hello"}]

        class FakeStream:
            def __init__(self):
                self._chunks = [
                    ChatResponse(content="Hello", tool_calls=[], finish_reason=None),
                    ChatResponse(content=" there", tool_calls=[], finish_reason=None),
                    ChatResponse(content="", tool_calls=[], finish_reason="stop"),
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

        async def _mock_inner(self, msgs, tools=None, **kw):
            async for c in FakeStream():
                yield c

        client._stream_chat_inner = _mock_inner.__get__(client)
        chunks = []
        async for chunk in client.stream_chat(messages):
            chunks.append(chunk)

        assert len(chunks) >= 2
        assert chunks[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_fails_falls_back_to_batch(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        messages = [{"role": "user", "content": "hello"}]

        async def failing_stream(self, msgs, tools=None, **kw):
            raise RuntimeError("stream failed")
            yield

        client._stream_chat_inner = failing_stream.__get__(client)
        with patch.object(client, "chat") as mock_chat:
            mock_chat.return_value = ChatResponse(
                content="batch fallback", tool_calls=[], finish_reason="stop"
            )
            chunks = []
            async for chunk in client.stream_chat(messages):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "batch fallback"

    @pytest.mark.asyncio
    async def test_stream_inner_basic(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")

        class MockStream:
            async def __aiter__(self):
                for i in range(3):
                    delta = SimpleNamespace(content=f"chunk{i}", tool_calls=None)
                    choice = SimpleNamespace(delta=delta, finish_reason=None)
                    yield SimpleNamespace(choices=[choice])
                final_delta = SimpleNamespace(content="", tool_calls=None)
                final_choice = SimpleNamespace(delta=final_delta, finish_reason="stop")
                yield SimpleNamespace(choices=[final_choice])

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=MockStream())
            chunks = []
            async for chunk in client._stream_chat_inner(
                [{"role": "user", "content": "hello"}], None
            ):
                chunks.append(chunk)

        assert len(chunks) >= 3

    @pytest.mark.asyncio
    async def test_stream_inner_no_finish_reason(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")

        class MockStream:
            async def __aiter__(self):
                delta = SimpleNamespace(content="partial", tool_calls=None)
                choice = SimpleNamespace(delta=delta, finish_reason=None)
                yield SimpleNamespace(choices=[choice])

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=MockStream())
            chunks = []
            async for chunk in client._stream_chat_inner(
                [{"role": "user", "content": "hello"}], None
            ):
                chunks.append(chunk)

        assert len(chunks) >= 1
        assert chunks[-1].finish_reason == "incomplete"

    @pytest.mark.asyncio
    async def test_stream_inner_error(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=RuntimeError("stream error")
            )
            with pytest.raises(RuntimeError, match="stream error"):
                async for _chunk in client._stream_chat_inner(
                    [{"role": "user", "content": "hello"}], None
                ):
                    pass

    @pytest.mark.asyncio
    async def test_stream_inner_thinking_budget(self) -> None:
        client = LLMClient(model="claude-sonnet-4-20250514", thinking_budget=500)

        class MockStream:
            async def __aiter__(self):
                delta = SimpleNamespace(content="ok", tool_calls=None)
                choice = SimpleNamespace(delta=delta, finish_reason="stop")
                yield SimpleNamespace(choices=[choice])

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=MockStream())
            chunks = []
            async for chunk in client._stream_chat_inner(
                [{"role": "user", "content": "think"}], None
            ):
                chunks.append(chunk)

        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert call_kwargs["thinking"]["budget_tokens"] == 500

    @pytest.mark.asyncio
    async def test_stream_inner_qwen_thinking(self) -> None:
        client = LLMClient(model="qwen3.6-4b", thinking_budget=500)

        class MockStream:
            async def __aiter__(self):
                delta = SimpleNamespace(content="ok", tool_calls=None)
                choice = SimpleNamespace(delta=delta, finish_reason="stop")
                yield SimpleNamespace(choices=[choice])

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=MockStream())
            chunks = []
            async for chunk in client._stream_chat_inner(
                [{"role": "user", "content": "think"}], None
            ):
                chunks.append(chunk)

        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert call_kwargs["extra_body"]["thinking"] is True

    @pytest.mark.asyncio
    async def test_stream_inner_with_tools(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        tools = [{"type": "function", "function": {"name": "file_read"}}]

        class MockStream:
            async def __aiter__(self):
                tc_delta = SimpleNamespace(
                    index=0,
                    id="call_1",
                    function=SimpleNamespace(name="file_read", arguments='{"file_path":'),
                )
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=None, tool_calls=[tc_delta]),
                            finish_reason=None,
                        )
                    ]
                )
                tc_delta2 = SimpleNamespace(
                    index=0,
                    id=None,
                    function=SimpleNamespace(name=None, arguments='"x.py"}'),
                )
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=None, tool_calls=[tc_delta2]),
                            finish_reason=None,
                        )
                    ]
                )
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="", tool_calls=None),
                            finish_reason="tool_calls",
                        )
                    ]
                )

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=MockStream())
            chunks = []
            async for chunk in client._stream_chat_inner(
                [{"role": "user", "content": "read x"}], tools
            ):
                chunks.append(chunk)

        final_chunk = chunks[-1]
        assert final_chunk.finish_reason == "tool_calls"
        assert len(final_chunk.tool_calls) > 0

    @pytest.mark.asyncio
    async def test_stream_inner_no_tool_calling_support(self) -> None:
        client = LLMClient(model="ollama/phi4:latest")
        tools = [{"type": "function", "function": {"name": "file_read"}}]

        class MockStream:
            async def __aiter__(self):
                delta = SimpleNamespace(content="no tools", tool_calls=None)
                choice = SimpleNamespace(delta=delta, finish_reason="stop")
                yield SimpleNamespace(choices=[choice])

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=MockStream())
            chunks = []
            async for chunk in client._stream_chat_inner(
                [{"role": "user", "content": "read"}], tools
            ):
                chunks.append(chunk)

        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_stream_inner_skip_empty_choices(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")

        class MockStream:
            async def __aiter__(self):
                yield SimpleNamespace(choices=[])
                delta = SimpleNamespace(content="ok", tool_calls=None)
                choice = SimpleNamespace(delta=delta, finish_reason="stop")
                yield SimpleNamespace(choices=[choice])

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=MockStream())
            chunks = []
            async for chunk in client._stream_chat_inner(
                [{"role": "user", "content": "hello"}], None
            ):
                chunks.append(chunk)

        assert len(chunks) >= 1


class TestLLMClientDeepSeek:
    """Tests for the DeepSeek direct API path."""

    @pytest.mark.asyncio
    async def test_deepseek_model_routes_to_direct(self) -> None:
        client = LLMClient(model="deepseek-v4-pro")
        mock_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None, reasoning_content=""),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
        )

        mock_openai = MagicMock()
        mock_openai.chat = MagicMock()
        mock_openai.chat.completions = MagicMock()
        mock_openai.chat.completions.create = MagicMock(return_value=mock_resp)

        with patch("openai.OpenAI", return_value=mock_openai):
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
                response = await client._call(
                    "deepseek-v4-pro",
                    [{"role": "user", "content": "hello"}],
                    None,
                )

        assert response.content == "ok"

    @pytest.mark.asyncio
    async def test_deepseek_with_tools(self) -> None:
        client = LLMClient(model="deepseek-v4-pro")
        tc_item = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="file_read", arguments='{"path": "x.py"}'),
        )
        mock_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="", tool_calls=[tc_item], reasoning_content=None
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=15),
        )

        mock_openai = MagicMock()
        mock_openai.chat = MagicMock()
        mock_openai.chat.completions = MagicMock()
        mock_openai.chat.completions.create = MagicMock(return_value=mock_resp)

        with patch("openai.OpenAI", return_value=mock_openai):
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
                response = await client._call(
                    "deepseek-v4-pro",
                    [{"role": "user", "content": "read x"}],
                    [{"type": "function", "function": {"name": "file_read"}}],
                )

        assert response.has_tool_calls
        assert response.tool_calls[0]["function"]["name"] == "file_read"

    @pytest.mark.asyncio
    async def test_deepseek_no_api_key(self) -> None:
        client = LLMClient(model="deepseek-v4-pro")

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
                await client._call(
                    "deepseek-v4-pro",
                    [{"role": "user", "content": "hello"}],
                    None,
                )

    @pytest.mark.asyncio
    async def test_deepseek_message_restructuring(self) -> None:
        client = LLMClient(model="deepseek-v4-pro")
        mock_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
        )

        mock_openai = MagicMock()
        mock_openai.chat = MagicMock()
        mock_openai.chat.completions = MagicMock()
        mock_openai.chat.completions.create = MagicMock(return_value=mock_resp)

        messages = [
            {"role": "user", "content": "do stuff"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    },
                    {
                        "id": "t2",
                        "type": "function",
                        "function": {"name": "grep", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "result1"},
            {"role": "tool", "tool_call_id": "t2", "content": "result2"},
        ]

        with patch("openai.OpenAI", return_value=mock_openai):
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
                response = await client._call("deepseek-v4-pro", messages, None)

        assert response.content == "ok"

    @pytest.mark.asyncio
    async def test_deepseek_with_reasoning_content_stripping(self) -> None:
        client = LLMClient(model="deepseek-v4-pro")
        mock_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
        )

        mock_openai = MagicMock()
        mock_openai.chat = MagicMock()
        mock_openai.chat.completions = MagicMock()
        mock_openai.chat.completions.create = MagicMock(return_value=mock_resp)

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "reasoning_content": "I should reply nicely."},
        ]

        with patch("openai.OpenAI", return_value=mock_openai):
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
                response = await client._call("deepseek-v4-pro", messages, None)

        assert response.content == "ok"


class TestLLMClientCallExtended:
    """Extended tests for the _call method."""

    @pytest.mark.asyncio
    async def test_call_with_unsupported_tools_model(self) -> None:
        client = LLMClient(model="ollama/phi4:latest")
        mock_resp = _mock_response(content="text response")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client._call(
                "ollama/phi4:latest",
                [{"role": "user", "content": "run cmd"}],
                [{"type": "function", "function": {"name": "shell"}}],
            )

        assert response.content == "text response"
        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_call_with_thinking(self) -> None:
        client = LLMClient(model="claude-sonnet-4-20250514", thinking_budget=2000)
        mock_resp = _mock_response(content="thoughtful")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client._call(
                "claude-sonnet-4-20250514",
                [{"role": "user", "content": "think deep"}],
                None,
            )

        assert response.content == "thoughtful"
        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert call_kwargs["thinking"]["budget_tokens"] == 2000
        assert call_kwargs["thinking"]["type"] == "enabled"

    @pytest.mark.asyncio
    async def test_call_with_qwen_thinking(self) -> None:
        client = LLMClient(model="qwen3.6-4b", thinking_budget=1000)
        mock_resp = _mock_response(content="reasoned")

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client._call(
                "qwen3.6-4b",
                [{"role": "user", "content": "think"}],
                None,
            )

        assert response.content == "reasoned"
        call_kwargs = mock_litellm.return_value.acompletion.call_args[1]
        assert call_kwargs["extra_body"]["thinking"] is True
        assert call_kwargs["extra_body"]["thinking_budget"] == 1000

    @pytest.mark.asyncio
    async def test_call_no_usage_data(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        mock_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hi", tool_calls=[], thinking=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            response = await client._call(
                "ollama_chat/qwen3:4b",
                [{"role": "user", "content": "hi"}],
                None,
            )

        assert response.usage == {}

    @pytest.mark.asyncio
    async def test_call_budget_check_after_cost(self) -> None:
        client = LLMClient(model="claude-sonnet-4-20250514", max_cost_usd=0.001)
        mock_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="costly", tool_calls=[], thinking=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=100000, completion_tokens=50000),
        )

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            with pytest.raises(RuntimeError, match="Budget exceeded"):
                await client._call(
                    "claude-sonnet-4-20250514",
                    [{"role": "user", "content": "expensive"}],
                    None,
                )

    @pytest.mark.asyncio
    async def test_deepseek_prefix_route(self) -> None:
        client = LLMClient(model="llamacpp/qwen2.5-coder")
        mock_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="deepseek ok", tool_calls=None, reasoning_content=""
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
        )
        mock_openai = MagicMock()
        mock_openai.chat = MagicMock()
        mock_openai.chat.completions = MagicMock()
        mock_openai.chat.completions.create = MagicMock(return_value=mock_resp)

        with patch("openai.OpenAI", return_value=mock_openai):
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
                response = await client._call(
                    "deepseek/deepseek-chat",
                    [{"role": "user", "content": "hello"}],
                    None,
                )

        assert response.content == "deepseek ok"

    @pytest.mark.asyncio
    async def test_chat_with_routing_preserves_model_on_error(self) -> None:
        router = ModelRouter({"plan": "claude-sonnet"})
        client = LLMClient(model="ollama/qwen3:4b", router=router)

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=RuntimeError("generic error")
            )
            with pytest.raises(RuntimeError, match="All models failed"):
                await client.chat(
                    [{"role": "user", "content": "plan"}],
                    task_type="plan",
                )

        assert client.model == "ollama/qwen3:4b"
        assert client._model_lower == "ollama/qwen3:4b"

    @pytest.mark.asyncio
    async def test_chat_with_fallback_chain_build_error(self) -> None:
        client = LLMClient(model="openai/qwen2.5-coder", fallback_models=[])

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=ConnectionError("Connect call failed")
            )
            with pytest.raises(RuntimeError, match=r"llama.cpp"):
                await client.chat([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_chat_raises_on_all_models_connection_error(self) -> None:
        client = LLMClient(model="ollama/missing-model")
        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(
                side_effect=ConnectionError("Connection refused")
            )
            with pytest.raises(RuntimeError, match="ollama serve"):
                await client.chat([{"role": "user", "content": "test"}])


class TestLLMClientPromptCaching:
    """Additional prompt caching edges."""

    def test_supports_prompt_caching(self) -> None:
        assert LLMClient._supports_prompt_caching("claude-sonnet") is True
        assert LLMClient._supports_prompt_caching("anthropic/claude-haiku") is True
        assert LLMClient._supports_prompt_caching("deepseek-chat") is True
        assert LLMClient._supports_prompt_caching("gpt-4o") is False
        assert LLMClient._supports_prompt_caching("ollama/qwen3") is False

    def test_apply_prompt_caching_short_messages(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        result = LLMClient._apply_prompt_caching("claude-sonnet", messages)
        assert result == messages

    def test_apply_prompt_caching_empty_content(self) -> None:
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "query"},
        ]
        result = LLMClient._apply_prompt_caching("claude-sonnet", messages)
        assert result[0].get("content") == ""

    def test_apply_prompt_caching_not_supported(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Query"},
        ]
        result = LLMClient._apply_prompt_caching("gpt-4o", messages)
        assert result == messages


class TestLLMClientTokenTracking:
    """Token and cost tracking during LLM calls."""

    @pytest.mark.asyncio
    async def test_chat_tracks_tokens_correctly(self) -> None:
        client = LLMClient(model="ollama/qwen3:4b")
        assert client.total_input_tokens == 0
        assert client.total_output_tokens == 0
        assert client.total_cost_usd == 0.0

        mock_resp1 = _mock_response(input_tokens=50, output_tokens=30)
        mock_resp2 = _mock_response(input_tokens=20, output_tokens=10)

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(side_effect=[mock_resp1, mock_resp2])
            await client.chat([{"role": "user", "content": "hi"}])
            await client.chat([{"role": "user", "content": "again"}])

        assert client.total_input_tokens == 70
        assert client.total_output_tokens == 40

    @pytest.mark.asyncio
    async def test_chat_cost_tracking(self) -> None:
        client = LLMClient(model="claude-sonnet-4-20250514")
        assert client.total_cost_usd == 0.0

        mock_resp = _mock_response(input_tokens=100, output_tokens=50)
        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=mock_resp)
            await client.chat([{"role": "user", "content": "hi"}])

        assert client.total_cost_usd > 0.0

    @pytest.mark.asyncio
    async def test_supports_thinking_detection(self) -> None:
        client = LLMClient(model="qwen3.6-4b")
        assert client._supports_thinking() is True
        assert client._supports_thinking("qwen3.6-8b") is True
        assert client._supports_thinking("qwen3-4b") is True
        assert client._supports_thinking("claude-sonnet") is False
        assert client._supports_thinking("gpt-4o") is False

    def test_is_anthropic_variants(self) -> None:
        client = LLMClient(model="claude-sonnet-4-20250514")
        assert client._is_anthropic_model() is True
        assert client._is_anthropic_model("anthropic/claude-opus") is True
        client2 = LLMClient(model="gpt-4o")
        assert client2._is_anthropic_model() is False


class TestStreamingCostNoDoubleCount:
    """Streaming path must not double-count cost when final usage is present."""

    @pytest.mark.asyncio
    async def test_cumulative_usage_replaces_heuristic_cost(self) -> None:
        from godspeed.llm.cost import estimate_cost

        model = "claude-sonnet-4-20250514"
        client = LLMClient(model=model)

        final_input = 100
        final_output = 200

        class CumulativeUsageStream:
            def __init__(self):
                self._idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx >= 4:
                    raise StopAsyncIteration
                self._idx += 1
                if self._idx < 4:
                    delta = SimpleNamespace(content=f"chunk{self._idx}", tool_calls=None)
                    return SimpleNamespace(
                        choices=[SimpleNamespace(delta=delta, finish_reason=None)]
                    )
                usage = MagicMock(
                    prompt_tokens=final_input,
                    completion_tokens=final_output,
                    cache_read_input_tokens=0,
                    __iter__=lambda s: iter(
                        [("prompt_tokens", final_input), ("completion_tokens", final_output)]
                    ),
                )
                delta = SimpleNamespace(content="", tool_calls=None)
                return SimpleNamespace(
                    choices=[SimpleNamespace(delta=delta, finish_reason="stop")],
                    usage=usage,
                )

            async def aclose(self):
                pass

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=CumulativeUsageStream())
            chunks = []
            async for chunk in client._stream_chat_inner([{"role": "user", "content": "hi"}], None):
                chunks.append(chunk)

        expected_cost = estimate_cost(model, final_input, final_output)
        assert abs(client.total_cost_usd - expected_cost) < 1e-9
        assert client.total_input_tokens == final_input
        assert client.total_output_tokens == final_output

    @pytest.mark.asyncio
    async def test_no_final_usage_keeps_heuristic_cost(self) -> None:
        from godspeed.llm.cost import estimate_cost

        model = "claude-sonnet-4-20250514"
        client = LLMClient(model=model)

        class NoUsageStream:
            def __init__(self):
                self._idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx >= 3:
                    raise StopAsyncIteration
                self._idx += 1
                delta = SimpleNamespace(content=f"tok{self._idx}", tool_calls=None)
                return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)])

            async def aclose(self):
                pass

        with patch("godspeed.llm.client._get_litellm") as mock_litellm:
            mock_litellm.return_value.acompletion = AsyncMock(return_value=NoUsageStream())
            chunks = []
            async for chunk in client._stream_chat_inner([{"role": "user", "content": "hi"}], None):
                chunks.append(chunk)

        heuristic_per_chunk = estimate_cost(model, 0, 1)
        assert abs(client.total_cost_usd - 3 * heuristic_per_chunk) < 1e-9
        assert client.total_input_tokens == 0
        assert client.total_output_tokens == 0


# ---------------------------------------------------------------------------
# Test: with_model contextmanager + immutable routing
# ---------------------------------------------------------------------------


class TestWithModel:
    """with_model() temporarily overrides self.model and restores it."""

    def test_restores_model_after_block(self) -> None:
        client = LLMClient(model="main-model")
        with client.with_model("plan-model"):
            assert client.model == "plan-model"
            assert client._model_lower == "plan-model"
        assert client.model == "main-model"
        assert client._model_lower == "main-model"

    def test_restores_model_on_exception(self) -> None:
        client = LLMClient(model="main-model")
        with pytest.raises(RuntimeError, match="boom"):
            with client.with_model("plan-model"):
                raise RuntimeError("boom")
        assert client.model == "main-model"
        assert client._model_lower == "main-model"

    def test_same_model_is_noop(self) -> None:
        client = LLMClient(model="main-model")
        with client.with_model("main-model"):
            assert client.model == "main-model"
        assert client.model == "main-model"


class TestResolveModel:
    """_resolve_model() returns routed model without mutating shared state."""

    def test_returns_routed_model(self) -> None:
        router = ModelRouter({"plan": "claude-sonnet-4"})
        client = LLMClient(model="gpt-4o", router=router)
        model, lower = client._resolve_model("plan")
        assert model == "claude-sonnet-4"
        assert lower == "claude-sonnet-4"
        # No mutation.
        assert client.model == "gpt-4o"
        assert client._model_lower == "gpt-4o"

    def test_returns_default_when_unrouted(self) -> None:
        router = ModelRouter({"plan": "claude-sonnet-4"})
        client = LLMClient(model="gpt-4o", router=router)
        model, lower = client._resolve_model("chat")
        assert model == "gpt-4o"
        assert lower == "gpt-4o"


class TestConcurrentRouting:
    """Concurrent chat() calls with different task types must not corrupt
    each other's model selection (regression for the shared-state race)."""

    @pytest.mark.asyncio
    async def test_concurrent_chat_uses_correct_model_per_call(self) -> None:
        router = ModelRouter({"plan": "claude-sonnet-4", "edit": "gpt-4o"})
        client = LLMClient(model="ollama/qwen3:4b", router=router)

        seen: dict[str, str] = {}

        async def _capture(*args: object, **kwargs: object) -> ChatResponse:
            model = kwargs.get("_model")
            assert model is not None
            seen[model] = model
            # Simulate an await point where another coroutine could interleave.
            await asyncio.sleep(0)
            return ChatResponse(content="ok", finish_reason="stop")

        client._chat_with_fallback = AsyncMock(side_effect=_capture)

        await asyncio.gather(
            client.chat([{"role": "user", "content": "plan"}], task_type="plan"),
            client.chat([{"role": "user", "content": "edit"}], task_type="edit"),
            client.chat([{"role": "user", "content": "chat"}]),
        )

        # Each call resolved its own model; no cross-contamination.
        assert seen == {
            "claude-sonnet-4": "claude-sonnet-4",
            "gpt-4o": "gpt-4o",
            "ollama/qwen3:4b": "ollama/qwen3:4b",
        }
        # Shared state untouched.
        assert client.model == "ollama/qwen3:4b"
        assert client._model_lower == "ollama/qwen3:4b"

    @pytest.mark.asyncio
    async def test_concurrent_stream_uses_correct_model_per_call(self) -> None:
        router = ModelRouter({"plan": "claude-sonnet-4", "edit": "gpt-4o"})
        client = LLMClient(model="ollama/qwen3:4b", router=router)

        seen: dict[str, str] = {}

        async def _capture(self, msgs, tools=None, **kwargs):
            model = kwargs.get("_model")
            assert model is not None
            seen[model] = model
            await asyncio.sleep(0)
            yield ChatResponse(content="ok", finish_reason="stop")

        client._stream_chat_inner = _capture.__get__(client)

        async def _consume(task_type: str | None) -> None:
            async for _chunk in client.stream_chat(
                [{"role": "user", "content": "x"}], task_type=task_type
            ):
                pass

        await asyncio.gather(
            _consume("plan"),
            _consume("edit"),
            _consume(None),
        )

        assert seen == {
            "claude-sonnet-4": "claude-sonnet-4",
            "gpt-4o": "gpt-4o",
            "ollama/qwen3:4b": "ollama/qwen3:4b",
        }
        assert client.model == "ollama/qwen3:4b"


class TestPromptCachingBreakpoints:
    def test_max_four_breakpoints_long_conversation(self) -> None:
        messages = [{"role": "user", "content": f"m{i}"} for i in range(12)]
        out = LLMClient._apply_prompt_caching("claude-sonnet-4-5", messages)
        marked = [m for m in out if isinstance(m.get("content"), list)]
        assert len(marked) == 3
        assert isinstance(out[-1]["content"], str)

    def test_short_conversation_caches_stable_prefix_only(self) -> None:
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        out = LLMClient._apply_prompt_caching("claude-sonnet-4-5", messages)
        assert isinstance(out[0]["content"], list)
        assert isinstance(out[1]["content"], str)


class TestDeriveClient:
    def test_derive_pins_model_and_caps_budget_to_remaining(self) -> None:
        parent = LLMClient(model="main-model", max_cost_usd=1.0)
        parent.total_cost_usd = 0.4
        child = parent.derive("architect-model")
        assert child.model == "architect-model"
        assert child.total_cost_usd == 0.0
        assert child.max_cost_usd == 0.6

    def test_adopt_folds_delta_into_parent(self) -> None:
        parent = LLMClient(model="main")
        child = parent.derive("other")
        child.total_input_tokens = 50
        child.total_output_tokens = 20
        child.total_cost_usd = 0.05
        parent.adopt(child)
        assert parent.total_input_tokens == 50
        assert parent.total_output_tokens == 20
        assert parent.total_cost_usd == 0.05
