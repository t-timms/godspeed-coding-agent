"""LLM client wrapping LiteLLM for unified model access."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import random
import re
import threading
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# Import estimate_cost at module level to avoid scoping issues in async methods
from godspeed.llm.cost import estimate_cost
from godspeed.llm.usage_ledger import UsageLedger

# Task-type contextvar for ledger attribution. Set around every public
# entry point (chat / stream_chat) so private call-chain methods can
# attribute recorded usage without threading new parameters.
_TASK_TYPE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "godspeed_llm_task_type", default=None
)

# Rate-limit retry policy
RATE_LIMIT_MAX_RETRIES = 4
RATE_LIMIT_BASE_DELAY = 1.0  # seconds — doubles each retry
RATE_LIMIT_MAX_DELAY = 60.0  # hard ceiling — past this, give up and fall over
RATE_LIMIT_JITTER = 0.25  # ±25% random jitter on each delay

_RATE_LIMIT_MARKERS = (
    "429",
    "rate_limit",
    "rate limit",
    "ratelimiterror",
    "too many requests",
    "quota",
)
_RETRY_AFTER_RE = re.compile(r"retry-?after[:\s]+(\d+)", re.IGNORECASE)

logger = logging.getLogger(__name__)

__all__ = ["BudgetExceededError", "ChatResponse", "LLMClient"]

# Lazy import — litellm pulls in 2000+ modules (~1.5s cold start).
# We defer it to first use so the TUI appears instantly.
_litellm = None
_litellm_lock = threading.Lock()

# DeepSeek API configuration
_DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
_DEEPSEEK_MODELS = {"deepseek-v4-pro", "deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner"}


def _get_litellm():
    """Import litellm on first use and cache it (thread-safe)."""
    global _litellm
    if _litellm is not None:
        return _litellm
    with _litellm_lock:
        if _litellm is None:
            import litellm

            litellm.suppress_debug_info = True
            _litellm = litellm
    return _litellm


class BudgetExceededError(RuntimeError):
    """Raised when session cost exceeds the configured budget."""

    def __init__(self, spent: float, limit: float) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(f"Budget exceeded: ${spent:.4f} / ${limit:.2f} limit")


@dataclass
class ChatResponse:
    """Parsed response from an LLM call."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = ""
    usage: dict[str, int] = field(default_factory=dict)
    thinking: str = ""  # Extended thinking content (Anthropic models)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class ModelRouter:
    """Routes LLM calls to different models based on task type.

    Config maps task types to model names. Unmatched task types
    fall back to the default model.
    """

    def __init__(self, routing: dict[str, str] | None = None) -> None:
        self._routing = routing or {}

    def route(self, default_model: str, task_type: str | None = None) -> str:
        """Select the model for a given task type.

        Args:
            default_model: The default model to use.
            task_type: Optional task hint (e.g., "plan", "edit", "chat").

        Returns:
            The model to use for this call.
        """
        if task_type and task_type in self._routing:
            routed = self._routing[task_type]
            logger.debug("Model routing task_type=%s model=%s", task_type, routed)
            return routed
        return default_model

    @property
    def has_routing(self) -> bool:
        return bool(self._routing)

    @property
    def routes(self) -> dict[str, str]:
        return dict(self._routing)


class LLMClient:
    """Unified LLM client via LiteLLM.

    Supports 200+ providers: Claude, GPT, Gemini, DeepSeek, Ollama, etc.
    Provides fallback chains, streaming, and token tracking.
    """

    def __init__(
        self,
        model: str,
        fallback_models: list[str] | None = None,
        timeout: int = 120,
        router: ModelRouter | None = None,
        thinking_budget: int = 0,
        max_cost_usd: float = 0.0,
        reasoning_effort: str = "",
        *,
        usage_ledger: UsageLedger | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        # Cache lowercased model name to avoid repeated .lower() calls
        self._model_lower = model.lower()
        self.fallback_models = fallback_models or []
        self.timeout = timeout
        self.router = router or ModelRouter()
        self.thinking_budget = thinking_budget
        self.max_cost_usd = max_cost_usd
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd: float = 0.0
        self.usage_ledger: UsageLedger = usage_ledger or UsageLedger()

    def _record_usage(
        self,
        *,
        task_type: str | None,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        """Record one completed call into the usage ledger (best effort)."""
        try:
            self.usage_ledger.record(
                task_type=task_type or "default",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )
        except Exception:
            logger.debug("Usage ledger recording failed", exc_info=True)

    def _resolve_model(self, task_type: str | None = None) -> tuple[str, str]:
        """Resolve which model to use for a call without mutating shared state.

        Args:
            task_type: Optional task hint for model routing.

        Returns:
            ``(model, model_lower)`` tuple — the routed model name and its
            pre-computed lowercase form.
        """
        routed = self.router.route(self.model, task_type)
        return routed, routed.lower()

    @contextmanager
    def with_model(self, model: str) -> Generator[None, None, None]:
        """Temporarily override ``self.model`` in an exception-safe block.

        Saves and restores ``self.model`` + ``self._model_lower``.  Use for
        synchronous-looking temporary swaps (e.g. the architect plan phase).
        **Not safe** across ``await`` points when multiple coroutines share
        the same client — use ``_resolve_model`` + parameter passing instead.
        """
        original = self.model
        original_lower = self._model_lower
        self.model = model
        self._model_lower = model.lower()
        try:
            yield
        finally:
            self.model = original
            self._model_lower = original_lower

    def derive(self, model: str) -> LLMClient:
        """Return a child client pinned to ``model`` sharing budget limits.

        Use for sub-phases that run a different model (e.g. architect
        planning) without mutating shared client state across ``await``
        points — mutating ``self.model`` mid-session also invalidates the
        provider prompt cache. The child starts with zeroed usage and a
        budget capped to the parent's *remaining* spend; call :meth:`adopt`
        afterwards to fold its delta into this client so session budgets
        stay accurate. The child shares the parent's usage ledger so its
        callbacks stay attributable in one place.
        """
        remaining_budget = (
            max(0.0, self.max_cost_usd - self.total_cost_usd) if self.max_cost_usd > 0 else 0.0
        )
        return LLMClient(
            model=model,
            fallback_models=self.fallback_models,
            timeout=self.timeout,
            router=self.router,
            thinking_budget=self.thinking_budget,
            max_cost_usd=remaining_budget,
            reasoning_effort=self.reasoning_effort,
            usage_ledger=self.usage_ledger,
        )

    def adopt(self, other: LLMClient) -> None:
        """Fold another client's usage delta into this client's totals."""
        self.total_input_tokens += other.total_input_tokens
        self.total_output_tokens += other.total_output_tokens
        self.total_cost_usd += other.total_cost_usd

    # Ollama models known to support native tool calling
    _TOOLS_CAPABLE_OLLAMA = (
        "qwen",
        "llama3",
        "mistral",
        "command-r",
        "firefunction",
        "hermes",
        "gemma",
    )

    def _supports_tool_calling(self, model_lower: str | None = None) -> bool:
        """Check if current model likely supports native function calling.

        Args:
            model_lower: Pre-lowercased model name. Falls back to
                ``self._model_lower`` when *None*.
        """
        name = model_lower or self._model_lower
        if name.startswith(("ollama/", "ollama_chat/")):
            model_name = name.split("/", 1)[-1].split(":")[0]
            return any(cap in model_name for cap in self._TOOLS_CAPABLE_OLLAMA)
        return True

    def _effective_model(
        self,
        model: str | None = None,
        model_lower: str | None = None,
    ) -> str:
        """Return the model string to use for API calls.

        Upgrades ``'ollama/'`` to ``'ollama_chat/'`` for tool-capable models,
        since LiteLLM's ``ollama_chat`` provider supports native tool calling
        while the plain ``ollama`` provider does not.

        Args:
            model: Override model name. Falls back to ``self.model``.
            model_lower: Pre-computed lowercase. Falls back to
                ``self._model_lower`` (or ``model.lower()`` when *model* is
                given but *model_lower* is not).
        """
        name = model or self.model
        name_lower = model_lower or name.lower()
        if name_lower.startswith("ollama/") and self._supports_tool_calling(name_lower):
            upgraded = "ollama_chat/" + name.split("/", 1)[1]
            logger.info("Upgrading model %s → %s for tool calling support", name, upgraded)
            return upgraded
        return name

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """Check if the error is a connection failure (server not running)."""
        exc_str = str(exc).lower()
        return any(
            marker in exc_str
            for marker in ("connection refused", "cannot connect", "connect call failed")
        )

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Check if the error is a 429 / rate limit / quota error.

        These are transient and should be retried with exponential backoff,
        not failed over immediately (falling over from rate-limited primary
        to a fallback often just rate-limits the fallback too).
        """
        exc_str = str(exc).lower()
        # "429" appears in many unrelated contexts (ports, request IDs) — only
        # count it when paired with a rate-limit word nearby.
        if "429" in exc_str and any(w in exc_str for w in ("too many", "rate", "quota", "throttl")):
            return True
        return any(marker in exc_str for marker in _RATE_LIMIT_MARKERS if marker != "429")

    @staticmethod
    def _parse_retry_after(error_message: str) -> float | None:
        """Extract a Retry-After hint (seconds) from the error message.

        Returns None when no hint is present. Clamps to RATE_LIMIT_MAX_DELAY
        so a misbehaving provider can't block the session for hours.
        """
        match = _RETRY_AFTER_RE.search(error_message)
        if match is None:
            return None
        try:
            hint = float(match.group(1))
        except ValueError:
            return None
        return min(hint, RATE_LIMIT_MAX_DELAY)

    @classmethod
    def _backoff_delay(cls, retry_index: int, retry_after: float | None) -> float:
        """Compute the sleep duration for the N-th retry (0-indexed).

        If the provider supplied Retry-After, treat it as a floor and add
        upward-only jitter (waiting *less* than the provider asked is
        counterproductive — we'd just trigger another 429).

        Otherwise use exponential backoff (base * 2^n) with ±25% jitter
        to break up thundering-herd retries across concurrent agents.
        Capped at RATE_LIMIT_MAX_DELAY.
        """
        if retry_after is not None:
            # Retry jitter for backoff — not a security context, so random.uniform is fine.
            jitter = 1.0 + random.uniform(0.0, RATE_LIMIT_JITTER)  # noqa: S311
            return min(retry_after * jitter, RATE_LIMIT_MAX_DELAY)
        base = min(RATE_LIMIT_BASE_DELAY * (2**retry_index), RATE_LIMIT_MAX_DELAY)
        jitter = 1.0 + random.uniform(-RATE_LIMIT_JITTER, RATE_LIMIT_JITTER)  # noqa: S311
        return min(max(base * jitter, 0.0), RATE_LIMIT_MAX_DELAY)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        task_type: str | None = None,
    ) -> ChatResponse:
        """Send messages to the LLM and return a parsed response.

        Uses LiteLLM's async completion with automatic provider routing.
        Falls back to alternate models on failure. Skips retries for
        connection-refused errors (e.g. Ollama not running).

        Args:
            messages: Conversation messages.
            tools: Tool schemas for function calling.
            task_type: Optional task hint for model routing
                (e.g., "plan", "edit", "chat").
        """
        # Resolve model immutably — no shared-state mutation.
        effective_model, effective_lower = self._resolve_model(task_type)
        token = _TASK_TYPE.set(task_type)
        try:
            return await self._chat_with_fallback(
                messages,
                tools,
                _model=effective_model,
                _model_lower=effective_lower,
            )
        finally:
            _TASK_TYPE.reset(token)

    async def _chat_with_fallback(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        _model: str | None = None,
        _model_lower: str | None = None,
    ) -> ChatResponse:
        """Internal: send messages with fallback chain.

        Args:
            _model: Resolved model name (avoids reading shared ``self.model``).
            _model_lower: Pre-computed lowercase of *_model*.

        Classification of failures:
        - Connection errors: server is down; skip retry, try next fallback.
        - Rate-limit / 429 / quota: retry the SAME model with exponential
          backoff + jitter up to RATE_LIMIT_MAX_RETRIES, honoring
          Retry-After when provided. Falling over to a fallback on
          rate-limit often just rate-limits the fallback too.
        - Other errors: one short retry on the primary model, then fall
          over to the next model in the chain.
        """
        model = _model or self.model
        model_lower = _model_lower or model.lower()
        models_to_try = [self._effective_model(model, model_lower), *self.fallback_models]

        last_error: Exception | None = None
        for idx, m in enumerate(models_to_try):
            try:
                return await self._call(m, messages, tools)
            except Exception as exc:
                logger.warning("LLM call failed model=%s error=%s", m, exc)
                last_error = exc

                if self._is_connection_error(exc):
                    # Server down — retrying is pointless, try next fallback.
                    continue

                if self._is_rate_limit_error(exc):
                    # Retry same model with exponential backoff + jitter.
                    recovered = await self._retry_on_rate_limit(m, messages, tools, exc)
                    if recovered is not None:
                        return recovered
                    # Exhausted rate-limit retries; move on to the next model.
                    last_error = exc
                    continue

                # Retry primary model once after short delay before trying fallbacks
                if idx == 0:
                    await asyncio.sleep(1)
                    try:
                        return await self._call(m, messages, tools)
                    except Exception as retry_exc:
                        logger.warning(
                            "Primary model retry failed model=%s error=%s",
                            m,
                            retry_exc,
                        )
                        last_error = retry_exc

        raise self._build_failure_error(
            last_error,
            model=model,
            model_lower=model_lower,
        )

    async def _retry_on_rate_limit(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        first_exc: Exception,
    ) -> ChatResponse | None:
        """Retry a rate-limited call with exponential backoff + jitter.

        Returns the successful ChatResponse, or None if retries are exhausted.
        Caller decides what to do with a None result (typically fall over to
        the next model in the chain).
        """
        current_exc: Exception = first_exc
        for attempt in range(RATE_LIMIT_MAX_RETRIES):
            retry_after = self._parse_retry_after(str(current_exc))
            delay = self._backoff_delay(attempt, retry_after)
            logger.warning(
                "Rate limit model=%s attempt=%d/%d delay=%.2fs retry_after=%s",
                model,
                attempt + 1,
                RATE_LIMIT_MAX_RETRIES,
                delay,
                retry_after,
            )
            await asyncio.sleep(delay)
            try:
                return await self._call(model, messages, tools)
            except Exception as exc:
                if not self._is_rate_limit_error(exc):
                    # Morphed into a different kind of failure — let the
                    # outer loop handle it (fall over, build failure, etc.).
                    raise
                current_exc = exc
        logger.warning(
            "Rate limit retries exhausted model=%s after %d attempts",
            model,
            RATE_LIMIT_MAX_RETRIES,
        )
        return None

    def _build_failure_error(
        self,
        last_error: Exception | None,
        *,
        model: str | None = None,
        model_lower: str | None = None,
    ) -> RuntimeError:
        """Build an actionable error message based on the failure type.

        Args:
            model: Model name for the error message (falls back to self.model).
            model_lower: Pre-lowercased model name (falls back to self._model_lower).
        """
        name_lower = model_lower or self._model_lower
        name = model or self.model
        if last_error and self._is_connection_error(last_error):
            if name_lower.startswith("ollama"):
                return RuntimeError(
                    "Ollama is not running. Fix with one of:\n"
                    "  1. Start Ollama:  ollama serve\n"
                    "  2. Use a cloud model:  godspeed -m claude-sonnet-4-20250514\n"
                    "  3. Set a fallback in ~/.godspeed/settings.yaml"
                )
            if name_lower.startswith(("llamacpp/", "openai/")):
                return RuntimeError(
                    "Local llama.cpp server is not running. Fix with one of:\n"
                    "  1. Start server:  python scripts/setup_qwen36_local.py\n"
                    "  2. Use a cloud model:  godspeed -m nvidia_nim/qwen/qwen3.5-397b-a17b\n"
                    "  3. Set a fallback in ~/.godspeed/settings.yaml"
                )
            return RuntimeError(
                f"Cannot connect to LLM provider for model '{name}'. "
                "Check that the server is running and the model name is correct."
            )
        return RuntimeError(f"All models failed. Last error: {last_error}")

    # Providers that support anthropic-style cache_control — use frozenset for O(1) lookup
    _ANTHROPIC_CACHING_PREFIXES: frozenset[str] = frozenset({"claude", "anthropic", "deepseek"})

    @classmethod
    def _supports_prompt_caching(cls, model: str) -> bool:
        """Check if model supports prompt caching via cache_control markers."""
        model_lower = model.lower()
        return any(prefix in model_lower for prefix in cls._ANTHROPIC_CACHING_PREFIXES)

    @staticmethod
    def _apply_prompt_caching(
        model: str,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply prompt caching markers for Anthropic-compatible providers.

        Uses the ``system_and_3`` strategy: at most 3 message breakpoints on
        the last stable messages, never on the newest message (which changes
        every turn). This leaves room for the system-prompt breakpoint within
        Anthropic's hard cap of 4 cache_control blocks per request. OpenAI
        handles caching automatically so we skip it there.
        """
        model_lower = model.lower()
        prefixes = LLMClient._ANTHROPIC_CACHING_PREFIXES
        supports_caching = any(prefix in model_lower for prefix in prefixes)
        if not supports_caching:
            return messages

        max_message_breakpoints = 3
        last_stable_idx = len(messages) - 1
        first_idx = max(0, last_stable_idx - max_message_breakpoints)
        if first_idx >= last_stable_idx:
            return messages

        cached = []
        for i, msg in enumerate(messages):
            if first_idx <= i < last_stable_idx:
                raw_content = msg.get("content", "")
                if isinstance(raw_content, str) and raw_content:
                    cached.append(
                        {
                            "role": msg["role"],
                            "content": [
                                {
                                    "type": "text",
                                    "text": raw_content,
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        }
                    )
                    continue
            cached.append(msg)
        return cached

    # Anthropic model prefixes for fast matching
    _ANTHROPIC_PREFIXES: frozenset[str] = frozenset({"claude", "anthropic"})

    # Models that support Qwen-style thinking via extra_body
    _THINKING_CAPABLE_PREFIXES: frozenset[str] = frozenset({"qwen3.6", "qwen3-"})

    def _is_anthropic_model(self, model: str | None = None) -> bool:
        """Check if the model is an Anthropic/Claude model."""
        name = (model or self._model_lower).lower() if model else self._model_lower
        return any(prefix in name for prefix in self._ANTHROPIC_PREFIXES)

    def _supports_thinking(self, model: str | None = None) -> bool:
        """Check if the model supports extended thinking mode."""
        name = (model or self._model_lower).lower() if model else self._model_lower
        return any(prefix in name for prefix in self._THINKING_CAPABLE_PREFIXES)

    def _check_budget(self) -> None:
        """Raise BudgetExceededError if session cost exceeds the limit."""
        if self.max_cost_usd > 0 and self.total_cost_usd > self.max_cost_usd:
            raise BudgetExceededError(self.total_cost_usd, self.max_cost_usd)

    async def _call_deepseek_direct(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> ChatResponse:
        """Make a direct API call to DeepSeek (bypassing LiteLLM).

        LiteLLM's deepseek provider has authentication issues ("Authentication Fails (governor)").
        This method calls the DeepSeek API directly using OpenAI library.

        DeepSeek V4 requires reasoning_content to be passed back in multi-turn conversations.
        We handle this by disabling thinking mode to simplify the interaction.
        """
        # Clean model name (remove provider prefix if present)
        clean_model = model.replace("deepseek/", "").strip()

        # DeepSeek V4: Use OpenAI library directly (handles formatting automatically)
        if clean_model in _DEEPSEEK_MODELS:
            import os

            from openai import OpenAI

            # Get API key
            api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPEK_API_KEY", "")
            if not api_key:
                raise RuntimeError(
                    "DEEPSEEK_API_KEY or DEEPEK_API_KEY environment variable not set. "
                    "DeepSeek direct API call requires this key."
                )

            # Create OpenAI client configured for DeepSeek API
            client = OpenAI(api_key=api_key, base_url=_DEEPSEEK_API_BASE)

            # Build kwargs for API call
            # DeepSeek V4 requires clean message ordering - strip reasoning_content
            # from assistant messages to avoid API rejection on multi-turn
            cleaned_messages = []
            for msg in messages:
                if msg.get("role") == "assistant" and "reasoning_content" in msg:
                    # Create clean copy without reasoning_content for API
                    clean_msg = {k: v for k, v in msg.items() if k != "reasoning_content"}
                    # Ensure content field exists (even if empty) for tool_calls messages
                    if "tool_calls" in clean_msg and "content" not in clean_msg:
                        clean_msg["content"] = ""
                    cleaned_messages.append(clean_msg)
                else:
                    cleaned_messages.append(msg)

            # DeepSeek V4 strictness: tool messages must immediately follow
            # an assistant message with tool_calls. Standard OpenAI format allows:
            #   assistant(tool_calls=[A,B]) -> tool(A) -> tool(B)
            # DeepSeek V4 requires:
            #   assistant(tool_calls=[A]) -> tool(A) -> assistant(tool_calls=[B]) -> tool(B)
            if "deepseek" in model.lower():
                restructured = []
                i = 0
                while i < len(cleaned_messages):
                    msg = cleaned_messages[i]
                    # Restructure ANY assistant with tool_calls (not just >1)
                    if (
                        msg.get("role") == "assistant"
                        and "tool_calls" in msg
                        and len(msg.get("tool_calls", [])) >= 1
                    ):
                        tool_calls = msg["tool_calls"]
                        content = msg.get("content", "")
                        # Collect all following tool messages
                        tool_results = {}
                        j = i + 1
                        while (
                            j < len(cleaned_messages) and cleaned_messages[j].get("role") == "tool"
                        ):
                            tc_id = cleaned_messages[j].get("tool_call_id", "")
                            tool_results[tc_id] = cleaned_messages[j]
                            j += 1
                        # Emit assistant+tool pair for each tool_call
                        for k, tc in enumerate(tool_calls):
                            a_msg: dict[str, Any] = {"role": "assistant"}
                            a_msg["content"] = content if k == 0 else ""
                            a_msg["tool_calls"] = [tc]
                            restructured.append(a_msg)
                            tc_id = tc.get("id", "")
                            if tc_id in tool_results:
                                restructured.append(tool_results[tc_id])
                        # Skip past original assistant + all its tool messages
                        i = j
                    else:
                        restructured.append(msg)
                        i += 1
                cleaned_messages = restructured
                logger.info(
                    "[DeepSeek] After restructuring: %s",
                    [m.get("role") for m in cleaned_messages[-12:]],
                )
                # Log restructured message roles for debugging
                _msg_roles = [m.get("role", "?") for m in cleaned_messages]
                logger.info("[DeepSeek] Restructured messages roles: %s", _msg_roles[-12:])

            # Debug: log message roles to check ordering
            _msg_roles = [m.get("role", "?") for m in cleaned_messages]
            logger.info("[DeepSeek] Sending messages roles: %s", _msg_roles[-10:])

            # Validate tool message ordering for DeepSeek API strictness
            for i, msg in enumerate(cleaned_messages):
                if msg.get("role") == "tool":
                    if i == 0:
                        logger.error("[DeepSeek] tool message at index 0 with no preceding message")
                    else:
                        prev = cleaned_messages[i - 1]
                        if prev.get("role") != "assistant" or "tool_calls" not in prev:
                            logger.error(
                                "[DeepSeek] tool message at index %d not preceded"
                                " by assistant+tool_calls. "
                                "Preceding: role=%s has_tool_calls=%s",
                                i,
                                prev.get("role"),
                                "tool_calls" in prev,
                            )

            # Apply prompt caching (Anthropic-compatible cache_control)
            # for ~75% input cost reduction. DeepSeek supports cache_control:
            # mark all messages except the last 2 as ephemeral cache.
            num_to_cache = max(0, len(cleaned_messages) - 2)
            for i in range(num_to_cache):
                content = cleaned_messages[i].get("content", "")
                if isinstance(content, str) and content:
                    cleaned_messages[i]["content"] = [
                        {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                    ]

            kwargs = {
                "model": clean_model,
                "messages": cleaned_messages,
                "max_tokens": 4096,
                # Disable thinking to avoid message ordering issues
                "extra_body": {"thinking": {"type": "disabled"}},
            }

            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            # Make API call using OpenAI library (handles formatting automatically)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(**kwargs),  # type: ignore[call-overload]
            )

            # Parse response
            choice = response.choices[0]
            message = choice.message

            # Extract content
            content_text = message.content or ""

            # Extract reasoning_content (DeepSeek V4 thinking/reasoning)
            reasoning_content = getattr(message, "reasoning_content", "") or ""

            # Extract tool calls
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )

            # Track usage and cost
            input_tokens = 0
            output_tokens = 0
            if response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens

            call_cost = estimate_cost(model, input_tokens, output_tokens)
            self.total_cost_usd += call_cost
            self._check_budget()
            self._record_usage(
                task_type=_TASK_TYPE.get(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=call_cost,
            )

            # Return response
            response_obj = ChatResponse(
                content=content_text,
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason or "",
                thinking=reasoning_content,  # Store in thinking field
                usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )

            return response_obj

        # Fallback: model not in DeepSeek set (should not happen due to caller guard)
        raise RuntimeError(  # pragma: no cover
            f"DeepSeek direct call not supported for model: {clean_model}"
        )

    async def _call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> ChatResponse:
        """Make a single LLM API call."""
        # Route DeepSeek models to direct API (bypasses LiteLLM's broken provider).
        # Only intercept bare DeepSeek names or deepseek/ provider prefix; models
        # served by other providers (nvidia_nim/, ollama/, etc.) go to LiteLLM.
        clean_model = model.replace("deepseek/", "").strip()
        if clean_model in _DEEPSEEK_MODELS or model.lower().startswith("deepseek/"):
            return await self._call_deepseek_direct(model, messages, tools)

        # Apply prompt caching for supported providers
        cached_messages = self._apply_prompt_caching(model, messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": cached_messages,
            "timeout": self.timeout,
        }
        if tools:
            if self._supports_tool_calling(model.lower()):
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            else:
                logger.info(
                    "Model %s may not support native tool calling; using text mode",
                    model,
                )

        # Extended thinking for Anthropic models and Qwen thinking mode
        if self.thinking_budget > 0:
            if self._is_anthropic_model(model):
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
            elif self._supports_thinking(model):
                # Qwen3.6 via llama.cpp OpenAI-compatible server uses extra_body
                kwargs["extra_body"] = {
                    "thinking": True,
                    "thinking_budget": self.thinking_budget,
                }

        if self.reasoning_effort:
            if "extra_body" in kwargs:
                kwargs["extra_body"]["reasoning_effort"] = self.reasoning_effort
            else:
                kwargs["reasoning_effort"] = self.reasoning_effort

        response = await _get_litellm().acompletion(**kwargs)

        # Parse response — guard against empty choices list
        if not response.choices:
            logger.warning("LLM returned empty choices list model=%s", model)
            return ChatResponse(content="", tool_calls=[], finish_reason="stop", usage={})

        choice = response.choices[0]
        message = choice.message

        # Extract thinking content from Anthropic responses
        thinking_text = ""
        if hasattr(message, "thinking") and message.thinking:
            thinking_text = message.thinking
        # Also check content blocks for thinking type
        if not thinking_text and hasattr(message, "content") and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking_text = block.get("thinking", "")
                    break

        # Extract text content (may be in content blocks)
        content_text = ""
        if isinstance(message.content, str):
            content_text = message.content
        elif isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    content_text += block.get("text", "")

        # Extract tool calls
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        # Qwen3-Coder-family models emit <function=...>...</function> XML that
        # Ollama's built-in parser (as of 0.20.x) doesn't extract — the call
        # ends up in the content field instead of tool_calls. When we see the
        # fingerprint and no structured tool_calls, parse and synthesize.
        if not tool_calls and content_text:
            from godspeed.llm.qwen3_coder_parser import extract_qwen3_coder_tool_calls

            parsed = extract_qwen3_coder_tool_calls(content_text)
            if parsed:
                tool_calls = parsed
                content_text = ""

        # ZAYA1-8B emits <zyphra_tool_call>{"name":"...","arguments":{...}}</zyphra_tool_call>
        # XML blocks. The vLLM zaya_xml parser extracts these server-side, but
        # when running locally via transformers or Ollama-style endpoints that
        # don't recognise the format, the tool calls end up in content.
        if not tool_calls and content_text:
            from godspeed.llm.zaya_xml_parser import extract_zaya_tool_calls

            parsed = extract_zaya_tool_calls(content_text)
            if parsed:
                tool_calls = parsed
                content_text = ""

        # Many local models (Qwen2.5-Coder, DeepSeek via Ollama) wrap tool
        # calls in markdown JSON blocks instead of native tool_calls arrays.
        # Parse those when no structured calls were found.
        if not tool_calls and content_text:
            from godspeed.llm.json_markdown_parser import extract_json_markdown_tool_calls

            parsed = extract_json_markdown_tool_calls(content_text)
            if parsed:
                tool_calls = parsed
                content_text = ""

        # Track usage and cost
        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

        call_cost = estimate_cost(model, input_tokens, output_tokens)
        self.total_cost_usd += call_cost
        self._record_usage(
            task_type=_TASK_TYPE.get(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=call_cost,
        )

        # Check budget after tracking
        self._check_budget()

        return ChatResponse(
            content=content_text or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            thinking=thinking_text,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            if response.usage
            else {},
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        task_type: str | None = None,
    ) -> AsyncGenerator[ChatResponse, None]:
        """Stream LLM response chunks with fallback to batch on failure.

        Yields ChatResponse objects as they arrive. On stream failure,
        retries once, then falls back to batch chat() with full retry
        logic. The final response has finish_reason set.
        """
        effective_model, _effective_lower = self._resolve_model(task_type)
        token = _TASK_TYPE.set(task_type)
        try:
            try:
                async for chunk in self._stream_chat_inner(
                    messages,
                    tools,
                    _model=effective_model,
                ):
                    yield chunk
                return
            except Exception:
                logger.warning("Streaming call failed, retrying once", exc_info=True)

            # Retry streaming once
            try:
                async for chunk in self._stream_chat_inner(
                    messages,
                    tools,
                    _model=effective_model,
                ):
                    yield chunk
                return
            except Exception:
                logger.warning("Streaming retry failed, falling back to batch", exc_info=True)

            # Fall back to batch with full retry/fallback chain; chat()
            # re-scopes the contextvar itself, so attribution stays correct.
            response = await self.chat(messages=messages, tools=tools, task_type=task_type)
            yield response
        finally:
            _TASK_TYPE.reset(token)

    async def _stream_chat_inner(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        _model: str | None = None,
    ) -> AsyncGenerator[ChatResponse, None]:
        """Inner streaming body — uses the resolved model passed via *_model*."""
        effective = self._effective_model(_model)
        effective_lower = effective.lower()

        # Apply prompt caching for supported providers (same as _call())
        cached_messages = self._apply_prompt_caching(effective, messages)

        kwargs: dict[str, Any] = {
            "model": effective,
            "messages": cached_messages,
            "stream": True,
            "timeout": self.timeout,
        }
        if tools:
            if self._supports_tool_calling(effective_lower):
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            else:
                logger.info(
                    "Model %s may not support native tool calling; using text mode",
                    effective,
                )

        # Extended thinking for Anthropic models and Qwen thinking mode
        if self.thinking_budget > 0:
            if self._is_anthropic_model(effective):
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
            elif self._supports_thinking(effective):
                kwargs["extra_body"] = {
                    "thinking": True,
                    "thinking_budget": self.thinking_budget,
                }

        if self.reasoning_effort:
            if "extra_body" in kwargs:
                kwargs["extra_body"]["reasoning_effort"] = self.reasoning_effort
            else:
                kwargs["reasoning_effort"] = self.reasoning_effort

        _running_input_tokens = 0
        _running_output_tokens = 0
        _streaming_heuristic_cost = 0.0
        _ledger_recorded = False
        try:
            response = await _get_litellm().acompletion(**kwargs)
            _content_chunks: list[str] = []
            collected_tool_calls: list[dict[str, Any]] = []

            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                if delta.content:
                    _content_chunks.append(delta.content)
                    _running_output_tokens += 1  # rough per-chunk heuristic
                    chunk_cost = estimate_cost(effective, 0, 1)
                    _streaming_heuristic_cost += chunk_cost
                    self.total_cost_usd += chunk_cost
                    self._check_budget()
                    yield ChatResponse(
                        content=delta.content,
                        tool_calls=[],
                        finish_reason=None,
                        usage={},
                    )

                # Collect tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        while len(collected_tool_calls) <= idx:
                            collected_tool_calls.append(
                                {"id": "", "function": {"name": "", "arguments": ""}}
                            )
                        if tc_delta.id:
                            collected_tool_calls[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                collected_tool_calls[idx]["function"]["name"] = (
                                    tc_delta.function.name
                                )
                            if tc_delta.function.arguments:
                                collected_tool_calls[idx]["function"]["arguments"] += (
                                    tc_delta.function.arguments
                                )

                if finish_reason:
                    # Final chunk — yield complete response
                    final_tool_calls = []
                    for tc in collected_tool_calls:
                        final_tool_calls.append(
                            {
                                "id": tc["id"],
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"],
                                },
                            }
                        )

                    chunk_usage: dict[str, int] = {}
                    if hasattr(chunk, "usage") and chunk.usage:
                        chunk_usage = dict(chunk.usage)
                        _running_input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                        _running_output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
                        cached_tokens = getattr(chunk.usage, "cache_read_input_tokens", 0) or 0
                        # Replace heuristic with authoritative: subtract interim, add real
                        self.total_cost_usd -= _streaming_heuristic_cost
                        self.total_input_tokens += _running_input_tokens
                        self.total_output_tokens += _running_output_tokens
                        call_cost = estimate_cost(
                            effective,
                            _running_input_tokens,
                            _running_output_tokens,
                            cached_input_tokens=cached_tokens,
                        )
                        self.total_cost_usd += call_cost
                        self._check_budget()
                        self._record_usage(
                            task_type=_TASK_TYPE.get(),
                            input_tokens=_running_input_tokens,
                            output_tokens=_running_output_tokens,
                            cost_usd=call_cost,
                        )
                        _ledger_recorded = True

                    collected_content = "".join(_content_chunks)

                    # Local models (Ollama) may embed tool calls in markdown
                    # JSON blocks rather than native tool_calls. Parse them
                    # when no structured calls were collected.
                    if not final_tool_calls and collected_content:
                        from godspeed.llm.json_markdown_parser import (
                            extract_json_markdown_tool_calls,
                        )

                        parsed = extract_json_markdown_tool_calls(collected_content)
                        if parsed:
                            final_tool_calls = parsed
                            collected_content = ""
                    yield ChatResponse(
                        content=collected_content,
                        tool_calls=final_tool_calls,
                        finish_reason=finish_reason,
                        usage=chunk_usage,
                    )
                    return

            # Stream ended without finish_reason — return collected content
            collected_content = "".join(_content_chunks)
            if collected_content or collected_tool_calls:
                yield ChatResponse(
                    content=collected_content,
                    tool_calls=[],
                    finish_reason="incomplete",
                    usage={},
                )
            if not _ledger_recorded:
                self._record_usage(
                    task_type=_TASK_TYPE.get(),
                    input_tokens=0,
                    output_tokens=_running_output_tokens,
                    cost_usd=_streaming_heuristic_cost,
                )

        except Exception as exc:
            logger.error("Streaming LLM call failed: %s", exc, exc_info=True)
            if not _ledger_recorded:
                self._record_usage(
                    task_type=_TASK_TYPE.get(),
                    input_tokens=_running_input_tokens,
                    output_tokens=_running_output_tokens,
                    cost_usd=_streaming_heuristic_cost,
                )
            raise
