"""A provider rejecting one optional request parameter must not end the session.

Seen with `openai/gpt-oss-20b` behind LiteLLM and `reasoning_effort: medium` in settings:
`UnsupportedParamsError: openai does not support parameters: ['reasoning_effort']` on every call, so
each of 8 benchmark tasks ended with `llm_error` after zero iterations, before any token was generated.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from godspeed.llm.client import LLMClient


class UnsupportedParamsError(Exception):
    """Same class name LiteLLM uses; matched by name so mocked LiteLLM works too."""


def _ok() -> SimpleNamespace:
    msg = SimpleNamespace(content="hi", tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice], usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2)
    )


def _err(*names: str) -> UnsupportedParamsError:
    listed = ", ".join(repr(n) for n in names)
    return UnsupportedParamsError(f"openai does not support parameters: [{listed}], for model=x")


class TestRejectedParams:
    def test_parses_listed_names_present_in_kwargs(self) -> None:
        kwargs = {"model": "m", "messages": [], "reasoning_effort": "medium", "top_k": 5}
        got = LLMClient._rejected_params(_err("reasoning_effort", "absent"), kwargs)
        assert got == {"reasoning_effort"}

    def test_essential_keys_are_never_dropped(self) -> None:
        kwargs = {"model": "m", "messages": [], "tools": [{}], "tool_choice": "auto"}
        assert LLMClient._rejected_params(_err("tools", "tool_choice", "model"), kwargs) == set()

    def test_other_exception_types_are_ignored(self) -> None:
        kwargs = {"model": "m", "reasoning_effort": "medium"}
        assert LLMClient._rejected_params(ValueError("['reasoning_effort']"), kwargs) == set()

    def test_message_without_a_list_is_ignored(self) -> None:
        assert LLMClient._rejected_params(UnsupportedParamsError("nope"), {"a": 1}) == set()


class TestAcompletion:
    @pytest.mark.asyncio
    async def test_retries_once_without_the_rejected_param(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = LLMClient(model="openai/gpt-oss-20b", reasoning_effort="medium")
        seen: list[dict] = []

        async def fake(**kwargs: object) -> SimpleNamespace:
            seen.append(dict(kwargs))
            if "reasoning_effort" in kwargs:
                raise _err("reasoning_effort")
            return _ok()

        with patch("godspeed.llm.client._get_litellm") as lit, caplog.at_level(logging.WARNING):
            lit.return_value.acompletion = AsyncMock(side_effect=fake)
            r1 = await client.chat([{"role": "user", "content": "x"}])
            r2 = await client.chat([{"role": "user", "content": "y"}])

        assert r1.content == "hi" and r2.content == "hi"
        # First chat: rejected call + retry. Second chat: parameter already left out -> one call.
        assert len(seen) == 3
        assert "reasoning_effort" in seen[0]
        assert "reasoning_effort" not in seen[1]
        assert "reasoning_effort" not in seen[2]
        warnings = [r for r in caplog.records if "rejected optional parameter" in r.getMessage()]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_unrelated_errors_propagate_unchanged(self) -> None:
        client = LLMClient(model="openai/gpt-oss-20b", reasoning_effort="medium")
        with patch("godspeed.llm.client._get_litellm") as lit:
            lit.return_value.acompletion = AsyncMock(side_effect=RuntimeError("boom"))
            with pytest.raises(Exception, match="boom"):
                await client.chat([{"role": "user", "content": "x"}])

    @pytest.mark.asyncio
    async def test_a_second_rejection_after_the_retry_is_not_swallowed(self) -> None:
        client = LLMClient(model="openai/gpt-oss-20b", reasoning_effort="medium")
        with patch("godspeed.llm.client._get_litellm") as lit:
            lit.return_value.acompletion = AsyncMock(side_effect=_err("reasoning_effort"))
            with pytest.raises(Exception, match="reasoning_effort"):
                await client.chat([{"role": "user", "content": "x"}])
