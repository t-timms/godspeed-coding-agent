"""Tests for godspeed.llm.token_counter.count_message_tokens.

The estimator drives context compaction. It used to skip nested payloads, so the ``arguments`` of
assistant ``tool_calls`` were never counted; on a local server with a hard 32K window a tool-heavy
session (dozens of long shell commands) then reached the limit before the 80% compaction threshold
fired, the reply was truncated mid tool call, and the server answered HTTP 500.
"""

from __future__ import annotations

from godspeed.llm.token_counter import (
    IMAGE_BLOCK_TOKEN_ESTIMATE,
    count_message_tokens,
    count_tokens,
)

MODEL = "openai/qwen3.8-27b"


def _tool_call(arguments: str, name: str = "shell") -> dict:
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}


def test_tool_call_arguments_are_counted() -> None:
    long_args = '{"command": "' + "python -m pytest tests/ -x -q --tb=short " * 40 + '"}'
    bare = [{"role": "assistant", "content": ""}]
    with_call = [{"role": "assistant", "content": "", "tool_calls": [_tool_call(long_args)]}]
    delta = count_message_tokens(with_call, MODEL) - count_message_tokens(bare, MODEL)
    assert delta >= count_tokens(long_args, MODEL)


def test_tool_call_name_is_counted() -> None:
    short = [{"role": "assistant", "content": "", "tool_calls": [_tool_call("{}", name="a")]}]
    long_name = "a_very_long_tool_name_" * 10
    longer = [
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("{}", name=long_name)]}
    ]
    assert count_message_tokens(longer, MODEL) > count_message_tokens(short, MODEL)


def test_plain_string_content_unchanged() -> None:
    msgs = [{"role": "user", "content": "hello world"}]
    expected = count_tokens("user", MODEL) + count_tokens("hello world", MODEL) + 2 + 4
    assert count_message_tokens(msgs, MODEL) == expected


def test_content_blocks_still_counted() -> None:
    msgs = [{"role": "user", "content": [{"type": "text", "text": "some block text"}]}]
    assert count_message_tokens(msgs, MODEL) > count_message_tokens([{"role": "user"}], MODEL)


def test_image_blocks_get_flat_estimate_and_url_is_not_encoded() -> None:
    url = "data:image/png;base64," + "A" * 5000
    msgs = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": url}}],
        }
    ]
    total = count_message_tokens(msgs, MODEL)
    assert total >= IMAGE_BLOCK_TOKEN_ESTIMATE
    assert total < IMAGE_BLOCK_TOKEN_ESTIMATE + 100  # the 5 KB base64 was not tokenized
