"""The agent-in-loop benchmark runner must hand the configured reasoning effort to the LLM client.

`experiments/swebench_lite/run_in_loop.py` built its `LLMClient` from `thinking_budget` only and
dropped `settings.reasoning_effort`. For a Qwen3.5+/3.8 model that means the client sends no
template kwargs, so the server's chat-template default (`xhigh` thinking) applies no matter what
the settings say -- the runner silently benchmarks a different configuration than the one requested.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

from godspeed.llm.client import LLMClient

_EXPERIMENTS_DIR = (Path(__file__).parent.parent / "experiments" / "swebench_lite").resolve()
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import run_in_loop  # noqa: E402


def test_reasoning_effort_reaches_llm_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    class _CapturingClient(LLMClient):
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            super().__init__(**kwargs)

    async def _fake_agent_loop(**_kwargs: Any) -> str:
        return "done"

    monkeypatch.setattr("godspeed.llm.client.LLMClient", _CapturingClient)
    monkeypatch.setattr("godspeed.agent.loop.agent_loop", _fake_agent_loop)
    # GodspeedSettings() reads the global settings.yaml in its constructor (env vars are only
    # applied by load_settings(), which this runner does not use).
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    (global_dir / "settings.yaml").write_text(
        f"reasoning_effort: medium\nglobal_dir: {global_dir.as_posix()}\n", encoding="utf-8"
    )
    monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", global_dir)

    payload = asyncio.run(
        run_in_loop._run_one_async(
            model="openai/qwen3.8-27b",
            prompt="fix the bug",
            project_dir=tmp_path,
            instance_id="acme__widget-1",
            split="dev",
            timeout_s=30,
            verify_workdir=tmp_path,
            max_iterations=1,
        )
    )

    assert payload["final_text"] == "done"
    assert captured["reasoning_effort"] == "medium"
    assert captured["model"] == "openai/qwen3.8-27b"
