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


def _run_with_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra_yaml: str, stop_with: str = "done"
) -> Path:
    """Run one in-loop session with a stub agent loop; return the global settings dir."""

    async def _fake_agent_loop(**_kwargs: Any) -> str:
        return stop_with

    monkeypatch.setattr("godspeed.agent.loop.agent_loop", _fake_agent_loop)
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    (global_dir / "settings.yaml").write_text(
        f"global_dir: {global_dir.as_posix()}\n{extra_yaml}", encoding="utf-8"
    )
    monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", global_dir)
    asyncio.run(
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
    return global_dir


def test_conversation_is_logged_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Benchmark runs produce the trajectories a tuning corpus needs; they must not be dropped."""
    import json

    global_dir = _run_with_settings(monkeypatch, tmp_path, "log_conversations: true\n")
    files = list((global_dir / "training").glob("*.conversation.jsonl"))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    assert records[0]["role"] == "system"
    end = [r for r in records if r.get("role") == "session_end" or "exit_reason" in r]
    assert end, records
    assert end[-1]["exit_reason"] == "stopped"
    assert end[-1]["exit_code"] == 0


def test_no_conversation_log_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    global_dir = _run_with_settings(monkeypatch, tmp_path, "log_conversations: false\n")
    assert not list((global_dir / "training").glob("*.conversation.jsonl"))
