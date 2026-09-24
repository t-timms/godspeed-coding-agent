"""Tests for the ``llamacpp:`` settings section and its CLI wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from godspeed.cli import _ensure_llamacpp
from godspeed.config import _KNOWN_TOP_LEVEL_KEYS, GodspeedSettings, LlamaCppSettings
from godspeed.tools.llamacpp_manager import start_kwargs_from_settings

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def test_qwen38_example_profile_parses_into_valid_settings() -> None:
    """The shipped example profile must load, use only known keys, and map to launch args."""
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(
        (SCRIPTS_DIR / "settings_local_llm_qwen38_27b.yaml").read_text(encoding="utf-8")
    )
    assert set(data) <= _KNOWN_TOP_LEVEL_KEYS
    settings = GodspeedSettings(**data)
    assert settings.llamacpp.spec_type == "draft-mtp"
    assert settings.llamacpp.no_kv_offload is False
    assert settings.reasoning_effort == "medium"
    kwargs = start_kwargs_from_settings(settings.llamacpp)
    assert kwargs["model_path"].name == "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
    assert "~" not in str(kwargs["model_path"])  # expanduser applied


class TestLlamaCppSettingsSchema:
    def test_defaults_preserve_original_launch_behavior(self) -> None:
        cfg = LlamaCppSettings()
        assert cfg.server_bin == ""
        assert cfg.model_path == ""
        assert cfg.context == 0
        assert cfg.no_kv_offload is True
        assert cfg.kv_cache_type == "q8_0"
        assert cfg.spec_type == ""
        assert cfg.spec_draft_n_max == 2
        assert cfg.n_cpu_moe == 0
        assert cfg.reasoning_budget is None
        assert cfg.extra_args == []

    def test_is_a_top_level_setting_with_defaults(self) -> None:
        settings = GodspeedSettings()
        assert isinstance(settings.llamacpp, LlamaCppSettings)
        assert settings.llamacpp == LlamaCppSettings()

    def test_key_is_registered_so_yaml_does_not_warn_unknown(self) -> None:
        assert "llamacpp" in _KNOWN_TOP_LEVEL_KEYS

    def test_parses_a_yaml_style_mapping(self) -> None:
        settings = GodspeedSettings(
            llamacpp={
                "spec_type": "draft-mtp",
                "n_cpu_moe": 8,
                "no_kv_offload": False,
                "reasoning_budget": -1,
                "extra_args": ["--chat-template-kwargs", '{"preserve_thinking": false}'],
            }
        )
        assert settings.llamacpp.spec_type == "draft-mtp"
        assert settings.llamacpp.n_cpu_moe == 8
        assert settings.llamacpp.no_kv_offload is False
        assert settings.llamacpp.reasoning_budget == -1
        assert settings.llamacpp.extra_args[0] == "--chat-template-kwargs"

    @pytest.mark.parametrize(
        "bad",
        [
            {"context": -1},
            {"n_cpu_moe": -1},
            {"spec_draft_n_max": 0},
            {"reasoning_budget": -2},
            {"spec_type": "Draft MTP"},
            {"spec_type": "draft_mtp; rm -rf /"},
            {"kv_cache_type": "q8 0"},
            {"kv_cache_type": "Q8_0"},
        ],
    )
    def test_invalid_values_are_rejected(self, bad: dict) -> None:
        with pytest.raises(ValidationError):
            LlamaCppSettings(**bad)

    def test_reasoning_budget_zero_and_unlimited_are_valid(self) -> None:
        assert LlamaCppSettings(reasoning_budget=0).reasoning_budget == 0
        assert LlamaCppSettings(reasoning_budget=-1).reasoning_budget == -1


class TestEnsureLlamaCppWithSettings:
    def test_settings_are_forwarded_to_start_server(self) -> None:
        cfg = LlamaCppSettings(spec_type="draft-mtp", n_cpu_moe=4, context=24576)
        with (
            patch("godspeed.tools.llamacpp_manager.is_server_running", side_effect=[False, True]),
            patch("godspeed.tools.llamacpp_manager.configure_litellm_env"),
            patch(
                "godspeed.tools.llamacpp_manager.start_server", return_value=MagicMock()
            ) as mock_start,
        ):
            assert _ensure_llamacpp(llamacpp_settings=cfg) is True
        kwargs = mock_start.call_args.kwargs
        assert kwargs["spec_type"] == "draft-mtp"
        assert kwargs["n_cpu_moe"] == 4
        assert kwargs["context"] == 24576
        assert "timeout" in kwargs

    def test_no_settings_keeps_the_original_call(self) -> None:
        with (
            patch("godspeed.tools.llamacpp_manager.is_server_running", side_effect=[False, True]),
            patch("godspeed.tools.llamacpp_manager.configure_litellm_env"),
            patch(
                "godspeed.tools.llamacpp_manager.start_server", return_value=MagicMock()
            ) as mock_start,
        ):
            assert _ensure_llamacpp() is True
        assert set(mock_start.call_args.kwargs) == {"timeout"}
