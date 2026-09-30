"""Tests for config loading, validation, and CLI-override merging. No GPU."""

import pytest

from pilora.config import (
    ConfigError,
    load_config,
    merge_overrides,
    set_dotted,
    validate_config,
)
from pilora.train import build_arg_parser, default_config, resolve_config

BASE_YAML = "configs/base.yaml"


def test_base_yaml_loads_and_validates():
    cfg = load_config(BASE_YAML)
    assert validate_config(cfg) is cfg
    assert cfg["model"].startswith("Qwen/")
    assert cfg["lora"]["r"] == 16


def test_load_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config("configs/does-not-exist.yaml")


def test_load_malformed_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("key: [unclosed\n")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_load_non_mapping_raises(tmp_path):
    bad = tmp_path / "list.yaml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError):
        load_config(bad)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("top", "model", ""),
        ("top", "seed", -1),
        ("top", "max_seq_length", 32),
        ("lora", "r", 0),
        ("lora", "lora_alpha", 0),
        ("lora", "lora_dropout", 1.5),
        ("lora", "target_modules", []),
        ("lora", "bias", "sometimes"),
        ("training", "learning_rate", -1e-4),
        ("training", "num_train_epochs", 0),
        ("eval", "max_new_tokens", 0),
        ("eval", "split", "sometimes"),
    ],
)
def test_validate_rejects_bad_values(section, key, value):
    cfg = default_config()
    if section == "top":
        cfg[key] = value
    else:
        cfg[section][key] = value
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_validate_rejects_missing_section():
    cfg = default_config()
    del cfg["lora"]
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_validate_allows_extra_keys():
    cfg = default_config()
    cfg["training"]["some_future_knob"] = 123
    assert validate_config(cfg) is cfg


def test_merge_overrides_dotted_keys():
    cfg = default_config()
    merged = merge_overrides(cfg, **{"training.learning_rate": 1e-4})
    assert merged["training"]["learning_rate"] == 1e-4
    assert merged["training"]["num_train_epochs"] == 3  # untouched
    assert cfg["training"]["learning_rate"] != 1e-4  # input not mutated


def test_merge_overrides_skips_none():
    cfg = default_config()
    merged = merge_overrides(cfg, model=None)
    assert merged["model"] == cfg["model"]


def test_set_dotted_creates_sections():
    cfg: dict = {}
    set_dotted(cfg, "a.b.c", 5)
    assert cfg == {"a": {"b": {"c": 5}}}


def test_resolve_config_defaults_match_builtin_constants():
    args = build_arg_parser().parse_args([])
    cfg = resolve_config(args)
    assert cfg["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg["lora"]["r"] == 16
    assert cfg["training"]["num_train_epochs"] == 3
    assert cfg["training"]["learning_rate"] == 2e-4


def test_resolve_config_file_plus_cli_override(tmp_path):
    args = build_arg_parser().parse_args(["--config", BASE_YAML, "--epochs", "1", "--lr", "0.0001"])
    cfg = resolve_config(args)
    assert cfg["training"]["num_train_epochs"] == 1
    assert cfg["training"]["learning_rate"] == pytest.approx(0.0001)
    # values from the file that the CLI did not touch survive
    assert cfg["lora"]["r"] == 16
    assert cfg["seed"] == 42


def test_resolve_config_bad_file_exits(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: ''\n")
    args = build_arg_parser().parse_args(["--config", str(bad)])
    with pytest.raises(SystemExit):
        resolve_config(args)


def test_resolve_eval_settings_defaults():
    from pilora.evaluate import build_arg_parser as eval_parser
    from pilora.evaluate import resolve_eval_settings

    args = eval_parser().parse_args([])
    settings = resolve_eval_settings(args)
    assert settings["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert settings["data"] == "data/prompt_injection_sft.jsonl"
    assert settings["max_new_tokens"] == 8
    assert settings["split"] == "test"


def test_resolve_eval_settings_config_plus_cli():
    from pilora.evaluate import build_arg_parser as eval_parser
    from pilora.evaluate import resolve_eval_settings

    args = eval_parser().parse_args(["--config", "configs/eval.yaml", "--max-new-tokens", "16"])
    settings = resolve_eval_settings(args)
    assert settings["max_new_tokens"] == 16
    assert settings["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert settings["split"] == "test"


def test_resolve_eval_settings_bad_split_exits(monkeypatch, tmp_path):
    from pilora.evaluate import build_arg_parser as eval_parser
    from pilora.evaluate import resolve_eval_settings

    bad = tmp_path / "eval-bad.yaml"
    bad.write_text(
        "model: Qwen/Qwen2.5-0.5B-Instruct\neval:\n  max_new_tokens: 8\n  split: whenever\n"
    )
    args = eval_parser().parse_args(["--config", str(bad)])
    with pytest.raises(SystemExit):
        resolve_eval_settings(args)
