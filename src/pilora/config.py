"""YAML config loading, validation, and CLI-override merging for pilora.

Every field the trainer and evaluator read is declared here. Validation
runs up front so a typo fails fast on your laptop instead of halfway
through a GPU run. Unknown extra keys are allowed (forward-compatible)
but never required.

Config layout (see configs/base.yaml):

    model: Qwen/Qwen2.5-0.5B-Instruct
    seed: 42
    max_seq_length: 512
    lora:
      r: 16
      lora_alpha: 32
      lora_dropout: 0.05
      target_modules: [q_proj, k_proj, v_proj, o_proj]
      bias: none
    training:
      learning_rate: 2.0e-4
      num_train_epochs: 3
      ...
    eval:
      max_new_tokens: 8
      split: test
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

LORA_BIASES = {"none", "all", "lora_only"}
EVAL_SPLITS = {"train", "val", "test"}


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict. Raises ConfigError on problems."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a YAML mapping")
    return data


def _require(cfg: dict, key: str, section: str) -> Any:
    if key not in cfg:
        raise ConfigError(f"config section '{section}' is missing required key '{key}'")
    return cfg[key]


def _check_int(value: Any, key: str, section: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"'{section}.{key}' must be an integer >= {minimum}, got {value!r}")
    return value


def _check_float(value: Any, key: str, section: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{section}.{key}' must be a number, got {value!r}")
    value = float(value)
    if not lo <= value <= hi:
        raise ConfigError(f"'{section}.{key}' must be within [{lo}, {hi}], got {value}")
    return value


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Check a config dict and return it unchanged. Raises ConfigError."""
    if not isinstance(cfg, dict):
        raise ConfigError("config must be a mapping")

    model = _require(cfg, "model", "top")
    if not isinstance(model, str) or not model.strip():
        raise ConfigError("'model' must be a non-empty string")

    seed = _require(cfg, "seed", "top")
    _check_int(seed, "seed", "top", minimum=0)

    max_seq = _require(cfg, "max_seq_length", "top")
    if isinstance(max_seq, bool) or not isinstance(max_seq, int):
        raise ConfigError("'max_seq_length' must be an integer")
    if not 64 <= max_seq <= 4096:
        raise ConfigError(f"'max_seq_length' must be within [64, 4096], got {max_seq}")

    lora = _require(cfg, "lora", "top")
    if not isinstance(lora, dict):
        raise ConfigError("'lora' must be a mapping")
    _check_int(_require(lora, "r", "lora"), "r", "lora")
    _check_int(_require(lora, "lora_alpha", "lora"), "lora_alpha", "lora")
    _check_float(_require(lora, "lora_dropout", "lora"), "lora_dropout", "lora", 0.0, 0.99)
    modules = _require(lora, "target_modules", "lora")
    if (
        not isinstance(modules, list)
        or not modules
        or not all(isinstance(m, str) and m for m in modules)
    ):
        raise ConfigError("'lora.target_modules' must be a non-empty list of strings")
    bias = _require(lora, "bias", "lora")
    if bias not in LORA_BIASES:
        raise ConfigError(f"'lora.bias' must be one of {sorted(LORA_BIASES)}, got {bias!r}")

    training = _require(cfg, "training", "top")
    if not isinstance(training, dict):
        raise ConfigError("'training' must be a mapping")
    _check_float(
        _require(training, "learning_rate", "training"), "learning_rate", "training", 0.0, 1.0
    )
    _check_int(_require(training, "num_train_epochs", "training"), "num_train_epochs", "training")
    _check_int(
        _require(training, "per_device_train_batch_size", "training"),
        "per_device_train_batch_size",
        "training",
    )
    _check_int(
        _require(training, "gradient_accumulation_steps", "training"),
        "gradient_accumulation_steps",
        "training",
    )
    if "warmup_ratio" in training:
        _check_float(training["warmup_ratio"], "warmup_ratio", "training", 0.0, 1.0)
    if "weight_decay" in training:
        _check_float(training["weight_decay"], "weight_decay", "training", 0.0, 10.0)
    if "bf16" in training and not isinstance(training["bf16"], bool):
        raise ConfigError("'training.bf16' must be a boolean")

    ev = _require(cfg, "eval", "top")
    if not isinstance(ev, dict):
        raise ConfigError("'eval' must be a mapping")
    _check_int(_require(ev, "max_new_tokens", "eval"), "max_new_tokens", "eval")
    split = _require(ev, "split", "eval")
    if split not in EVAL_SPLITS:
        raise ConfigError(f"'eval.split' must be one of {sorted(EVAL_SPLITS)}, got {split!r}")

    return cfg


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    """Set a dotted key like 'training.learning_rate', creating sections as needed."""
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value
    return cfg


def merge_overrides(cfg: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Merge overrides into a config. Dotted keys set nested values.

    Example: merge_overrides(cfg, **{"training.learning_rate": 1e-4})
    """
    merged = deepcopy(cfg)
    for dotted, value in overrides.items():
        if value is None:
            continue
        if "." in dotted:
            set_dotted(merged, dotted, value)
        elif isinstance(value, dict) and isinstance(merged.get(dotted), dict):
            merged[dotted] = _deep_merge(merged[dotted], value)
        else:
            merged[dotted] = deepcopy(value)
    return merged
