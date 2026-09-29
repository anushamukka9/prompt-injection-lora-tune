"""Tests for training configuration. Imports no heavy deps."""

from pilora.train import BASE_MODEL, FALLBACK_MODEL, LORA_CONFIG, TRAINING_CONFIG


def test_lora_config_sane():
    assert LORA_CONFIG["r"] == 16
    assert LORA_CONFIG["lora_alpha"] == 2 * LORA_CONFIG["r"]
    assert 0 < LORA_CONFIG["lora_dropout"] < 1
    assert set(LORA_CONFIG["target_modules"]) >= {"q_proj", "v_proj"}
    assert LORA_CONFIG["bias"] == "none"


def test_training_config_sane():
    assert TRAINING_CONFIG["num_train_epochs"] >= 1
    assert TRAINING_CONFIG["per_device_train_batch_size"] >= 1
    eff = (TRAINING_CONFIG["per_device_train_batch_size"]
           * TRAINING_CONFIG["gradient_accumulation_steps"])
    assert eff == 16
    assert TRAINING_CONFIG["max_seq_length"] <= 2048
    assert TRAINING_CONFIG["bf16"] is True


def test_model_defaults():
    assert "Qwen2.5-0.5B" in BASE_MODEL
    assert "Llama-3.2-1B" in FALLBACK_MODEL
