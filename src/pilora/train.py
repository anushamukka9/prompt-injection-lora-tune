"""LoRA fine-tune for prompt-injection classification.

Heavy imports (torch, transformers, peft, trl) live inside main() so the
module imports cleanly on machines without a GPU; only the config dicts below
are importable anywhere.

Usage (on a GPU machine after running setup.sh):
    pilora-train --data data/prompt_injection_sft.jsonl --out runs/qwen05b-lora

Loss is computed on the assistant completion only, via
DataCollatorForCompletionOnlyLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pilora.config import (
    ConfigError,
    load_config,
    merge_overrides,
    validate_config,
)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# LoRA hyperparams. Documented in docs/method.md with the reasoning.
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
}

TRAINING_CONFIG = {
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,  # effective batch 16
    "max_seq_length": 512,
    "bf16": True,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "seed": 42,
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA fine-tune for prompt-injection classification")
    p.add_argument(
        "--config",
        default=None,
        help="YAML config file (see configs/base.yaml); CLI flags override it",
    )
    p.add_argument("--data", default="data/prompt_injection_sft.jsonl")
    p.add_argument("--out", default="runs/qwen05b-lora")
    p.add_argument("--model", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


def default_config() -> dict:
    """Built-in defaults, mirroring LORA_CONFIG / TRAINING_CONFIG below."""
    return {
        "model": BASE_MODEL,
        "seed": TRAINING_CONFIG["seed"],
        "max_seq_length": TRAINING_CONFIG["max_seq_length"],
        "lora": dict(LORA_CONFIG),
        "training": dict(TRAINING_CONFIG),
        "eval": {"max_new_tokens": 8, "split": "test"},
    }


def resolve_config(args: argparse.Namespace) -> dict:
    """Merge --config file (if given) with explicit CLI flags. No heavy imports."""
    if args.config:
        try:
            cfg = load_config(args.config)
        except ConfigError as exc:
            raise SystemExit(f"bad --config: {exc}") from exc
        cfg = merge_overrides(default_config(), **cfg)
    else:
        cfg = default_config()
    cfg = merge_overrides(
        cfg,
        **{
            "model": args.model,
            "seed": args.seed,
            "max_seq_length": args.max_seq_length,
            "training.num_train_epochs": args.epochs,
            "training.learning_rate": args.lr,
        },
    )
    try:
        return validate_config(cfg)
    except ConfigError as exc:
        raise SystemExit(f"invalid config: {exc}") from exc


def main() -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    args = build_arg_parser().parse_args()
    cfg = resolve_config(args)
    model_name = cfg["model"]
    training = cfg["training"]
    lora_cfg = cfg["lora"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_text(record: dict) -> dict:
        text = tokenizer.apply_chat_template(
            record["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    raw = load_dataset("json", data_files=args.data, split="train")
    train_ds = raw.filter(lambda r: r["split"] == "train").map(to_text)
    val_ds = raw.filter(lambda r: r["split"] == "val").map(to_text)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    )

    sft_args = SFTConfig(
        output_dir=args.out,
        num_train_epochs=training["num_train_epochs"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        max_seq_length=cfg["max_seq_length"],
        bf16=training.get("bf16", True),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training.get("warmup_ratio", 0.05),
        weight_decay=training.get("weight_decay", 0.01),
        logging_steps=training.get("logging_steps", 10),
        eval_strategy=training.get("eval_strategy", "epoch"),
        save_strategy=training.get("save_strategy", "epoch"),
        load_best_model_at_end=training.get("load_best_model_at_end", True),
        metric_for_best_model=training.get("metric_for_best_model", "eval_loss"),
        seed=cfg["seed"],
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    summary = {
        "base_model": model_name,
        "lora": lora_cfg,
        "training": training,
        "seed": cfg["seed"],
        "max_seq_length": cfg["max_seq_length"],
        "train_n": len(train_ds),
        "val_n": len(val_ds),
    }
    Path(args.out, "run_config.json").write_text(json.dumps(summary, indent=2))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
