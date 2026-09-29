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
    p.add_argument("--data", default="data/prompt_injection_sft.jsonl")
    p.add_argument("--out", default="runs/qwen05b-lora")
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--epochs", type=int, default=TRAINING_CONFIG["num_train_epochs"])
    p.add_argument("--lr", type=float, default=TRAINING_CONFIG["learning_rate"])
    p.add_argument("--max-seq-length", type=int, default=TRAINING_CONFIG["max_seq_length"])
    p.add_argument("--seed", type=int, default=TRAINING_CONFIG["seed"])
    return p


def main() -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    args = build_arg_parser().parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    peft_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        bias=LORA_CONFIG["bias"],
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
        num_train_epochs=args.epochs,
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        bf16=TRAINING_CONFIG["bf16"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        eval_strategy=TRAINING_CONFIG["eval_strategy"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        load_best_model_at_end=TRAINING_CONFIG["load_best_model_at_end"],
        metric_for_best_model=TRAINING_CONFIG["metric_for_best_model"],
        seed=args.seed,
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
        "base_model": args.model,
        "lora": LORA_CONFIG,
        "training": {**TRAINING_CONFIG, "num_train_epochs": args.epochs,
                     "learning_rate": args.lr, "seed": args.seed},
        "train_n": len(train_ds),
        "val_n": len(val_ds),
    }
    Path(args.out, "run_config.json").write_text(json.dumps(summary, indent=2))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
