"""Evaluate a model on the prompt-injection test split: base vs fine-tuned.

Writes report.json and report.md in the llm-eval-harness report style, plus
a one-line summary (accuracy / FPR / FNR). A --compare mode renders a
side-by-side markdown table from two report.json files.

Usage:
    pilora-eval --model Qwen/Qwen2.5-0.5B-Instruct --out reports/base
    pilora-eval --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/qwen05b-lora \\
        --out reports/lora
    pilora-eval --compare reports/base/report.json reports/lora/report.json \\
        --out reports/comparison.md

Heavy model imports live inside run_eval() so that report rendering and
--compare work without torch installed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import load_config
from .dataset import SYSTEM_PROMPT, load_jsonl
from .metrics import (
    classification_report,
    parse_label,
    summarize_latency,
    summary_line,
)

MAX_NEW_TOKENS = 8
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATA = "data/prompt_injection_sft.jsonl"


def build_prompt(tokenizer, text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_eval(
    model_name: str,
    data_path: str,
    adapter_path: str | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    split: str = "test",
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records = [r for r in load_jsonl(Path(data_path)) if r["split"] == split]
    if not records:
        raise ValueError(f"no records with split={split!r} in {data_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()

    preds: list[str | None] = []
    labels: list[str] = []
    latencies_ms: list[float] = []
    per_category: dict[str, dict[str, list]] = {}
    with torch.no_grad():
        for rec in records:
            prompt = build_prompt(tokenizer, rec["text"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            start = time.perf_counter()
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            preds.append(parse_label(gen))
            labels.append(rec["label"])
            latencies_ms.append(elapsed_ms)
            bucket = per_category.setdefault(rec["category"], {"preds": [], "labels": []})
            bucket["preds"].append(preds[-1])
            bucket["labels"].append(rec["label"])

    report = classification_report(preds, labels)
    report["latency"] = summarize_latency(latencies_ms)
    report["model"] = model_name
    report["adapter"] = adapter_path
    report["split"] = split
    report["by_category"] = {
        cat: classification_report(b["preds"], b["labels"]) for cat, b in per_category.items()
    }
    return report


def render_markdown(report: dict, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"Model: `{report.get('model', '?')}`",
    ]
    if report.get("adapter"):
        lines.append(f"Adapter: `{report['adapter']}`")
    lines += [
        "",
        "## Summary",
        "",
        f"- n = {report['n']}",
        f"- accuracy = {report['accuracy'] * 100:.1f}%",
        f"- precision = {report['precision'] * 100:.1f}%",
        f"- recall = {report['recall'] * 100:.1f}%",
        f"- false positive rate = {report['fpr'] * 100:.1f}%",
        f"- false negative rate = {report['fnr'] * 100:.1f}%",
        f"- unparseable outputs = {report['unparseable']}",
        (
            f"- latency mean = {report['latency']['mean_ms']:.1f} ms, "
            f"p95 = {report['latency']['p95_ms']:.1f} ms"
        ),
        "",
        "## By category",
        "",
        "| category | n | accuracy | FPR | FNR |",
        "|---|---|---|---|---|",
    ]
    for cat, sub in report["by_category"].items():
        lines.append(
            f"| {cat} | {sub['n']} | {sub['accuracy'] * 100:.1f}% | "
            f"{sub['fpr'] * 100:.1f}% | {sub['fnr'] * 100:.1f}% |"
        )
    lines += [
        "",
        (
            "_Dataset is fully synthetic (see data/build notes in README). "
            "Treat these numbers as a training-dynamics check, not a security claim._"
        ),
    ]
    return "\n".join(lines) + "\n"


def render_comparison(a: dict, b: dict, name_a: str, name_b: str) -> str:
    def row(label: str, key: str, fmt="{:.1f}%", scale: float = 100.0) -> str:
        return f"| {label} | {fmt.format(a[key] * scale)} | {fmt.format(b[key] * scale)} |"

    lines = [
        "# Base vs fine-tuned",
        "",
        f"A = {name_a} (`{a.get('model', '?')}`)",
        f"B = {name_b} (`{b.get('model', '?')}`"
        + (f" + `{b.get('adapter')}`" if b.get("adapter") else ""),
        "",
        "| metric | A (base) | B (fine-tuned) |",
        "|---|---|---|",
        row("accuracy", "accuracy"),
        row("precision", "precision"),
        row("recall", "recall"),
        row("false positive rate", "fpr"),
        row("false negative rate", "fnr"),
        f"| unparseable | {a['unparseable']} | {b['unparseable']} |",
        f"| latency mean (ms) | {a['latency']['mean_ms']:.1f} | {b['latency']['mean_ms']:.1f} |",
        "",
        (
            "_Dataset is fully synthetic. These numbers measure fit to the synthetic "
            "distribution, not real-world detection performance._"
        ),
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate prompt-injection classifier")
    p.add_argument(
        "--config",
        default=None,
        help="YAML config file (see configs/eval.yaml); CLI flags override it",
    )
    p.add_argument("--model", default=None)
    p.add_argument("--adapter", default=None, help="path to LoRA adapter (optional)")
    p.add_argument("--data", default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--out", default="reports/eval")
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("A_JSON", "B_JSON"),
        help="render comparison markdown from two report.json files",
    )
    return p


def resolve_eval_settings(args: argparse.Namespace) -> dict:
    """Merge --config file (if given) with explicit CLI flags. No heavy imports."""
    model, data, max_new_tokens, split = DEFAULT_MODEL, DEFAULT_DATA, MAX_NEW_TOKENS, "test"
    if args.config:
        raw = load_config(args.config)
        model = raw.get("model") or model
        data = raw.get("data") or data
        ev = raw.get("eval") or {}
        max_new_tokens = ev.get("max_new_tokens", max_new_tokens)
        split = ev.get("split", split)
    if args.model:
        model = args.model
    if args.data:
        data = args.data
    if args.max_new_tokens is not None:
        max_new_tokens = args.max_new_tokens
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise SystemExit(f"invalid max_new_tokens: {max_new_tokens!r}")
    if split not in ("train", "val", "test"):
        raise SystemExit(f"invalid eval split: {split!r}")
    return {"model": model, "data": data, "max_new_tokens": max_new_tokens, "split": split}


def main() -> None:
    p = build_arg_parser()
    args = p.parse_args()

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text())
        b = json.loads(Path(args.compare[1]).read_text())
        md = render_comparison(a, b, "A", "B")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        print(f"wrote {out_path}")
        return

    settings = resolve_eval_settings(args)
    report = run_eval(
        settings["model"],
        settings["data"],
        args.adapter,
        max_new_tokens=settings["max_new_tokens"],
        split=settings["split"],
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    title = "Prompt-injection eval" + (" (fine-tuned)" if args.adapter else " (base)")
    (out_dir / "report.md").write_text(render_markdown(report, title))
    print(summary_line(report))
    print(f"wrote {out_dir / 'report.json'} and {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
