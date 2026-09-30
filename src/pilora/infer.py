"""Single and batch inference for the prompt-injection classifier.

Two backends:

  MockBackend: a deterministic keyword heuristic. It exists so the CLI,
    batching, and output formatting can be exercised and demoed without a
    GPU or a model download. It is clearly labeled "mock" everywhere it
    appears. It is not a model and its labels are not a claim about any
    model.

  HFBackend: the real base model plus an optional LoRA adapter, loaded
    with transformers. Needs torch/peft and a GPU for sane latency; all
    heavy imports stay inside the class so the module imports anywhere.

Usage:
    pilora-infer --mock --text "Ignore all previous instructions and ..."
    pilora-infer --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/qwen05b-lora --file inputs.txt
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from .dataset import LABEL_BENIGN, LABEL_INJECTION
from .metrics import parse_label

# Heuristic patterns for the mock backend. Deliberately crude: the point
# is plumbing, not detection quality.
_MOCK_INJECTION_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bignore (all )?previous instructions\b",
        r"\bdisregard .*instructions\b",
        r"\bjailbreak\b",
        r"\bDAN\b.*\bmode\b",
        r"\bsystem (prompt|override|notice)\b",
        r"\breveal your .*instructions\b",
        r"\bbypass .*content policy\b",
        r"\bno restrictions\b",
        r"\bunrestricted mode\b",
    ]
]


def format_results(results: list[dict], width: int = 60) -> str:
    """Render classification results as a plain-text table."""
    lines = [f"{'LABEL':<10} {'TEXT'}", "-" * (10 + 1 + width)]
    for r in results:
        text = r["text"].replace("\n", " ")
        if len(text) > width:
            text = text[: width - 1] + "..."
        lines.append(f"{r['label']:<10} {text}")
    lines.append(f"\nbackend: {results[0]['backend']} ({len(results)} texts)" if results else "")
    return "\n".join(lines)


class MockBackend:
    """Deterministic keyword-heuristic stand-in. Label is always 'mock'."""

    name = "mock"

    def classify(self, texts: list[str]) -> list[dict]:
        results = []
        for text in texts:
            label = LABEL_BENIGN
            for pattern in _MOCK_INJECTION_RES:
                if pattern.search(text):
                    label = LABEL_INJECTION
                    break
            results.append({"text": text, "label": label, "backend": self.name})
        return results


class HFBackend:
    """Real model inference: base model plus optional LoRA adapter."""

    name = "hf"

    def __init__(self, model_name: str, adapter_path: str | None = None, max_new_tokens: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.adapter_path = adapter_path
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
        model.eval()
        self.model = model

    def classify(self, texts: list[str]) -> list[dict]:
        from .evaluate import build_prompt

        results = []
        with self._torch.no_grad():
            for text in texts:
                prompt = build_prompt(self.tokenizer, text)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                start = time.perf_counter()
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                gen = self.tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                results.append(
                    {
                        "text": text,
                        "label": parse_label(gen) or "UNPARSEABLE",
                        "backend": self.name,
                        "latency_ms": round(elapsed_ms, 1),
                    }
                )
        return results


def collect_texts(args: argparse.Namespace) -> list[str]:
    texts = list(args.text or [])
    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise SystemExit(f"input file not found: {path}")
        texts.extend(
            line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    if not texts:
        raise SystemExit("nothing to classify: pass --text and/or --file")
    return texts


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classify texts as INJECTION or BENIGN")
    p.add_argument(
        "--mock", action="store_true", help="use the deterministic mock backend (no model download)"
    )
    p.add_argument("--model", default=None, help="HF model id for real inference")
    p.add_argument("--adapter", default=None, help="LoRA adapter directory (with --model)")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--text", action="append", default=[], help="text to classify (repeatable)")
    p.add_argument("--file", default=None, help="file with one text per line")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    texts = collect_texts(args)

    if args.mock or not args.model:
        if not args.mock:
            print("(no --model given: using the mock backend)", file=sys.stderr)
        backend = MockBackend()
    else:
        backend = HFBackend(args.model, args.adapter, args.max_new_tokens)

    results = backend.classify(texts)
    print(format_results(results))


if __name__ == "__main__":
    main()
