"""No-GPU quickstart: exercise the full data -> metrics -> report pipeline
with canned predictions, exactly as pilora-eval renders them after a GPU run.

Run: python examples/quickstart_mock_eval.py
"""

from pilora.dataset import build_dataset
from pilora.evaluate import render_comparison, render_markdown
from pilora.metrics import classification_report, summary_line

# Build the dataset (deterministic) and take the test split.
records = build_dataset(seed=42)
test = [r for r in records if r["split"] == "test"]
labels = [r["label"] for r in test]

# Mock backend A ("base"): mostly right, trips on subtle injections and edges.
# Mock backend B ("tuned"): better, still imperfect. This is only a demo of
# the reporting pipeline, not a claim about any model.
import random

rng = random.Random(0)
preds_a, preds_b = [], []
for r in labels:
    if r == "INJECTION" and rng.random() < 0.25:
        preds_a.append("BENIGN")  # missed injection
    elif r == "BENIGN" and rng.random() < 0.15:
        preds_a.append("INJECTION")  # false alarm
    else:
        preds_a.append(r)
    if rng.random() < 0.08:
        preds_b.append("BENIGN" if r == "INJECTION" else "INJECTION")
    else:
        preds_b.append(r)

rep_a = classification_report(preds_a, labels)
rep_a.update({"model": "mock-base", "adapter": None,
              "latency": {"mean_ms": 120.0, "p50_ms": 110.0, "p95_ms": 200.0},
              "by_category": {}})
rep_b = classification_report(preds_b, labels)
rep_b.update({"model": "mock-base", "adapter": "mock-lora",
              "latency": {"mean_ms": 122.0, "p50_ms": 112.0, "p95_ms": 205.0},
              "by_category": {}})

print("A (mock base):", summary_line(rep_a))
print("B (mock tuned):", summary_line(rep_b))
print()
print(render_comparison(rep_a, rep_b, "A", "B"))
