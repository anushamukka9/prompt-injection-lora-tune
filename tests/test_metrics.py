"""Tests for metrics and report rendering. No GPU, no torch needed."""

import pytest

from pilora.evaluate import render_comparison, render_markdown
from pilora.metrics import (
    classification_report,
    confusion_matrix,
    parse_label,
    summarize_latency,
    summary_line,
)


def test_parse_label_exact():
    assert parse_label("INJECTION") == "INJECTION"
    assert parse_label("benign") == "BENIGN"


def test_parse_label_embedded():
    assert parse_label("My verdict: INJECTION because it overrides.") == "INJECTION"
    assert parse_label("I think this is BENIGN.") == "BENIGN"


def test_parse_label_first_wins():
    assert parse_label("INJECTION, not BENIGN") == "INJECTION"
    assert parse_label("BENIGN, not INJECTION") == "BENIGN"


def test_parse_label_none():
    assert parse_label("I cannot decide.") is None
    assert parse_label("") is None


def test_confusion_matrix():
    cm = confusion_matrix(
        ["INJECTION", "BENIGN", None, "INJECTION"],
        ["INJECTION", "INJECTION", "BENIGN", "BENIGN"],
    )
    assert cm == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_classification_report_perfect():
    rep = classification_report(["INJECTION", "BENIGN"], ["INJECTION", "BENIGN"])
    assert rep["accuracy"] == 1.0
    assert rep["fpr"] == 0.0
    assert rep["fnr"] == 0.0
    assert rep["unparseable"] == 0


def test_classification_report_counts_unparseable_as_wrong():
    rep = classification_report([None, None], ["INJECTION", "BENIGN"])
    assert rep["accuracy"] == 0.0
    assert rep["unparseable"] == 2


def test_classification_report_empty_raises():
    with pytest.raises(ValueError):
        classification_report([], [])


def test_summarize_latency():
    lat = summarize_latency([10.0, 20.0, 30.0, 40.0])
    assert lat["mean_ms"] == 25.0
    assert lat["p50_ms"] == 30.0
    assert lat["p95_ms"] == 40.0
    assert summarize_latency([])["mean_ms"] == 0.0


def test_summary_line_format():
    rep = classification_report(["INJECTION", "BENIGN"], ["INJECTION", "BENIGN"])
    line = summary_line(rep)
    assert "accuracy=100.0%" in line and "fpr=0.0%" in line and "n=2" in line


def _fake_report(**over):
    base = {
        "n": 200,
        "accuracy": 0.9,
        "precision": 0.88,
        "recall": 0.92,
        "fpr": 0.12,
        "fnr": 0.08,
        "unparseable": 3,
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter": None,
        "latency": {"mean_ms": 45.0, "p50_ms": 42.0, "p95_ms": 80.0},
        "by_category": {
            "obvious_injection": {"n": 50, "accuracy": 1.0, "fpr": 0.0, "fnr": 0.0},
            "benign_edge": {"n": 40, "accuracy": 0.8, "fpr": 0.2, "fnr": 0.0},
        },
    }
    base.update(over)
    return base


def test_render_markdown_contains_key_sections():
    md = render_markdown(_fake_report(), "Eval (base)")
    assert "# Eval (base)" in md
    assert "accuracy = 90.0%" in md
    assert "false positive rate = 12.0%" in md
    assert "benign_edge" in md
    assert "synthetic" in md.lower()


def test_render_comparison_table():
    a = _fake_report()
    b = _fake_report(accuracy=0.95, fpr=0.05, adapter="runs/qwen05b-lora")
    md = render_comparison(a, b, "A", "B")
    assert "| accuracy | 90.0% | 95.0% |" in md
    assert "| false positive rate | 12.0% | 5.0% |" in md
    assert "runs/qwen05b-lora" in md
