"""Tests for the synthetic dataset pipeline. No GPU, no torch needed."""

from collections import Counter

from pilora.dataset import (
    CATEGORY_SPECS,
    LABEL_BENIGN,
    LABEL_INJECTION,
    SYSTEM_PROMPT,
    build_dataset,
    format_for_sft,
)


def test_total_size_and_labels():
    records = build_dataset(seed=42)
    expected = sum(count for _, _, _, count in CATEGORY_SPECS)
    assert len(records) == expected == 2000
    counts = Counter(r["label"] for r in records)
    assert counts[LABEL_INJECTION] == 1000
    assert counts[LABEL_BENIGN] == 1000


def test_categories_match_labels():
    records = build_dataset(seed=42)
    for r in records:
        if r["category"] in ("obvious_injection", "subtle_injection"):
            assert r["label"] == LABEL_INJECTION
        else:
            assert r["label"] == LABEL_BENIGN


def test_splits_stratified_and_disjoint():
    records = build_dataset(seed=42)
    ids_by_split: dict[str, set[str]] = {}
    for r in records:
        ids_by_split.setdefault(r["split"], set()).add(r["id"])
    assert set(ids_by_split) == {"train", "val", "test"}
    assert ids_by_split["train"].isdisjoint(ids_by_split["val"])
    assert ids_by_split["train"].isdisjoint(ids_by_split["test"])
    assert ids_by_split["val"].isdisjoint(ids_by_split["test"])
    # every category appears in every split (stratified)
    for split in ("train", "val", "test"):
        cats = {r["category"] for r in records if r["split"] == split}
        assert cats == {"obvious_injection", "subtle_injection", "benign", "benign_edge"}
    # roughly 80/10/10
    n = len(records)
    assert abs(len(ids_by_split["train"]) / n - 0.8) < 0.02


def test_deterministic_and_unique_texts():
    a = build_dataset(seed=7)
    b = build_dataset(seed=7)
    assert [r["text"] for r in a] == [r["text"] for r in b]
    texts = [r["text"] for r in a]
    assert len(set(texts)) == len(texts), "duplicate texts generated"


def test_different_seeds_differ():
    a = build_dataset(seed=1)
    b = build_dataset(seed=2)
    assert [r["text"] for r in a] != [r["text"] for r in b]


def test_records_have_required_fields():
    for r in build_dataset(seed=42):
        assert set(r) >= {"id", "text", "label", "category", "split"}
        assert r["text"].strip()
        assert r["label"] in (LABEL_INJECTION, LABEL_BENIGN)


def test_sft_format():
    rec = build_dataset(seed=42)[0]
    sft = format_for_sft(rec)
    roles = [m["role"] for m in sft["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert sft["messages"][0]["content"] == SYSTEM_PROMPT
    assert sft["messages"][1]["content"] == rec["text"]
    assert sft["messages"][2]["content"] == rec["label"]
    assert sft["label"] == rec["label"]
