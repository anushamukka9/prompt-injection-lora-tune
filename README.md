# prompt-injection-lora-tune

LoRA fine-tuning of a small open model (Qwen2.5-0.5B-Instruct) for binary
prompt-injection classification, with an honest, fully synthetic eval. This
is the training complement to my rule-based repos
[llm-sentinel](https://github.com/anushamukka9/llm-sentinel) and
[agent-policy-kit](https://github.com/anushamukka9/agent-policy-kit): rules
catch what you can name, a tuned small model can catch phrasing you have not
seen. The interesting question is how much it actually learns, so this repo
is built around measuring that instead of asserting it.

## Quickstart (no GPU needed)

```bash
pip install -e ".[test]"
pilora-build-data --out data/prompt_injection.jsonl --seed 42
python examples/quickstart_mock_eval.py
```

The mock eval runs the full metrics and reporting pipeline on canned
predictions, so you can see exactly what the GPU-run reports will look like.

## Training (one command, on a GPU machine)

```bash
bash setup.sh            # once per machine: torch, deps, accelerate config
pilora-train --data data/prompt_injection_sft.jsonl --out runs/qwen05b-lora
```

Then evaluate base vs tuned and render the comparison:

```bash
pilora-eval --out reports/base
pilora-eval --adapter runs/qwen05b-lora --out reports/lora
pilora-eval --compare reports/base/report.json reports/lora/report.json --out reports/comparison.md
```

See [docs/gpu-guide.md](docs/gpu-guide.md) for GPU requirements, time and
cost estimates, and launch steps. TL;DR: 12 GB VRAM, ~15-90 minutes, well
under $1 on spot GPUs.

## Method

- Task: classify a user message as INJECTION or BENIGN (single-token output).
- Base: Qwen2.5-0.5B-Instruct (fallback: Llama-3.2-1B-Instruct).
- LoRA r=16, alpha=32, dropout 0.05, all attention projections; 3 epochs,
  effective batch 16, bf16, cosine schedule. Loss on the assistant completion
  only.
- Eval: 200-example stratified test split; accuracy, precision, recall, FPR,
  FNR, unparseable count, latency; per-category breakdowns. Reports mirror my
  [llm-eval-harness](https://github.com/anushamukka9/llm-eval-harness) format
  (`report.json` + `report.md`).

Full details in [docs/method.md](docs/method.md). Reference config in
`configs/base.yaml`.

## Dataset

2,000 fully synthetic records, built by `src/pilora/dataset.py` (deterministic,
seed 42): 500 obvious injections, 500 subtle injections, 600 benign, 400
tricky-but-benign edge cases (discussions *about* prompt injection, quoted
instructions, code with instruction-like strings). Split 1600/200/200,
stratified by category. No real user data, no PII. Documented honestly in
[data/README.md](data/README.md), including what the synthetic approach
cannot tell you.

## Results

**PENDING GPU RUN.** No training has been executed yet; this section gets
the base-vs-tuned comparison table once a real run lands. I am not
publishing numbers I have not measured. The `examples/quickstart_mock_eval.py`
output shows the report format with placeholder numbers only.

Expected (not promised): the base instruct model should already do well on
obvious injections from pretraining; the fine-tune should mainly move the
needle on subtle injections and benign_edge false positives. If the tuned
model does not beat the base, that goes in the table too. Negative results
are results.

## Limitations

- The dataset is synthetic and template-generated. Good scores measure fit
  to the templates, not real-world detection. Real attacks are adversarial
  and adaptive; this is a fixed target.
- A 0.5B classifier is a triage signal, not a security boundary. In any real
  deployment it sits behind or beside deterministic checks, never alone.
- English only. Short inputs only (512 tokens).
- Nothing here has been red-teamed. The eval measures what I built it to
  measure.

## Repo map

```
src/pilora/
  dataset.py    synthetic dataset builder (the source of truth)
  train.py      LoRA SFT training (GPU only; config importable anywhere)
  evaluate.py   base-vs-tuned eval, report.json/md rendering, --compare
  metrics.py    pure-Python metrics (accuracy, FPR/FNR, latency)
data/           built dataset + honest build notes
docs/           method.md, gpu-guide.md
examples/       no-GPU mock eval, dataset sample
configs/        base.yaml reference training config
tests/          data pipeline, metrics, and config tests (no GPU)
setup.sh        one-time GPU machine setup
```

## Author

Anusha Mukka, [anushamukka.com](https://anushamukka.com)

## License

MIT, see [LICENSE](LICENSE).
