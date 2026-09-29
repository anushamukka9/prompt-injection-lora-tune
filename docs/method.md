# Method

## What this is

A LoRA fine-tune of Qwen2.5-0.5B-Instruct for binary prompt-injection
classification. Input: a user message. Output: exactly one word, INJECTION or
BENIGN. The point is not to ship a production detector. The point is to
measure, honestly, what a small open model learns about this task from
synthetic data, and to publish the full recipe so anyone can reproduce it.

## Why binary classification

It is the smallest task that still matters. My other repos (llm-sentinel,
agent-policy-kit) use deterministic, rule-based checks for prompt injection.
A learned classifier is a natural complement: rules catch what you can name,
a model can catch phrasing you have not seen. But a classifier also fails in
ways rules do not, which is why the eval reports false positive and false
negative rates separately instead of hiding behind one accuracy number.

## Why Qwen2.5-0.5B-Instruct

- 0.5B parameters train in minutes on one consumer GPU, so the whole
  experiment costs less than a dollar and anyone can rerun it.
- The Instruct variant already follows the "respond with one word" format,
  which keeps unparseable outputs low and makes the base-vs-tuned comparison
  meaningful instead of a formatting contest.
- Apache 2.0 license, no gated access.

Fallback: Llama-3.2-1B-Instruct (`--model meta-llama/Llama-3.2-1B-Instruct`,
gated, needs a Hugging Face token). It roughly doubles training time and
needs the same 12 GB of VRAM.

## Training setup

Supervised fine-tuning with TRL's SFTTrainer. Chat-formatted records
(system + user + assistant), loss computed on the assistant completion only
via DataCollatorForCompletionOnlyLM, so the model is not rewarded for
memorizing prompts.

| Hyperparameter | Value | Why |
|---|---|---|
| LoRA rank r | 16 | Small task, small model; 16 is plenty and keeps adapters tiny |
| LoRA alpha | 32 | alpha = 2r, the standard scaling |
| LoRA dropout | 0.05 | Light regularization on a 2k-example set |
| Target modules | q_proj, k_proj, v_proj, o_proj | All attention projections; cheap at 0.5B |
| Learning rate | 2e-4 | Standard LoRA LR for small models |
| Epochs | 3 | 1600 train examples; more epochs overfit the templates |
| Effective batch | 16 (4 x 4 grad accum) | Fits comfortably in 12 GB |
| Max seq length | 512 | Injection attempts are short; longer wastes VRAM |
| Scheduler | cosine, 5% warmup | Smooth decay, standard |
| Precision | bf16 | Halves memory vs fp32, no accuracy cost here |

Full config: `configs/base.yaml`. CLI flags on `pilora-train` override everything.

## Evaluation

`pilora-eval` runs the test split (200 examples, stratified across all four
categories) through the model with greedy decoding, parses the first
INJECTION/BENIGN token, and reports accuracy, precision, recall, FPR, FNR,
unparseable-output count, and latency. Reports land in `reports/<name>/` as
`report.json` + `report.md`, the same shape as my
[llm-eval-harness](https://github.com/anushamukka9/llm-eval-harness) reports,
so suites can be compared with the same tooling.

Scoring rules worth knowing:

- The first whole-word label in the generation wins; trailing punctuation is
  fine ("INJECTION.").
- An unparseable generation is always scored as wrong, even when the true
  label is BENIGN. A model that cannot follow the output format has failed
  the task.
- Per-category breakdowns are in every report, because aggregate accuracy
  hides the interesting failure: models usually ace obvious injections and
  stumble on benign_edge.

## What the numbers can and cannot say

The dataset is 100% synthetic and template-generated. Good numbers here mean
the model learned the synthetic distribution. They do not mean it detects
real prompt injection in the wild. Real attacks are adversarial, adaptive,
and nothing like templates. I say this in the README too, because a
benchmark that flatters its author is worse than no benchmark.

The honest use of this repo: a cheap, reproducible harness for asking
"does this training change move FPR/FNR on a fixed synthetic suite", plus a
starting point you can extend with real, labeled data.
