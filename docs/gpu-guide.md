# GPU run guide

Everything below is an estimate, labeled as such. Nothing here has been run
yet; the results section in the README says PENDING until a real run lands.

## What you need

- One NVIDIA GPU with **12 GB+ VRAM** (24 GB comfortable, 12 GB sufficient).
  The 0.5B model in bf16 is ~1 GB of weights; LoRA trains only the adapters,
  so optimizer state stays small. Peak usage lands around 6-8 GB with the
  default batch settings.
- CUDA 12.1+ driver, ~20 GB free disk (torch + model cache).

Known-good cheap options: RTX 3060 12GB, RTX 3090, RTX 4090, L4, A10.

## Time and cost estimates

Dataset: 1600 train / 200 val / 200 test, 3 epochs, effective batch 16
= 300 optimizer steps. On an RTX 4090 that is roughly **15-30 minutes**
including eval; on a 3060 12GB, roughly **45-90 minutes**.

| Provider | GPU | Approx. $/hr (spot) | Est. cost per full run |
|---|---|---|---|
| vast.ai | RTX 4090 | $0.25 - 0.45 | under $0.50 |
| vast.ai | RTX 3090 | $0.15 - 0.30 | under $0.50 |
| runpod | RTX 4090 | ~$0.39 | under $0.50 |
| runpod | A10 24GB | ~$0.50 | under $1.00 |

These are spot/preemptible prices as of late 2026 and move around; check the
marketplace before launching. On-demand is roughly 2-3x. Even at on-demand
prices a full train+eval stays under about $2.

## Launch steps (vast.ai example)

1. Rent a PyTorch CUDA 12.1 template with 12 GB+ VRAM.
2. `git clone https://github.com/anushamukka9/prompt-injection-lora-tune.git`
   and `cd` in. Checkout `develop`.
3. Run setup once:
   `bash setup.sh`
4. Train (the one command):
   `pilora-train --data data/prompt_injection_sft.jsonl --out runs/qwen05b-lora`
5. Evaluate base and tuned, then compare:
   ```
   pilora-eval --out reports/base
   pilora-eval --adapter runs/qwen05b-lora --out reports/lora
   pilora-eval --compare reports/base/report.json reports/lora/report.json --out reports/comparison.md
   ```
6. Copy `reports/` and `runs/qwen05b-lora/run_config.json` back, paste the
   comparison table into the README results section, commit.

## If the run OOMs

Halve `--per-device` pressure by raising gradient accumulation instead of
lowering the batch: add `--help` on pilora-train for the flags, or edit
`configs/base.yaml`. Dropping max-seq-length to 256 also helps; injection
texts are short.
