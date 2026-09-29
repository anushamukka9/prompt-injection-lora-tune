#!/usr/bin/env bash
# One-time GPU machine setup for prompt-injection-lora-tune.
# Tested on CUDA 12.1 images (vast.ai / runpod pytorch templates).
# After this, training is a single command (see README).
set -euo pipefail

python3 --version
nvidia-smi --query-gpu=name,memory.total --format=csv

pip install --upgrade pip
# CUDA 12.1 torch build; adjust the index URL if your image uses CUDA 11.8/12.4
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Project + GPU extras (transformers, peft, trl, datasets, accelerate)
pip install -e ".[gpu]"

# Single-GPU accelerate config (no interactive questions)
mkdir -p ~/.cache/huggingface/accelerate
cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "Setup complete. Verify with: python -c 'import torch; print(torch.cuda.is_available())'"
echo "Then build data and train:"
echo "  pilora-build-data --out data/prompt_injection.jsonl"
echo "  pilora-train --data data/prompt_injection_sft.jsonl --out runs/qwen05b-lora"
