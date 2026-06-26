#!/bin/bash
set -e

echo "=== DFlash Worker Setup ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

# Upgrade vLLM to 0.23.0+
echo ">>> Upgrading vLLM..."
pip3 install --upgrade "vllm>=0.23.0" 2>&1 | tail -5

echo ">>> vLLM version: $(python3 -c 'import vllm; print(vllm.__version__)')"

# Download Qwen3.5-27B (main model)
echo ">>> Downloading Qwen3.5-27B..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-27B', ignore_patterns=['*.gguf'])
print('Qwen3.5-27B downloaded')
"

# Download DFlash draft model
echo ">>> Downloading z-lab/Qwen3.5-27B-DFlash..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('z-lab/Qwen3.5-27B-DFlash')
print('DFlash draft model downloaded')
"

echo ">>> Setup complete: $(date)"
echo ">>> Disk: $(df -h / | tail -1)"
