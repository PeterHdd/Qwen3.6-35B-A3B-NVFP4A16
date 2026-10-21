#!/usr/bin/env bash
# One-time setup on a fresh 1x RTX PRO 6000 VM (Ubuntu + CUDA drivers preinstalled), using uv.
#
# Two separate environments, because llm-compressor (from main) and vLLM pin
# different versions of compressed-tensors and would overwrite each other:
#   ~/quant-env  -> quantize.py
#   ~/qenv       -> vllm serve + benchmark.sh
set -euo pipefail

# System packages: Python headers + compiler. Triton compiles a small C helper
# at runtime (used when packing NVFP4 weights) and needs Python.h for that.
sudo apt update
sudo apt install -y python3.12-dev build-essential tmux

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# ---- Quantization environment ----------------------------------------------
uv venv ~/quant-env --python 3.12
source ~/quant-env/bin/activate
# compressed-tensors comes from PyPI as a pre-release: llm-compressor main needs
# >=0.19.1a2, and the GitHub version labels itself lower than that.
uv pip install --prerelease=allow \
  "compressed-tensors>=0.19.1a2" \
  "git+https://github.com/vllm-project/llm-compressor.git" \
  "transformers>=5" datasets accelerate huggingface_hub
# Qwen3.6's processor imports torchvision (it handles images/video);
# --torch-backend=auto picks the build matching the installed torch/CUDA.
uv pip install torchvision --torch-backend=auto
deactivate

# ---- Serving + benchmarking environment ------------------------------------
uv venv ~/qenv --python 3.12
source ~/qenv/bin/activate
uv pip install -U vllm "lm-eval[api]" huggingface_hub

# Log in so you can download the model and later upload your build
hf auth login

# Pre-download the BF16 original (~67 GiB) into the shared Hugging Face cache
hf download Qwen/Qwen3.6-35B-A3B
deactivate

nvidia-smi
df -h ~
echo "Setup done."
echo "Quantize:  source ~/quant-env/bin/activate && python quantize.py"
echo "Benchmark: source ~/qenv/bin/activate && ./benchmark.sh ./Qwen3.6-35B-A3B-NVFP4A16"