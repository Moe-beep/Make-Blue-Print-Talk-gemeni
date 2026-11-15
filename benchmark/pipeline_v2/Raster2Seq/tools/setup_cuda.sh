#!/usr/bin/env bash
# Set up Raster2Seq inference on a Linux machine with an NVIDIA GPU.
# Needs: NVIDIA driver, a CUDA toolkit (nvcc) matching CUDA_VERSION, python 3.10, run from the repo root.
#   CUDA_VERSION=cu121 bash tools/setup_cuda.sh   # default cu118
set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-cu118}"
TORCH_VERSION="${TORCH_VERSION:-2.3.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.18.1}"

cd "$(dirname "$0")/.."

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver first." >&2; exit 1; }
command -v nvcc >/dev/null || { echo "nvcc not found: install the CUDA toolkit (or set CUDA_HOME / PATH)." >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvcc --version | tail -1

pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
pip install -r requirements.txt

sh models/ops/make.sh
python tools/check_cuda.py
