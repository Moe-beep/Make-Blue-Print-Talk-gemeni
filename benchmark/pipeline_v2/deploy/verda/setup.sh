#!/usr/bin/env bash
# Set up main.py on a Verda GPU VM (Ubuntu 24.04 + CUDA 12.x image). Run from the project root:
#   bash deploy/verda/setup.sh
# Creates one .venv per model (each needs its own torch build, see pipelines/runners.py),
# builds the two deformable-attention CUDA ops and checks that every env sees the GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REQUIRED_NVCC_MAJOR=12   # Raster2Seq's torch 2.3.1+cu121 op build needs nvcc 12.x

# torch builds per environment (all available for Python 3.10-3.12 on Linux x86_64)
RASTER2SEQ_TORCH=(torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121)
RASTER2GRAPH_TORCH=(torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126)
PLAN2STRCT_TORCH=(torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126)
TEXT_OCR_TORCH=(torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126)
ROOT_TORCH=(torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126)

fail() { echo "ERROR: $*" >&2; exit 1; }

check_system() {
    command -v nvidia-smi >/dev/null || fail "nvidia-smi not found: use a Verda image with the NVIDIA driver."
    export PATH="/usr/local/cuda/bin:$PATH"
    command -v nvcc >/dev/null || fail "nvcc not found: pick an 'Ubuntu 24.04 + CUDA 12.x' image."
    local nvcc_major
    nvcc_major="$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
    [ "$nvcc_major" = "$REQUIRED_NVCC_MAJOR" ] \
        || fail "nvcc $nvcc_major.x found, need $REQUIRED_NVCC_MAJOR.x: pick an 'Ubuntu 24.04 + CUDA 12.x' image."
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
    nvcc --version | tail -1
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
}

install_system_packages() {
    local python_minor
    python_minor="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    local sudo=""
    [ "$(id -u)" -eq 0 ] || sudo="sudo"
    $sudo apt-get update -q
    $sudo apt-get install -yq "python${python_minor}-venv" "python${python_minor}-dev" build-essential libgl1 libglib2.0-0
}

# new_venv <dir>: fresh .venv in <dir>, prints its python path
new_venv() {
    rm -rf "$1/.venv"
    python3 -m venv "$1/.venv"
    "$1/.venv/bin/python" -m pip install -q --upgrade pip setuptools wheel ninja >&2
    echo "$1/.venv/bin/python"
}

check_gpu() {
    "$1" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; \
print('  torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"
}

setup_raster2seq() {
    echo "== Raster2Seq"
    local py; py="$(new_venv "$ROOT/Raster2Seq")"
    "$py" -m pip install -q "${RASTER2SEQ_TORCH[@]}"
    "$py" -m pip install -q -r "$ROOT/Raster2Seq/requirements.txt"
    (cd "$ROOT/Raster2Seq/models/ops" && "$py" -m pip install -q --no-build-isolation .)
    check_gpu "$py"
    (cd "$ROOT/Raster2Seq" && "$py" tools/check_cuda.py)
}

setup_raster2graph() {
    echo "== Raster-to-Graph"
    local py; py="$(new_venv "$ROOT/Raster-to-Graph")"
    "$py" -m pip install -q "${RASTER2GRAPH_TORCH[@]}"
    "$py" -m pip install -q -r "$ROOT/Raster-to-Graph/requirements.txt"
    (cd "$ROOT/Raster-to-Graph/models/ops" && "$py" -m pip install -q --no-build-isolation .)
    check_gpu "$py"
}

setup_plan2strct() {
    echo "== 2DPlan2Strct (RF-DETR)"
    local py; py="$(new_venv "$ROOT/models/2dplan2strct")"
    "$py" -m pip install -q "${PLAN2STRCT_TORCH[@]}"
    "$py" -m pip install -q -r "$ROOT/models/2dplan2strct/requirements.txt"
    check_gpu "$py"
}

setup_root() {
    echo "== project root (main.py, CubiCasa5K layout model, YOLO detectors)"
    local py; py="$(new_venv "$ROOT")"
    "$py" -m pip install -q "${ROOT_TORCH[@]}"
    # onnxruntime: the two YOLO detectors (models/yolo-weights/*.onnx) run in this env
    "$py" -m pip install -q numpy opencv-python-headless pillow "pydantic>=2" pypdfium2 python-dotenv \n        "onnxruntime-gpu[cuda]>=1.26,<1.27"
    check_gpu "$py"
}

setup_text_ocr() {
    echo "== YOLO26-Text + PaddleOCR (Ultralytics OCR solution)"
    mkdir -p "$ROOT/models/text_ocr"
    local py; py="$(new_venv "$ROOT/models/text_ocr")"
    "$py" -m pip install -q "${TEXT_OCR_TORCH[@]}"
    # recognizer PP-OCRv6 runs on ONNX Runtime (>= 1.26 on Python 3.11+); weights download on first run
    # onnxruntime-gpu 1.26.x with its CUDA 12 wheels: newer builds look for CUDA 13 libs and fall back to CPU
    "$py" -m pip install -q ultralytics "onnxruntime-gpu[cuda]>=1.26,<1.27" opencv-python-headless
    "$py" -c "import ultralytics; print('  ultralytics', ultralytics.__version__)"
    check_gpu "$py"
}

main() {
    check_system
    install_system_packages
    setup_raster2seq
    setup_raster2graph
    setup_plan2strct
    setup_text_ocr
    setup_root
    echo "Done. Run:  .venv/bin/python main.py <images> --device cuda"
}

main "$@"
