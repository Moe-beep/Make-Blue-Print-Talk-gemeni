"""Check that the CUDA setup is ready for Raster2Seq inference.

Verifies the GPU is visible to torch, the MultiScaleDeformableAttention op is compiled, and that the
CUDA op matches the pure-PyTorch reference implementation. Run from the repository root:

    python tools/check_cuda.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ops.functions import MSDA, MSDeformAttnFunction, ms_deform_attn_core_pytorch  # noqa: E402

ATOL = 1e-4
IM2COL_STEP = 2


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def check_op_matches_reference(device: torch.device) -> float:
    torch.manual_seed(0)
    batch, heads, channels, levels, points, queries = 2, 8, 32, 4, 4, 300
    shapes = torch.as_tensor([(32, 32), (16, 16), (8, 8), (4, 4)], dtype=torch.long, device=device)
    level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    length = int(shapes.prod(1).sum())

    value = torch.rand(batch, length, heads, channels, device=device)
    sampling_locations = torch.rand(batch, queries, heads, levels, points, 2, device=device)
    attention_weights = torch.rand(batch, queries, heads, levels, points, device=device) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)

    expected = ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights)
    actual = MSDeformAttnFunction.apply(
        value, shapes, level_start_index, sampling_locations, attention_weights, IM2COL_STEP
    )
    return (actual - expected).abs().max().item()


def main() -> None:
    print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
    if torch.version.cuda is None:
        fail("this is a CPU-only torch build; install a CUDA build (see README).")
    if not torch.cuda.is_available():
        fail("torch cannot see a GPU (check the NVIDIA driver / `nvidia-smi`, or `docker run --gpus all`).")

    device = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(device)
    print(f"[OK] GPU: {torch.cuda.get_device_name(device)} (compute capability {major}.{minor})")

    if MSDA is None:
        fail("MultiScaleDeformableAttention is not built: run `sh models/ops/make.sh`.")
    print("[OK] MultiScaleDeformableAttention CUDA op imported")

    max_error = check_op_matches_reference(device)
    if max_error > ATOL:
        fail(f"CUDA op differs from the PyTorch reference: max abs error {max_error:.2e} > {ATOL:.0e}")
    print(f"[OK] CUDA op matches the PyTorch reference (max abs error {max_error:.2e})")
    print("CUDA setup is ready.")


if __name__ == "__main__":
    main()
