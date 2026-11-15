"""Benchmark RF-DETR inference on CPU and, when available, CUDA.

Example:
    python benchmark.py plan.jpg --device both --repeats 20 --output benchmark.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inference import DEFAULT_REPO_ID, FloorPlanDetector


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "CPU"


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def run_device(
    image: np.ndarray,
    *,
    device: str,
    repo_id: str,
    revision: str | None,
    model_dir: Path | None,
    threshold: float,
    warmups: int,
    repeats: int,
) -> dict:
    start = time.perf_counter()
    detector = FloorPlanDetector(repo_id, revision=revision, model_dir=model_dir, device=device)
    load_seconds = time.perf_counter() - start

    for _ in range(warmups):
        detector.model.predict(image, threshold=threshold)
    synchronize(device)

    samples_ms = []
    result = None
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        result = detector.model.predict(image, threshold=threshold)
        synchronize(device)
        samples_ms.append((time.perf_counter() - start) * 1000)

    median_ms = statistics.median(samples_ms)
    return {
        "device": device,
        "hardware": torch.cuda.get_device_name(0) if device == "cuda" else cpu_name(),
        "precision": "float32",
        "model_load_seconds": round(load_seconds, 3),
        "median_ms_per_image": round(median_ms, 2),
        "p95_ms_per_image": round(float(np.percentile(samples_ms, 95)), 2),
        "images_per_second_at_median": round(1000 / median_ms, 2),
        "detections": len(result.confidence),
        "samples_ms": [round(value, 2) for value in samples_ms],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", help="Hub commit SHA or tag to pin the checkpoint")
    parser.add_argument("--model-dir", type=Path, help="Use a locally downloaded model")
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="both")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.warmups < 1 or args.repeats < 2:
        parser.error("use at least one warm-up and two timed runs")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA GPU is unavailable on this machine")

    with Image.open(args.image) as source:
        image = np.array(source.convert("RGB"))
    devices = ("cpu", "cuda") if args.device == "both" else (args.device,)
    report = {
        "model": args.repo_id,
        "revision": args.revision or "main",
        "image": args.image.name,
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "input_resolution": 576,
        "threshold": args.threshold,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "results": {},
    }
    for device in devices:
        if device == "cuda" and not torch.cuda.is_available():
            report["results"][device] = {"status": "unavailable"}
            print("CUDA: unavailable on this machine")
            continue
        result = run_device(
            image,
            device=device,
            repo_id=args.repo_id,
            revision=args.revision,
            model_dir=args.model_dir,
            threshold=args.threshold,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        report["results"][device] = result
        print(f"{device.upper()}: {result['median_ms_per_image']} ms median, "
              f"{result['p95_ms_per_image']} ms p95, "
              f"{result['images_per_second_at_median']} images/s, "
              f"{result['detections']} detections")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
