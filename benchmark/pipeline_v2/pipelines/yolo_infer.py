"""YOLO floor-plan detectors exported to ONNX (models/yolo-weights): boxes per image -> <output>/<stem>.json.

Each model has a JSON spec next to it (<model>.json: classes, preprocessing, per-class thresholds). The
preprocessing is the spec's letterbox: scale = min(S/width, S/height), bilinear resize to
round(width*scale) x round(height*scale), pasted centred on an S x S canvas filled with 114, RGB 0..1.
NMS is inside the graph; rows with confidence 0 are padding. Every detection with confidence > 0 is
written (boxes mapped back to image pixels); the per-class thresholds are applied when converting
(pipelines/to_graph.py), so they can change without rerunning the model.

    python pipelines/yolo_infer.py work/input --output work/yolo_general --model models/yolo-weights/floorplan-yolo11m-1280.onnx
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger("yolo")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PAD_VALUE = 114


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO floor-plan detector (ONNX)")
    parser.add_argument("input", type=Path, help="folder of images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help=".onnx file; its spec is the .json next to it",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu, cuda or cuda:N (CUDA needs onnxruntime-gpu)",
    )
    return parser.parse_args()


def letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, float, int, int]:
    """(1x3xSxS float32 RGB 0..1, scale, left, top) exactly as in the model spec."""
    width, height = image.size
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    canvas = Image.new("RGB", (size, size), (PAD_VALUE, PAD_VALUE, PAD_VALUE))
    left, top = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(image.resize((new_w, new_h), Image.BILINEAR), (left, top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return array.transpose(2, 0, 1)[None], scale, left, top


def session(model: Path, device: str):
    import onnxruntime as ort

    if (
        device.startswith("cuda")
        and "CUDAExecutionProvider" in ort.get_available_providers()
    ):
        index = int(device.split(":")[1]) if ":" in device else 0
        providers = [
            ("CUDAExecutionProvider", {"device_id": index}),
            "CPUExecutionProvider",
        ]
    else:
        if device.startswith("cuda"):
            logger.warning(
                "CUDA requested but onnxruntime has no CUDAExecutionProvider: using CPU"
            )
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model), providers=providers)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    spec_path = args.model.with_suffix(".json")
    if not args.model.is_file() or not spec_path.is_file():
        raise FileNotFoundError(f"Model or spec not found: {args.model}, {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    size = spec["input"]["shape"][2]
    classes = spec["classes"]
    images = sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No images found in {args.input}")

    sess = session(args.model, args.device)
    input_name = spec["input"]["name"]
    args.output.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        tensor, scale, left, top = letterbox(image, size)
        rows = sess.run(None, {input_name: tensor})[0][0]
        detections = []
        for x1, y1, x2, y2, conf, cls in rows:
            if conf <= 0:
                continue  # padding
            box = [
                (x1 - left) / scale,
                (y1 - top) / scale,
                (x2 - left) / scale,
                (y2 - top) / scale,
            ]
            detections.append(
                {
                    "label": classes[int(cls)],
                    "score": round(float(conf), 4),
                    "box_xyxy": [
                        round(float(min(max(v, 0), limit)), 1)
                        for v, limit in zip(box, (width, height) * 2)
                    ],
                }
            )
        out = {
            "image": image_path.name,
            "width": width,
            "height": height,
            "model": args.model.name,
            "detections": detections,
        }
        out_path = args.output / f"{image_path.stem}.json"
        out_path.write_text(json.dumps(out), encoding="utf-8")
        logger.info(
            "%s -> %s (%d detections)", image_path.name, out_path, len(detections)
        )


if __name__ == "__main__":
    main()
