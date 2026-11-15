"""2DPlan2Strct (RF-DETR Medium: wall, room, door, window boxes) as a standalone job.

Loads the local checkpoint once (no network) and writes <output>/<stem>.json per image:
{image, width, height, min_score, detections: [{label, score, box_xyxy}]}, boxes in original pixels.
Detections down to SAVE_MIN_SCORE are kept; the model's own threshold (0.35) is applied when merging
(pipelines/parsers.py), so it can be changed without rerunning the model.

Runs in the model's own environment (models/2dplan2strct/.venv, needs rfdetr 1.10):
    python pipelines/plan2strct_infer.py work/input --output work/plan2strct --model-dir models/2dplan2strct
Model terms: non-commercial research and evaluation only (models/2dplan2strct/USAGE_TERMS.md).
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger("plan2strct")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SAVE_MIN_SCORE = 0.2
SUPPORTED_VARIANT = "RFDETRMedium"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2DPlan2Strct: wall, room, door and window boxes"
    )
    parser.add_argument("input", type=Path, help="folder of images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="folder with config.json and the checkpoint",
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda or cuda:N")
    return parser.parse_args()


def load_model(model_dir: Path, device: str):
    from rfdetr import RFDETRMedium

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config["variant"] != SUPPORTED_VARIANT:
        raise ValueError(f"Unsupported model variant: {config['variant']}")
    weights = model_dir / config["checkpoint"]
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    model = RFDETRMedium(
        pretrain_weights=str(weights),
        resolution=config["resolution"],
        num_classes=config["num_classes"],
        device=device,
    )
    return model, config["class_names"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    images = sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No images found in {args.input}")

    model, class_names = load_model(args.model_dir, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        with Image.open(image_path) as source:
            image = np.array(source.convert("RGB"))
        height, width = image.shape[:2]
        result = model.predict(image, threshold=SAVE_MIN_SCORE)
        detections = [
            {
                "label": class_names[int(class_id)],
                "score": float(score),
                "box_xyxy": [float(v) for v in box],
            }
            for box, score, class_id in zip(
                result.xyxy, result.confidence, result.class_id
            )
        ]
        detections.sort(key=lambda d: d["score"], reverse=True)
        out = {
            "image": image_path.name,
            "width": width,
            "height": height,
            "min_score": SAVE_MIN_SCORE,
            "detections": detections,
        }
        out_path = args.output / f"{image_path.stem}.json"
        out_path.write_text(json.dumps(out), encoding="utf-8")
        logger.info(
            "%s -> %s (%d detections)", image_path.name, out_path, len(detections)
        )


if __name__ == "__main__":
    main()
