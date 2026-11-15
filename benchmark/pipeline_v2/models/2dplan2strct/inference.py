"""Run the 2D floor-plan component detector from Hugging Face Hub or local files.

Example:
    python inference.py plan.png --output-dir results/plan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from rfdetr import RFDETRMedium


DEFAULT_REPO_ID = "OsamaMo/2dplan2strct"
COLORS = {
    "wall": "#e63946",
    "room": "#457b9d",
    "door": "#f4a261",
    "window": "#2a9d8f",
}


class FloorPlanDetector:
    """Load RF-DETR once, then run it on one or more floor-plan images."""

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        *,
        revision: str | None = None,
        model_dir: Path | None = None,
        device: str = "auto",
    ) -> None:
        self.repo_id = repo_id
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA GPU is available")
        if model_dir is None:
            config_path = Path(hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            weights = hf_hub_download(repo_id=repo_id, filename=config["checkpoint"], revision=revision)
        else:
            model_dir = Path(model_dir)
            config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
            weights = str(model_dir / config["checkpoint"])

        if config["variant"] != "RFDETRMedium":
            raise ValueError(f"Unsupported model variant: {config['variant']}")
        self.class_names = config["class_names"]
        self.model = RFDETRMedium(
            pretrain_weights=weights,
            resolution=config["resolution"],
            num_classes=config["num_classes"],
            device=self.device,
        )
        if self.model.class_names and self.model.class_names != self.class_names:
            raise ValueError("Checkpoint class names do not match config.json")

    def predict(self, image_path: str | Path, *, threshold: float = 0.35) -> dict:
        """Return boxes in original-image pixel coordinates, sorted by confidence."""
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        image_path = Path(image_path)
        with Image.open(image_path) as source:
            image = np.array(source.convert("RGB"))
        height, width = image.shape[:2]
        result = self.model.predict(image, threshold=threshold)
        detections = [
            {
                "label": self.class_names[int(class_id)],
                "score": float(score),
                "box_xyxy": [float(value) for value in box],
            }
            for box, score, class_id in zip(result.xyxy, result.confidence, result.class_id)
        ]
        detections.sort(key=lambda item: item["score"], reverse=True)
        return {
            "model": self.repo_id,
            "image": str(image_path),
            "width": width,
            "height": height,
            "threshold": threshold,
            "detections": detections,
        }


def save_prediction(prediction: dict, image_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    """Write a JSON result and an annotated PNG without changing the source image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "predictions.json"
    image_out = output_dir / "annotated.png"
    json_path.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, min(image.size) // 300)
    for item in reversed(prediction["detections"]):
        color = COLORS.get(item["label"], "#ffffff")
        box = item["box_xyxy"]
        draw.rectangle(box, outline=color, width=line_width)
        label = f"{item['label']} {item['score']:.2f}"
        text_box = draw.textbbox((box[0], box[1]), label)
        text_height = text_box[3] - text_box[1]
        label_y = max(0, box[1] - text_height - 4)
        draw.rectangle((box[0], label_y, box[0] + text_box[2] - text_box[0] + 4, label_y + text_height + 4), fill=color)
        draw.text((box[0] + 2, label_y + 2), label, fill="white")
    image.save(image_out)
    return json_path, image_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="PNG or JPEG floor-plan image")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face model repository")
    parser.add_argument("--revision", help="Optional Hub commit SHA or tag for reproducible downloads")
    parser.add_argument("--model-dir", type=Path, help="Use previously downloaded config and weights offline")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Run on CPU or CUDA; auto uses CUDA when available")
    parser.add_argument("--threshold", type=float, default=0.35, help="Confidence threshold from 0 to 1")
    parser.add_argument("--output-dir", type=Path, default=Path("output/prediction"))
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    detector = FloorPlanDetector(args.repo_id, revision=args.revision, model_dir=args.model_dir,
                                 device=args.device)
    prediction = detector.predict(args.image, threshold=args.threshold)
    json_path, image_path = save_prediction(prediction, args.image, args.output_dir)
    print(f"{len(prediction['detections'])} detections")
    print(f"JSON: {json_path}")
    print(f"Annotated image: {image_path}")


if __name__ == "__main__":
    main()
