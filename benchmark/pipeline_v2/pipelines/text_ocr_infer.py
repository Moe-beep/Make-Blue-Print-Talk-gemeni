"""YOLO26-Text (finds words) + PaddleOCR PP-OCRv6 (reads them), via the Ultralytics OCR solution.

Writes <output>/<stem>.json per image: {image, width, height, model, texts: [{text, confidence,
detection_confidence, line, polygon}]}, polygons in original-image pixels. Every word is kept; which
words are room labels is decided later (pipelines/to_graph.py), so the vocabulary can change without
rerunning the models.

Runs in its own env (models/text_ocr/.venv: ultralytics + onnxruntime, deploy/verda/setup.sh); the
YOLO26-Text and PP-OCRv6 weights download on the first run:
    python pipelines/text_ocr_infer.py work/input --output work/text_ocr --device cuda
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("text_ocr")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DETECTOR = "yolo26x-text.pt"  # largest YOLO26-Text: plan labels are short and small
RECOGNIZER = "paddleocr"  # PP-OCRv6 on ONNX Runtime (Ultralytics default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26-Text + PaddleOCR: every word on the plan"
    )
    parser.add_argument("input", type=Path, help="folder of images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or cuda:N")
    return parser.parse_args()


def main() -> None:
    import cv2
    from ultralytics import solutions

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    images = sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No images found in {args.input}")

    ocr = solutions.OCR(
        model=DETECTOR,
        recognizer=RECOGNIZER,
        device=args.device,
        show=False,
        verbose=False,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Skipping unreadable image: %s", image_path)
            continue
        height, width = image.shape[:2]
        result = ocr(image)
        texts = [
            {
                "text": t["text"],
                "confidence": float(t["confidence"]),
                "detection_confidence": float(t["detection_confidence"]),
                "line": int(t["line"]),
                "polygon": [[float(x), float(y)] for x, y in t["polygon"]],
            }
            for t in result.texts
        ]
        out = {
            "image": image_path.name,
            "width": width,
            "height": height,
            "model": f"{DETECTOR} + {RECOGNIZER}",
            "texts": texts,
        }
        out_path = args.output / f"{image_path.stem}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        logger.info("%s -> %s (%d words)", image_path.name, out_path, len(texts))


if __name__ == "__main__":
    main()
