"""Image loading, OCR preprocessing and PaddleOCR text reading.

All boxes are returned in ORIGINAL image pixels, so evidence shown later is
never the enhanced image.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 300_000_000  # large plan scans are legitimate


@dataclass
class OcrWord:
    text: str
    box: list[float]          # [x0, y0, x1, y1] original px
    conf: float
    vertical: bool = False    # rotated by ~90 degrees


# ------------------------------------------------------------------ image

def load_image(path: str | Path, page: int = 0, pdf_dpi: int = 200) -> Image.Image:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium
        return pdfium.PdfDocument(str(path))[page].render(scale=pdf_dpi / 72).to_pil().convert("RGB")
    return Image.open(path).convert("RGB")


def enhance(img: Image.Image, method: str = "sharpen", target_long_side: int = 2000) -> tuple[Image.Image, float]:
    """OCR input only. Returns (image, scale). Never downscales.
    lanczos: upscale. sharpen: upscale + denoise + local contrast + unsharp mask."""
    if method == "none":
        return img, 1.0
    up = max(1.0, target_long_side / max(img.size))
    rgb = np.asarray(img.convert("RGB"))
    big = cv2.resize(rgb, None, fx=up, fy=up, interpolation=cv2.INTER_LANCZOS4) if up > 1 else rgb
    if method == "sharpen":
        g = cv2.cvtColor(big, cv2.COLOR_RGB2GRAY)
        g = cv2.fastNlMeansDenoising(g, None, h=7, templateWindowSize=7, searchWindowSize=21)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        g = cv2.addWeighted(g, 1.7, cv2.GaussianBlur(g, (0, 0), sigmaX=up), -0.7, 0)
        big = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(big), up


# ------------------------------------------------------------------ PaddleOCR

class PaddleReader:
    """PaddleOCR 3.x.

    model: 'small' (default), 'tiny', 'medium' = PP-OCRv6 detector+recognizer of that size
           (multilingual incl. latin scripts); 'auto' = PaddleOCR's choice for `lang`.
    lang:  language code for model='auto': 'de' (latin), 'en', 'fr', 'ru', ... or '' (server model).
    threads: CPU threads for this reader (default: all cores).
    textline_orientation: extra classifier per text line (0/180 deg only); off = faster.
    """

    def __init__(self, lang: Optional[str] = None, model: str = "small", threads: Optional[int] = None,
                 textline_orientation: bool = False):
        from paddleocr import PaddleOCR

        kw = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=textline_orientation,
                  # tuned for small, faint, dense text in drawings
                  text_det_thresh=0.2, text_det_box_thresh=0.45, text_det_unclip_ratio=2.0,
                  text_det_limit_type="max", text_det_limit_side_len=4000,
                  cpu_threads=threads or os.cpu_count() or 4,
                  # oneDNN crashes on CPU with PaddlePaddle >= 3.3; PLANQA_OCR_MKLDNN=1 re-enables it (3.2.x)
                  enable_mkldnn=os.environ.get("PLANQA_OCR_MKLDNN", "0") == "1")
        if model == "auto":
            lang = os.environ.get("PLANQA_OCR_LANG", "de") if lang is None else lang
            if lang:
                kw["lang"] = lang
            self.name = f"auto(lang={lang or 'server'})"
        else:
            kw.update(text_detection_model_name=f"PP-OCRv6_{model}_det", text_recognition_model_name=f"PP-OCRv6_{model}_rec")
            self.name = f"PP-OCRv6_{model}"
        self.engine = PaddleOCR(**kw)

    def read(self, img: Image.Image) -> list[tuple[str, float, list]]:
        """(text, score, polygon) in pixels of `img`."""
        arr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()  # BGR
        out = []
        for res in self.engine.predict(arr):
            try:
                res["rec_texts"]
            except (KeyError, TypeError):  # older result objects expose the dict via .json
                res = res.json.get("res", res.json)
            for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
                if text.strip():
                    out.append((text.strip(), float(score), [list(map(float, p)) for p in poly]))
        return out


def read_text(img: Image.Image, reader: PaddleReader, enhance_method: str = "sharpen", scale_to: int = 1600,
              tile_px: int = 3000, overlap: int = 300) -> list[OcrWord]:
    """Enhance (upscale small images so the long side is scale_to px), OCR (tiled with
    overlap if very large), map boxes back to original px. OCR time grows with pixel count."""
    work, up = enhance(img, enhance_method, scale_to)
    W, H = work.size
    xs = list(range(0, max(1, W - overlap), tile_px - overlap)) if W > tile_px else [0]
    ys = list(range(0, max(1, H - overlap), tile_px - overlap)) if H > tile_px else [0]

    words = []
    for y0 in ys:
        for x0 in xs:
            tile = work.crop((x0, y0, min(W, x0 + tile_px), min(H, y0 + tile_px)))
            for text, score, poly in reader.read(tile):
                px, py = [p[0] for p in poly], [p[1] for p in poly]
                box = [(x0 + min(px)) / up, (y0 + min(py)) / up, (x0 + max(px)) / up, (y0 + max(py)) / up]
                w, h = box[2] - box[0], box[3] - box[1]
                words.append(OcrWord(text, box, score, vertical=h > 1.5 * w and len(text) > 1))
    return _dedupe(words) if len(xs) * len(ys) > 1 else words


def _overlap(a, b) -> float:
    """Intersection over the smaller box."""
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def _dedupe(words: list[OcrWord]) -> list[OcrWord]:
    kept: list[OcrWord] = []
    for w in sorted(words, key=lambda w: w.conf, reverse=True):
        if not any(_overlap(w.box, k.box) > 0.6 for k in kept):
            kept.append(w)
    return kept


def group_words(words: list[OcrWord]) -> list[OcrWord]:
    """Join fragments of one horizontal line ('5,8' + 'm2' -> '5,8 m2'): vertical
    overlap > 50 %, similar height, gap < 0.8 x height. Stacked texts stay separate."""
    parent = list(range(len(words)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(words):
        for j in range(i + 1, len(words)):
            b = words[j]
            if a.vertical or b.vertical:
                continue
            ha, hb = a.box[3] - a.box[1], b.box[3] - b.box[1]
            h = min(ha, hb)
            if h <= 0 or max(ha, hb) / h > 1.6:
                continue
            v_overlap = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
            gap = max(b.box[0] - a.box[2], a.box[0] - b.box[2])
            if v_overlap > 0.5 * h and gap < 0.8 * h:
                parent[root(i)] = root(j)

    groups: dict[int, list[OcrWord]] = {}
    for i, w in enumerate(words):
        groups.setdefault(root(i), []).append(w)
    out = []
    for g in groups.values():
        g.sort(key=lambda w: w.box[0])
        out.append(g[0] if len(g) == 1 else OcrWord(
            " ".join(w.text for w in g),
            [min(w.box[0] for w in g), min(w.box[1] for w in g), max(w.box[2] for w in g), max(w.box[3] for w in g)],
            min(w.conf for w in g)))  # a group is only as sure as its weakest word
    return out
