"""Image -> plan graph (+ report). Models are loaded once and reused for many plans."""
from __future__ import annotations

import copy
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .build_graph import build_graph
from .ocr import PaddleReader, group_words, load_image, read_text
from .schemas import PlanGraph
from .visualize import html_report, layout_blend


class Timer:
    """Wall-clock seconds per coarse step."""

    def __init__(self):
        self.steps, self.t0 = {}, time.perf_counter()

    @contextmanager
    def __call__(self, name):
        t = time.perf_counter()
        yield
        self.steps[name] = self.steps.get(name, 0) + time.perf_counter() - t

    def summary(self) -> dict:
        return {**{k: round(v, 2) for k, v in self.steps.items()}, "total": round(time.perf_counter() - self.t0, 2)}


PROB_LEVELS = 255   # raw maps are stored as uint8 probabilities (x/255): 4x smaller than float32, enough to re-threshold


def save_layout_maps(path: Path, room_prob, icon_prob, heatmaps=None) -> None:
    """All network outputs (rooms, icons, junction heatmaps) so they can be re-used without the model.
    Load with np.load(path): arrays room_prob [12,H,W], icon_prob [11,H,W], heatmaps [21,h,w] (working size)."""
    arrays = {"room_prob": room_prob, "icon_prob": icon_prob}
    if heatmaps is not None:
        arrays["heatmaps"] = heatmaps
    np.savez_compressed(path, **{k: np.round(np.clip(v, 0, 1) * PROB_LEVELS).astype(np.uint8) for k, v in arrays.items()})


def graph_from_detections(det: dict, min_conf=None) -> PlanGraph:
    """Build the graph from raw model output with given thresholds (no model calls, milliseconds)."""
    return build_graph(det["source"], det["page"], det["size"], det["words"], copy.deepcopy(det["rooms"]),
                       min_conf, copy.deepcopy(det["objects"]), det["floors"], det.get("walls"))


class Pipeline:
    """min_conf: one number or {texts, rooms, objects}; detections below are dropped from the graph
    (default planqa.build_graph.DEFAULT_MIN_CONF).
    ocr_model: small | tiny | medium | auto; ocr_scale: long side (px) small images are upscaled to
    for OCR; threads: CPU threads for OCR and layout (default all cores)."""

    def __init__(self, ocr_lang=None, enhance="sharpen", min_conf=None, layout=True, tta=True,
                 layout_max_side=1600, weights="models/model_best_val_loss_var.pkl",
                 ocr_model="small", ocr_scale=1600, threads=None, textline_orientation=False):
        self.enhance, self.min_conf, self.tta, self.layout_max_side = enhance, min_conf, tta, layout_max_side
        self.ocr_scale = ocr_scale
        self.load_times = Timer()
        with self.load_times("load OCR model"):
            self.reader = PaddleReader(lang=ocr_lang, model=ocr_model, threads=threads,
                                       textline_orientation=textline_orientation)
        self.layout = None
        if layout:
            try:
                from .layout import LayoutModel
                with self.load_times("load layout model"):
                    self.layout = LayoutModel(weights, threads=threads)
            except (ImportError, FileNotFoundError) as e:
                print(f"layout model unavailable, continuing without rooms/objects: {e}")

    def process(self, path: Path, out_dir: Path, page: int = 0, report: bool = True) -> tuple[PlanGraph, dict]:
        from .layout import (floors_from_segmentation, objects_from_segmentation, rooms_from_segmentation,
                             walls_from_segmentation)

        t = Timer()
        out_dir.mkdir(parents=True, exist_ok=True)
        img = load_image(path, page)
        with t("OCR"):
            words = group_words(read_text(img, self.reader, self.enhance, self.ocr_scale))
        rooms = objects = floors = walls = room_prob = None
        self.last_layout_img = None
        if self.layout:
            with t("layout"):
                room_prob, icon_prob = self.layout.segment(img, self.layout_max_side, tta=self.tta)
                rooms = rooms_from_segmentation(room_prob, icon_prob)
                objects = objects_from_segmentation(icon_prob)
                floors = floors_from_segmentation(room_prob)
                walls = walls_from_segmentation(room_prob)
            with t("save maps"):
                save_layout_maps(out_dir / "layout_maps.npz", room_prob, icon_prob,
                                 getattr(self.layout, "last_heatmaps", None))
        with t("graph + report"):
            det = {"source": path.name, "page": page, "size": img.size, "words": words, "rooms": rooms,
                   "objects": objects, "floors": floors, "walls": walls}
            self.last_detections = det   # raw output of this run, e.g. for answer.py --sweep
            graph = graph_from_detections(det, self.min_conf)
            (out_dir / "plan_graph.json").write_text(graph.model_dump_json(indent=2), encoding="utf-8")
            from .plan_to_graph import to_building_graph   # common schema (same as IFC), what the agent reads
            (out_dir / "building_graph.json").write_text(
                to_building_graph(graph).model_dump_json(indent=1, exclude_none=True), encoding="utf-8")
            self.last_layout_img = layout_blend(img, room_prob) if room_prob is not None else None
            if report:
                html_report(img, graph, out_dir / "report.html", layout_img=self.last_layout_img)
        timing = t.summary()
        (out_dir / "timings.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        return graph, timing
