"""Fill schema_campi facts from floor plans: raster images through seven models in parallel, vector PDFs by geometry.

Usage:
    python main.py path/to/plan.png
    python main.py path/to/plan.pdf                          # vector PDF: room/apartment areas, no model
    python main.py path/to/folder --output facts --device cuda
    python main.py plans/ --models layout plan2strct         # run only some models
    python main.py plans/ --raster2seq-profile r2g512        # local Raster2Seq.pth (rooms only, no doors/windows)
    python main.py plans/ --remote root@1.2.3.4              # run on a GPU VM over SSH (or set JUNCTION_REMOTE)

Images (PNG, JPG, ...) go to Raster2Seq, Raster-to-Graph, the CubiCasa5K layout model, 2DPlan2Strct and
YOLO26-Text + PaddleOCR (every word, room labels) and two YOLO detectors (ONNX, models/yolo-weights),
run in parallel. PDFs go to pipelines/pdf_plan.py, which reads rooms, labels, apartments and net
areas from the vector geometry and the printed scale.
Each model, as soon as it finishes, has its output converted to building-graph/1 (pipelines/to_graph.py,
keeping all its data) and the rules of planqa/campi_map.py fill schema_campi.json from it
(pipelines/campi_fill.py). Per input plan, in <output>/<stem>/:
  <model>.building_graph.json   everything the model produced, in the format the rules read
  <model>.campi.json            schema_campi.json filled by the rules from that one model (null = unknown)
  house.json                    everything in one file: the merged answer plus every model's campi and graph
  merged.campi.json             the models combined field by field, best model first (pipelines/merge_campi.py);
                                a PDF and an image with the same stem are one drawing: one merged file
Model logs and raw outputs stay in <work-dir>.
Override a repo's interpreter with RASTER2SEQ_PYTHON / RASTER2GRAPH_PYTHON / LAYOUT_PYTHON /
PLAN2STRCT_PYTHON / TEXT_OCR_PYTHON / YOLO_PYTHON,
the layout weights path with CUBICASA_WEIGHTS (default models/model_best_val_loss_var.pkl).
"""

import argparse
import json
import logging
import os
import re
import shutil
from pathlib import Path

import pypdfium2 as pdfium
from dotenv import load_dotenv

from pipelines.campi_fill import fill_campi
from pipelines.house import HOUSE_NAME, build_house
from pipelines.merge_campi import merge_campi
from pipelines.pdf_plan import PDF_PLAN, extract_pdf_plan
from pipelines.raster2seq_profiles import DEFAULT_PROFILE, PROFILES
from pipelines.remote import (
    DEFAULT_REMOTE_DIR,
    REMOTE_DIR_ENV,
    REMOTE_TARGET_ENV,
    SSH_KEY_ENV,
    remote_host,
    run_remote,
)
from pipelines.runners import (
    LAYOUT,
    MODEL_NAMES,
    PLAN2STRCT,
    RASTER2GRAPH,
    RASTER2SEQ,
    TEXT_OCR,
    YOLO_GENERAL,
    YOLO_MODELS,
    YOLO_SYMBOLS,
    JobResult,
    build_jobs,
    run_parallel,
)
from pipelines.schema_fill import load_schema
from pipelines.to_graph import (
    RASTER2GRAPH_LABELS,
    layout_graph,
    pdf_plan_graph,
    plan2strct_graph,
    raster2graph_graph,
    raster2seq_graph,
    text_ocr_graph,
    yolo_graph,
)
from planqa.building_schema import BuildingGraph

logger = logging.getLogger("main")

ENV_FILE = Path(__file__).resolve().parent / ".env"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PDF_EXTENSION = ".pdf"
MERGED_NAME = "merged.campi.json"


def collect_inputs(input_path: Path) -> tuple[list[Path], list[Path]]:
    """(images, pdfs) from one file or the top level of a folder."""
    if input_path.is_file():
        candidates = [input_path]
    elif input_path.is_dir():
        candidates = sorted(input_path.iterdir())
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")
    images = [p for p in candidates if p.suffix.lower() in IMAGE_EXTENSIONS]
    pdfs = [p for p in candidates if p.suffix.lower() == PDF_EXTENSION]
    return images, pdfs


def safe_stem(path: Path) -> str:
    """Stem without dots: Raster2Seq names outputs after the text before the first dot."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", path.stem)


def stage_images(images: list[Path], stage_dir: Path) -> dict[str, Path]:
    """Copy the images into one folder holding only images (Raster-to-Graph reads every file in it).

    Returns staged stem -> original path.
    """
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    staged = {}
    for image in images:
        stem = safe_stem(image)
        if stem in staged:
            raise ValueError(
                f"Two images map to the same name '{stem}': {staged[stem]} and {image}"
            )
        shutil.copy2(image, stage_dir / f"{stem}{image.suffix.lower()}")
        staged[stem] = image
    return staged


def write_json(data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_model_result(graph: BuildingGraph, model: str, schema: dict, plan_dir: Path) -> dict:
    """One model on one plan: its building graph and the schema_campi facts the rules derive from it."""
    doc = json.loads(graph.model_dump_json(exclude_none=True))
    campi = fill_campi(doc, schema, model)
    write_json(doc, plan_dir / f"{model}.building_graph.json")
    write_json(campi, plan_dir / f"{model}.campi.json")
    logger.info("[%s] %s -> %s (%d fields filled)", model, graph.source.file, plan_dir, len(campi["_meta"]["filled"]))
    return campi


def plan_entry(plans: dict, stem: str) -> dict:
    """Results of one drawing, collected over all its formats (same stem = same drawing, README)."""
    return plans.setdefault(stem, {"files": [], "models": {}, "runs": {}})


def write_merged(plans: dict, schema: dict, output: Path) -> None:
    """One merged.campi.json per drawing, from every model of every format of that drawing."""
    for stem, entry in plans.items():
        file = " + ".join(sorted(set(entry["files"])))
        merged = merge_campi(entry["models"], schema, file)
        merged["_meta"]["runs"] = entry["runs"]
        write_json(merged, output / stem / MERGED_NAME)
        logger.info("%s -> %s (from %s; %d uncertain)", file, output / stem / MERGED_NAME,
                    ", ".join(merged["_meta"]["models"]) or "no model", len(merged["_meta"]["uncertain_fields"]))
        write_house(output / stem, file, merged, entry["models"])


def write_house(plan_dir: Path, file: str, merged: dict, campi: dict[str, dict]) -> None:
    """house.json: the merged answer plus every model's complete output (graphs read back from disk)."""
    graphs = {}
    for model in campi:
        path = plan_dir / f"{model}.building_graph.json"
        if path.is_file():
            graphs[model] = json.loads(path.read_text(encoding="utf-8"))
    write_json(build_house(file, merged, campi, graphs), plan_dir / HOUSE_NAME)


def yolo_builder(name: str):
    """Graph builder of one YOLO model, with that model's JSON spec (classes, per-class thresholds)."""
    spec = json.loads(YOLO_MODELS[name].with_suffix(".json").read_text(encoding="utf-8"))
    return lambda out, stem, image: yolo_graph(out, stem, image, spec, name)


def process_images(images: list[Path], args: argparse.Namespace, schema: dict, plans: dict) -> list[str]:
    """Run the models in parallel on all images; return the failed models.

    As soon as one model finishes, each plan's output of that model is converted to a building graph
    and filled into schema_campi by the rules (<output>/<plan>/<model>.*.json). When all are done,
    <output>/<plan>/merged.campi.json combines the per-model files field by field (pipelines/merge_campi.py).
    """
    staged = stage_images(images, args.work_dir / "input")
    logger.info("%d image(s), models: %s", len(staged), ", ".join(args.models))

    profile = PROFILES[args.raster2seq_profile]
    jobs = build_jobs(args.work_dir / "input", args.work_dir, args.device, tuple(args.models), profile)
    output_dirs = {job.name: job.output_dir for job in jobs}
    graph_builders = {
        RASTER2SEQ: lambda out, stem, image: raster2seq_graph(out, stem, image, profile),
        RASTER2GRAPH: lambda out, stem, image: raster2graph_graph(out, stem, image, RASTER2GRAPH_LABELS),
        LAYOUT: layout_graph,
        PLAN2STRCT: plan2strct_graph,
        TEXT_OCR: text_ocr_graph,
        **{name: yolo_builder(name) for name in (YOLO_GENERAL, YOLO_SYMBOLS)},
    }
    for stem, original in staged.items():
        plan_entry(plans, stem)["files"].append(original.name)

    def convert(result: JobResult) -> None:
        if not result.ok:
            return
        for stem, original in staged.items():
            graph = graph_builders[result.name](output_dirs[result.name], stem, original)
            if graph is None:
                logger.warning("[%s] no output for %s", result.name, original.name)
                continue
            campi = write_model_result(graph, result.name, schema, args.output / stem)
            plans[stem]["models"][result.name] = campi

    results = run_parallel(jobs, args.work_dir / "logs", on_done=convert)
    runs = {
        name: {"ok": r.ok, "returncode": r.returncode, "seconds": round(r.seconds, 1), "log": str(r.log_path)}
        for name, r in results.items()
    }
    for stem in staged:
        plans[stem]["runs"].update(runs)
    return [name for name, r in results.items() if not r.ok]


def process_pdfs(pdfs: list[Path], args: argparse.Namespace, schema: dict, plans: dict) -> list[str]:
    """Read every PDF's geometry -> building graph -> schema_campi; return the PDFs that could not be read."""
    failed = []
    for pdf in pdfs:
        plan_dir = args.output / safe_stem(pdf)
        entry = plan_entry(plans, safe_stem(pdf))
        entry["files"].append(pdf.name)
        try:
            plan = extract_pdf_plan(pdf)
        except (OSError, ValueError, pdfium.PdfiumError) as error:
            logger.error("%s: %s", pdf.name, error)
            failed.append(pdf.name)
            entry["runs"][PDF_PLAN] = {"ok": False, "error": str(error)}
            continue
        entry["models"][PDF_PLAN] = write_model_result(pdf_plan_graph(plan, pdf), PDF_PLAN, schema, plan_dir)
        entry["runs"][PDF_PLAN] = {"ok": True}
    return failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input", type=Path, help="floor-plan image or PDF, or a folder of them"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("facts"),
        help="folder for the filled JSON files",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("work/raster_models"),
        help="raw model outputs and logs",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda, cuda:N or cpu (Raster-to-Graph needs CUDA)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
        help="models to run on images",
    )
    parser.add_argument(
        "--raster2seq-profile",
        choices=sorted(PROFILES),
        default=DEFAULT_PROFILE,
        help="r2g512 = local Raster2Seq.pth (rooms only); cubicasa = hf:cubicasa5k (rooms, doors, windows)",
    )
    parser.add_argument(
        "--remote",
        default=os.environ.get(REMOTE_TARGET_ENV),
        help=f"run on a GPU VM over SSH, e.g. root@1.2.3.4 (default: ${REMOTE_TARGET_ENV}; unset = run here)",
    )
    parser.add_argument(
        "--remote-dir",
        default=os.environ.get(REMOTE_DIR_ENV, DEFAULT_REMOTE_DIR),
        help=f"project folder on the VM, relative to the SSH home (default: ${REMOTE_DIR_ENV} or {DEFAULT_REMOTE_DIR})",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=f"run on this machine even if ${REMOTE_TARGET_ENV} is set",
    )
    return parser.parse_args()


def remote_main_args(args: argparse.Namespace) -> list[str]:
    """The options main.py gets on the VM (input and output are set by pipelines/remote.py)."""
    return [
        "--local",  # never hop again, even if the VM has its own JUNCTION_REMOTE
        "--device",
        args.device,
        "--models",
        *args.models,
        "--raster2seq-profile",
        args.raster2seq_profile,
    ]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    load_dotenv(
        ENV_FILE
    )  # JUNCTION_REMOTE etc.; variables already set in the shell win
    args = parse_args()

    images, pdfs = collect_inputs(args.input)
    if not images and not pdfs:
        raise ValueError(f"No images or PDFs found in {args.input}")
    if args.remote and not args.local:
        host = remote_host(args.remote, args.remote_dir, os.environ.get(SSH_KEY_ENV))
        run_remote(host, images + pdfs, args.output, remote_main_args(args))
        return
    schema = load_schema()
    args.output.mkdir(parents=True, exist_ok=True)

    plans: dict[str, dict] = {}  # stem -> files, per-model campi, runs (a PDF and a PNG of one drawing share it)
    failed_pdfs = process_pdfs(pdfs, args, schema, plans) if pdfs else []
    failed_models = process_images(images, args, schema, plans) if images else []
    write_merged(plans, schema, args.output)

    if failed_pdfs:
        logger.error("PDFs without results: %s", ", ".join(failed_pdfs))
    if failed_models:
        logger.error(
            "Failed models: %s - see logs in %s",
            ", ".join(failed_models),
            args.work_dir / "logs",
        )


if __name__ == "__main__":
    main()
