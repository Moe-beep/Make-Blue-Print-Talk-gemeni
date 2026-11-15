"""Build and run the model inference scripts as parallel subprocesses.

Each repository keeps its own Python environment (different torch/CUDA builds), so every model
runs as a separate process with its repository as working directory. The interpreter for each
repo can be overridden with an environment variable (see PYTHON_ENV_VARS). The CubiCasa5K layout
model (planqa) runs from the project root with the project's interpreter; 2DPlan2Strct runs our
wrapper pipelines/plan2strct_infer.py with the interpreter of models/2dplan2strct/.venv.
"""

import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pipelines.raster2seq_profiles import Raster2SeqProfile

logger = logging.getLogger(__name__)

# checkpoints the model script downloads itself (no local file to check)
DOWNLOADED_PREFIXES = ("hf:", "ultralytics:")

ROOT = Path(__file__).resolve().parents[1]

RASTER2SEQ = "raster2seq"
RASTER2GRAPH = "raster2graph"
LAYOUT = "layout"
PLAN2STRCT = "plan2strct"
TEXT_OCR = "text_ocr"
YOLO_GENERAL = "yolo_general"
YOLO_SYMBOLS = "yolo_symbols"
MODEL_NAMES = (RASTER2SEQ, RASTER2GRAPH, LAYOUT, PLAN2STRCT, TEXT_OCR, YOLO_GENERAL, YOLO_SYMBOLS)

PLAN2STRCT_DIR = ROOT / "models" / "2dplan2strct"  # config.json, checkpoint and its own .venv
YOLO_DIR = ROOT / "models" / "yolo-weights"  # ONNX + JSON spec per model
YOLO_MODELS = {
    YOLO_GENERAL: YOLO_DIR / "floorplan-yolo11m-1280.onnx",  # 13 classes: symbols, room types, floors
    YOLO_SYMBOLS: YOLO_DIR / "floorplan-symbols-yolo11l-1280.onnx",  # symbols + doorway
}
TEXT_OCR_DIR = ROOT / "models" / "text_ocr"  # its .venv; YOLO26-Text + PP-OCRv6 weights download on first run

REPO_DIRS = {
    RASTER2SEQ: ROOT / "Raster2Seq",
    RASTER2GRAPH: ROOT / "Raster-to-Graph",
    LAYOUT: ROOT,
    PLAN2STRCT: PLAN2STRCT_DIR,
    TEXT_OCR: TEXT_OCR_DIR,
    YOLO_GENERAL: ROOT,
    YOLO_SYMBOLS: ROOT,
}
# Raster2Seq's checkpoint comes from its profile (pipelines/raster2seq_profiles.py)
CHECKPOINTS = {
    RASTER2GRAPH: ROOT / "models" / "graph_topology" / "graph.pth",
    # official CubiCasa5K checkpoint (link in the CubiCasa5k README), same env var as planqa/layout.py
    LAYOUT: Path(os.environ.get("CUBICASA_WEIGHTS", ROOT / "models" / "model_best_val_loss_var.pkl")),
    PLAN2STRCT: PLAN2STRCT_DIR / "checkpoint_best_ema.pth",
    **YOLO_MODELS,
}
PYTHON_ENV_VARS = {
    RASTER2SEQ: "RASTER2SEQ_PYTHON",
    RASTER2GRAPH: "RASTER2GRAPH_PYTHON",
    LAYOUT: "LAYOUT_PYTHON",
    PLAN2STRCT: "PLAN2STRCT_PYTHON",
    TEXT_OCR: "TEXT_OCR_PYTHON",
    YOLO_GENERAL: "YOLO_PYTHON",
    YOLO_SYMBOLS: "YOLO_PYTHON",
}
VENV_PYTHONS = (
    Path(".venv") / "Scripts" / "python.exe",
    Path(".venv") / "bin" / "python",
)


@dataclass(frozen=True)
class ModelJob:
    name: str
    repo_dir: Path
    command: tuple[str, ...]
    output_dir: Path
    checkpoint: str  # local path, or hf:<key> downloaded by the model script


@dataclass(frozen=True)
class JobResult:
    name: str
    returncode: int
    seconds: float
    log_path: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def python_for(name: str) -> str:
    """Interpreter for a repo: env var override, else the repo's .venv, else the current interpreter."""
    override = os.environ.get(PYTHON_ENV_VARS[name])
    if override:
        return override
    for venv_python in VENV_PYTHONS:
        candidate = REPO_DIRS[name] / venv_python
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def build_jobs(
    input_dir: Path,
    work_dir: Path,
    device: str,
    names: tuple[str, ...],
    raster2seq_profile: Raster2SeqProfile,
) -> list[ModelJob]:
    """Commands for the requested models; every path is absolute because each job runs in its repo dir."""
    input_dir = input_dir.resolve()
    work_dir = work_dir.resolve()
    checkpoints = {
        **{name: str(path) for name, path in CHECKPOINTS.items()},
        RASTER2SEQ: raster2seq_profile.checkpoint,
        TEXT_OCR: "ultralytics:yolo26x-text + PP-OCRv6",
    }
    builders = {
        RASTER2SEQ: lambda out: (
            "predict.py",
            f"--dataset_root={input_dir}",
            f"--checkpoint={checkpoints[RASTER2SEQ]}",
            f"--output_dir={out}",
            f"--device={device}",
            *raster2seq_profile.args,
        ),
        RASTER2GRAPH: lambda out: (
            "inference.py",
            f"--input_dir={input_dir}",
            f"--output_dir={out}",
            f"--resume={CHECKPOINTS[RASTER2GRAPH]}",
            f"--device={device}",
        ),
        LAYOUT: lambda out: (
            "-m",
            "pipelines.layout_infer",
            str(input_dir),
            f"--output={out}",
            f"--weights={CHECKPOINTS[LAYOUT]}",
            f"--device={device}",
        ),
        # our own wrapper, run by path with the model's interpreter (it needs rfdetr)
        PLAN2STRCT: lambda out: (
            str(ROOT / "pipelines" / "plan2strct_infer.py"),
            str(input_dir),
            f"--output={out}",
            f"--model-dir={PLAN2STRCT_DIR}",
            f"--device={device}",
        ),
        TEXT_OCR: lambda out: (
            str(ROOT / "pipelines" / "text_ocr_infer.py"),
            str(input_dir),
            f"--output={out}",
            f"--device={device}",
        ),
        **{
            name: (lambda out, model=model: (
                str(ROOT / "pipelines" / "yolo_infer.py"),
                str(input_dir),
                f"--output={out}",
                f"--model={model}",
                f"--device={device}",
            ))
            for name, model in YOLO_MODELS.items()
        },
    }
    jobs = []
    for name in names:
        output_dir = work_dir / name
        jobs.append(
            ModelJob(
                name=name,
                repo_dir=REPO_DIRS[name],
                command=(python_for(name), *builders[name](output_dir)),
                output_dir=output_dir,
                checkpoint=checkpoints[name],
            )
        )
    return jobs


def check_job_inputs(job: ModelJob) -> None:
    if not job.repo_dir.is_dir():
        raise FileNotFoundError(f"{job.name}: repository not found: {job.repo_dir}")
    if job.checkpoint.startswith(DOWNLOADED_PREFIXES):
        return
    if not Path(job.checkpoint).is_file():
        raise FileNotFoundError(f"{job.name}: checkpoint not found: {job.checkpoint}")


def run_job(job: ModelJob, log_dir: Path) -> JobResult:
    """Run one model to completion; stdout+stderr go to <log_dir>/<name>.log."""
    check_job_inputs(job)
    # start clean so a failed run can never be merged with an earlier run's outputs
    if job.output_dir.exists():
        shutil.rmtree(job.output_dir)
    job.output_dir.mkdir(parents=True)
    log_path = log_dir / f"{job.name}.log"
    logger.info("[%s] start: %s", job.name, " ".join(job.command))
    start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            job.command,
            cwd=job.repo_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    seconds = time.perf_counter() - start
    level = logging.INFO if completed.returncode == 0 else logging.ERROR
    logger.log(
        level,
        "[%s] exit %d after %.1f s (log: %s)",
        job.name,
        completed.returncode,
        seconds,
        log_path,
    )
    return JobResult(job.name, completed.returncode, seconds, log_path)


def run_parallel(
    jobs: list[ModelJob], log_dir: Path, on_done: Callable[[JobResult], None] | None = None
) -> dict[str, JobResult]:
    """Run all jobs at the same time. A failing model does not stop the others.

    on_done is called in this thread for each job as soon as it finishes (in finishing order), so its
    outputs can be processed while the other models are still running.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    with ThreadPoolExecutor(max_workers=max(len(jobs), 1)) as pool:
        futures = {pool.submit(run_job, job, log_dir): job.name for job in jobs}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except OSError as error:
                # missing repo, checkpoint or interpreter: report it and keep the other models' results
                logger.error("[%s] could not start: %s", name, error)
                results[name] = JobResult(name, -1, 0.0, log_dir / f"{name}.log")
            if on_done is not None:
                on_done(results[name])
    return {job.name: results[job.name] for job in jobs}
