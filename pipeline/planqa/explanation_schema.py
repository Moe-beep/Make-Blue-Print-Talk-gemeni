"""explanation.json ("explanation/1"): everything the frontend needs to explain answers.

One file per document (floor-plan image or IFC model), built from data that already exists:

  source / elements   <- plan_graph.json (rooms, texts, objects) + raw detections incl. those
                         below the thresholds (status below_threshold) | ifc_graph.json (guid, storey)
  hotspots            <- layout-model probabilities (pixel entropy) -> regions -> element ids
  blind_spots         <- heuristics: wall openings without a door symbol, rooms without a label,
                         IFC elements with missing properties (ifc_graph notes / null columns)
  answers[].sources   <- rules (planqa/qa.py), LLM agent (planqa/agent.py), Gemini Decisions API
  answers[].evidence  <- evidence ids of rules + agent, Decisions checks; counterfactual =
                         rules re-run without that element (leave-one-out, milliseconds)
  answers[].steps     <- agent trace (tool, args, result) + routing + Decisions calls
  answers[].coverage  <- ids found in the agent's tool results, per element kind
  answers[].uncertainty <- Decisions choice distribution, or Poisson-binomial over element
                         confidences (counted / below threshold / checked)
  calibration         <- benchmark run (predictions.jsonl): stated confidence vs. accuracy

Ids are the ids of plan_graph.json / ifc_graph.json (r*, t*, o*, S*, W*, D*, ...), so the
frontend can link every explanation to the drawing (bbox / polygon) or the 3D model (guid).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "explanation/1"

# ----------------------------------------------------------------------------- shared


class Geometry(BaseModel):
    """Where an element is: pixels for drawings, guid (+ storey) for IFC models."""
    bbox: Optional[list[float]] = Field(None, description="[x0, y0, x1, y1] in image pixels")
    polygon: Optional[list[list[float]]] = Field(None, description="outline in image pixels")
    guid: Optional[str] = Field(None, description="IFC GlobalId (3D viewer selection)")
    storey: Optional[str] = Field(None, description="storey id (L0, ...) for IFC")
    centroid: Optional[list[float]] = None


class ClassProb(BaseModel):
    label: str
    p: float


class Check(BaseModel):
    """Independent verification of one element (e.g. Gemini Decisions API on an image crop)."""
    by: Literal["decision", "ocr", "rule", "human"] = "decision"
    question: str = Field(description="what was asked, e.g. 'Ist das ein Türsymbol?'")
    kind: Literal["yes_no", "choice", "score"] = "yes_no"
    p_yes: Optional[float] = Field(None, description="yes_no: probability of yes")
    probabilities: list[ClassProb] = Field(default_factory=list, description="choice / score levels")
    verdict: Literal["confirmed", "rejected", "unclear"]
    crop: Optional[str] = Field(None, description="path/URL of the image crop that was checked")


class Element(BaseModel):
    """One thing on the drawing / in the model, including candidates the pipeline did not keep."""
    id: str
    kind: str = Field(description="room | text | door | window | toilet | sink | ... | wall | space")
    label: Optional[str] = Field(None, description="human readable, e.g. 'Tür MH1 ↔ ET', 'KH (Bad)'")
    geometry: Geometry
    conf: Optional[float] = Field(None, description="confidence of the producing model (layout / OCR)")
    class_probs: list[ClassProb] = Field(default_factory=list, description="top classes of the layout model")
    status: Literal["kept", "below_threshold", "rejected", "confirmed"] = Field(
        description="kept = above threshold; below_threshold = dropped from plan_graph; "
                    "confirmed / rejected = after an independent check")
    room: Optional[str] = Field(None, description="containing room / space id")
    checks: list[Check] = Field(default_factory=list)


# ----------------------------------------------------------------------------- document level


class Hotspot(BaseModel):
    """A region where the layout model is unsure (high entropy), linked to elements."""
    id: str
    rank: int
    entropy: float = Field(description="0..1, normalised mean entropy of the region")
    geometry: Geometry
    channel: Literal["rooms", "symbols", "text"]
    top_classes: list[ClassProb]
    elements: list[str] = Field(default_factory=list)
    resolution: Optional[str] = Field(None, description="how it was resolved, e.g. 'Gemini Decision: Tür 0,94'")
    resolved_by: Literal["decision", "ocr_label", "rule", "unresolved"] = "unresolved"


class BlindSpot(BaseModel):
    """Something the pipeline cannot see, so no answer can use it."""
    id: str
    geometry: Geometry
    reason: str = Field(description="e.g. 'Wandöffnung ohne Tür-Detektion'")
    suggestion: Optional[str] = None
    heuristic: str = Field(description="which rule found it (wall_gap_without_door, room_without_label, ...)")


class Assets(BaseModel):
    plan_image: Optional[str] = None
    uncertainty_images: dict[str, str] = Field(default_factory=dict,
                                               description="channel (combined, rooms, symbols, text) -> PNG file")
    layout_image: Optional[str] = None
    uncertainty_image: Optional[str] = Field(None, description="entropy heat map (PNG, same size as the plan)")
    uncertainty_share: Optional[dict[str, float]] = Field(
        None, description="share of the uncertain area per channel, e.g. {symbols: 0.54, rooms: 0.38, text: 0.08}")
    uncertain_area: Optional[float] = Field(None, description="fraction of the plan above the entropy threshold")


class Source(BaseModel):
    file: str
    kind: Literal["plan", "ifc"]
    page: Optional[int] = None
    image_size: Optional[list[int]] = None
    graph_file: str = Field(description="plan_graph.json or ifc_graph.json this was built from")


class Run(BaseModel):
    created: str
    models: dict[str, str] = Field(description="ocr, layout, agent, decision -> model id")
    thresholds: dict[str, float] = Field(default_factory=dict)


# ----------------------------------------------------------------------------- per answer


class SourceAnswer(BaseModel):
    value: Optional[float] = None
    text: Optional[str] = None
    model: Optional[str] = None
    explanation: Optional[str] = None
    distribution: Optional[dict[str, float]] = Field(None, description="Decisions: value -> probability")


class Uncertainty(BaseModel):
    distribution: dict[str, float] = Field(description="possible answer -> probability (sums to 1)")
    interval: list[float] = Field(description="[low, high] of the central interval")
    level: float = 0.9
    method: Literal["decision", "poisson_binomial", "agreement", "none"]
    composition: dict[str, int] = Field(default_factory=dict,
                                        description="e.g. {sure: 6, confirmed: 2, rejected: 1}")


class Evidence(BaseModel):
    element: str = Field(description="id in elements")
    role: Literal["counted", "excluded", "considered", "context"]
    reason: Optional[str] = None
    order: Optional[int] = Field(None, description="number shown on the plan (counting animation)")
    decisive: bool = Field(False, description="the answer would change without this element")
    counterfactual: Optional[float] = Field(None, description="answer without this element")


class Step(BaseModel):
    n: int
    kind: Literal["route", "read", "tool", "view", "check", "final"]
    title: str
    tool: Optional[str] = None
    args: Optional[dict] = None
    ids: list[str] = Field(default_factory=list, description="element ids that appeared in this step")
    result: Optional[str] = Field(None, description="short summary of the result")
    region: Optional[list[float]] = Field(None, description="view: crop of the drawing the agent looked at [x0,y0,x1,y1]")
    ms: Optional[int] = None


class Coverage(BaseModel):
    by_kind: dict[str, list[int]] = Field(description="kind -> [seen, total], e.g. door: [9, 9]")
    first_seen: dict[str, int] = Field(description="element id -> step number where it first appeared")
    referenced_only: list[str] = Field(default_factory=list, description="seen only as a reference (room_id)")
    blind_spots: list[str] = Field(default_factory=list, description="relevant BlindSpot ids")
    views: list[list[float]] = Field(default_factory=list, description="regions of the drawing the agent looked at")


class Answer(BaseModel):
    id: str
    question: str
    language: Optional[str] = None
    intent: Optional[str] = None
    intent_conf: Optional[float] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    text: str = Field(description="answer sentence in the language of the question")
    summary: str = Field(description="one-sentence 'why' for the answer card")
    confidence: float
    confidence_label: Literal["hoch", "mittel", "niedrig"]
    abstained: bool = False
    abstain_reason: Optional[str] = None
    agreement: Literal["agree", "partial", "conflict", "single"] = Field(
        description="traffic light: do rules / agent / decision agree?")
    sources: dict[str, SourceAnswer] = Field(description="rules | agent | decision")
    uncertainty: Optional[Uncertainty] = None
    evidence: list[Evidence]
    references: list[str] = Field(default_factory=list, description="non-element evidence: table paths, facts, views")
    steps: list[Step]
    coverage: Optional[Coverage] = None
    limitations: list[str] = Field(default_factory=list)
    truth: Optional[float] = Field(None, description="ground truth, benchmark only")
    correct: Optional[bool] = None


class CalibrationBin(BaseModel):
    lo: float
    hi: float
    n: int
    stated: float = Field(description="mean stated confidence in the bin")
    accuracy: float


class Calibration(BaseModel):
    benchmark: str
    n: int
    bins: list[CalibrationBin]
    ece: float = Field(description="expected calibration error (0 = perfectly calibrated)")


class ExplanationDoc(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source: Source
    run: Run
    assets: Assets = Assets()
    elements: dict[str, Element] = Field(description="id -> element; answers reference these ids")
    hotspots: list[Hotspot] = Field(default_factory=list)
    blind_spots: list[BlindSpot] = Field(default_factory=list)
    answers: list[Answer]
    calibration: Optional[Calibration] = None
