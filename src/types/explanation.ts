// explanation/1 (Junction/planqa/explanation_schema.py) as returned by the PlanQA service (Junction/serve.py).
// Pixel coordinates refer to the drawing as the pipeline read it (source.image_size).

export interface ExplGeometry {
  bbox?: number[]; // [x0, y0, x1, y1] px
  polygon?: number[][];
  guid?: string;
  storey?: string;
  centroid?: number[];
}

export interface ExplClassProb {
  label: string;
  p: number;
}

export interface ExplElement {
  id: string;
  kind: string;
  label?: string;
  geometry: ExplGeometry;
  conf?: number;
  class_probs?: ExplClassProb[];
  status: 'kept' | 'below_threshold' | 'rejected' | 'confirmed';
  room?: string;
}

export interface ExplHotspot {
  id: string;
  rank: number;
  entropy: number;
  geometry: ExplGeometry;
  channel: 'rooms' | 'symbols' | 'text';
  top_classes: ExplClassProb[];
  elements?: string[];
  resolution?: string;
  resolved_by?: string;
}

export interface ExplBlindSpot {
  id: string;
  geometry: ExplGeometry;
  reason: string;
  suggestion?: string;
  heuristic: string;
}

export interface ExplSourceAnswer {
  value?: number | null;
  text?: string;
  model?: string;
  explanation?: string;
}

export interface ExplUncertainty {
  distribution: Record<string, number>;
  interval: number[];
  level?: number;
  method: string;
}

export type EvidenceRole = 'counted' | 'excluded' | 'considered' | 'context';

export interface ExplEvidence {
  element: string;
  role: EvidenceRole;
  reason?: string;
  order?: number;
  decisive?: boolean;
  counterfactual?: number;
}

export interface ExplStep {
  n: number;
  kind: 'route' | 'read' | 'tool' | 'view' | 'check' | 'final';
  title: string;
  tool?: string;
  args?: Record<string, unknown>;
  ids?: string[];
  result?: string;
  region?: number[];
  ms?: number;
}

export interface ExplCoverage {
  by_kind: Record<string, [number, number]>;
  first_seen: Record<string, number>;
  referenced_only?: string[];
  blind_spots?: string[];
  views?: number[][];
}

export interface ExplAnswer {
  id: string;
  question: string;
  intent?: string;
  value?: number | null;
  unit?: string;
  text: string;
  summary: string;
  confidence: number;
  confidence_label: 'hoch' | 'mittel' | 'niedrig';
  abstained?: boolean;
  abstain_reason?: string;
  agreement: 'agree' | 'partial' | 'conflict' | 'single';
  sources: Record<string, ExplSourceAnswer>;
  uncertainty?: ExplUncertainty;
  evidence: ExplEvidence[];
  references?: string[];
  steps: ExplStep[];
  coverage?: ExplCoverage;
  limitations?: string[];
}

export interface ExplDocument {
  schema_version: string;
  source: { file: string; kind: 'plan' | 'ifc'; page?: number; image_size?: number[] };
  run: { created: string; models: Record<string, string>; thresholds?: Record<string, number> };
  assets?: {
    uncertainty_share?: Record<string, number>;
    uncertain_area?: number;
  };
  elements: Record<string, ExplElement>;
  hotspots?: ExplHotspot[];
  blind_spots?: ExplBlindSpot[];
}

export type PlanqaStatus =
  | 'uploading'
  | 'queued'
  | 'processing'
  | 'ready'
  | 'error'
  | 'offline';

export interface PlanqaDocInfo {
  id: string;
  name: string;
  kind: 'plan' | 'ifc';
  page: number;
  status: PlanqaStatus;
  stage?: string;
  error?: string | null;
  seconds?: number | null;
  n_answers?: number;
  explanation?: ExplDocument;
  assets?: Record<string, string>; // channel -> service path (/documents/<id>/assets/...)
  summary?: {
    storeys: number;
    spaces: number;
    elements: Record<string, number>;
    notes: string[];
  };
}

/** Pipeline state of one project file. */
export interface PlanqaEntryState {
  key: string; // asset id + page
  docId?: string;
  status: PlanqaStatus;
  stage?: string;
  error?: string;
  seconds?: number | null;
  info?: PlanqaDocInfo;
}

export interface PlanqaAnswerBundle {
  document: string;
  kind: 'plan' | 'ifc';
  answer: ExplAnswer;
  elements: Record<string, ExplElement>;
  viewer_ids: Record<string, string>;
  image_size: number[] | null;
  agent?: string | null;
  seconds?: number;
}

/** What the user picked in an explanation (evidence group, one element, a step, a hotspot ...). */
export interface ExplainSelection {
  key: string;
  label: string;
  items: Array<{ id: string; role: EvidenceRole | 'step' | 'hotspot' | 'blind'; order?: number }>;
  regions?: Array<{ bbox: number[]; role: 'step' | 'hotspot' | 'blind'; label?: string }>;
}

/** One rectangle drawn by Viewport2D on top of the plan (0..1000 normalised like Box2D). */
export interface ExplainMark {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  color: string;
  dashed?: boolean;
  order?: number;
  label?: string;
  /** background tint (0..1, default 0.12); thick = focus */
  fillOpacity?: number;
  thick?: boolean;
}

export interface ExplainOverlay {
  heatmapUrl?: string | null;
  heatmapOpacity?: number;
  marks?: ExplainMark[];
}
