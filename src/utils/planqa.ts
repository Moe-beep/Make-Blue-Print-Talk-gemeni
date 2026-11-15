// Client for the PlanQA pipeline (Python service behind /api/planqa) + mapping of explanations onto the viewers.
import type { HighlightTarget, LoadedAsset } from '../types/analysis';
import type {
  EvidenceRole,
  ExplainMark,
  ExplainSelection,
  ExplElement,
  PlanqaAnswerBundle,
  PlanqaDocInfo,
} from '../types/explanation';

export const PLANQA_BASE = '/api/planqa';

export const ROLE_STYLE: Record<
  EvidenceRole | 'step' | 'hotspot' | 'blind' | 'seen' | 'unseen' | 'view',
  { color: string; dashed?: boolean; label: string }
> = {
  counted: { color: '#16A34A', label: 'counted' },
  considered: { color: '#EA8A0C', dashed: true, label: 'below threshold, not counted' },
  excluded: { color: '#DC2626', label: 'excluded by the agent' },
  context: { color: '#16A34A', label: 'evidence' },
  step: { color: '#2563EB', label: 'selected step' },
  hotspot: { color: '#7C3AED', label: 'uncertain spot' },
  blind: { color: '#9333EA', dashed: true, label: 'blind spot' },
  seen: { color: '#2563EB', label: 'seen' },
  unseen: { color: '#9CA3AF', dashed: true, label: 'not seen' },
  view: { color: '#EA580C', label: 'viewed crop' },
};

/** Which project files the pipeline can read. */
export function planqaSupported(asset: LoadedAsset): boolean {
  const n = asset.name.toLowerCase();
  if (asset.fileKind === '3d-ifc') return true;
  if (asset.fileKind === '2d-pdf') return true;
  if (asset.fileKind === '2d-image') return /\.(png|jpe?g|webp|bmp|tiff?)$/.test(n) || !/\.[a-z0-9]+$/.test(n);
  return false;
}

async function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(',', 2)[1] ?? '');
    r.onerror = () => reject(r.error ?? new Error('Could not read the file'));
    r.readAsDataURL(blob);
  });
}

/** Original bytes of the file as loaded in the browser (not the downscaled viewer capture). */
export async function assetBase64(asset: LoadedAsset): Promise<string> {
  if (asset.arrayBuffer) return blobToBase64(new Blob([asset.arrayBuffer]));
  if (asset.rawText !== undefined) return blobToBase64(new Blob([asset.rawText], { type: 'text/plain' }));
  if (asset.previewUrl) {
    const resp = await fetch(asset.previewUrl);
    return blobToBase64(await resp.blob());
  }
  throw new Error('No file data available in the browser');
}

async function json<T>(resp: Response): Promise<T & { error?: string; offline?: boolean }> {
  const text = await resp.text();
  try {
    return JSON.parse(text);
  } catch {
    return { error: text.slice(0, 300) || `HTTP ${resp.status}` } as T & { error?: string };
  }
}

export class PlanqaError extends Error {
  constructor(
    message: string,
    public offline = false,
    public status = 0,
  ) {
    super(message);
  }
}

export async function planqaUpload(asset: LoadedAsset, page: number): Promise<PlanqaDocInfo> {
  const dataBase64 = await assetBase64(asset);
  const resp = await fetch(`${PLANQA_BASE}/documents`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: asset.name, page, dataBase64 }),
  });
  const data = await json<PlanqaDocInfo>(resp);
  if (!resp.ok) throw new PlanqaError(data.error || `HTTP ${resp.status}`, Boolean(data.offline), resp.status);
  return data;
}

export async function planqaInfo(docId: string): Promise<PlanqaDocInfo> {
  const resp = await fetch(`${PLANQA_BASE}/documents/${docId}`);
  const data = await json<PlanqaDocInfo>(resp);
  if (!resp.ok) throw new PlanqaError(data.error || `HTTP ${resp.status}`, Boolean(data.offline), resp.status);
  return data;
}

export async function planqaWaitReady(
  docId: string,
  onProgress?: (info: PlanqaDocInfo) => void,
  timeoutMs = 20 * 60 * 1000,
): Promise<PlanqaDocInfo> {
  const t0 = Date.now();
  for (;;) {
    const info = await planqaInfo(docId);
    onProgress?.(info);
    if (info.status === 'ready' || info.status === 'error') return info;
    if (Date.now() - t0 > timeoutMs) throw new PlanqaError('Timed out waiting for the pipeline');
    await new Promise((r) => setTimeout(r, 1500));
  }
}

export async function planqaAsk(docId: string, question: string): Promise<PlanqaAnswerBundle> {
  const resp = await fetch(`${PLANQA_BASE}/documents/${docId}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
  const data = await json<PlanqaAnswerBundle>(resp);
  if (!resp.ok) throw new PlanqaError(data.error || `HTTP ${resp.status}`, Boolean(data.offline), resp.status);
  return data;
}

export function planqaAssetUrl(servicePath: string | undefined | null): string | null {
  return servicePath ? `${PLANQA_BASE}${servicePath}` : null;
}

// ------------------------------------------------------------------------------- geometry

export function bboxOf(el: ExplElement | undefined): number[] | null {
  if (!el) return null;
  if (el.geometry.bbox && el.geometry.bbox.length >= 4) return el.geometry.bbox;
  const poly = el.geometry.polygon;
  if (poly && poly.length > 0) {
    const xs = poly.map((p) => p[0]);
    const ys = poly.map((p) => p[1]);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  }
  return null;
}

function norm(b: number[], size: number[]) {
  const [w, h] = size;
  const c = (v: number) => Math.max(0, Math.min(1000, v));
  return { xmin: c((b[0] / w) * 1000), ymin: c((b[1] / h) * 1000), xmax: c((b[2] / w) * 1000), ymax: c((b[3] / h) * 1000) };
}

/** 2D: selection -> coloured, numbered rectangles for Viewport2D. */
export function marksForSelection(
  sel: ExplainSelection,
  elements: Record<string, ExplElement>,
  imageSize: number[] | null | undefined,
): ExplainMark[] {
  if (!imageSize || imageSize.length < 2) return [];
  const marks: ExplainMark[] = [];
  for (const it of sel.items) {
    const b = bboxOf(elements[it.id]);
    if (!b) continue;
    const st = ROLE_STYLE[it.role];
    marks.push({ ...norm(b, imageSize), color: st.color, dashed: st.dashed, order: it.order, label: elements[it.id]?.label });
  }
  for (const r of sel.regions ?? []) {
    const st = ROLE_STYLE[r.role];
    marks.push({ ...norm(r.bbox, imageSize), color: st.color, dashed: st.dashed ?? r.role !== 'hotspot', label: r.label });
  }
  return marks;
}

/** One rectangle for a pixel bbox (null if the image size is unknown). */
export function markFor(
  bbox: number[] | null | undefined,
  imageSize: number[] | null | undefined,
  style: { color: string; dashed?: boolean },
  extra: Partial<ExplainMark> = {},
): ExplainMark | null {
  if (!bbox || !imageSize || imageSize.length < 2) return null;
  return { ...norm(bbox, imageSize), color: style.color, dashed: style.dashed, ...extra };
}

/** 3D: selection -> highlight target with exact '#<STEP id>' ids of the viewer meshes. */
export function highlightForSelection(sel: ExplainSelection, viewerIds: Record<string, string>): HighlightTarget {
  const ids = sel.items.map((it) => viewerIds[it.id]).filter((x): x is string => Boolean(x));
  return {
    id: `expl-${sel.key}`,
    label: sel.label,
    category: 'All',
    keywords: [],
    ifcIds: ids.length > 0 ? ids : ['#none'],
    expectedCount: ids.length || undefined,
  };
}

export function fmtNumber(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '–';
  return Number.isInteger(v) ? String(v) : v.toLocaleString('de-DE', { maximumFractionDigits: digits });
}
