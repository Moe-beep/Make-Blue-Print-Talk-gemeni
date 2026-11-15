import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import type { ExplainOverlay } from '../types/explanation';
import type {
  HighlightTarget,
  LoadedAsset,
  SpatialTilePayload,
} from '../types/analysis';
import {
  type DxfBlockInstance,
  type DxfPrimitive,
  type DxfRoomPolygon,
  parseDxfFile,
} from '../utils/dxfParser';
import {
  ChevronLeft,
  ChevronRight,
  Crosshair,
  Focus,
  Grid3X3,
  Loader2,
  RotateCcw,
  Ruler,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString();

export interface Viewport2DHandle {
  captureImagePayload: () => Promise<{
    mimeType: string;
    dataBase64: string;
    tiles?: SpatialTilePayload[];
    calibrationNote?: string;
  } | null>;
}

interface Viewport2DProps {
  asset: LoadedAsset;
  activeHighlight?: HighlightTarget | null;
  locating?: boolean;
  onClearHighlight?: () => void;
  onParsedSummary: (summary: Record<string, unknown>) => void;
  onCalibrationChange?: (note: string | null) => void;
  onImageReady?: () => void;
  /** PlanQA explanation layer: uncertainty heat map + coloured, numbered evidence boxes. */
  explanationOverlay?: ExplainOverlay | null;
}

interface NormalizedOverlayBox {
  xmin: number; // 0..1000
  ymin: number; // 0..1000
  xmax: number; // 0..1000
  ymax: number; // 0..1000
  label?: string;
  isSpace?: boolean;
}

interface InkReferenceGrid {
  key: string;
  w: number;
  h: number;
  inkMask: Uint8Array;
  wallMask: Uint8Array;
  isDarkBlueprint: boolean;
}

interface CachedCapturePayload {
  key: string;
  mimeType: string;
  dataBase64: string;
  tiles: SpatialTilePayload[];
}

const SYNONYM_GROUPS: string[][] = [
  ['lamp', 'light', 'luminaire', 'sconce', 'pendant', 'chandelier', 'fixture', 'leuchte'],
  ['door', 'doors', 'tuer', 'tür', 'porte', 'ovi'],
  ['window', 'windows', 'fenster', 'fenetre', 'ikkuna'],
  ['kitchen', 'küche', 'kuche', 'cuisine', 'keittiö', 'kueche'],
  ['bath', 'bathroom', 'bad', 'badezimmer', 'wc', 'toilet', 'lavatory', 'kylpyhuone'],
  ['living', 'wohnen', 'wohnzimmer', 'salon', 'olohuone'],
  ['bedroom', 'schlafen', 'schlafzimmer', 'chambre', 'makuuhuone'],
  ['balcony', 'balkon', 'loggia', 'terrace', 'terrasse', 'porch', 'patio'],
  ['hall', 'hallway', 'corridor', 'korridor', 'flur', 'entree', 'entry', 'eteinen'],
  ['wall', 'walls', 'wand', 'wände', 'seinä'],
  ['stair', 'stairs', 'treppe', 'escalier'],
  ['furniture', 'möbel', 'chair', 'table', 'sofa', 'couch', 'bed', 'desk', 'wardrobe', 'cabinet'],
  ['sink', 'basin', 'lavabo', 'washbasin'],
  ['shower', 'dusche', 'bathtub', 'tub', 'wanne'],
];

function buildSynonymGroup2D(term: string): string[] {
  const t = term.trim().toLowerCase();
  const group = new Set<string>([t]);
  for (const grp of SYNONYM_GROUPS) {
    if (
      grp.some(
        (g) => g === t || (t.length >= 4 && (g.includes(t) || t.includes(g))),
      )
    ) {
      for (const g of grp) group.add(g);
    }
  }
  return Array.from(group);
}

function buildKeywordGroups2D(rawKeywords: string[]): string[][] {
  const cleaned = rawKeywords
    .map((s) => s.trim().toLowerCase())
    .filter((s) => s.length >= 2 && !/^\d+$/.test(s));
  const groups: string[][] = [];
  for (const term of cleaned) {
    const sg = buildSynonymGroup2D(term);
    const existing = groups.find((g) => g.some((item) => sg.includes(item)));
    if (existing) {
      for (const item of sg) {
        if (!existing.includes(item)) existing.push(item);
      }
    } else {
      groups.push(sg);
    }
  }
  return groups;
}

function matchesSingleDxfPolygon(
  rp: DxfRoomPolygon,
  target: HighlightTarget,
): boolean {
  const hay = `${rp.label} ${rp.code} ${rp.apartment ?? ''}`.toLowerCase();
  if (
    target.excludeKeywords?.some(
      (ex) => ex.trim().length >= 2 && hay.includes(ex.trim().toLowerCase()),
    )
  ) {
    return false;
  }
  if (target.exactNames && target.exactNames.length > 0) {
    return target.exactNames.some((en) => {
      const low = en.trim().toLowerCase();
      return (
        low.length > 0 &&
        (rp.label.toLowerCase() === low ||
          rp.code.toLowerCase() === low ||
          hay.includes(low))
      );
    });
  }
  const groups = buildKeywordGroups2D(target.keywords ?? []);
  if (groups.length === 0) return false;
  return groups.every((g) => g.some((t) => hay.includes(t)));
}

function matchesSingleDxfBlock(
  blk: DxfBlockInstance,
  target: HighlightTarget,
): boolean {
  if (target.category === 'Space') return false;
  const hay = `${blk.blockName} ${blk.layer}`.toLowerCase();
  if (
    target.excludeKeywords?.some(
      (ex) => ex.trim().length >= 2 && hay.includes(ex.trim().toLowerCase()),
    )
  ) {
    return false;
  }
  if (target.exactNames && target.exactNames.length > 0) {
    const exactMatched = target.exactNames.some((en) => {
      const low = en.trim().toLowerCase();
      return low.length > 0 && hay.includes(low);
    });
    if (exactMatched) return true;
  }
  const groups = buildKeywordGroups2D(target.keywords ?? []);
  if (groups.length === 0) return false;
  return groups.every((g) => g.some((t) => hay.includes(t)));
}

function matchesSingleDxfPrimitive(
  p: DxfPrimitive,
  target: HighlightTarget,
): boolean {
  if (target.category === 'Space') return false;
  const hay = `${p.layer} ${p.blockName ?? ''} ${p.kind === 'text' ? p.text : ''}`.toLowerCase();
  if (
    target.excludeKeywords?.some(
      (ex) => ex.trim().length >= 2 && hay.includes(ex.trim().toLowerCase()),
    )
  ) {
    return false;
  }
  if (target.exactNames && target.exactNames.length > 0) {
    const exactMatched = target.exactNames.some((en) => {
      const low = en.trim().toLowerCase();
      return low.length > 0 && hay.includes(low);
    });
    if (exactMatched) return true;
  }
  const groups = buildKeywordGroups2D(target.keywords ?? []);
  if (groups.length === 0) return false;
  return groups.every((g) => g.some((t) => hay.includes(t)));
}

/**
 * Ensures an <svg> string has explicit pixel width & height matching its viewBox
 * aspect ratio so browser <img> rendering and offscreen <canvas> capture share
 * identical proportions with zero letterboxing.
 */
function ensureSvgExplicitDimensions(rawSvg: string): string {
  try {
    const doc = new DOMParser().parseFromString(rawSvg, 'image/svg+xml');
    const svgEl = doc.querySelector('svg');
    if (!svgEl) return rawSvg;

    const vb = svgEl.getAttribute('viewBox');
    if (vb) {
      const parts = vb
        .trim()
        .split(/[\s,]+/)
        .map(Number);
      if (
        parts.length === 4 &&
        Number.isFinite(parts[2]) &&
        Number.isFinite(parts[3]) &&
        parts[2] > 0 &&
        parts[3] > 0
      ) {
        const targetW = 1800;
        const targetH = Math.max(
          240,
          Math.min(4800, Math.round((targetW * parts[3]) / parts[2])),
        );
        svgEl.setAttribute('width', String(targetW));
        svgEl.setAttribute('height', String(targetH));
        if (!svgEl.getAttribute('preserveAspectRatio')) {
          svgEl.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        }
        return new XMLSerializer().serializeToString(doc);
      }
    }
    if (!svgEl.getAttribute('width') || !svgEl.getAttribute('height')) {
      svgEl.setAttribute('width', '1600');
      svgEl.setAttribute('height', '1200');
      return new XMLSerializer().serializeToString(doc);
    }
    return rawSvg;
  } catch {
    return rawSvg;
  }
}

function buildOverlappingSpatialTiles(
  source: HTMLImageElement | HTMLCanvasElement,
  srcW: number,
  srcH: number,
): SpatialTilePayload[] {
  if (srcW < 240 || srcH < 240) return [];
  const xBands: Array<[number, number]> =
    srcW >= srcH
      ? [
          [0, 400],
          [300, 700],
          [600, 1000],
        ]
      : [
          [0, 560],
          [440, 1000],
        ];
  const yBands: Array<[number, number]> =
    srcW >= srcH
      ? [
          [0, 560],
          [440, 1000],
        ]
      : [
          [0, 400],
          [300, 700],
          [600, 1000],
        ];

  const tiles: SpatialTilePayload[] = [];
  const tileCanvas = document.createElement('canvas');
  const tileCtx = tileCanvas.getContext('2d');
  if (!tileCtx) return [];

  let idx = 0;
  for (const [y0, y1] of yBands) {
    for (const [x0, x1] of xBands) {
      const sx = Math.round((x0 / 1000) * srcW);
      const sy = Math.round((y0 / 1000) * srcH);
      const sw = Math.max(16, Math.round(((x1 - x0) / 1000) * srcW));
      const sh = Math.max(16, Math.round(((y1 - y0) / 1000) * srcH));
      const scale = Math.min(1, 1024 / Math.max(sw, sh));
      const dw = Math.max(16, Math.round(sw * scale));
      const dh = Math.max(16, Math.round(sh * scale));

      tileCanvas.width = dw;
      tileCanvas.height = dh;
      tileCtx.fillStyle = '#ffffff';
      tileCtx.fillRect(0, 0, dw, dh);
      tileCtx.drawImage(source, sx, sy, sw, sh, 0, 0, dw, dh);
      const url = tileCanvas.toDataURL('image/jpeg', 0.86);
      tiles.push({
        id: `tile-${idx++}`,
        x0,
        y0,
        x1,
        y1,
        mimeType: 'image/jpeg',
        dataBase64: url.replace(/^data:image\/jpeg;base64,/, ''),
      });
    }
  }
  return tiles;
}

function encodeImageElementToJpegPayload(img: HTMLImageElement): {
  mimeType: string;
  dataBase64: string;
  tiles: SpatialTilePayload[];
} | null {
  try {
    const maxSide = 1536;
    const w = img.naturalWidth || 1600;
    const h = img.naturalHeight || 1200;
    const scale = Math.min(1, maxSide / Math.max(w, h));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(w * scale));
    canvas.height = Math.max(1, Math.round(h * scale));
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.88);
    const tiles = buildOverlappingSpatialTiles(img, w, h);
    return {
      mimeType: 'image/jpeg',
      dataBase64: dataUrl.replace(/^data:image\/jpeg;base64,/, ''),
      tiles,
    };
  } catch {
    return null;
  }
}

function buildInkReferenceGrid(
  img: HTMLImageElement,
  key: string,
): InkReferenceGrid | null {
  try {
    const nw = img.naturalWidth;
    const nh = img.naturalHeight;
    if (nw <= 0 || nh <= 0) return null;
    const maxSide = 900;
    const scale = Math.min(1, maxSide / Math.max(nw, nh));
    const w = Math.max(64, Math.round(nw * scale));
    const h = Math.max(64, Math.round(nh * scale));

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(img, 0, 0, w, h);

    const data = ctx.getImageData(0, 0, w, h).data;

    // Sample global background color across a 16x16 grid
    const lums: Array<{ lum: number; r: number; g: number; b: number }> = [];
    const stepX = Math.max(1, Math.floor(w / 16));
    const stepY = Math.max(1, Math.floor(h / 16));
    for (let y = Math.floor(stepY / 2); y < h; y += stepY) {
      for (let x = Math.floor(stepX / 2); x < w; x += stepX) {
        const idx = (y * w + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        lums.push({
          lum: 0.299 * r + 0.587 * g + 0.114 * b,
          r,
          g,
          b,
        });
      }
    }
    lums.sort((a, b) => a.lum - b.lum);
    const medianLum = lums[Math.floor(lums.length * 0.5)]?.lum ?? 245;
    const isDarkBlueprint = medianLum < 110;
    const bgRef = isDarkBlueprint
      ? (lums[Math.floor(lums.length * 0.2)] ?? { lum: 20, r: 20, g: 20, b: 20 })
      : (lums[Math.floor(lums.length * 0.78)] ?? {
          lum: 248,
          r: 252,
          g: 252,
          b: 250,
        });

    const inkMask = new Uint8Array(w * h);
    const wallMask = new Uint8Array(w * h);

    for (let i = 0, p = 0; i < w * h; i++, p += 4) {
      const r = data[p];
      const g = data[p + 1];
      const b = data[p + 2];
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      const colorDist = Math.hypot(r - bgRef.r, g - bgRef.g, b - bgRef.b);
      if (isDarkBlueprint) {
        if (lum - bgRef.lum > 28 || colorDist > 36) {
          inkMask[i] = 1;
        }
        if (lum > 215) {
          wallMask[i] = 1;
        }
      } else {
        if (bgRef.lum - lum > 22 || colorDist > 32) {
          inkMask[i] = 1;
        }
        if (lum < 82) {
          wallMask[i] = 1;
        }
      }
    }

    return { key, w, h, inkMask, wallMask, isDarkBlueprint };
  } catch {
    return null;
  }
}

/**
 * Tightens a normalized [0..1000] box using the blueprint's actual stroke ink:
 * - For objects (doors, furniture, windows, fixtures): trims empty paper/floor
 *   margins (up to 28% per side) and pulls off full-edge structural wall bands
 *   so the highlight hugs the drawn symbol cleanly.
 * - For spaces (rooms): gently pulls 1-2 cells inward if the outer box edge sits
 *   directly on a thick black structural wall line.
 */
function tightenNormalizedBoxWithInk(
  box: NormalizedOverlayBox,
  grid: InkReferenceGrid | null,
): NormalizedOverlayBox {
  if (!grid) return box;
  const { w, h, inkMask, wallMask } = grid;

  let x0 = Math.max(0, Math.min(w - 1, Math.floor((box.xmin / 1000) * w)));
  let x1 = Math.max(0, Math.min(w - 1, Math.ceil((box.xmax / 1000) * w)));
  let y0 = Math.max(0, Math.min(h - 1, Math.floor((box.ymin / 1000) * h)));
  let y1 = Math.max(0, Math.min(h - 1, Math.ceil((box.ymax / 1000) * h)));

  const origW = x1 - x0 + 1;
  const origH = y1 - y0 + 1;
  if (origW < 4 || origH < 4) return box;

  const rowDensity = (
    mask: Uint8Array,
    ry: number,
    cx0: number,
    cx1: number,
  ): number => {
    const span = cx1 - cx0 + 1;
    if (span <= 0) return 0;
    let sum = 0;
    const base = ry * w;
    for (let x = cx0; x <= cx1; x++) sum += mask[base + x];
    return sum / span;
  };

  const colDensity = (
    mask: Uint8Array,
    cx: number,
    cy0: number,
    cy1: number,
  ): number => {
    const span = cy1 - cy0 + 1;
    if (span <= 0) return 0;
    let sum = 0;
    for (let y = cy0; y <= cy1; y++) sum += mask[y * w + cx];
    return sum / span;
  };

  if (box.isSpace) {
    // Pull room edge slightly inside full-length structural walls so walls stay crisp
    const maxWallTrimX = Math.min(4, Math.floor(origW * 0.06));
    const maxWallTrimY = Math.min(4, Math.floor(origH * 0.06));
    for (let i = 0; i < maxWallTrimY && y1 - y0 > 6; i++) {
      if (rowDensity(wallMask, y0, x0, x1) > 0.62) y0++;
      else break;
    }
    for (let i = 0; i < maxWallTrimY && y1 - y0 > 6; i++) {
      if (rowDensity(wallMask, y1, x0, x1) > 0.62) y1--;
      else break;
    }
    for (let i = 0; i < maxWallTrimX && x1 - x0 > 6; i++) {
      if (colDensity(wallMask, x0, y0, y1) > 0.62) x0++;
      else break;
    }
    for (let i = 0; i < maxWallTrimX && x1 - x0 > 6; i++) {
      if (colDensity(wallMask, x1, y0, y1) > 0.62) x1--;
      else break;
    }
    return {
      ...box,
      xmin: Math.max(0, Math.min(1000, (x0 / w) * 1000)),
      xmax: Math.max(0, Math.min(1000, ((x1 + 1) / w) * 1000)),
      ymin: Math.max(0, Math.min(1000, (y0 / h) * 1000)),
      ymax: Math.max(0, Math.min(1000, ((y1 + 1) / h) * 1000)),
    };
  }

  // Object symbol tightening:
  // 1) Strip outer wall bands if box edge landed on a continuous thick wall
  const maxWallStrip = 3;
  for (let i = 0; i < maxWallStrip && y1 - y0 > 5; i++) {
    if (rowDensity(wallMask, y0, x0, x1) > 0.8) y0++;
    else break;
  }
  for (let i = 0; i < maxWallStrip && y1 - y0 > 5; i++) {
    if (rowDensity(wallMask, y1, x0, x1) > 0.8) y1--;
    else break;
  }
  for (let i = 0; i < maxWallStrip && x1 - x0 > 5; i++) {
    if (colDensity(wallMask, x0, y0, y1) > 0.8) x0++;
    else break;
  }
  for (let i = 0; i < maxWallStrip && x1 - x0 > 5; i++) {
    if (colDensity(wallMask, x1, y0, y1) > 0.8) x1--;
    else break;
  }

  // 2) Trim empty floor/paper margins (up to 28% per side) so box snaps tightly onto the symbol
  const maxTrimX = Math.floor(origW * 0.28);
  const maxTrimY = Math.floor(origH * 0.28);
  let trimmedTop = 0;
  while (
    trimmedTop < maxTrimY &&
    y1 - y0 > 4 &&
    rowDensity(inkMask, y0, x0, x1) < 0.018
  ) {
    y0++;
    trimmedTop++;
  }
  let trimmedBottom = 0;
  while (
    trimmedBottom < maxTrimY &&
    y1 - y0 > 4 &&
    rowDensity(inkMask, y1, x0, x1) < 0.018
  ) {
    y1--;
    trimmedBottom++;
  }
  let trimmedLeft = 0;
  while (
    trimmedLeft < maxTrimX &&
    x1 - x0 > 4 &&
    colDensity(inkMask, x0, y0, y1) < 0.018
  ) {
    x0++;
    trimmedLeft++;
  }
  let trimmedRight = 0;
  while (
    trimmedRight < maxTrimX &&
    x1 - x0 > 4 &&
    colDensity(inkMask, x1, y0, y1) < 0.018
  ) {
    x1--;
    trimmedRight++;
  }

  // Small 1.5-unit cushion clamped within original box so strokes aren't clipped
  let xmin = Math.max(box.xmin, (x0 / w) * 1000 - 1.5);
  let xmax = Math.min(box.xmax, ((x1 + 1) / w) * 1000 + 1.5);
  let ymin = Math.max(box.ymin, (y0 / h) * 1000 - 1.5);
  let ymax = Math.min(box.ymax, ((y1 + 1) / h) * 1000 + 1.5);

  if (xmax - xmin < 9) {
    const cx = (xmin + xmax) * 0.5;
    xmin = Math.max(0, cx - 4.5);
    xmax = Math.min(1000, cx + 4.5);
  }
  if (ymax - ymin < 9) {
    const cy = (ymin + ymax) * 0.5;
    ymin = Math.max(0, cy - 4.5);
    ymax = Math.min(1000, cy + 4.5);
  }

  return {
    ...box,
    xmin,
    ymin,
    xmax,
    ymax,
  };
}

interface Point2D {
  x: number;
  y: number;
}

function computeMonotoneConvexHull(points: Point2D[]): Point2D[] {
  if (points.length <= 2) return points.slice();
  const sorted = points
    .slice()
    .sort((a, b) => (a.x !== b.x ? a.x - b.x : a.y - b.y));

  const cross = (o: Point2D, a: Point2D, b: Point2D) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower: Point2D[] = [];
  for (const p of sorted) {
    while (
      lower.length >= 2 &&
      cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0
    ) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper: Point2D[] = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (
      upper.length >= 2 &&
      cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0
    ) {
      upper.pop();
    }
    upper.push(p);
  }

  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function buildRoundedPolygonSvgPath(pts: Point2D[], maxRadius = 6): string {
  const n = pts.length;
  if (n < 3) return '';
  const cmds: string[] = [];
  for (let i = 0; i < n; i++) {
    const prev = pts[(i - 1 + n) % n];
    const curr = pts[i];
    const next = pts[(i + 1) % n];

    const dx1 = prev.x - curr.x;
    const dy1 = prev.y - curr.y;
    const len1 = Math.hypot(dx1, dy1);

    const dx2 = next.x - curr.x;
    const dy2 = next.y - curr.y;
    const len2 = Math.hypot(dx2, dy2);

    const r = Math.min(maxRadius, len1 * 0.28, len2 * 0.28);
    if (r < 0.4 || len1 < 1e-3 || len2 < 1e-3) {
      cmds.push(
        `${i === 0 ? 'M' : 'L'} ${curr.x.toFixed(2)} ${curr.y.toFixed(2)}`,
      );
      continue;
    }

    const p1x = curr.x + (dx1 / len1) * r;
    const p1y = curr.y + (dy1 / len1) * r;
    const p2x = curr.x + (dx2 / len2) * r;
    const p2y = curr.y + (dy2 / len2) * r;

    cmds.push(`${i === 0 ? 'M' : 'L'} ${p1x.toFixed(2)} ${p1y.toFixed(2)}`);
    cmds.push(
      `Q ${curr.x.toFixed(2)} ${curr.y.toFixed(2)} ${p2x.toFixed(2)} ${p2y.toFixed(2)}`,
    );
  }
  cmds.push('Z');
  return cmds.join(' ');
}

/**
 * Extracts a contour-hugging convex-hull SVG path around actual CAD symbol ink
 * (eliminating empty floor corners on quarter-circle door swings, diagonal chairs,
 * round tables, and oval plumbing fixtures while filtering out outer wall bands).
 * Returns null when a rounded rectangle is already optimal (e.g. rooms or thin linear windows).
 */
function buildObjectSilhouettePath(
  box: NormalizedOverlayBox,
  grid: InkReferenceGrid | null,
): string | null {
  if (!grid || box.isSpace) return null;
  const bw = box.xmax - box.xmin;
  const bh = box.ymax - box.ymin;
  if (bw < 8 || bh < 8) return null;

  const aspect = Math.max(bw, bh) / Math.max(1, Math.min(bw, bh));
  const lowerLabel = (box.label ?? '').toLowerCase();
  if (aspect >= 3.2 || lowerLabel.includes('window')) {
    return null;
  }

  const { w, h, inkMask, wallMask } = grid;
  const x0 = Math.max(0, Math.min(w - 1, Math.floor((box.xmin / 1000) * w)));
  const x1 = Math.max(0, Math.min(w - 1, Math.ceil((box.xmax / 1000) * w)));
  const y0 = Math.max(0, Math.min(h - 1, Math.floor((box.ymin / 1000) * h)));
  const y1 = Math.max(0, Math.min(h - 1, Math.ceil((box.ymax / 1000) * h)));

  const pw = x1 - x0 + 1;
  const ph = y1 - y0 + 1;
  if (pw < 6 || ph < 6) return null;

  // Identify full-span structural wall rows/cols near the outer 3px perimeter so wall edges don't skew the symbol hull
  const excludedRows = new Set<number>();
  const excludedCols = new Set<number>();
  const edgeBandY = Math.min(3, Math.floor(ph * 0.2));
  const edgeBandX = Math.min(3, Math.floor(pw * 0.2));

  for (let y = y0; y <= y1; y++) {
    if (y - y0 < edgeBandY || y1 - y < edgeBandY) {
      let wallCount = 0;
      const base = y * w;
      for (let x = x0; x <= x1; x++) wallCount += wallMask[base + x];
      if (wallCount / pw > 0.72) excludedRows.add(y);
    }
  }
  for (let x = x0; x <= x1; x++) {
    if (x - x0 < edgeBandX || x1 - x < edgeBandX) {
      let wallCount = 0;
      for (let y = y0; y <= y1; y++) wallCount += wallMask[y * w + x];
      if (wallCount / ph > 0.72) excludedCols.add(x);
    }
  }

  // Collect extreme clean ink points per row (mathematically identical 2D convex hull to all interior points)
  const extremePoints: Point2D[] = [];
  let cleanInkCount = 0;
  let minPxX = Infinity;
  let maxPxX = -Infinity;
  let minPxY = Infinity;
  let maxPxY = -Infinity;

  for (let y = y0; y <= y1; y++) {
    if (excludedRows.has(y)) continue;
    let rowMinX = -1;
    let rowMaxX = -1;
    const base = y * w;
    for (let x = x0; x <= x1; x++) {
      if (excludedCols.has(x)) continue;
      if (inkMask[base + x] !== 1) continue;

      // Require >= 1 neighbor with ink to reject 1px JPEG noise specks
      let hasNeighbor = false;
      for (let dy = -1; dy <= 1 && !hasNeighbor; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        const nBase = ny * w;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          if (nx >= 0 && nx < w && inkMask[nBase + nx] === 1) {
            hasNeighbor = true;
            break;
          }
        }
      }
      if (!hasNeighbor) continue;

      cleanInkCount++;
      if (rowMinX === -1) rowMinX = x;
      rowMaxX = x;
    }

    if (rowMinX !== -1) {
      minPxX = Math.min(minPxX, rowMinX);
      maxPxX = Math.max(maxPxX, rowMaxX);
      minPxY = Math.min(minPxY, y);
      maxPxY = Math.max(maxPxY, y);
      extremePoints.push({
        x: ((rowMinX + 0.5) / w) * 1000,
        y: ((y + 0.5) / h) * 1000,
      });
      if (rowMaxX !== rowMinX) {
        extremePoints.push({
          x: ((rowMaxX + 0.5) / w) * 1000,
          y: ((y + 0.5) / h) * 1000,
        });
      }
    }
  }

  if (cleanInkCount < 10 || extremePoints.length < 4) return null;

  const spanRatioX = (maxPxX - minPxX + 1) / pw;
  const spanRatioY = (maxPxY - minPxY + 1) / ph;
  if (spanRatioX < 0.55 || spanRatioY < 0.55) return null;

  const rawHull = computeMonotoneConvexHull(extremePoints);
  if (rawHull.length < 3) return null;

  // Compute Shoelace area and centroid
  let area2 = 0;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < rawHull.length; i++) {
    const p1 = rawHull[i];
    const p2 = rawHull[(i + 1) % rawHull.length];
    area2 += p1.x * p2.y - p2.x * p1.y;
    cx += p1.x;
    cy += p1.y;
  }
  const hullArea = Math.abs(area2) * 0.5;
  const boxArea = bw * bh;
  if (hullArea < boxArea * 0.2 || hullArea > boxArea * 0.95) {
    return null;
  }

  cx /= rawHull.length;
  cy /= rawHull.length;

  // Gently pad vertices outward by 2.0 normalized units so outer CAD strokes are cleanly wrapped
  const pad = 2.0;
  const minClampX = Math.max(0, box.xmin - 1.5);
  const maxClampX = Math.min(1000, box.xmax + 1.5);
  const minClampY = Math.max(0, box.ymin - 1.5);
  const maxClampY = Math.min(1000, box.ymax + 1.5);

  const paddedHull: Point2D[] = [];
  for (const pt of rawHull) {
    const vx = pt.x - cx;
    const vy = pt.y - cy;
    const vlen = Math.hypot(vx, vy);
    const px =
      vlen > 1e-3
        ? Math.max(minClampX, Math.min(maxClampX, pt.x + (vx / vlen) * pad))
        : pt.x;
    const py =
      vlen > 1e-3
        ? Math.max(minClampY, Math.min(maxClampY, pt.y + (vy / vlen) * pad))
        : pt.y;
    if (
      paddedHull.length === 0 ||
      Math.hypot(
        px - paddedHull[paddedHull.length - 1].x,
        py - paddedHull[paddedHull.length - 1].y,
      ) >= 2.2
    ) {
      paddedHull.push({ x: px, y: py });
    }
  }

  if (paddedHull.length < 3) return null;
  const cornerRadius = Math.min(8, Math.min(bw, bh) * 0.16);
  return buildRoundedPolygonSvgPath(paddedHull, cornerRadius);
}

async function renderPdfPageToDataUrl(
  buffer: ArrayBuffer,
  pageNumber: number,
): Promise<{
  dataUrl: string;
  jpegBase64: string;
  tiles: SpatialTilePayload[];
  pageCount: number;
  width: number;
  height: number;
  extractedTexts: string[];
}> {
  const cloned = buffer.slice(0);
  const loadingTask = pdfjsLib.getDocument({ data: new Uint8Array(cloned) });
  const pdfDoc = await loadingTask.promise;
  const pageCount = pdfDoc.numPages;
  const safePage = Math.max(1, Math.min(pageCount, pageNumber));
  const page = await pdfDoc.getPage(safePage);

  const baseVp = page.getViewport({ scale: 1 });
  const maxDim = Math.max(baseVp.width, baseVp.height, 1);
  const renderScale = Math.max(1.5, Math.min(2.6, 1900 / maxDim));
  const viewport = page.getViewport({ scale: renderScale });

  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(viewport.width));
  canvas.height = Math.max(1, Math.round(viewport.height));
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Canvas 2D context unavailable for PDF sheet.');
  }
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  await page.render({
    canvas,
    canvasContext: ctx,
    viewport,
  }).promise;

  const extractedTexts: string[] = [];
  try {
    const textContent = await page.getTextContent();
    for (const item of textContent.items) {
      if ('str' in item && typeof item.str === 'string') {
        const trimmed = item.str.trim();
        if (trimmed) extractedTexts.push(trimmed);
      }
    }
  } catch {
    // Non-blocking
  }

  const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
  const jpegBase64 = dataUrl.replace(/^data:image\/jpeg;base64,/, '');
  const tiles = buildOverlappingSpatialTiles(canvas, canvas.width, canvas.height);
  void loadingTask.destroy();
  return {
    dataUrl,
    jpegBase64,
    tiles,
    pageCount,
    width: canvas.width,
    height: canvas.height,
    extractedTexts: Array.from(new Set(extractedTexts)).slice(0, 300),
  };
}

export const Viewport2D = forwardRef<Viewport2DHandle, Viewport2DProps>(
  function Viewport2D(
    {
      asset,
      activeHighlight,
      locating = false,
      onClearHighlight,
      onParsedSummary,
      onCalibrationChange,
      onImageReady,
      explanationOverlay,
    },
    ref,
  ) {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const imgRef = useRef<HTMLImageElement | null>(null);
    const cachedPayloadRef = useRef<CachedCapturePayload | null>(null);
    const inkGridRef = useRef<InkReferenceGrid | null>(null);

    const [zoom, setZoom] = useState(1);
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [dragging, setDragging] = useState(false);
    const dragStartRef = useRef({ x: 0, y: 0, px: 0, py: 0 });
    const [showGrid, setShowGrid] = useState(false);
    const [imgNaturalSize, setImgNaturalSize] = useState({
      width: 1600,
      height: 1200,
    });
    const [imgLoadTick, setImgLoadTick] = useState(0);

    // DXF / SVG / PDF rendered URL + CAD overlay data
    const [renderedSvgUrl, setRenderedSvgUrl] = useState<string | null>(null);
    const [svgMarkupStr, setSvgMarkupStr] = useState<string | null>(null);
    const [renderedPdfUrl, setRenderedPdfUrl] = useState<string | null>(null);
    const [pdfPage, setPdfPage] = useState(1);
    const [pdfPageCount, setPdfPageCount] = useState(1);
    const [pdfRendering, setPdfRendering] = useState(false);

    const [dxfOverlay, setDxfOverlay] = useState<{
      roomPolygons: DxfRoomPolygon[];
      blockInstances: DxfBlockInstance[];
      primitives: DxfPrimitive[];
      viewBox: { x: number; y: number; width: number; height: number };
    } | null>(null);

    // Interactive 2-point scale calibration tool
    const [calibrating, setCalibrating] = useState(false);
    const [pts, setPts] = useState<Array<{ x: number; y: number }>>([]);
    const [realMetersInput, setRealMetersInput] = useState('1.0');
    const [calibrationNote, setCalibrationNote] = useState<string | null>(null);

    const onParsedRef = useRef(onParsedSummary);
    onParsedRef.current = onParsedSummary;
    const onCalibRef = useRef(onCalibrationChange);
    onCalibRef.current = onCalibrationChange;
    const onReadyRef = useRef(onImageReady);
    onReadyRef.current = onImageReady;

    // Reset viewport state when switching files
    useEffect(() => {
      setZoom(1);
      setPan({ x: 0, y: 0 });
      setCalibrating(false);
      setPts([]);
      setCalibrationNote(null);
      setPdfPage(1);
      setPdfPageCount(1);
      setRenderedPdfUrl(null);
      cachedPayloadRef.current = null;
      inkGridRef.current = null;
      onCalibRef.current?.(null);

      if (asset.fileKind === '2d-dxf' && asset.rawText) {
        const parsed = parseDxfFile(asset.rawText);
        setSvgMarkupStr(parsed.svgMarkup);
        setDxfOverlay({
          roomPolygons: parsed.roomPolygons,
          blockInstances: parsed.blockInstances,
          primitives: parsed.primitives,
          viewBox: parsed.viewBox,
        });
        const blob = new Blob([parsed.svgMarkup], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        setRenderedSvgUrl(url);
        onParsedRef.current({ format: 'DXF', ...parsed.summary });
        return () => URL.revokeObjectURL(url);
      }

      setDxfOverlay(null);

      if (asset.fileKind === '2d-svg' && asset.rawText) {
        const normalizedSvg = ensureSvgExplicitDimensions(asset.rawText);
        setSvgMarkupStr(normalizedSvg);
        const blob = new Blob([normalizedSvg], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        setRenderedSvgUrl(url);
        try {
          const doc = new DOMParser().parseFromString(
            normalizedSvg,
            'image/svg+xml',
          );
          const svgEl = doc.querySelector('svg');
          const textNodes = Array.from(doc.querySelectorAll('text, tspan'))
            .map((el) => el.textContent?.trim() ?? '')
            .filter(Boolean);
          onParsedRef.current({
            format: 'SVG',
            viewBox: svgEl?.getAttribute('viewBox') ?? null,
            texts: Array.from(new Set(textNodes)).slice(0, 250),
          });
        } catch {
          // Non-blocking
        }
        return () => URL.revokeObjectURL(url);
      }

      setSvgMarkupStr(null);
      setRenderedSvgUrl(null);
    }, [asset]);

    // Render PDF page onto high-DPI canvas so pan, zoom, calibration, and 2D highlights work identically to images
    useEffect(() => {
      if (asset.fileKind !== '2d-pdf' || !asset.arrayBuffer) return;
      let cancelled = false;
      setPdfRendering(true);
      renderPdfPageToDataUrl(asset.arrayBuffer, pdfPage)
        .then((res) => {
          if (cancelled) return;
          setRenderedPdfUrl(res.dataUrl);
          setPdfPageCount(res.pageCount);
          setImgNaturalSize({ width: res.width, height: res.height });
          cachedPayloadRef.current = {
            key: `${asset.id}::${pdfPage}`,
            mimeType: 'image/jpeg',
            dataBase64: res.jpegBase64,
            tiles: res.tiles,
          };
          onParsedRef.current({
            format: 'PDF',
            page: pdfPage,
            pageCount: res.pageCount,
            texts: res.extractedTexts,
          });
        })
        .catch(() => {
          if (!cancelled) {
            setRenderedPdfUrl(null);
          }
        })
        .finally(() => {
          if (!cancelled) setPdfRendering(false);
        });
      return () => {
        cancelled = true;
      };
    }, [asset, pdfPage]);

    useImperativeHandle(ref, () => ({
      captureImagePayload: async () => {
        const expectedKey = `${asset.id}::${pdfPage}`;
        if (
          cachedPayloadRef.current &&
          cachedPayloadRef.current.key === expectedKey
        ) {
          return {
            mimeType: cachedPayloadRef.current.mimeType,
            dataBase64: cachedPayloadRef.current.dataBase64,
            tiles: cachedPayloadRef.current.tiles,
            calibrationNote: calibrationNote ?? undefined,
          };
        }

        // Fast synchronous encode if <img> is already loaded in DOM
        if (
          imgRef.current &&
          imgRef.current.complete &&
          imgRef.current.naturalWidth > 0
        ) {
          const encoded = encodeImageElementToJpegPayload(imgRef.current);
          if (encoded) {
            cachedPayloadRef.current = {
              key: expectedKey,
              mimeType: encoded.mimeType,
              dataBase64: encoded.dataBase64,
              tiles: encoded.tiles,
            };
            return {
              ...encoded,
              calibrationNote: calibrationNote ?? undefined,
            };
          }
        }

        // Fallback for PDFs if called before initial render finishes
        if (asset.fileKind === '2d-pdf' && asset.arrayBuffer) {
          try {
            const rendered = await renderPdfPageToDataUrl(
              asset.arrayBuffer,
              pdfPage,
            );
            setRenderedPdfUrl(rendered.dataUrl);
            setPdfPageCount(rendered.pageCount);
            cachedPayloadRef.current = {
              key: expectedKey,
              mimeType: 'image/jpeg',
              dataBase64: rendered.jpegBase64,
              tiles: rendered.tiles,
            };
            return {
              mimeType: 'image/jpeg',
              dataBase64: rendered.jpegBase64,
              tiles: rendered.tiles,
              calibrationNote: calibrationNote ?? undefined,
            };
          } catch {
            const bytes = new Uint8Array(asset.arrayBuffer);
            let binary = '';
            const chunk = 0x8000;
            for (let i = 0; i < bytes.length; i += chunk) {
              binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
            }
            return {
              mimeType: 'application/pdf',
              dataBase64: btoa(binary),
              calibrationNote: calibrationNote ?? undefined,
            };
          }
        }

        let tempObjectUrl: string | null = null;
        let srcUrl = renderedSvgUrl || asset.previewUrl || null;

        if (!srcUrl && asset.rawText) {
          if (asset.fileKind === '2d-dxf') {
            const svgStr = svgMarkupStr ?? parseDxfFile(asset.rawText).svgMarkup;
            tempObjectUrl = URL.createObjectURL(
              new Blob([svgStr], { type: 'image/svg+xml' }),
            );
            srcUrl = tempObjectUrl;
          } else if (asset.fileKind === '2d-svg') {
            const normalizedSvg = ensureSvgExplicitDimensions(asset.rawText);
            tempObjectUrl = URL.createObjectURL(
              new Blob([normalizedSvg], { type: 'image/svg+xml' }),
            );
            srcUrl = tempObjectUrl;
          }
        }

        if (!srcUrl) return null;

        return new Promise((resolve) => {
          const cleanup = () => {
            if (tempObjectUrl) {
              URL.revokeObjectURL(tempObjectUrl);
              tempObjectUrl = null;
            }
          };
          const img = new Image();
          img.crossOrigin = 'anonymous';
          img.onload = () => {
            const encoded = encodeImageElementToJpegPayload(img);
            cleanup();
            if (!encoded) {
              resolve(null);
              return;
            }
            cachedPayloadRef.current = {
              key: expectedKey,
              mimeType: encoded.mimeType,
              dataBase64: encoded.dataBase64,
              tiles: encoded.tiles,
            };
            resolve({
              ...encoded,
              calibrationNote: calibrationNote ?? undefined,
            });
          };
          img.onerror = () => {
            cleanup();
            resolve(null);
          };
          img.src = srcUrl;
        });
      },
    }));

    // Normalized [0..1000] regions from Gemini / cached highlight target (preserving Space vs Object per subTarget)
    const box2DMatches = useMemo<NormalizedOverlayBox[]>(() => {
      if (!activeHighlight) return [];

      const sanitizeList = (
        list: NonNullable<HighlightTarget['boxes2D']>,
        isSpace: boolean,
        expectedCount?: number,
      ): NormalizedOverlayBox[] => {
        const valid = list
          .map((b) => {
            const xmin = Math.max(0, Math.min(1000, Math.min(b.xmin, b.xmax)));
            const xmax = Math.max(0, Math.min(1000, Math.max(b.xmin, b.xmax)));
            const ymin = Math.max(0, Math.min(1000, Math.min(b.ymin, b.ymax)));
            const ymax = Math.max(0, Math.min(1000, Math.max(b.ymin, b.ymax)));
            return { xmin, ymin, xmax, ymax, label: b.label, isSpace };
          })
          .filter((b) => b.xmax - b.xmin >= 4 && b.ymax - b.ymin >= 4);

        if (!isSpace && expectedCount && expectedCount > 0 && valid.length > expectedCount) {
          return valid.slice(0, expectedCount);
        }
        return valid;
      };

      if (activeHighlight.subTargets && activeHighlight.subTargets.length > 0) {
        const fromSubs = activeHighlight.subTargets.flatMap((sub) =>
          sanitizeList(
            sub.boxes2D ?? [],
            sub.category === 'Space',
            sub.expectedCount,
          ),
        );
        if (fromSubs.length > 0) return fromSubs;
      }

      return sanitizeList(
        activeHighlight.boxes2D ?? [],
        activeHighlight.category === 'Space',
        activeHighlight.expectedCount,
      );
    }, [activeHighlight]);

    // Match DXF CAD roomPolygons and exact vector primitives (lines/arcs/circles) without rectangular borders
    const dxfMatches = useMemo(() => {
      if (!activeHighlight || !dxfOverlay) {
        return {
          polygons: [] as DxfRoomPolygon[],
          blocks: [] as DxfBlockInstance[],
          colorPrimitives: [] as DxfPrimitive[],
          matchedItemCount: 0,
        };
      }
      const subList =
        activeHighlight.subTargets && activeHighlight.subTargets.length > 0
          ? activeHighlight.subTargets
          : [activeHighlight];

      const polygons = dxfOverlay.roomPolygons.filter((rp) =>
        subList.some((sub) => matchesSingleDxfPolygon(rp, sub)),
      );

      let blocks = dxfOverlay.blockInstances.filter((blk) =>
        subList.some((sub) => matchesSingleDxfBlock(blk, sub)),
      );

      if (
        activeHighlight.expectedCount &&
        activeHighlight.expectedCount > 0 &&
        blocks.length !== activeHighlight.expectedCount &&
        box2DMatches.length > 0
      ) {
        blocks = [];
      }

      const objectBoxes = box2DMatches.filter((b) => !b.isSpace);
      const { x: vbX, y: vbY, width: vbW, height: vbH } = dxfOverlay.viewBox;

      const getPrimSvgGeom = (
        p: DxfPrimitive,
      ): { cx: number; cy: number; span: number } | null => {
        if (p.kind === 'line') {
          return {
            cx: (p.x1 + p.x2) * 0.5,
            cy: -(p.y1 + p.y2) * 0.5,
            span: Math.hypot(p.x2 - p.x1, p.y2 - p.y1),
          };
        }
        if (p.kind === 'circle' || p.kind === 'arc') {
          return {
            cx: p.cx,
            cy: -p.cy,
            span: p.r * 2,
          };
        }
        return null;
      };

      let colorPrimitives: DxfPrimitive[] = [];
      let matchedItemCount = 0;

      if (blocks.length > 0) {
        matchedItemCount = blocks.length;
        const svgBlockRects = blocks.map((blk) => {
          const w = Math.max(1e-3, blk.xmax - blk.xmin);
          const h = Math.max(1e-3, blk.ymax - blk.ymin);
          const pad = Math.max(w, h) * 0.06;
          return {
            x0: blk.xmin - pad,
            x1: blk.xmax + pad,
            y0: -blk.ymax - pad,
            y1: -blk.ymin + pad,
            maxSpan: Math.max(w, h) * 1.25,
          };
        });
        colorPrimitives = dxfOverlay.primitives
          .filter((p) => {
            const g = getPrimSvgGeom(p);
            if (!g) return false;
            return svgBlockRects.some(
              (r) =>
                g.cx >= r.x0 &&
                g.cx <= r.x1 &&
                g.cy >= r.y0 &&
                g.cy <= r.y1 &&
                g.span <= r.maxSpan,
            );
          })
          .slice(0, 2400);
      } else if (objectBoxes.length > 0 && vbW > 0 && vbH > 0) {
        matchedItemCount = objectBoxes.length;
        const svgVisionRects = objectBoxes.map((b) => {
          const x0 = vbX + (b.xmin / 1000) * vbW;
          const x1 = vbX + (b.xmax / 1000) * vbW;
          const y0 = vbY + (b.ymin / 1000) * vbH;
          const y1 = vbY + (b.ymax / 1000) * vbH;
          const maxDim = Math.max(x1 - x0, y1 - y0);
          const pad = maxDim * 0.05;
          return {
            x0: x0 - pad,
            x1: x1 + pad,
            y0: y0 - pad,
            y1: y1 + pad,
            maxSpan: maxDim * 1.3,
          };
        });
        colorPrimitives = dxfOverlay.primitives
          .filter((p) => {
            const g = getPrimSvgGeom(p);
            if (!g) return false;
            return svgVisionRects.some(
              (r) =>
                g.cx >= r.x0 &&
                g.cx <= r.x1 &&
                g.cy >= r.y0 &&
                g.cy <= r.y1 &&
                g.span <= r.maxSpan,
            );
          })
          .slice(0, 2400);
      } else if (polygons.length === 0 && box2DMatches.length === 0) {
        colorPrimitives = dxfOverlay.primitives
          .filter((p) =>
            subList.some((sub) => matchesSingleDxfPrimitive(p, sub)),
          )
          .slice(0, 1800);
        matchedItemCount = colorPrimitives.length;
      }

      return { polygons, blocks, colorPrimitives, matchedItemCount };
    }, [activeHighlight, box2DMatches, dxfOverlay]);

    // Convert DXF blockInstances into normalized [0..1000] bounds for Focus & fallback
    const dxfBlockOverlayBoxes = useMemo<NormalizedOverlayBox[]>(() => {
      if (!dxfOverlay || dxfMatches.blocks.length === 0) return [];
      const { x: vbX, y: vbY, width: vbW, height: vbH } = dxfOverlay.viewBox;
      if (vbW <= 0 || vbH <= 0) return [];
      return dxfMatches.blocks.map((blk) => {
        const svgXMin = blk.xmin;
        const svgXMax = blk.xmax;
        const svgYMin = -blk.ymax;
        const svgYMax = -blk.ymin;
        const xmin = Math.max(
          0,
          Math.min(1000, ((svgXMin - vbX) / vbW) * 1000),
        );
        const xmax = Math.max(
          0,
          Math.min(1000, ((svgXMax - vbX) / vbW) * 1000),
        );
        const ymin = Math.max(
          0,
          Math.min(1000, ((svgYMin - vbY) / vbH) * 1000),
        );
        const ymax = Math.max(
          0,
          Math.min(1000, ((svgYMax - vbY) / vbH) * 1000),
        );
        return {
          xmin,
          ymin,
          xmax: Math.max(xmin + 10, xmax),
          ymax: Math.max(ymin + 10, ymax),
          label: blk.blockName,
          isSpace: false,
        };
      });
    }, [dxfMatches.blocks, dxfOverlay]);

    // Regions rendered by the clean SVG multiply/screen marker wash (all non-DXF blueprints, plus any DXF items not already covered by CAD vector polygons/primitives)
    const rawColorRegions = useMemo<NormalizedOverlayBox[]>(() => {
      if (!activeHighlight) return [];
      if (!dxfOverlay) return box2DMatches;

      const remaining: NormalizedOverlayBox[] = [];
      for (const b of box2DMatches) {
        if (b.isSpace && dxfMatches.polygons.length > 0) continue;
        if (!b.isSpace && dxfMatches.colorPrimitives.length > 0) continue;
        remaining.push(b);
      }
      if (
        remaining.length === 0 &&
        dxfBlockOverlayBoxes.length > 0 &&
        dxfMatches.colorPrimitives.length === 0
      ) {
        return dxfBlockOverlayBoxes;
      }
      return remaining;
    }, [
      activeHighlight,
      box2DMatches,
      dxfBlockOverlayBoxes,
      dxfMatches.colorPrimitives.length,
      dxfMatches.polygons.length,
      dxfOverlay,
    ]);

    // Snap each highlight region tightly onto actual blueprint ink and extract contour-hugging symbol hulls (< 1ms)
    const tightenedColorShapes = useMemo<
      Array<NormalizedOverlayBox & { silhouettePath: string | null }>
    >(() => {
      void imgLoadTick;
      const grid = inkGridRef.current;
      return rawColorRegions.map((reg) => {
        const tightened = tightenNormalizedBoxWithInk(reg, grid);
        const silhouettePath = buildObjectSilhouettePath(tightened, grid);
        return { ...tightened, silhouettePath };
      });
    }, [rawColorRegions, imgLoadTick]);

    const tightenedColorRegions = tightenedColorShapes;

    const isDarkBlueprint = useMemo(() => {
      void imgLoadTick;
      return inkGridRef.current?.isDarkBlueprint ?? false;
    }, [imgLoadTick]);

    const effectiveBoxes = useMemo<NormalizedOverlayBox[]>(() => {
      if (dxfBlockOverlayBoxes.length > 0) return dxfBlockOverlayBoxes;
      return tightenedColorRegions.length > 0
        ? tightenedColorRegions
        : box2DMatches;
    }, [box2DMatches, dxfBlockOverlayBoxes, tightenedColorRegions]);

    const totalHighlightCount = useMemo(() => {
      if (!activeHighlight) return 0;
      if (dxfOverlay) {
        const polyCount = dxfMatches.polygons.length;
        const primItemCount = dxfMatches.matchedItemCount;
        if (polyCount > 0 || primItemCount > 0) {
          return polyCount + primItemCount + tightenedColorRegions.length;
        }
      }
      return effectiveBoxes.length;
    }, [
      activeHighlight,
      dxfMatches.matchedItemCount,
      dxfMatches.polygons.length,
      dxfOverlay,
      effectiveBoxes.length,
      tightenedColorRegions.length,
    ]);

    // Compute union bounds in [0..1] image fraction for "Focus" zoom-to-highlight
    const highlightNormalizedBounds = useMemo<{
      minX: number;
      minY: number;
      maxX: number;
      maxY: number;
    } | null>(() => {
      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;

      for (const b of effectiveBoxes) {
        minX = Math.min(minX, b.xmin / 1000);
        minY = Math.min(minY, b.ymin / 1000);
        maxX = Math.max(maxX, b.xmax / 1000);
        maxY = Math.max(maxY, b.ymax / 1000);
      }

      if (dxfOverlay) {
        const { x: vbX, y: vbY, width: vbW, height: vbH } = dxfOverlay.viewBox;
        if (vbW > 0 && vbH > 0) {
          for (const rp of dxfMatches.polygons) {
            for (const pt of rp.pts) {
              const nx = (pt.x - vbX) / vbW;
              const ny = (-pt.y - vbY) / vbH;
              minX = Math.min(minX, nx);
              minY = Math.min(minY, ny);
              maxX = Math.max(maxX, nx);
              maxY = Math.max(maxY, ny);
            }
          }
          for (const p of dxfMatches.colorPrimitives) {
            if (p.kind === 'line') {
              const nx1 = (p.x1 - vbX) / vbW;
              const ny1 = (-p.y1 - vbY) / vbH;
              const nx2 = (p.x2 - vbX) / vbW;
              const ny2 = (-p.y2 - vbY) / vbH;
              minX = Math.min(minX, nx1, nx2);
              minY = Math.min(minY, ny1, ny2);
              maxX = Math.max(maxX, nx1, nx2);
              maxY = Math.max(maxY, ny1, ny2);
            } else if (p.kind === 'circle' || p.kind === 'arc') {
              const nx1 = (p.cx - p.r - vbX) / vbW;
              const nx2 = (p.cx + p.r - vbX) / vbW;
              const ny1 = (-p.cy - p.r - vbY) / vbH;
              const ny2 = (-p.cy + p.r - vbY) / vbH;
              minX = Math.min(minX, nx1, nx2);
              minY = Math.min(minY, ny1, ny2);
              maxX = Math.max(maxX, nx1, nx2);
              maxY = Math.max(maxY, ny1, ny2);
            }
          }
        }
      }

      if (!Number.isFinite(minX) || !Number.isFinite(minY)) return null;
      return {
        minX: Math.max(0, minX),
        minY: Math.max(0, minY),
        maxX: Math.min(1, maxX),
        maxY: Math.min(1, maxY),
      };
    }, [dxfMatches.colorPrimitives, dxfMatches.polygons, dxfOverlay, effectiveBoxes]);

    const focusOnHighlight = useCallback(() => {
      if (!highlightNormalizedBounds || !imgRef.current || !containerRef.current) {
        return;
      }
      const baseW = imgRef.current.offsetWidth;
      const baseH = imgRef.current.offsetHeight;
      const stageRect = containerRef.current.getBoundingClientRect();
      if (baseW <= 0 || baseH <= 0 || stageRect.width <= 0 || stageRect.height <= 0) {
        return;
      }
      const spanX = Math.max(
        0.14,
        highlightNormalizedBounds.maxX - highlightNormalizedBounds.minX,
      );
      const spanY = Math.max(
        0.14,
        highlightNormalizedBounds.maxY - highlightNormalizedBounds.minY,
      );
      const targetZoomX = (stageRect.width * 0.62) / (baseW * spanX);
      const targetZoomY = (stageRect.height * 0.62) / (baseH * spanY);
      const nextZoom = Math.max(
        1,
        Math.min(2.8, Number(Math.min(targetZoomX, targetZoomY).toFixed(2))),
      );

      const cx =
        (highlightNormalizedBounds.minX + highlightNormalizedBounds.maxX) / 2;
      const cy =
        (highlightNormalizedBounds.minY + highlightNormalizedBounds.maxY) / 2;
      const offsetX = (0.5 - cx) * baseW * nextZoom;
      const offsetY = (0.5 - cy) * baseH * nextZoom;

      setZoom(nextZoom);
      setPan({ x: Math.round(offsetX), y: Math.round(offsetY) });
    }, [highlightNormalizedBounds]);

    const handleWheel = (e: React.WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.12 : 0.9;
      setZoom((z) =>
        Math.min(8, Math.max(0.35, Number((z * factor).toFixed(2)))),
      );
    };

    const handleMouseDown = (e: React.MouseEvent) => {
      if (calibrating) return;
      setDragging(true);
      dragStartRef.current = {
        x: e.clientX,
        y: e.clientY,
        px: pan.x,
        py: pan.y,
      };
    };

    const handleMouseMove = (e: React.MouseEvent) => {
      if (!dragging) return;
      setPan({
        x: dragStartRef.current.px + (e.clientX - dragStartRef.current.x),
        y: dragStartRef.current.py + (e.clientY - dragStartRef.current.y),
      });
    };

    const handleMouseUp = () => {
      setDragging(false);
    };

    const handleStageClick = (e: React.MouseEvent) => {
      if (!calibrating || !imgRef.current) return;
      const rect = imgRef.current.getBoundingClientRect();
      if (
        e.clientX < rect.left ||
        e.clientX > rect.right ||
        e.clientY < rect.top ||
        e.clientY > rect.bottom
      ) {
        return;
      }
      const nx = (e.clientX - rect.left) / rect.width;
      const ny = (e.clientY - rect.top) / rect.height;
      setPts((prev) =>
        prev.length >= 2 ? [{ x: nx, y: ny }] : [...prev, { x: nx, y: ny }],
      );
    };

    const applyCalibration = () => {
      if (pts.length !== 2 || !imgRef.current) return;
      const realM = parseFloat(realMetersInput);
      if (!Number.isFinite(realM) || realM <= 0) return;
      const natW = imgRef.current.naturalWidth || imgNaturalSize.width || 1000;
      const natH = imgRef.current.naturalHeight || imgNaturalSize.height || 1000;
      const dx = (pts[1].x - pts[0].x) * natW;
      const dy = (pts[1].y - pts[0].y) * natH;
      const pxDist = Math.hypot(dx, dy);
      if (pxDist < 4) return;
      const metersPerPx = realM / pxDist;
      const imgWidthM = natW * metersPerPx;
      const imgHeightM = natH * metersPerPx;
      const note = `User-calibrated reference line: ${pxDist.toFixed(0)} px = ${realM.toFixed(2)} m (full canvas spans ${imgWidthM.toFixed(2)} m × ${imgHeightM.toFixed(2)} m, ${(imgWidthM * imgHeightM).toFixed(1)} m² bounding rectangle).`;
      setCalibrationNote(note);
      onCalibRef.current?.(note);
      setCalibrating(false);
    };

    const displayUrl =
      asset.fileKind === '2d-pdf'
        ? renderedPdfUrl
        : renderedSvgUrl || asset.previewUrl;

    // Fallback iframe ONLY if a PDF fails canvas rasterization
    if (
      asset.fileKind === '2d-pdf' &&
      !pdfRendering &&
      !renderedPdfUrl &&
      asset.previewUrl
    ) {
      return (
        <div className="relative w-full h-full min-h-[480px] bg-[#F3F1EB] flex flex-col">
          <div className="flex items-center justify-between px-4 py-2.5 bg-white border-b border-[#E5E2DC]">
            <span className="text-xs font-medium text-[#161614] truncate">
              {asset.name}
            </span>
            <span className="text-xs font-mono text-[#5C5A54]">
              PDF Drawing Sheet
            </span>
          </div>
          <iframe
            src={asset.previewUrl}
            title={asset.name}
            className="w-full flex-1 border-0 bg-white"
          />
        </div>
      );
    }

    const natW = Math.max(100, imgNaturalSize.width);
    const natH = Math.max(100, imgNaturalSize.height);

    return (
      <div
        ref={containerRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={handleStageClick}
        className={`relative w-full h-full min-h-[480px] bg-[#EFECE5] overflow-hidden select-none flex items-center justify-center ${
          showGrid ? 'drafting-grid' : ''
        } ${calibrating ? 'cursor-crosshair' : dragging ? 'cursor-grabbing' : 'cursor-grab'}`}
      >
        {/* Top-left file readout */}
        <div className="absolute top-3.5 left-3.5 z-10 bg-white/95 border border-[#E2DFD7] rounded-lg px-3 py-1.5 shadow-sm pointer-events-none">
          <div className="text-xs font-medium text-[#161614] truncate max-w-[260px]">
            {asset.name}
          </div>
          <div className="text-[11px] font-mono tabular-nums text-[#5C5A54] mt-0.5">
            {Math.round(zoom * 100)}% zoom
            {asset.fileKind === '2d-pdf' && pdfPageCount > 1
              ? ` · Page ${pdfPage}/${pdfPageCount}`
              : ''}
            {calibrationNote ? ' · Scale calibrated' : ''}
          </div>
        </div>

        {/* Top-right active highlight banner */}
        {activeHighlight && (
          <div
            onClick={(e) => e.stopPropagation()}
            className="absolute top-3.5 right-3.5 z-20 flex items-center gap-2 bg-[#161614] text-white rounded-lg px-3 py-1.5 shadow-md max-w-[360px]"
          >
            {locating && totalHighlightCount === 0 ? (
              <Loader2 className="w-3.5 h-3.5 text-[#F59E0B] animate-spin shrink-0" />
            ) : (
              <Crosshair className="w-3.5 h-3.5 text-[#F59E0B] shrink-0" />
            )}
            <div className="min-w-0">
              <div className="text-[11px] font-medium truncate">
                {activeHighlight.label}
              </div>
              <div className="text-[10px] font-mono text-[#D6D3CB] tabular-nums">
                {locating && totalHighlightCount === 0
                  ? 'Locating symbols in blueprint…'
                  : totalHighlightCount > 0
                    ? `${totalHighlightCount} matched in blueprint`
                    : 'No matching symbols located'}
              </div>
            </div>
            {highlightNormalizedBounds && totalHighlightCount > 0 && (
              <button
                type="button"
                onClick={focusOnHighlight}
                title="Zoom to highlighted elements"
                className="inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-medium rounded bg-white/12 hover:bg-white/22 text-white transition-colors cursor-pointer shrink-0"
              >
                <Focus className="w-3 h-3" />
                Focus
              </button>
            )}
            {onClearHighlight && (
              <button
                type="button"
                onClick={() => {
                  onClearHighlight();
                }}
                title="Clear highlight"
                className="p-1 rounded hover:bg-white/10 text-[#D6D3CB] hover:text-white transition-colors cursor-pointer shrink-0"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        )}

        {/* PDF sheet loading state */}
        {asset.fileKind === '2d-pdf' && pdfRendering && !displayUrl && (
          <div className="flex items-center gap-2 text-xs font-medium text-[#5C5A54] bg-white/90 px-4 py-2.5 rounded-lg border border-[#E2DFD7] shadow-xs">
            <Loader2 className="w-4 h-4 animate-spin text-[#1F4E3D]" />
            Rendering PDF blueprint sheet…
          </div>
        )}

        {/* Pixel-locked Image / SVG / PDF Canvas + Clean Architectural Marker Layers */}
        {displayUrl && (
          <div
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: 'center center',
            }}
            className="relative inline-block transition-transform duration-100 ease-out"
          >
            <img
              ref={imgRef}
              src={displayUrl}
              alt={asset.name}
              referrerPolicy="no-referrer"
              draggable={false}
              onLoad={(e) => {
                const imgEl = e.currentTarget;
                const nw = imgEl.naturalWidth;
                const nh = imgEl.naturalHeight;
                if (nw > 0 && nh > 0) {
                  setImgNaturalSize({ width: nw, height: nh });
                }
                const cacheKey = `${asset.id}::${pdfPage}`;
                if (
                  !cachedPayloadRef.current ||
                  cachedPayloadRef.current.key !== cacheKey
                ) {
                  const encoded = encodeImageElementToJpegPayload(imgEl);
                  if (encoded) {
                    cachedPayloadRef.current = {
                      key: cacheKey,
                      mimeType: encoded.mimeType,
                      dataBase64: encoded.dataBase64,
                      tiles: encoded.tiles,
                    };
                  }
                }
                const grid = buildInkReferenceGrid(imgEl, cacheKey);
                if (grid) {
                  inkGridRef.current = grid;
                }
                setImgLoadTick((t) => t + 1);
                onReadyRef.current?.();
              }}
              className="block max-h-[68vh] max-w-[82vw] w-auto h-auto bg-white shadow-sm border border-[#E2DFD7]"
            />

            {/* Crisp Architectural Marker Highlight Layer (multiply/screen blend: contour-hugging symbol silhouettes, seam-free room groups, zero borders, zero numbers, razor-sharp black CAD lines) */}
            {activeHighlight && tightenedColorShapes.length > 0 && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox="0 0 1000 1000"
                preserveAspectRatio="none"
                style={{
                  mixBlendMode: isDarkBlueprint ? 'screen' : 'multiply',
                }}
              >
                {/* Space/Room Wash Group: uniform opacity on <g> eliminates seams across L-shaped multi-rectangle rooms */}
                <g opacity={0.34}>
                  {tightenedColorShapes
                    .filter((reg) => reg.isSpace)
                    .map((reg, idx) => {
                      const bw = Math.max(4, reg.xmax - reg.xmin);
                      const bh = Math.max(4, reg.ymax - reg.ymin);
                      const rx = Math.min(8, Math.min(bw, bh) * 0.03);
                      return (
                        <rect
                          key={`hl-space-${idx}`}
                          x={reg.xmin}
                          y={reg.ymin}
                          width={bw}
                          height={bh}
                          rx={rx}
                          ry={rx}
                          fill="#FBBF24"
                          fillOpacity={1}
                          stroke="none"
                        />
                      );
                    })}
                </g>

                {/* Object/Symbol Wash Group: contour-hugging convex-hull silhouettes + uniform opacity on <g> so adjacent chairs/leaves never double-darken */}
                <g opacity={0.48}>
                  {tightenedColorShapes
                    .filter((reg) => !reg.isSpace)
                    .map((reg, idx) => {
                      if (reg.silhouettePath) {
                        return (
                          <path
                            key={`hl-obj-${idx}`}
                            d={reg.silhouettePath}
                            fill="#FF7A1A"
                            fillOpacity={1}
                            stroke="none"
                          />
                        );
                      }
                      const bw = Math.max(4, reg.xmax - reg.xmin);
                      const bh = Math.max(4, reg.ymax - reg.ymin);
                      const rx = Math.min(14, Math.min(bw, bh) * 0.16);
                      return (
                        <rect
                          key={`hl-obj-${idx}`}
                          x={reg.xmin}
                          y={reg.ymin}
                          width={bw}
                          height={bh}
                          rx={rx}
                          ry={rx}
                          fill="#FF7A1A"
                          fillOpacity={1}
                          stroke="none"
                        />
                      );
                    })}
                </g>
              </svg>
            )}

            {/* PlanQA explanation layer: heat map of model uncertainty + evidence boxes (role colour, counting order) */}
            {explanationOverlay?.heatmapUrl && (
              <img
                src={explanationOverlay.heatmapUrl}
                alt=""
                draggable={false}
                className="absolute inset-0 w-full h-full pointer-events-none"
                style={{ opacity: explanationOverlay.heatmapOpacity ?? 0.75 }}
              />
            )}
            {explanationOverlay?.marks && explanationOverlay.marks.length > 0 && (
              <div className="absolute inset-0 pointer-events-none">
                {explanationOverlay.marks.map((m, i) => (
                  <div
                    key={`expl-${i}`}
                    title={m.label}
                    className="absolute rounded-[2px]"
                    style={{
                      left: `${m.xmin / 10}%`,
                      top: `${m.ymin / 10}%`,
                      width: `${Math.max(0.4, (m.xmax - m.xmin) / 10)}%`,
                      height: `${Math.max(0.4, (m.ymax - m.ymin) / 10)}%`,
                      border: `${Math.max(1, (m.thick ? 3.5 : 2) / zoom)}px ${m.dashed ? 'dashed' : 'solid'} ${m.color}`,
                      background: `${m.color}${Math.round(Math.min(1, Math.max(0, m.fillOpacity ?? 0.12)) * 255)
                        .toString(16)
                        .padStart(2, '0')}`,
                      boxShadow: m.thick ? `0 0 0 ${2 / zoom}px #fff` : undefined,
                    }}
                  >
                    {m.order !== undefined && (
                      <span
                        className="absolute -top-2 -left-2 rounded-full text-white font-bold flex items-center justify-center shadow-sm"
                        style={{
                          background: m.color,
                          minWidth: `${16 / zoom}px`,
                          height: `${16 / zoom}px`,
                          fontSize: `${9.5 / zoom}px`,
                          padding: `0 ${3 / zoom}px`,
                        }}
                      >
                        {m.order}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* DXF Direct Vector Color Highlighting (exact room polygon wash + crisp CAD line/arc/circle recoloring, zero borders or numbers) */}
            {dxfOverlay &&
              activeHighlight &&
              (dxfMatches.polygons.length > 0 ||
                dxfMatches.colorPrimitives.length > 0) && (
                <svg
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  viewBox={`${dxfOverlay.viewBox.x} ${dxfOverlay.viewBox.y} ${dxfOverlay.viewBox.width} ${dxfOverlay.viewBox.height}`}
                  preserveAspectRatio="none"
                >
                  {dxfMatches.polygons.map((rp, i) => {
                    const pointsAttr = rp.pts
                      .map((p) => `${p.x.toFixed(3)},${(-p.y).toFixed(3)}`)
                      .join(' ');
                    return (
                      <polygon
                        key={`poly-${i}`}
                        points={pointsAttr}
                        fill="#FBBF24"
                        fillOpacity="0.35"
                        stroke="none"
                        style={{ mixBlendMode: 'multiply' }}
                      />
                    );
                  })}

                  {dxfMatches.colorPrimitives.map((p, i) => {
                    const sw =
                      Math.max(
                        dxfOverlay.viewBox.width,
                        dxfOverlay.viewBox.height,
                      ) * 0.003;
                    if (p.kind === 'line') {
                      return (
                        <line
                          key={`prim-${i}`}
                          x1={p.x1}
                          y1={-p.y1}
                          x2={p.x2}
                          y2={-p.y2}
                          stroke="#EA580C"
                          strokeWidth="2.4"
                          vectorEffect="non-scaling-stroke"
                          strokeLinecap="round"
                        />
                      );
                    }
                    if (p.kind === 'circle') {
                      return (
                        <circle
                          key={`prim-${i}`}
                          cx={p.cx}
                          cy={-p.cy}
                          r={Math.max(p.r, sw * 1.5)}
                          fill="#FF7A1A"
                          fillOpacity="0.3"
                          stroke="#EA580C"
                          strokeWidth="2.2"
                          vectorEffect="non-scaling-stroke"
                        />
                      );
                    }
                    if (p.kind === 'arc') {
                      const a1 = (p.startDeg * Math.PI) / 180;
                      const a2 = (p.endDeg * Math.PI) / 180;
                      const x1 = p.cx + p.r * Math.cos(a1);
                      const y1 = -(p.cy + p.r * Math.sin(a1));
                      const x2 = p.cx + p.r * Math.cos(a2);
                      const y2 = -(p.cy + p.r * Math.sin(a2));
                      const sweep = (p.endDeg - p.startDeg + 360) % 360;
                      const largeArc = sweep > 180 ? 1 : 0;
                      const dAttr = `M ${x1} ${y1} A ${p.r} ${p.r} 0 ${largeArc} 0 ${x2} ${y2}`;
                      return (
                        <path
                          key={`prim-${i}`}
                          d={dAttr}
                          fill="none"
                          stroke="#EA580C"
                          strokeWidth="2.4"
                          vectorEffect="non-scaling-stroke"
                          strokeLinecap="round"
                        />
                      );
                    }
                    return null;
                  })}
                </svg>
              )}

            {/* Calibration line overlay */}
            {pts.length > 0 && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox={`0 0 ${natW} ${natH}`}
                preserveAspectRatio="none"
              >
                {pts.length === 2 && (
                  <line
                    x1={pts[0].x * natW}
                    y1={pts[0].y * natH}
                    x2={pts[1].x * natW}
                    y2={pts[1].y * natH}
                    stroke="#1F4E3D"
                    strokeWidth="2.5"
                    vectorEffect="non-scaling-stroke"
                  />
                )}
                {pts.map((p, i) => (
                  <circle
                    key={i}
                    cx={p.x * natW}
                    cy={p.y * natH}
                    r={Math.max(6, Math.min(natW, natH) * 0.007)}
                    fill="#1F4E3D"
                    stroke="#ffffff"
                    strokeWidth={Math.max(2, Math.min(natW, natH) * 0.0025)}
                  />
                ))}
              </svg>
            )}
          </div>
        )}

        {/* Calibration prompt bar */}
        {calibrating && (
          <div
            onClick={(e) => e.stopPropagation()}
            className="absolute top-14 right-3.5 z-20 bg-white border border-[#1F4E3D]/30 rounded-xl p-3 shadow-md max-w-xs"
          >
            <p className="text-xs font-semibold text-[#161614]">
              Calibrate Drawing Scale
            </p>
            <p className="text-[11px] text-[#5C5A54] mt-0.5">
              Click 2 points on a known wall, doorway (~0.9 m), or scale bar.
            </p>
            {pts.length === 2 && (
              <div className="mt-2 flex items-center gap-2">
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  value={realMetersInput}
                  onChange={(e) => setRealMetersInput(e.target.value)}
                  className="w-20 px-2 py-1 text-xs font-mono border border-[#D8D4CB] rounded-md focus:outline-none focus:ring-2 focus:ring-[#1F4E3D]"
                />
                <span className="text-xs text-[#5C5A54]">metres</span>
                <button
                  type="button"
                  onClick={applyCalibration}
                  className="ml-auto px-2.5 py-1 text-xs font-medium bg-[#1F4E3D] text-white rounded-md hover:bg-[#183E30] cursor-pointer whitespace-nowrap shrink-0"
                >
                  Apply
                </button>
              </div>
            )}
          </div>
        )}

        {/* Bottom toolbar */}
        <div
          onClick={(e) => e.stopPropagation()}
          className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/95 border border-[#E2DFD7] shadow-md"
        >
          {asset.fileKind === '2d-pdf' && pdfPageCount > 1 && (
            <>
              <button
                type="button"
                disabled={pdfPage <= 1 || pdfRendering}
                onClick={() => {
                  setPdfPage((p) => Math.max(1, p - 1));
                  onClearHighlight?.();
                }}
                title="Previous PDF page"
                className="p-1.5 rounded-md text-[#3B3934] hover:bg-[#F3F1EB] disabled:opacity-40 cursor-pointer"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="text-xs font-mono tabular-nums text-[#3B3934] px-1">
                {pdfPage}/{pdfPageCount}
              </span>
              <button
                type="button"
                disabled={pdfPage >= pdfPageCount || pdfRendering}
                onClick={() => {
                  setPdfPage((p) => Math.min(pdfPageCount, p + 1));
                  onClearHighlight?.();
                }}
                title="Next PDF page"
                className="p-1.5 rounded-md text-[#3B3934] hover:bg-[#F3F1EB] disabled:opacity-40 cursor-pointer"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
              <div className="h-4 w-px bg-[#E2DFD7] mx-0.5" />
            </>
          )}

          <button
            type="button"
            onClick={() =>
              setZoom((z) => Math.max(0.35, Number((z - 0.2).toFixed(2))))
            }
            title="Zoom out"
            className="p-1.5 rounded-md text-[#3B3934] hover:bg-[#F3F1EB] cursor-pointer"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-xs font-mono tabular-nums text-[#3B3934] w-11 text-center">
            {Math.round(zoom * 100)}%
          </span>
          <button
            type="button"
            onClick={() =>
              setZoom((z) => Math.min(8, Number((z + 0.2).toFixed(2))))
            }
            title="Zoom in"
            className="p-1.5 rounded-md text-[#3B3934] hover:bg-[#F3F1EB] cursor-pointer"
          >
            <ZoomIn className="w-4 h-4" />
          </button>

          <div className="h-4 w-px bg-[#E2DFD7] mx-0.5" />

          <button
            type="button"
            onClick={() => setShowGrid((v) => !v)}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              showGrid
                ? 'bg-[#161614] text-white border-[#161614]'
                : 'bg-white text-[#3B3934] border-[#E2DFD7] hover:bg-[#F7F6F2]'
            }`}
          >
            <Grid3X3 className="w-3.5 h-3.5" />
            Grid
          </button>

          <button
            type="button"
            onClick={() => {
              setCalibrating((v) => !v);
              setPts([]);
            }}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              calibrating
                ? 'bg-[#1F4E3D] text-white border-[#1F4E3D]'
                : 'bg-white text-[#3B3934] border-[#E2DFD7] hover:bg-[#F7F6F2]'
            }`}
          >
            <Ruler className="w-3.5 h-3.5" />
            Calibrate Scale
          </button>

          <button
            type="button"
            onClick={() => {
              setZoom(1);
              setPan({ x: 0, y: 0 });
            }}
            title="Reset view"
            className="p-1.5 rounded-md text-[#3B3934] hover:bg-[#F3F1EB] cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    );
  },
);
