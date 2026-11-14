import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';
import type { HighlightTarget, LoadedAsset } from '../types/analysis';
import {
  type DxfPrimitive,
  type DxfRoomPolygon,
  parseDxfFile,
} from '../utils/dxfParser';
import {
  Crosshair,
  Grid3X3,
  RotateCcw,
  Ruler,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';

export interface Viewport2DHandle {
  captureImagePayload: () => Promise<{
    mimeType: string;
    dataBase64: string;
    calibrationNote?: string;
  } | null>;
}

interface Viewport2DProps {
  asset: LoadedAsset;
  activeHighlight?: HighlightTarget | null;
  onClearHighlight?: () => void;
  onParsedSummary: (summary: Record<string, unknown>) => void;
  onCalibrationChange?: (note: string | null) => void;
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
  ['furniture', 'möbel', 'chair', 'table', 'sofa', 'bed', 'desk'],
  ['sink', 'basin', 'lavabo', 'washbasin'],
  ['shower', 'dusche', 'bathtub', 'tub', 'wanne'],
];

function buildSynonymGroup2D(term: string): string[] {
  const t = term.trim().toLowerCase();
  const group = new Set<string>([t]);
  for (const grp of SYNONYM_GROUPS) {
    if (grp.some((g) => g === t || (t.length >= 4 && (g.includes(t) || t.includes(g))))) {
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
  if (target.excludeKeywords?.some((ex) => ex.trim().length >= 2 && hay.includes(ex.trim().toLowerCase()))) {
    return false;
  }
  if (target.exactNames && target.exactNames.length > 0) {
    return target.exactNames.some((en) => {
      const low = en.trim().toLowerCase();
      return low.length > 0 && (rp.label.toLowerCase() === low || rp.code.toLowerCase() === low || hay.includes(low));
    });
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
  if (target.excludeKeywords?.some((ex) => ex.trim().length >= 2 && hay.includes(ex.trim().toLowerCase()))) {
    return false;
  }
  if (target.exactNames && target.exactNames.length > 0) {
    return target.exactNames.some((en) => {
      const low = en.trim().toLowerCase();
      return low.length > 0 && hay.includes(low);
    });
  }
  const groups = buildKeywordGroups2D(target.keywords ?? []);
  if (groups.length === 0) return false;
  return groups.every((g) => g.some((t) => hay.includes(t)));
}

export const Viewport2D = forwardRef<Viewport2DHandle, Viewport2DProps>(
  function Viewport2D(
    {
      asset,
      activeHighlight,
      onClearHighlight,
      onParsedSummary,
      onCalibrationChange,
    },
    ref,
  ) {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const imgRef = useRef<HTMLImageElement | null>(null);
    const [zoom, setZoom] = useState(1);
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [dragging, setDragging] = useState(false);
    const dragStartRef = useRef({ x: 0, y: 0, px: 0, py: 0 });
    const [showGrid, setShowGrid] = useState(false);

    // DXF / SVG rendered URL + CAD overlay data
    const [renderedSvgUrl, setRenderedSvgUrl] = useState<string | null>(null);
    const [svgMarkupStr, setSvgMarkupStr] = useState<string | null>(null);
    const [dxfOverlay, setDxfOverlay] = useState<{
      roomPolygons: DxfRoomPolygon[];
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

    useEffect(() => {
      setZoom(1);
      setPan({ x: 0, y: 0 });
      setCalibrating(false);
      setPts([]);
      setCalibrationNote(null);
      onCalibRef.current?.(null);

      if (asset.fileKind === '2d-dxf' && asset.rawText) {
        const parsed = parseDxfFile(asset.rawText);
        setSvgMarkupStr(parsed.svgMarkup);
        setDxfOverlay({
          roomPolygons: parsed.roomPolygons,
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
        setSvgMarkupStr(asset.rawText);
        const blob = new Blob([asset.rawText], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        setRenderedSvgUrl(url);
        try {
          const doc = new DOMParser().parseFromString(
            asset.rawText,
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

    useImperativeHandle(ref, () => ({
      captureImagePayload: async () => {
        if (asset.fileKind === '2d-pdf' && asset.arrayBuffer) {
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

        const srcUrl = renderedSvgUrl || asset.previewUrl;
        if (!srcUrl) return null;

        return new Promise((resolve) => {
          const img = new Image();
          img.crossOrigin = 'anonymous';
          img.onload = () => {
            const maxSide = 2400;
            const w = img.naturalWidth || 1600;
            const h = img.naturalHeight || 1200;
            const scale = Math.min(1, maxSide / Math.max(w, h));
            const canvas = document.createElement('canvas');
            canvas.width = Math.max(1, Math.round(w * scale));
            canvas.height = Math.max(1, Math.round(h * scale));
            const ctx = canvas.getContext('2d');
            if (!ctx) {
              resolve(null);
              return;
            }
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            const dataUrl = canvas.toDataURL('image/png');
            resolve({
              mimeType: 'image/png',
              dataBase64: dataUrl.replace(/^data:image\/png;base64,/, ''),
              calibrationNote: calibrationNote ?? undefined,
            });
          };
          img.onerror = () => resolve(null);
          img.src = srcUrl;
        });
      },
    }));

    // Match DXF polygons & primitives for activeHighlight
    const dxfMatches = useMemo(() => {
      if (!activeHighlight || !dxfOverlay) {
        return { polygons: [] as DxfRoomPolygon[], primitives: [] as DxfPrimitive[] };
      }
      const subList =
        activeHighlight.subTargets && activeHighlight.subTargets.length > 0
          ? activeHighlight.subTargets
          : [activeHighlight];

      const polygons = dxfOverlay.roomPolygons.filter((rp) =>
        subList.some((sub) => matchesSingleDxfPolygon(rp, sub)),
      );

      const primitives = dxfOverlay.primitives
        .filter((p) => subList.some((sub) => matchesSingleDxfPrimitive(p, sub)))
        .slice(0, 1500);

      return { polygons, primitives };
    }, [activeHighlight, dxfOverlay]);

    const box2DMatches = useMemo(() => {
      if (!activeHighlight) return [];
      const raw = activeHighlight.boxes2D ?? [];
      if (
        activeHighlight.expectedCount &&
        activeHighlight.expectedCount > 0 &&
        raw.length > activeHighlight.expectedCount &&
        (!activeHighlight.subTargets || activeHighlight.subTargets.length === 0)
      ) {
        return raw.slice(0, activeHighlight.expectedCount);
      }
      return raw;
    }, [activeHighlight]);
    const totalHighlightCount =
      dxfOverlay && (dxfMatches.polygons.length > 0 || dxfMatches.primitives.length > 0)
        ? dxfMatches.polygons.length + dxfMatches.primitives.length
        : box2DMatches.length;

    const handleWheel = (e: React.WheelEvent) => {
      if (asset.fileKind === '2d-pdf') return;
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.12 : 0.9;
      setZoom((z) =>
        Math.min(8, Math.max(0.35, Number((z * factor).toFixed(2)))),
      );
    };

    const handleMouseDown = (e: React.MouseEvent) => {
      if (calibrating || asset.fileKind === '2d-pdf') return;
      setDragging(true);
      dragStartRef.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
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
      const natW = imgRef.current.naturalWidth || 1000;
      const natH = imgRef.current.naturalHeight || 1000;
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

    const displayUrl = renderedSvgUrl || asset.previewUrl;

    if (asset.fileKind === '2d-pdf' && asset.previewUrl) {
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
            {calibrationNote ? ' · Scale calibrated' : ''}
          </div>
        </div>

        {/* Top-right active highlight banner */}
        {activeHighlight && (
          <div
            onClick={(e) => e.stopPropagation()}
            className="absolute top-3.5 right-3.5 z-20 flex items-center gap-2 bg-[#161614] text-white rounded-lg px-3 py-1.5 shadow-md max-w-[320px]"
          >
            <Crosshair className="w-3.5 h-3.5 text-[#F59E0B] shrink-0" />
            <div className="min-w-0">
              <div className="text-[11px] font-medium truncate">
                {activeHighlight.label}
              </div>
              <div className="text-[10px] font-mono text-[#D6D3CB] tabular-nums">
                {totalHighlightCount > 0
                  ? `${totalHighlightCount} matched in drawing`
                  : 'No geometric bounds mapped'}
              </div>
            </div>
            {onClearHighlight && (
              <button
                type="button"
                onClick={onClearHighlight}
                title="Clear highlight"
                className="p-1 rounded hover:bg-white/10 text-[#D6D3CB] hover:text-white transition-colors cursor-pointer shrink-0"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        )}

        {/* Image / SVG Canvas + Highlight Layers */}
        {displayUrl && (
          <div
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: 'center center',
            }}
            className="relative transition-transform duration-75 ease-out"
          >
            <img
              ref={imgRef}
              src={displayUrl}
              alt={asset.name}
              referrerPolicy="no-referrer"
              draggable={false}
              className="max-h-[68vh] max-w-[82vw] object-contain bg-white shadow-sm border border-[#E2DFD7]"
            />

            {/* DXF Vector Highlight Overlay */}
            {dxfOverlay &&
              activeHighlight &&
              (dxfMatches.polygons.length > 0 || dxfMatches.primitives.length > 0) && (
                <svg
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  viewBox={`${dxfOverlay.viewBox.x} ${dxfOverlay.viewBox.y} ${dxfOverlay.viewBox.width} ${dxfOverlay.viewBox.height}`}
                  preserveAspectRatio="xMidYMid meet"
                >
                  {dxfMatches.polygons.map((rp, i) => {
                    const pointsAttr = rp.pts
                      .map((p) => `${p.x.toFixed(3)},${(-p.y).toFixed(3)}`)
                      .join(' ');
                    return (
                      <polygon
                        key={`poly-${i}`}
                        points={pointsAttr}
                        fill="#C26D2D"
                        fillOpacity="0.28"
                        stroke="#C26D2D"
                        strokeWidth={Math.max(
                          dxfOverlay.viewBox.width,
                          dxfOverlay.viewBox.height,
                        ) * 0.0045}
                      />
                    );
                  })}
                  {dxfMatches.primitives.map((p, i) => {
                    const sw =
                      Math.max(
                        dxfOverlay.viewBox.width,
                        dxfOverlay.viewBox.height,
                      ) * 0.0055;
                    if (p.kind === 'line') {
                      return (
                        <line
                          key={`prim-${i}`}
                          x1={p.x1}
                          y1={-p.y1}
                          x2={p.x2}
                          y2={-p.y2}
                          stroke="#C26D2D"
                          strokeWidth={sw}
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
                          r={Math.max(p.r, sw * 2.2)}
                          fill="#C26D2D"
                          fillOpacity="0.24"
                          stroke="#C26D2D"
                          strokeWidth={sw}
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
                      return (
                        <path
                          key={`prim-${i}`}
                          d={`M ${x1} ${y1} A ${p.r} ${p.r} 0 ${largeArc} 0 ${x2} ${y2}`}
                          fill="none"
                          stroke="#C26D2D"
                          strokeWidth={sw}
                        />
                      );
                    }
                    return null;
                  })}
                </svg>
              )}

            {/* 2D Normalized Bounding Box Overlay (for PNG/JPG/SVG/PDF) */}
            {activeHighlight && box2DMatches.length > 0 && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox="0 0 1000 1000"
                preserveAspectRatio="none"
              >
                {box2DMatches.map((b, i) => {
                  const x = Math.max(0, Math.min(990, b.xmin));
                  const y = Math.max(0, Math.min(990, b.ymin));
                  const w = Math.max(18, Math.min(1000 - x, b.xmax - b.xmin));
                  const h = Math.max(18, Math.min(1000 - y, b.ymax - b.ymin));
                  return (
                    <g key={`box-${i}`}>
                      <rect
                        x={x}
                        y={y}
                        width={w}
                        height={h}
                        rx="6"
                        fill="#C26D2D"
                        fillOpacity="0.22"
                        stroke="#C26D2D"
                        strokeWidth="4"
                      />
                      <circle
                        cx={x + w / 2}
                        cy={y + h / 2}
                        r="7"
                        fill="#C26D2D"
                        stroke="#FFFFFF"
                        strokeWidth="2.5"
                      />
                    </g>
                  );
                })}
              </svg>
            )}

            {/* Calibration line overlay */}
            {pts.length > 0 && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
              >
                {pts.length === 2 && (
                  <line
                    x1={pts[0].x * 100}
                    y1={pts[0].y * 100}
                    x2={pts[1].x * 100}
                    y2={pts[1].y * 100}
                    stroke="#1F4E3D"
                    strokeWidth="0.6"
                  />
                )}
                {pts.map((p, i) => (
                  <circle
                    key={i}
                    cx={p.x * 100}
                    cy={p.y * 100}
                    r="1.1"
                    fill="#1F4E3D"
                    stroke="#ffffff"
                    strokeWidth="0.4"
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

          {!svgMarkupStr ||
          asset.fileKind === '2d-dxf' ||
          asset.fileKind === '2d-svg' ? (
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
          ) : null}

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
