import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react';
import type { LoadedAsset } from '../types/analysis';
import { parseDxfFile } from '../utils/dxfParser';
import {
  Grid3X3,
  RotateCcw,
  Ruler,
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
  onParsedSummary: (summary: Record<string, unknown>) => void;
  onCalibrationChange?: (note: string | null) => void;
}

export const Viewport2D = forwardRef<Viewport2DHandle, Viewport2DProps>(
  function Viewport2D({ asset, onParsedSummary, onCalibrationChange }, ref) {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const imgRef = useRef<HTMLImageElement | null>(null);
    const [zoom, setZoom] = useState(1);
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [dragging, setDragging] = useState(false);
    const dragStartRef = useRef({ x: 0, y: 0, px: 0, py: 0 });
    const [showGrid, setShowGrid] = useState(false);

    // DXF / SVG rendered URL
    const [renderedSvgUrl, setRenderedSvgUrl] = useState<string | null>(null);
    const [svgMarkupStr, setSvgMarkupStr] = useState<string | null>(null);

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
        const blob = new Blob([parsed.svgMarkup], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        setRenderedSvgUrl(url);
        onParsedRef.current({ format: 'DXF', ...parsed.summary });
        return () => URL.revokeObjectURL(url);
      }

      if (asset.fileKind === '2d-svg' && asset.rawText) {
        setSvgMarkupStr(asset.rawText);
        const blob = new Blob([asset.rawText], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        setRenderedSvgUrl(url);
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

        // Rasterize onto an offscreen canvas (normalizes SVG, DXF, WebP, PNG, JPG)
        return new Promise((resolve) => {
          const img = new Image();
          img.crossOrigin = 'anonymous';
          img.onload = () => {
            const maxSide = 1600;
            const w = img.naturalWidth || 1200;
            const h = img.naturalHeight || 900;
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

            // If user calibrated scale, draw the calibration bar visually? Keep original clean and pass exact text calibration!
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

    const handleWheel = (e: React.WheelEvent) => {
      if (asset.fileKind === '2d-pdf') return;
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.12 : 0.9;
      setZoom((z) => Math.min(8, Math.max(0.35, Number((z * factor).toFixed(2)))));
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
      setPts((prev) => (prev.length >= 2 ? [{ x: nx, y: ny }] : [...prev, { x: nx, y: ny }]));
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
        <div className="relative w-full h-full min-h-[460px] bg-slate-100 flex flex-col">
          <div className="flex items-center justify-between px-4 py-2 bg-white border-b border-slate-200">
            <span className="text-xs font-medium text-slate-800 truncate">
              {asset.name}
            </span>
            <span className="text-xs font-mono text-slate-500">
              PDF Vector / Drawing Sheet
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
        className={`relative w-full h-full min-h-[460px] bg-slate-100 overflow-hidden select-none flex items-center justify-center ${
          showGrid ? 'drafting-grid' : ''
        } ${calibrating ? 'cursor-crosshair' : dragging ? 'cursor-grabbing' : 'cursor-grab'}`}
      >
        {/* Top-left file readout */}
        <div className="absolute top-3 left-3 z-10 bg-white/95 border border-slate-200 rounded-lg px-3 py-1.5 shadow-sm pointer-events-none">
          <div className="text-xs font-medium text-slate-900 truncate max-w-[260px]">
            {asset.name}
          </div>
          <div className="text-[11px] font-mono tabular-nums text-slate-500 mt-0.5">
            {Math.round(zoom * 100)}% zoom
            {calibrationNote ? ' · Scale calibrated' : ''}
          </div>
        </div>

        {/* Image / SVG Canvas */}
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
              className="max-h-[68vh] max-w-[85vw] object-contain bg-white shadow-sm border border-slate-200"
            />
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
                    stroke="#2563eb"
                    strokeWidth="0.6"
                  />
                )}
                {pts.map((p, i) => (
                  <circle
                    key={i}
                    cx={p.x * 100}
                    cy={p.y * 100}
                    r="1.1"
                    fill="#2563eb"
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
            className="absolute top-3 right-3 z-20 bg-white border border-blue-200 rounded-xl p-3 shadow-md max-w-xs"
          >
            <p className="text-xs font-semibold text-slate-900">
              Calibrate Drawing Scale
            </p>
            <p className="text-[11px] text-slate-600 mt-0.5">
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
                  className="w-20 px-2 py-1 text-xs font-mono border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-600"
                />
                <span className="text-xs text-slate-600">metres</span>
                <button
                  type="button"
                  onClick={applyCalibration}
                  className="ml-auto px-2.5 py-1 text-xs font-medium bg-blue-600 text-white rounded-md hover:bg-blue-700 cursor-pointer whitespace-nowrap shrink-0"
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
          className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/95 border border-slate-200 shadow-md"
        >
          <button
            type="button"
            onClick={() => setZoom((z) => Math.max(0.35, Number((z - 0.2).toFixed(2))))}
            title="Zoom out"
            className="p-1.5 rounded-md text-slate-700 hover:bg-slate-100 cursor-pointer"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-xs font-mono tabular-nums text-slate-700 w-11 text-center">
            {Math.round(zoom * 100)}%
          </span>
          <button
            type="button"
            onClick={() => setZoom((z) => Math.min(8, Number((z + 0.2).toFixed(2))))}
            title="Zoom in"
            className="p-1.5 rounded-md text-slate-700 hover:bg-slate-100 cursor-pointer"
          >
            <ZoomIn className="w-4 h-4" />
          </button>

          <div className="h-4 w-px bg-slate-200 mx-0.5" />

          <button
            type="button"
            onClick={() => setShowGrid((v) => !v)}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              showGrid
                ? 'bg-slate-900 text-white border-slate-900'
                : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-50'
            }`}
          >
            <Grid3X3 className="w-3.5 h-3.5" />
            Grid
          </button>

          {!svgMarkupStr || asset.fileKind === '2d-dxf' || asset.fileKind === '2d-svg' ? (
            <button
              type="button"
              onClick={() => {
                setCalibrating((v) => !v);
                setPts([]);
              }}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                calibrating
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-50'
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
            className="p-1.5 rounded-md text-slate-700 hover:bg-slate-100 cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    );
  },
);
