import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  ArrowUpRight,
  CheckCircle2,
  Database,
  Download,
  FileSpreadsheet,
  FolderOpen,
  Layers,
  Lightbulb,
  Loader2,
  Maximize2,
  MessageSquare,
  Search,
  Send,
  Sparkles,
  Upload,
} from 'lucide-react';
import { Viewport2D, type Viewport2DHandle } from './components/Viewport2D';
import { Viewport3D, type Viewport3DHandle } from './components/Viewport3D';
import type {
  ChatMessage,
  DatasetOverview,
  DatasetSampleMeta,
  DetectedObject,
  FileKind,
  LoadedAsset,
  TakeoffAnalysis,
} from './types/analysis';

type InspectorTab = 'takeoff' | 'assistant' | 'dataset';
type ObjectCategoryFilter =
  | 'All'
  | 'Lighting'
  | 'Openings'
  | 'Furniture'
  | 'Plumbing & Appliances'
  | 'Structural';

function inferFileKind(fileName: string, mimeType: string): FileKind | null {
  const lower = fileName.toLowerCase();
  if (lower.endsWith('.ifc')) return '3d-ifc';
  if (
    lower.endsWith('.glb') ||
    lower.endsWith('.gltf') ||
    lower.endsWith('.obj') ||
    lower.endsWith('.stl') ||
    lower.endsWith('.ply')
  ) {
    return '3d-mesh';
  }
  if (lower.endsWith('.dxf')) return '2d-dxf';
  if (lower.endsWith('.svg') || mimeType === 'image/svg+xml') return '2d-svg';
  if (lower.endsWith('.pdf') || mimeType === 'application/pdf') return '2d-pdf';
  if (
    lower.endsWith('.png') ||
    lower.endsWith('.jpg') ||
    lower.endsWith('.jpeg') ||
    lower.endsWith('.webp') ||
    mimeType.startsWith('image/')
  ) {
    return '2d-image';
  }
  return null;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function buildProceduralSampleRoomObj(): string {
  // Valid Wavefront OBJ (5.2m x 4.4m x 2.7m studio room with 2 pendant lamps, 1 floor lamp, table, 4 chairs, sofa, window)
  const lines: string[] = ['# Studio Room Sample (metres)'];
  let vOffset = 1;
  const addBox = (
    name: string,
    cx: number,
    cy: number,
    cz: number,
    w: number,
    h: number,
    d: number,
  ) => {
    lines.push(`o ${name}`);
    const hw = w / 2;
    const hh = h / 2;
    const hd = d / 2;
    const verts = [
      [cx - hw, cy - hh, cz - hd],
      [cx + hw, cy - hh, cz - hd],
      [cx + hw, cy + hh, cz - hd],
      [cx - hw, cy + hh, cz - hd],
      [cx - hw, cy - hh, cz + hd],
      [cx + hw, cy - hh, cz + hd],
      [cx + hw, cy + hh, cz + hd],
      [cx - hw, cy + hh, cz + hd],
    ];
    for (const v of verts) {
      lines.push(`v ${v[0].toFixed(3)} ${v[1].toFixed(3)} ${v[2].toFixed(3)}`);
    }
    const o = vOffset;
    const faces = [
      [o, o + 1, o + 2, o + 3],
      [o + 4, o + 7, o + 6, o + 5],
      [o, o + 4, o + 5, o + 1],
      [o + 1, o + 5, o + 6, o + 2],
      [o + 2, o + 6, o + 7, o + 3],
      [o + 3, o + 7, o + 4, o],
    ];
    for (const f of faces) {
      lines.push(`f ${f[0]} ${f[1]} ${f[2]} ${f[3]}`);
    }
    vOffset += 8;
  };

  // Floor slab (5.2m x 4.4m) & 3 cutaway walls for clear interior inspection
  addBox('Floor_Slab', 0, 0.05, 0, 5.2, 0.1, 4.4);
  addBox('North_Wall', 0, 1.4, -2.15, 5.2, 2.7, 0.12);
  addBox('West_Wall', -2.55, 1.4, 0, 0.12, 2.7, 4.4);
  addBox('Door_Frame_Entry', -2.48, 1.15, 1.35, 0.16, 2.1, 0.9);
  addBox('Window_Opening_North', 0.8, 1.5, -2.12, 1.8, 1.3, 0.18);

  // Dining table & 4 chairs
  addBox('Dining_Table', 0.9, 0.76, -0.5, 1.6, 0.06, 0.9);
  addBox('Table_Leg_1', 0.2, 0.38, -0.85, 0.06, 0.72, 0.06);
  addBox('Table_Leg_2', 1.6, 0.38, -0.85, 0.06, 0.72, 0.06);
  addBox('Table_Leg_3', 0.2, 0.38, -0.15, 0.06, 0.72, 0.06);
  addBox('Table_Leg_4', 1.6, 0.38, -0.15, 0.06, 0.72, 0.06);

  addBox('Dining_Chair_1', 0.55, 0.45, -1.15, 0.45, 0.85, 0.45);
  addBox('Dining_Chair_2', 1.25, 0.45, -1.15, 0.45, 0.85, 0.45);
  addBox('Dining_Chair_3', 0.55, 0.45, 0.15, 0.45, 0.85, 0.45);
  addBox('Dining_Chair_4', 1.25, 0.45, 0.15, 0.45, 0.85, 0.45);

  // Lounge sofa + coffee table + desk
  addBox('Lounge_Sofa', -1.45, 0.42, 0.4, 0.9, 0.78, 2.1);
  addBox('Coffee_Table', -0.4, 0.24, 0.4, 0.65, 0.42, 1.1);
  addBox('Bookshelf_Unit', -1.7, 1.05, -1.85, 1.3, 2.0, 0.38);

  // 3 Lamps (2 ceiling pendant lamps + 1 standing floor lamp)
  addBox('Ceiling_Pendant_Lamp_1', 0.55, 2.25, -0.5, 0.32, 0.28, 0.32);
  addBox('Ceiling_Pendant_Lamp_2', 1.25, 2.25, -0.5, 0.32, 0.28, 0.32);
  addBox('Standing_Floor_Lamp_Base', -2.05, 0.08, 1.65, 0.28, 0.04, 0.28);
  addBox('Standing_Floor_Lamp_Stem', -2.05, 0.9, 1.65, 0.05, 1.6, 0.05);
  addBox('Standing_Floor_Lamp_Shade', -2.05, 1.78, 1.65, 0.38, 0.32, 0.38);

  return lines.join('\n');
}

export default function App() {
  const [overview, setOverview] = useState<DatasetOverview | null>(null);
  const [asset, setAsset] = useState<LoadedAsset | null>(null);
  const [analysis, setAnalysis] = useState<TakeoffAnalysis | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [loadingSampleId, setLoadingSampleId] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<InspectorTab>('takeoff');
  const [objectFilter, setObjectFilter] = useState<ObjectCategoryFilter>('All');
  const [objectSearch, setObjectSearch] = useState('');
  const [areaUnit, setAreaUnit] = useState<'m2' | 'ft2'>('m2');
  const [dragActive, setDragActive] = useState(false);

  const [calibrationNote, setCalibrationNote] = useState<string | null>(null);
  const [parsedSummary, setParsedSummary] = useState<Record<string, unknown>>({});

  // Chat Q&A state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [asking, setAsking] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const viewport2DRef = useRef<Viewport2DHandle | null>(null);
  const viewport3DRef = useRef<Viewport3DHandle | null>(null);

  // Cleanup blob preview URLs
  useEffect(() => {
    return () => {
      if (asset?.previewUrl) {
        URL.revokeObjectURL(asset.previewUrl);
      }
    };
  }, [asset]);

  useEffect(() => {
    fetch('/api/dataset/overview')
      .then((r) => r.json())
      .then((data: DatasetOverview) => setOverview(data))
      .catch(() => {
        // Non-blocking
      });
  }, []);

  const buildAnalyzePayload = useCallback(
    async (targetAsset: LoadedAsset, currentSummary: Record<string, unknown>, calib: string | null) => {
      if (targetAsset.fileKind === '3d-ifc' || targetAsset.fileKind === '3d-mesh') {
        const snaps = await viewport3DRef.current?.captureSnapshots();
        return {
          fileName: targetAsset.name,
          fileKind: targetAsset.fileKind,
          mimeType: snaps ? 'image/png' : undefined,
          dataBase64: snaps?.isometricBase64,
          secondaryImageBase64: snaps?.topPlanBase64,
          structuredSummary: currentSummary,
          calibrationNote: calib ?? undefined,
        };
      }

      const payload2d = await viewport2DRef.current?.captureImagePayload();
      return {
        fileName: targetAsset.name,
        fileKind: targetAsset.fileKind,
        mimeType: payload2d?.mimeType,
        dataBase64: payload2d?.dataBase64,
        structuredSummary: currentSummary,
        calibrationNote: payload2d?.calibrationNote ?? calib ?? undefined,
      };
    },
    [],
  );

  const runAnalysis = useCallback(
    async (
      targetAsset?: LoadedAsset,
      overrideSummary?: Record<string, unknown>,
      overrideCalib?: string | null,
    ) => {
      const active = targetAsset ?? asset;
      if (!active) return;
      setAnalyzing(true);
      setErrorMsg(null);
      try {
        const payload = await buildAnalyzePayload(
          active,
          overrideSummary ?? parsedSummary,
          overrideCalib !== undefined ? overrideCalib : calibrationNote,
        );
        const resp = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.error || 'Analysis failed.');
        }
        setAnalysis(data.analysis);
        setActiveTab('takeoff');
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : 'Analysis failed.');
      } finally {
        setAnalyzing(false);
      }
    },
    [asset, buildAnalyzePayload, calibrationNote, parsedSummary],
  );

  const loadLocalFile = async (file: File) => {
    const kind = inferFileKind(file.name, file.type);
    if (!kind) {
      setErrorMsg(
        `Unsupported format "${file.name}". Please upload PNG, JPG, WEBP, SVG, PDF, DXF, IFC, GLB, GLTF, OBJ, STL, or PLY.`,
      );
      return;
    }

    setErrorMsg(null);
    setAnalysis(null);
    setChatMessages([]);
    setParsedSummary({});
    setCalibrationNote(null);

    let previewUrl: string | undefined;
    let rawText: string | undefined;
    let arrayBuffer: ArrayBuffer | undefined;

    if (kind === '2d-image' || kind === '2d-pdf') {
      previewUrl = URL.createObjectURL(file);
      if (kind === '2d-pdf') {
        arrayBuffer = await file.arrayBuffer();
      }
    } else if (kind === '2d-svg' || kind === '2d-dxf' || kind === '3d-ifc') {
      rawText = await file.text();
    } else if (kind === '3d-mesh') {
      if (file.name.toLowerCase().endsWith('.obj')) {
        rawText = await file.text();
      } else {
        arrayBuffer = await file.arrayBuffer();
      }
    }

    const nextAsset: LoadedAsset = {
      id: `${file.name}-${Date.now()}`,
      name: file.name,
      fileKind: kind,
      mimeType: file.type || 'application/octet-stream',
      sizeBytes: file.size,
      previewUrl,
      rawText,
      arrayBuffer,
    };
    setAsset(nextAsset);
  };

  const loadDatasetSample = async (sample: DatasetSampleMeta) => {
    setLoadingSampleId(sample.id);
    setErrorMsg(null);
    setAnalysis(null);
    setChatMessages([]);
    setParsedSummary({});
    setCalibrationNote(null);

    try {
      const resp = await fetch(`/api/dataset/sample/${encodeURIComponent(sample.id)}`);
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || 'Failed to load sample');
      }
      const binaryStr = atob(data.dataBase64);
      const bytes = new Uint8Array(binaryStr.length);
      for (let i = 0; i < binaryStr.length; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
      }

      const kind: FileKind =
        sample.format === 'png'
          ? '2d-image'
          : sample.format === 'dxf'
            ? '2d-dxf'
            : '3d-ifc';

      let previewUrl: string | undefined;
      let rawText: string | undefined;
      if (kind === '2d-image') {
        const blob = new Blob([bytes], { type: 'image/png' });
        previewUrl = URL.createObjectURL(blob);
      } else {
        rawText = new TextDecoder().decode(bytes);
      }

      setAsset({
        id: `${sample.id}-${Date.now()}`,
        name: sample.id,
        fileKind: kind,
        mimeType: data.mimeType,
        sizeBytes: bytes.byteLength,
        previewUrl,
        rawText,
        sampleMeta: sample,
      });
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Sample fetch failed.');
    } finally {
      setLoadingSampleId(null);
    }
  };

  const loadSample3DRoom = () => {
    const objText = buildProceduralSampleRoomObj();
    setErrorMsg(null);
    setAnalysis(null);
    setChatMessages([]);
    setParsedSummary({});
    setCalibrationNote(null);
    setAsset({
      id: `studio-room-${Date.now()}`,
      name: 'furnished-studio-room.obj',
      fileKind: '3d-mesh',
      mimeType: 'text/plain',
      sizeBytes: objText.length,
      rawText: objText,
    });
  };

  const handleAskQuestion = async (questionText?: string) => {
    const q = (questionText ?? chatInput).trim();
    if (!q || !asset || asking) return;
    setChatInput('');
    setActiveTab('assistant');

    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: q,
      timestamp: new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
      }),
    };
    setChatMessages((prev) => [...prev, userMsg]);
    setAsking(true);

    try {
      const payload = await buildAnalyzePayload(asset, parsedSummary, calibrationNote);
      const resp = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...payload,
          question: q,
          currentAnalysis: analysis,
          history: chatMessages.map((m) => ({ role: m.role, content: m.content })),
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || 'Failed to answer question.');
      }
      setChatMessages((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: 'assistant',
          content: data.answer,
          timestamp: new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          }),
        },
      ]);
    } catch (err) {
      setChatMessages((prev) => [
        ...prev,
        {
          id: `e-${Date.now()}`,
          role: 'assistant',
          content:
            err instanceof Error ? err.message : 'Unable to process question.',
          timestamp: new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          }),
        },
      ]);
    } finally {
      setAsking(false);
    }
  };

  const handleExportJson = () => {
    if (!analysis || !asset) return;
    const blob = new Blob(
      [JSON.stringify({ file: asset.name, analysis }, null, 2)],
      { type: 'application/json' },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${asset.name.replace(/\.[^.]+$/, '')}-takeoff.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportCsv = () => {
    if (!analysis || !asset) return;
    const rows = [
      ['Section', 'Category', 'Item', 'Count / Area', 'Unit', 'Notes'],
      ...analysis.rooms.map((r) => [
        'Room',
        'Space',
        r.name,
        areaUnit === 'm2'
          ? String(r.estimatedAreaSqM ?? '')
          : String(r.estimatedAreaSqFt ?? ''),
        areaUnit === 'm2' ? 'm²' : 'ft²',
        `${r.dimensionsText} ${r.notes}`.trim(),
      ]),
      ...analysis.detectedObjects.map((o) => [
        'Object',
        o.category,
        o.name,
        String(o.count),
        o.unit,
        o.locationsOrNotes,
      ]),
    ];
    const csv = rows
      .map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(','))
      .join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${asset.name.replace(/\.[^.]+$/, '')}-takeoff.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const lightingTotal = useMemo(() => {
    if (!analysis) return 0;
    return analysis.detectedObjects
      .filter((o) => o.category === 'Lighting')
      .reduce((acc, o) => acc + o.count, 0);
  }, [analysis]);

  const totalObjectsCount = useMemo(() => {
    if (!analysis) return 0;
    return analysis.detectedObjects.reduce((acc, o) => acc + o.count, 0);
  }, [analysis]);

  const filteredObjects = useMemo(() => {
    if (!analysis) return [];
    return analysis.detectedObjects.filter((obj: DetectedObject) => {
      const catOk =
        objectFilter === 'All' || obj.category === objectFilter;
      const q = objectSearch.trim().toLowerCase();
      const searchOk =
        !q ||
        obj.name.toLowerCase().includes(q) ||
        obj.locationsOrNotes.toLowerCase().includes(q) ||
        obj.category.toLowerCase().includes(q);
      return catOk && searchOk;
    });
  }, [analysis, objectFilter, objectSearch]);

  const is3D = asset?.fileKind === '3d-ifc' || asset?.fileKind === '3d-mesh';

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      {/* Hidden shared file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".png,.jpg,.jpeg,.webp,.svg,.pdf,.dxf,.ifc,.glb,.gltf,.obj,.stl,.ply"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void loadLocalFile(f);
          e.target.value = '';
        }}
      />

      {/* Top Bar Contract: 3 zones separated by gap-8 */}
      <header className="sticky top-0 z-30 bg-white border-b border-slate-200 px-6 py-3.5 flex items-center justify-between gap-8">
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }}
          className="text-lg font-bold tracking-tight text-slate-900 font-display whitespace-nowrap shrink-0"
        >
          Stoa
        </a>

        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600">
          <a
            href="#workbench"
            onClick={(e) => {
              e.preventDefault();
              document
                .getElementById('workbench')
                ?.scrollIntoView({ behavior: 'smooth' });
            }}
            className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            Analyzer
          </a>
          <button
            type="button"
            onClick={() => setActiveTab('takeoff')}
            className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            Quantity Takeoff
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('assistant')}
            className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            Ask AI
          </button>
          <a
            href="#dataset-library"
            onClick={(e) => {
              e.preventDefault();
              document
                .getElementById('dataset-library')
                ?.scrollIntoView({ behavior: 'smooth' });
            }}
            className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
          >
            Dataset Samples
          </a>
        </nav>

        <div className="shrink-0">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Upload className="w-3.5 h-3.5" />
            Upload File
          </button>
        </div>
      </header>

      {/* Main Spatial Workbench */}
      <main id="workbench" className="flex-1 flex flex-col">
        {/* Context Action Strip below header */}
        <div className="bg-white border-b border-slate-200 px-6 py-3 flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
            <span className="font-medium text-slate-900">
              {asset ? asset.name : 'No active drawing or 3D model'}
            </span>
            <span aria-hidden="true">·</span>
            <span>
              {asset
                ? `${asset.fileKind.toUpperCase()} · ${formatBytes(asset.sizeBytes)}`
                : 'Supports PNG, JPG, SVG, PDF, DXF, IFC, GLB, OBJ, STL'}
            </span>
            {overview && (
              <>
                <span aria-hidden="true">·</span>
                <span>
                  Grounded on{' '}
                  <span className="font-mono tabular-nums text-slate-800">
                    {overview.datasetStats.totalFilesInZip.toLocaleString()}
                  </span>{' '}
                  Junction 2026 graphics &amp;{' '}
                  <span className="font-mono tabular-nums text-slate-800">
                    {overview.datasetStats.totalQuestions.toLocaleString()}
                  </span>{' '}
                  takeoff Q&amp;As
                </span>
              </>
            )}
          </div>

          <div className="flex items-center gap-2">
            {asset && (
              <button
                type="button"
                onClick={() => void runAnalysis()}
                disabled={analyzing}
                className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-slate-900 rounded-lg hover:bg-slate-800 disabled:opacity-50 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                {analyzing ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Analyzing Space &amp; Objects…
                  </>
                ) : (
                  <>
                    <Sparkles className="w-3.5 h-3.5" />
                    {analysis ? 'Re-Analyze File' : 'Analyze Space & Objects'}
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        {errorMsg && (
          <div className="mx-6 mt-4 px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-800 flex items-center justify-between gap-4">
            <span>{errorMsg}</span>
            <button
              type="button"
              onClick={() => setErrorMsg(null)}
              className="font-semibold underline cursor-pointer shrink-0"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Primary Split Workspace (Left: 2D/3D Viewport, Right: Takeoff & AI Inspector) */}
        <section className="grid grid-cols-1 lg:grid-cols-12 flex-1 min-h-[600px] border-b border-slate-200">
          {/* Left 7 cols: Spatial Viewer / Dropzone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragActive(false);
              const f = e.dataTransfer.files?.[0];
              if (f) void loadLocalFile(f);
            }}
            className={`lg:col-span-7 relative border-b lg:border-b-0 lg:border-r border-slate-200 flex flex-col ${
              dragActive ? 'ring-2 ring-inset ring-blue-600 bg-blue-50/40' : ''
            }`}
          >
            {!asset ? (
              <div className="flex-1 flex flex-col items-center justify-center p-8 drafting-grid">
                <div className="max-w-lg w-full bg-white border border-slate-200 rounded-xl p-8 text-center shadow-sm">
                  <div className="w-11 h-11 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center mx-auto">
                    <FolderOpen className="w-5 h-5" />
                  </div>
                  <h1 className="text-xl font-bold text-slate-900 mt-4 font-display">
                    Inspect Room Size, Lamps &amp; Objects in 2D or 3D
                  </h1>
                  <p className="text-sm text-slate-600 mt-2 leading-relaxed">
                    Drop a room photo, PNG/PDF floor plan, CAD DXF, or 3D BIM/mesh
                    file (.ifc, .glb, .obj, .stl). Stoa measures room areas,
                    counts lighting fixtures, doors, windows, and furniture, and
                    answers spatial questions.
                  </p>

                  <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <Upload className="w-4 h-4" />
                      Choose 2D or 3D File
                    </button>
                    <button
                      type="button"
                      onClick={loadSample3DRoom}
                      className="inline-flex items-center gap-2 px-4 py-2.5 text-sm font-medium text-slate-700 bg-slate-100 rounded-lg hover:bg-slate-200 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <Layers className="w-4 h-4" />
                      Load 3D Furnished Room
                    </button>
                  </div>

                  <p className="text-xs text-slate-500 mt-5">
                    PNG · JPG · WEBP · SVG · PDF · DXF · IFC · GLB · GLTF · OBJ ·
                    STL · PLY
                  </p>
                </div>
              </div>
            ) : is3D ? (
              <Viewport3D
                ref={viewport3DRef}
                asset={asset}
                onParsedSummary={(s) => setParsedSummary(s)}
              />
            ) : (
              <Viewport2D
                ref={viewport2DRef}
                asset={asset}
                onParsedSummary={(s) => setParsedSummary(s)}
                onCalibrationChange={(note) => setCalibrationNote(note)}
              />
            )}
          </div>

          {/* Right 5 cols: Quantity Takeoff + AI Chat + Ground-Truth */}
          <div className="lg:col-span-5 bg-white flex flex-col min-h-[540px] max-h-[820px]">
            {/* Tab Bar */}
            <div className="px-4 py-2.5 border-b border-slate-200 flex items-center justify-between gap-2 bg-slate-50/70">
              <div className="flex items-center gap-1 p-1 bg-slate-200/70 rounded-lg">
                <button
                  type="button"
                  onClick={() => setActiveTab('takeoff')}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    activeTab === 'takeoff'
                      ? 'bg-white text-slate-900 shadow-sm'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <FileSpreadsheet className="w-3.5 h-3.5" />
                  Takeoff &amp; Specs
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('assistant')}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    activeTab === 'assistant'
                      ? 'bg-white text-slate-900 shadow-sm'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                  AI Assistant
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('dataset')}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    activeTab === 'dataset'
                      ? 'bg-white text-slate-900 shadow-sm'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <Database className="w-3.5 h-3.5" />
                  Dataset Truth
                </button>
              </div>

              {analysis && activeTab === 'takeoff' && (
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={handleExportCsv}
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-slate-700 border border-slate-200 rounded-md hover:bg-slate-50 whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <Download className="w-3 h-3" />
                    CSV
                  </button>
                  <button
                    type="button"
                    onClick={handleExportJson}
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-slate-700 border border-slate-200 rounded-md hover:bg-slate-50 whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <Download className="w-3 h-3" />
                    JSON
                  </button>
                </div>
              )}
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto">
              {activeTab === 'takeoff' && (
                <div className="p-5 space-y-6">
                  {!asset ? (
                    <div className="py-16 text-center max-w-sm mx-auto">
                      <Maximize2 className="w-8 h-8 text-slate-300 mx-auto" />
                      <h2 className="text-sm font-semibold text-slate-900 mt-3">
                        Ready for Spatial Inspection
                      </h2>
                      <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                        Upload a photo, floor plan, or 3D file on the left, or
                        choose a verified Junction 2026 sample below to extract
                        room sizes, lamp counts, and object schedules.
                      </p>
                    </div>
                  ) : analyzing ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                        {[0, 1, 2, 3].map((i) => (
                          <div
                            key={i}
                            className="h-16 bg-slate-100 rounded-lg animate-pulse"
                          />
                        ))}
                      </div>
                      <div className="h-24 bg-slate-100 rounded-lg animate-pulse" />
                      <div className="h-44 bg-slate-100 rounded-lg animate-pulse" />
                    </div>
                  ) : !analysis ? (
                    <div className="py-10 text-center max-w-sm mx-auto space-y-4">
                      <div className="w-10 h-10 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center mx-auto">
                        <Sparkles className="w-5 h-5" />
                      </div>
                      <div>
                        <h2 className="text-sm font-semibold text-slate-900">
                          File Loaded in Viewport
                        </h2>
                        <p className="text-xs text-slate-600 mt-1">
                          {is3D
                            ? 'Inspect the 3D geometry on the left, then run AI takeoff to combine BIM/mesh metadata with visual inspection.'
                            : 'Optionally calibrate scale with the ruler tool, then run AI takeoff to measure rooms and count lamps & objects.'}
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() => void runAnalysis()}
                        className="inline-flex items-center gap-2 px-5 py-2.5 text-xs font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors cursor-pointer"
                      >
                        <Sparkles className="w-3.5 h-3.5" />
                        Analyze Space &amp; Objects
                      </button>

                      {/* Deterministic pre-parsed summary preview if IFC/DXF */}
                      {Object.keys(parsedSummary).length > 0 && (
                        <div className="text-left mt-6 pt-5 border-t border-slate-200">
                          <h3 className="text-xs font-semibold text-slate-800">
                            Pre-Parsed File Telemetry
                          </h3>
                          <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                            {'projectName' in parsedSummary && (
                              <div>
                                <dt className="text-slate-500">Project</dt>
                                <dd className="font-medium text-slate-900 truncate">
                                  {String(parsedSummary.projectName)}
                                </dd>
                              </div>
                            )}
                            {'totalMeshCount' in parsedSummary && (
                              <div>
                                <dt className="text-slate-500">3D Meshes</dt>
                                <dd className="font-mono tabular-nums text-slate-900">
                                  {String(parsedSummary.totalMeshCount)}
                                </dd>
                              </div>
                            )}
                            {'spaces' in parsedSummary &&
                              Array.isArray(parsedSummary.spaces) && (
                                <div>
                                  <dt className="text-slate-500">IFC Spaces</dt>
                                  <dd className="font-mono tabular-nums text-slate-900">
                                    {parsedSummary.spaces.length}
                                  </dd>
                                </div>
                              )}
                            {'lightingFixtures' in parsedSummary &&
                              Array.isArray(parsedSummary.lightingFixtures) && (
                                <div>
                                  <dt className="text-slate-500">
                                    Lamp Families
                                  </dt>
                                  <dd className="font-mono tabular-nums text-slate-900">
                                    {parsedSummary.lightingFixtures.length}
                                  </dd>
                                </div>
                              )}
                          </dl>
                        </div>
                      )}
                    </div>
                  ) : (
                    <>
                      {/* Headline Spatial KPIs (flat bordered strip, tabular numerals) */}
                      <div className="grid grid-cols-2 sm:grid-cols-4 border border-slate-200 rounded-lg divide-y sm:divide-y-0 sm:divide-x divide-slate-200">
                        <div className="p-3">
                          <div className="text-[11px] text-slate-500">
                            Total Area
                          </div>
                          <div className="text-lg font-bold font-mono tabular-nums text-slate-900 mt-0.5">
                            {areaUnit === 'm2'
                              ? analysis.dimensions.totalAreaSqM !== null
                                ? `${analysis.dimensions.totalAreaSqM.toFixed(1)} m²`
                                : '—'
                              : analysis.dimensions.totalAreaSqFt !== null
                                ? `${analysis.dimensions.totalAreaSqFt.toFixed(0)} ft²`
                                : analysis.dimensions.totalAreaSqM !== null
                                  ? `${(analysis.dimensions.totalAreaSqM * 10.7639).toFixed(0)} ft²`
                                  : '—'}
                          </div>
                        </div>
                        <div className="p-3">
                          <div className="text-[11px] text-slate-500">
                            Rooms
                          </div>
                          <div className="text-lg font-bold font-mono tabular-nums text-slate-900 mt-0.5">
                            {analysis.rooms.length}
                          </div>
                        </div>
                        <div className="p-3">
                          <div className="text-[11px] text-slate-500 flex items-center gap-1">
                            <Lightbulb className="w-3 h-3 text-amber-600" />
                            Lamps
                          </div>
                          <div className="text-lg font-bold font-mono tabular-nums text-slate-900 mt-0.5">
                            {lightingTotal}
                          </div>
                        </div>
                        <div className="p-3">
                          <div className="text-[11px] text-slate-500">
                            Total Objects
                          </div>
                          <div className="text-lg font-bold font-mono tabular-nums text-slate-900 mt-0.5">
                            {totalObjectsCount}
                          </div>
                        </div>
                      </div>

                      {/* Summary & Calibration */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between gap-2 text-xs text-slate-500">
                          <span>{analysis.sceneType}</span>
                          <span>{analysis.dimensions.footprintSummary}</span>
                        </div>
                        <p className="text-sm text-slate-800 leading-relaxed">
                          {analysis.summary}
                        </p>
                        <p className="text-xs text-slate-500 leading-relaxed">
                          <span className="font-medium text-slate-700">
                            Calibration:
                          </span>{' '}
                          {analysis.scaleAndCalibration}
                        </p>
                      </div>

                      {/* Room Dimensions & Area Table */}
                      <div>
                        <div className="flex items-center justify-between gap-2 mb-2.5">
                          <h3 className="text-xs font-semibold text-slate-900">
                            Room Sizes &amp; Areas
                          </h3>
                          <div className="flex items-center gap-1 p-0.5 bg-slate-100 rounded-md">
                            <button
                              type="button"
                              onClick={() => setAreaUnit('m2')}
                              className={`px-2 py-0.5 text-[11px] font-mono rounded cursor-pointer ${
                                areaUnit === 'm2'
                                  ? 'bg-white text-slate-900 shadow-sm'
                                  : 'text-slate-600'
                              }`}
                            >
                              m²
                            </button>
                            <button
                              type="button"
                              onClick={() => setAreaUnit('ft2')}
                              className={`px-2 py-0.5 text-[11px] font-mono rounded cursor-pointer ${
                                areaUnit === 'ft2'
                                  ? 'bg-white text-slate-900 shadow-sm'
                                  : 'text-slate-600'
                              }`}
                            >
                              ft²
                            </button>
                          </div>
                        </div>

                        {analysis.rooms.length === 0 ? (
                          <p className="text-xs text-slate-500 py-3 border-y border-slate-100">
                            No distinct room boundaries identified.
                          </p>
                        ) : (
                          <div className="border border-slate-200 rounded-lg overflow-hidden">
                            <table className="w-full text-left text-xs">
                              <thead className="bg-slate-50 border-b border-slate-200 text-slate-500">
                                <tr>
                                  <th className="py-2 px-3 font-medium">Room</th>
                                  <th className="py-2 px-3 font-medium">
                                    Dimensions
                                  </th>
                                  <th className="py-2 px-3 font-medium text-right">
                                    Area ({areaUnit === 'm2' ? 'm²' : 'ft²'})
                                  </th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-100">
                                {analysis.rooms.map((room, idx) => {
                                  const areaVal =
                                    areaUnit === 'm2'
                                      ? room.estimatedAreaSqM
                                      : (room.estimatedAreaSqFt ??
                                        (room.estimatedAreaSqM !== null
                                          ? room.estimatedAreaSqM * 10.7639
                                          : null));
                                  return (
                                    <tr
                                      key={`${room.name}-${idx}`}
                                      className="hover:bg-slate-50/70"
                                    >
                                      <td className="py-2 px-3">
                                        <div className="font-medium text-slate-900">
                                          {room.name}
                                        </div>
                                        {room.labelInDrawing &&
                                          room.labelInDrawing !== room.name && (
                                            <div className="text-[11px] text-slate-500">
                                              Label: {room.labelInDrawing}
                                            </div>
                                          )}
                                      </td>
                                      <td className="py-2 px-3 text-slate-600 font-mono text-[11px]">
                                        {room.dimensionsText}
                                      </td>
                                      <td className="py-2 px-3 text-right font-mono tabular-nums font-medium text-slate-900">
                                        {areaVal !== null
                                          ? areaVal.toFixed(2)
                                          : '—'}
                                      </td>
                                    </tr>
                                  );
                                })}
                              </tbody>
                            </table>
                          </div>
                        )}
                      </div>

                      {/* Detected Objects, Lamps & Fixtures Table */}
                      <div>
                        <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
                          <h3 className="text-xs font-semibold text-slate-900">
                            Lamps, Fixtures &amp; Object Schedule
                          </h3>
                          <div className="relative">
                            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 -translate-y-1/2" />
                            <input
                              type="text"
                              value={objectSearch}
                              onChange={(e) => setObjectSearch(e.target.value)}
                              placeholder="Filter objects…"
                              className="pl-7 pr-2.5 py-1 text-xs border border-slate-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-600 w-40"
                            />
                          </div>
                        </div>

                        {/* Category filter buttons */}
                        <div className="flex flex-wrap items-center gap-1 mb-3">
                          {(
                            [
                              'All',
                              'Lighting',
                              'Openings',
                              'Furniture',
                              'Plumbing & Appliances',
                              'Structural',
                            ] as const
                          ).map((cat) => (
                            <button
                              key={cat}
                              type="button"
                              onClick={() => setObjectFilter(cat)}
                              className={`px-2.5 py-1 text-[11px] font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                                objectFilter === cat
                                  ? 'bg-slate-900 text-white'
                                  : 'bg-slate-100 text-slate-600 hover:text-slate-900'
                              }`}
                            >
                              {cat}
                            </button>
                          ))}
                        </div>

                        <div className="border border-slate-200 rounded-lg overflow-hidden">
                          <table className="w-full text-left text-xs">
                            <thead className="bg-slate-50 border-b border-slate-200 text-slate-500">
                              <tr>
                                <th className="py-2 px-3 font-medium">Item</th>
                                <th className="py-2 px-3 font-medium">
                                  Category
                                </th>
                                <th className="py-2 px-3 font-medium text-right">
                                  Qty
                                </th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                              {filteredObjects.length === 0 ? (
                                <tr>
                                  <td
                                    colSpan={3}
                                    className="py-4 px-3 text-center text-slate-500"
                                  >
                                    No matching items in this filter.
                                  </td>
                                </tr>
                              ) : (
                                filteredObjects.map((obj, idx) => (
                                  <tr
                                    key={`${obj.name}-${idx}`}
                                    className="hover:bg-slate-50/70"
                                  >
                                    <td className="py-2 px-3">
                                      <div className="font-medium text-slate-900">
                                        {obj.name}
                                      </div>
                                      <div className="text-[11px] text-slate-500">
                                        {obj.locationsOrNotes} ·{' '}
                                        {obj.confidence}
                                      </div>
                                    </td>
                                    <td className="py-2 px-3 text-slate-600">
                                      {obj.category}
                                    </td>
                                    <td className="py-2 px-3 text-right font-mono tabular-nums font-semibold text-slate-900">
                                      {obj.count}
                                    </td>
                                  </tr>
                                ))
                              )}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* Tab 2: Multi-Turn AI Spatial Q&A */}
              {activeTab === 'assistant' && (
                <div className="flex flex-col h-full min-h-[480px]">
                  <div className="flex-1 p-5 space-y-4 overflow-y-auto">
                    {!asset ? (
                      <p className="text-xs text-slate-500 text-center py-12">
                        Upload or select a 2D/3D file first to ask questions
                        about room sizes, lamps, and contents.
                      </p>
                    ) : chatMessages.length === 0 ? (
                      <div className="space-y-4">
                        <p className="text-xs text-slate-600 leading-relaxed">
                          Ask anything about{' '}
                          <span className="font-medium text-slate-900">
                            {asset.name}
                          </span>
                          . Stoa inspects the drawing or 3D geometry and
                          applies learned Junction 2026 rules.
                        </p>
                        <div className="space-y-2">
                          <div className="text-[11px] font-medium text-slate-500">
                            Quick Questions
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {[
                              'How large is each room in m² and ft²?',
                              'How many lamps and lighting fixtures are there?',
                              'Count all doors, windows, and furniture items.',
                              'What is the overall footprint and ceiling height?',
                            ].map((q) => (
                              <button
                                key={q}
                                type="button"
                                onClick={() => void handleAskQuestion(q)}
                                className="text-left px-3 py-1.5 text-xs bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-md transition-colors cursor-pointer"
                              >
                                {q}
                              </button>
                            ))}
                          </div>
                        </div>
                      </div>
                    ) : (
                      chatMessages.map((msg) => (
                        <div
                          key={msg.id}
                          className={`flex flex-col ${
                            msg.role === 'user' ? 'items-end' : 'items-start'
                          }`}
                        >
                          <div
                            className={`max-w-[90%] rounded-lg px-3.5 py-2.5 text-xs leading-relaxed whitespace-pre-wrap ${
                              msg.role === 'user'
                                ? 'bg-blue-600 text-white'
                                : 'bg-slate-100 text-slate-900 border border-slate-200'
                            }`}
                          >
                            {msg.content}
                          </div>
                          <span className="text-[10px] text-slate-400 mt-1 font-mono tabular-nums">
                            {msg.timestamp}
                          </span>
                        </div>
                      ))
                    )}

                    {asking && (
                      <div className="flex items-center gap-2 text-xs text-slate-500">
                        <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-600" />
                        Inspecting spatial geometry…
                      </div>
                    )}
                  </div>

                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      void handleAskQuestion();
                    }}
                    className="p-3 border-t border-slate-200 bg-slate-50 flex items-center gap-2"
                  >
                    <input
                      type="text"
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      disabled={!asset || asking}
                      placeholder={
                        asset
                          ? 'Ask about room size, lamps, doors, furniture…'
                          : 'Load a file to start asking…'
                      }
                      className="flex-1 px-3 py-2 text-xs bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600 disabled:opacity-50"
                    />
                    <button
                      type="submit"
                      disabled={!asset || asking || !chatInput.trim()}
                      className="inline-flex items-center justify-center px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-40 transition-colors cursor-pointer shrink-0"
                    >
                      <Send className="w-3.5 h-3.5" />
                    </button>
                  </form>
                </div>
              )}

              {/* Tab 3: Junction 2026 Dataset Grounding & Ground-Truth Verification */}
              {activeTab === 'dataset' && (
                <div className="p-5 space-y-5">
                  {analysis?.datasetPatternsApplied &&
                    analysis.datasetPatternsApplied.length > 0 && (
                      <div>
                        <h3 className="text-xs font-semibold text-slate-900">
                          Learned Dataset Patterns Applied
                        </h3>
                        <ul className="mt-2 space-y-1.5 text-xs text-slate-700">
                          {analysis.datasetPatternsApplied.map((p, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <CheckCircle2 className="w-3.5 h-3.5 text-blue-600 mt-0.5 shrink-0" />
                              <span>{p}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                  {asset?.sampleMeta ? (
                    <div className="pt-4 border-t border-slate-200">
                      <h3 className="text-xs font-semibold text-slate-900">
                        Ground-Truth Questions from Junction 2026 (
                        {asset.sampleMeta.source})
                      </h3>
                      <p className="text-[11px] text-slate-500 mt-0.5">
                        Click &ldquo;Test AI&rdquo; on any benchmark question to
                        run Stoa AI live against this drawing and compare
                        with the dataset answer.
                      </p>
                      <div className="mt-3 divide-y divide-slate-200 border border-slate-200 rounded-lg">
                        {asset.sampleMeta.groundTruthQuestions.map((gt, i) => (
                          <div key={i} className="p-3 space-y-1.5">
                            <div className="text-xs font-medium text-slate-900">
                              {gt.question}
                            </div>
                            <div className="flex items-center justify-between gap-2">
                              <div className="text-xs text-slate-600">
                                Ground Truth:{' '}
                                <span className="font-mono tabular-nums font-semibold text-slate-900">
                                  {String(gt.answer)}{' '}
                                  {gt.unit ? gt.unit : ''}
                                </span>
                              </div>
                              <button
                                type="button"
                                onClick={() => void handleAskQuestion(gt.question)}
                                className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-medium text-blue-600 hover:bg-blue-50 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                              >
                                Test AI
                                <ArrowUpRight className="w-3 h-3" />
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <h3 className="text-xs font-semibold text-slate-900">
                        Junction 2026 Knowledge Base
                      </h3>
                      <p className="text-xs text-slate-600 leading-relaxed">
                        Stoa incorporates quantity-takeoff rules,
                        multilingual room taxonomies, and BIM/CAD scale
                        calibrations distilled from{' '}
                        {overview
                          ? overview.datasetStats.totalQuestions.toLocaleString()
                          : '127,044'}{' '}
                        verified annotations across CubiCasa5k, Swiss Dwellings,
                        IFC-Bench, IFC Models, and OpeningDesign CAD sets.
                      </p>
                      {overview && (
                        <div className="grid grid-cols-2 gap-3 pt-2 text-xs">
                          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                            <div className="text-slate-500">Count Q&amp;As</div>
                            <div className="font-mono tabular-nums font-semibold text-slate-900 mt-0.5">
                              {overview.datasetStats.questionTypes.count?.toLocaleString()}
                            </div>
                          </div>
                          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                            <div className="text-slate-500">Area Q&amp;As</div>
                            <div className="font-mono tabular-nums font-semibold text-slate-900 mt-0.5">
                              {overview.datasetStats.questionTypes.area?.toLocaleString()}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Bottom Section: Streamable Samples from Junction 2026 Dataset */}
        <section
          id="dataset-library"
          className="px-6 py-8 bg-white max-w-[1440px] w-full mx-auto"
        >
          <div className="flex flex-wrap items-end justify-between gap-4 mb-5">
            <div>
              <h2 className="text-base font-bold text-slate-900 font-display">
                Junction 2026 Dataset Benchmark Samples
              </h2>
              <p className="text-xs text-slate-600 mt-1">
                Stream real 2D PNG floor plans, CAD DXF drawings, and 3D IFC
                architectural &amp; electrical models directly from the 9.77 GB
                dataset archive.
              </p>
            </div>
            <button
              type="button"
              onClick={loadSample3DRoom}
              className="text-xs font-medium text-blue-600 hover:underline underline-offset-4 cursor-pointer whitespace-nowrap shrink-0"
            >
              Load Procedural 3D Room (.obj)
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {(overview?.samples ?? []).map((sample) => {
              const isLoading = loadingSampleId === sample.id;
              return (
                <div
                  key={sample.id}
                  className="border border-slate-200 rounded-xl p-4 flex flex-col justify-between gap-3 hover:border-slate-300 transition-colors bg-slate-50/50"
                >
                  <div>
                    <div className="text-xs text-slate-500 flex items-center gap-1.5">
                      <span>{sample.source}</span>
                      <span aria-hidden="true">·</span>
                      <span className="font-mono uppercase">
                        {sample.format}
                      </span>
                      <span aria-hidden="true">·</span>
                      <span className="font-mono tabular-nums">
                        {formatBytes(sample.uncomp_sz)}
                      </span>
                    </div>
                    <h3 className="text-sm font-semibold text-slate-900 mt-1">
                      {sample.title}
                    </h3>
                    <p className="text-xs text-slate-600 mt-1">
                      {sample.groundTruthQuestions.length} verified ground-truth
                      questions included
                    </p>
                  </div>

                  <button
                    type="button"
                    disabled={Boolean(loadingSampleId)}
                    onClick={() => void loadDatasetSample(sample)}
                    className="inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-semibold text-slate-900 bg-white border border-slate-200 rounded-lg hover:bg-slate-100 disabled:opacity-50 transition-colors cursor-pointer"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        Streaming from Archive…
                      </>
                    ) : (
                      <>
                        <ArrowUpRight className="w-3.5 h-3.5" />
                        Open in Viewport
                      </>
                    )}
                  </button>
                </div>
              );
            })}
          </div>
        </section>
      </main>
    </div>
  );
}
