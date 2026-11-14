import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { AnimatePresence, motion } from 'motion/react';
import {
  AlertTriangle,
  Box,
  Crosshair,
  Download,
  FileCode2,
  FileImage,
  FileSpreadsheet,
  FileText,
  FolderOpen,
  Lightbulb,
  Loader2,
  Maximize2,
  MessageSquare,
  Mic,
  MicOff,
  Plus,
  Search,
  Send,
  Sparkles,
  Square,
  Trash2,
  Upload,
  Volume2,
  X,
} from 'lucide-react';
import { Viewport2D, type Viewport2DHandle } from './components/Viewport2D';
import { Viewport3D, type Viewport3DHandle } from './components/Viewport3D';
import type {
  ChatMessage,
  DetectedObject,
  FileKind,
  HighlightTarget,
  LoadedAsset,
  ProjectFileEntry,
  RoomMetric,
  TakeoffAnalysis,
} from './types/analysis';

type InspectorTab = 'takeoff' | 'assistant';
type ObjectCategoryFilter =
  | 'All'
  | 'Lighting'
  | 'Openings'
  | 'Furniture'
  | 'Plumbing & Appliances'
  | 'Structural';

function isFiniteNum(val: unknown): val is number {
  return typeof val === 'number' && Number.isFinite(val);
}

async function readJsonResponse<T extends Record<string, any>>(
  resp: Response,
): Promise<T> {
  const raw = await resp.text();
  if (!raw) return {} as T;
  try {
    return JSON.parse(raw) as T;
  } catch {
    throw new Error(
      resp.ok
        ? 'Received an invalid JSON response from the server.'
        : `Server error (${resp.status}). Please try again.`,
    );
  }
}

function formatCleanAnswer(raw: string): string {
  if (!raw) return '';
  let text = raw.trim();
  const unfenced = text
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();
  if (unfenced.startsWith('{') && unfenced.includes('"answer"')) {
    try {
      const parsed = JSON.parse(unfenced) as { answer?: unknown };
      if (typeof parsed.answer === 'string') {
        text = parsed.answer;
      }
    } catch {
      const m = unfenced.match(/"answer"\s*:\s*"((?:\\.|[^"\\])*)"/);
      if (m) {
        try {
          text = JSON.parse(`"${m[1]}"`);
        } catch {
          text = m[1].replace(/\\n/g, '\n').replace(/\\"/g, '"');
        }
      }
    }
  }
  return text
    .replace(/\r\n/g, '\n')
    .replace(/(\d)\s*\*\s*(\d)/g, '$1 × $2')
    .replace(/^[ \t]*[•*]+[ \t]+/gm, '- ')
    .replace(/\*+/g, '')
    .replace(/^[ \t]*#{1,6}[ \t]+/gm, '')
    .split('\n')
    .map((line) => line.replace(/[ \t]{2,}/g, ' ').trim())
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function StoaLogo() {
  return (
    <span className="inline-flex items-center text-[20px] font-semibold tracking-[-0.03em] text-[#161614] font-display select-none">
      Stoa
    </span>
  );
}

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

function shortFormatLabel(kind: FileKind): string {
  switch (kind) {
    case '3d-ifc':
      return 'IFC BIM';
    case '3d-mesh':
      return '3D Mesh';
    case '2d-dxf':
      return 'CAD DXF';
    case '2d-svg':
      return 'Vector SVG';
    case '2d-pdf':
      return 'PDF Sheet';
    default:
      return '2D Image';
  }
}

function FileKindIcon({ kind }: { kind: FileKind }) {
  if (kind === '3d-ifc' || kind === '3d-mesh') {
    return <Box className="w-3.5 h-3.5 text-[#1F4E3D] shrink-0" />;
  }
  if (kind === '2d-dxf' || kind === '2d-svg') {
    return <FileCode2 className="w-3.5 h-3.5 text-[#C26D2D] shrink-0" />;
  }
  if (kind === '2d-pdf') {
    return <FileText className="w-3.5 h-3.5 text-[#5C5A54] shrink-0" />;
  }
  return <FileImage className="w-3.5 h-3.5 text-[#1F4E3D] shrink-0" />;
}

function buildFallbackHighlights(
  text: string,
  analysis: TakeoffAnalysis | null,
): HighlightTarget[] {
  const lower = text.toLowerCase();
  const candidates: HighlightTarget[] = [];
  const seen = new Set<string>();

  if (analysis) {
    for (const room of analysis.rooms ?? []) {
      const nLower = (room.name || '').toLowerCase();
      const labelLower = (room.labelInDrawing || '').toLowerCase();
      if (
        (nLower.length > 2 && lower.includes(nLower)) ||
        (labelLower.length > 2 && lower.includes(labelLower))
      ) {
        const key = `room-${room.name}`;
        if (!seen.has(key)) {
          seen.add(key);
          candidates.push({
            id: key,
            label: room.name,
            category: 'Space',
            keywords: [nLower, labelLower].filter(Boolean),
            exactNames: [room.name, room.labelInDrawing].filter(Boolean),
            boxes2D: room.boxes2D,
          });
        }
      }
    }
    for (const obj of analysis.detectedObjects ?? []) {
      const oLower = (obj.name || '').toLowerCase();
      const firstWord = oLower.split(/[\s:(]+/)[0];
      if (
        (oLower.length > 2 && lower.includes(oLower)) ||
        (firstWord.length > 3 && lower.includes(firstWord))
      ) {
        const key = `obj-${obj.name}`;
        if (!seen.has(key)) {
          seen.add(key);
          candidates.push({
            id: key,
            label: `${obj.name} (${obj.count})`,
            category: obj.category === 'Other' ? 'All' : obj.category,
            keywords: [oLower, firstWord],
            exactNames: [obj.name],
            boxes2D: obj.boxes2D,
          });
        }
      }
    }
  }

  const rulePatterns: Array<{
    regex: RegExp;
    id: string;
    label: string;
    category: HighlightTarget['category'];
    keywords: string[];
  }> = [
    {
      regex: /\b(lamp|lamps|light|lights|luminaire|luminaires|pendant|sconce)/i,
      id: 'kw-lamps',
      label: 'Lamps & Luminaires',
      category: 'Lighting',
      keywords: ['lamp', 'light', 'luminaire', 'pendant', 'sconce'],
    },
    {
      regex: /\b(switch|switches|dimmer)/i,
      id: 'kw-switches',
      label: 'Lighting Switches',
      category: 'All',
      keywords: ['switch', 'dimmer'],
    },
    {
      regex: /\b(door|doors)/i,
      id: 'kw-doors',
      label: 'Doors',
      category: 'Openings',
      keywords: ['door', 'ifcdoor', 'tür'],
    },
    {
      regex: /\b(window|windows)/i,
      id: 'kw-windows',
      label: 'Windows',
      category: 'Openings',
      keywords: ['window', 'ifcwindow', 'fenster'],
    },
    {
      regex: /\b(kitchen|küche)/i,
      id: 'kw-kitchen',
      label: 'Kitchen',
      category: 'Space',
      keywords: ['kitchen', 'küche'],
    },
    {
      regex: /\b(bath|bathroom|toilet|wc|sink|vanity)/i,
      id: 'kw-bath-plumbing',
      label: 'Bath & Sanitary',
      category: 'Plumbing & Appliances',
      keywords: ['bath', 'toilet', 'wc', 'sink', 'shower', 'sanitary'],
    },
  ];

  for (const r of rulePatterns) {
    if (r.regex.test(lower) && !seen.has(r.id) && candidates.length < 5) {
      seen.add(r.id);
      candidates.push({
        id: r.id,
        label: r.label,
        category: r.category,
        keywords: r.keywords,
      });
    }
  }

  return candidates;
}

function mergeHighlights(
  msgId: string,
  items: HighlightTarget[],
): HighlightTarget {
  const keywords = new Set<string>();
  const exactNames = new Set<string>();
  const ifcTypes = new Set<string>();
  const ifcIds = new Set<string>();
  const boxes2D: NonNullable<HighlightTarget['boxes2D']> = [];

  for (const it of items) {
    for (const k of it.keywords ?? []) keywords.add(k);
    for (const e of it.exactNames ?? []) exactNames.add(e);
    for (const t of it.ifcTypes ?? []) ifcTypes.add(t);
    for (const id of it.ifcIds ?? []) ifcIds.add(id);
    for (const b of it.boxes2D ?? []) boxes2D.push(b);
  }

  return {
    id: `${msgId}-all`,
    label: `All Mentioned (${items.length})`,
    category: 'All',
    keywords: Array.from(keywords),
    exactNames: exactNames.size > 0 ? Array.from(exactNames) : undefined,
    ifcTypes: ifcTypes.size > 0 ? Array.from(ifcTypes) : undefined,
    ifcIds: ifcIds.size > 0 ? Array.from(ifcIds) : undefined,
    boxes2D: boxes2D.length > 0 ? boxes2D : undefined,
    subTargets: items,
  };
}

export default function App() {
  const [projectFiles, setProjectFiles] = useState<ProjectFileEntry[]>([]);
  const [activeFileId, setActiveFileId] = useState<string | null>(null);

  const [analyzing, setAnalyzing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<InspectorTab>('takeoff');
  const [objectFilter, setObjectFilter] = useState<ObjectCategoryFilter>('All');
  const [objectSearch, setObjectSearch] = useState('');
  const [areaUnit, setAreaUnit] = useState<'m2' | 'ft2'>('m2');
  const [dragActive, setDragActive] = useState(false);

  // Viewport active object/room highlight state
  const [activeHighlight, setActiveHighlight] = useState<HighlightTarget | null>(
    null,
  );
  const [locatingHighlightId, setLocatingHighlightId] = useState<string | null>(
    null,
  );

  // Chat Q&A + Voice I/O state (English only, ElevenLabs API)
  const [chatInput, setChatInput] = useState('');
  const [asking, setAsking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [transcribingVoice, setTranscribingVoice] = useState(false);
  const [elevenLabsReady, setElevenLabsReady] = useState(false);
  const [speakingMsgId, setSpeakingMsgId] = useState<string | null>(null);
  const [loadingTtsMsgId, setLoadingTtsMsgId] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const viewport2DRef = useRef<Viewport2DHandle | null>(null);
  const viewport3DRef = useRef<Viewport3DHandle | null>(null);
  const projectFilesRef = useRef<ProjectFileEntry[]>([]);
  projectFilesRef.current = projectFiles;

  // Voice dictation & TTS refs
  const speechRecRef = useRef<any>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const baseDictationTextRef = useRef<string>('');
  const voiceSessionActiveRef = useRef(false);
  const chunkInFlightRef = useRef(false);
  const chunkTimerRef = useRef<number | null>(null);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
  const audioObjectUrlRef = useRef<string | null>(null);

  // Derived active file entry
  const activeEntry = useMemo(
    () => projectFiles.find((f) => f.asset.id === activeFileId) ?? null,
    [projectFiles, activeFileId],
  );
  const asset = activeEntry?.asset ?? null;
  const analysis = activeEntry?.analysis ?? null;
  const parsedSummary = activeEntry?.parsedSummary ?? {};
  const chatMessages = activeEntry?.chatMessages ?? [];
  const calibrationNote = activeEntry?.calibrationNote ?? null;

  // Cleanup all blob URLs, microphone, and TTS audio strictly on unmount
  useEffect(() => {
    return () => {
      voiceSessionActiveRef.current = false;
      if (chunkTimerRef.current !== null) {
        window.clearInterval(chunkTimerRef.current);
      }
      for (const entry of projectFilesRef.current) {
        if (entry.asset.previewUrl) {
          URL.revokeObjectURL(entry.asset.previewUrl);
        }
      }
      try {
        speechRecRef.current?.stop();
      } catch {
        // ignore
      }
      if (mediaRecorderRef.current?.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
      audioPlayerRef.current?.pause();
      if (audioObjectUrlRef.current) {
        URL.revokeObjectURL(audioObjectUrlRef.current);
      }
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  useEffect(() => {
    fetch('/api/voice/status')
      .then((r) => r.json())
      .then((data: { configured?: boolean }) =>
        setElevenLabsReady(Boolean(data.configured)),
      )
      .catch(() => {});
  }, []);

  const updateEntry = useCallback(
    (fileId: string, updater: (prev: ProjectFileEntry) => ProjectFileEntry) => {
      setProjectFiles((prev) =>
        prev.map((entry) =>
          entry.asset.id === fileId ? updater(entry) : entry,
        ),
      );
    },
    [],
  );

  const selectProjectFile = (fileId: string) => {
    setActiveFileId(fileId);
    setActiveHighlight(null);
    setErrorMsg(null);
  };

  const removeProjectFile = (fileId: string) => {
    const target = projectFiles.find((f) => f.asset.id === fileId);
    if (target?.asset.previewUrl) {
      URL.revokeObjectURL(target.asset.previewUrl);
    }
    const remaining = projectFiles.filter((f) => f.asset.id !== fileId);
    setProjectFiles(remaining);
    if (activeFileId === fileId) {
      setActiveFileId(remaining[0]?.asset.id ?? null);
      setActiveHighlight(null);
    }
  };

  const buildAnalyzePayload = useCallback(
    async (
      targetAsset: LoadedAsset,
      currentSummary: Record<string, unknown>,
      calib: string | null,
    ) => {
      if (
        targetAsset.fileKind === '3d-ifc' ||
        targetAsset.fileKind === '3d-mesh'
      ) {
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
    async (overrideFileId?: string) => {
      const targetId = overrideFileId ?? activeFileId;
      const entry = projectFilesRef.current.find(
        (f) => f.asset.id === targetId,
      );
      if (!entry) return;

      setAnalyzing(true);
      setErrorMsg(null);
      try {
        const payload = await buildAnalyzePayload(
          entry.asset,
          entry.parsedSummary,
          entry.calibrationNote,
        );
        const resp = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await readJsonResponse<{
          analysis?: TakeoffAnalysis;
          matchedGroundTruth?: ProjectFileEntry['matchedGroundTruth'];
          error?: string;
        }>(resp);
        if (!resp.ok || !data.analysis) {
          throw new Error(data.error || 'Analysis failed.');
        }
        updateEntry(entry.asset.id, (prev) => ({
          ...prev,
          analysis: data.analysis!,
          matchedGroundTruth: Array.isArray(data.matchedGroundTruth)
            ? data.matchedGroundTruth
            : prev.matchedGroundTruth,
        }));
        setActiveTab('takeoff');
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : 'Analysis failed.');
      } finally {
        setAnalyzing(false);
      }
    },
    [activeFileId, buildAnalyzePayload, updateEntry],
  );

  const loadLocalFiles = async (fileList: FileList | File[]) => {
    const files = Array.from(fileList);
    if (files.length === 0) return;

    setErrorMsg(null);
    const newEntries: ProjectFileEntry[] = [];
    const unsupported: string[] = [];

    for (const file of files) {
      const lowerName = file.name.toLowerCase();

      const kind = inferFileKind(file.name, file.type);
      if (!kind) {
        unsupported.push(file.name);
        continue;
      }

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
        if (lowerName.endsWith('.obj')) {
          rawText = await file.text();
        } else {
          arrayBuffer = await file.arrayBuffer();
        }
      }

      const nextAsset: LoadedAsset = {
        id: `${file.name}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        name: file.name,
        fileKind: kind,
        mimeType: file.type || 'application/octet-stream',
        sizeBytes: file.size,
        previewUrl,
        rawText,
        arrayBuffer,
      };

      newEntries.push({
        asset: nextAsset,
        parsedSummary: {},
        analysis: null,
        chatMessages: [],
        matchedGroundTruth: [],
        calibrationNote: null,
        addedAt: new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        }),
      });
    }

    if (unsupported.length > 0) {
      setErrorMsg(
        `Unsupported format: ${unsupported.join(', ')}. Supported: PNG, JPG, WEBP, SVG, PDF, DXF, IFC, GLB, GLTF, OBJ, STL, PLY.`,
      );
    }

    if (newEntries.length > 0) {
      setProjectFiles((prev) => [...prev, ...newEntries]);
      setActiveFileId(newEntries[0].asset.id);
      setActiveHighlight(null);
    }
  };

  const clearProjectFiles = () => {
    for (const f of projectFiles) {
      if (f.asset.previewUrl) URL.revokeObjectURL(f.asset.previewUrl);
    }
    setProjectFiles([]);
    setActiveFileId(null);
    setActiveHighlight(null);
    setErrorMsg(null);
  };

  const handleToggleHighlight = async (
    target: HighlightTarget,
    answerContext?: string,
  ) => {
    if (activeHighlight?.id === target.id) {
      setActiveHighlight(null);
      return;
    }
    setActiveHighlight(target);

    // For 2D raster/SVG images without pre-computed boxes2D, locate exact matching subset live
    if (
      asset &&
      (asset.fileKind === '2d-image' || asset.fileKind === '2d-svg') &&
      (!target.boxes2D || target.boxes2D.length === 0)
    ) {
      setLocatingHighlightId(target.id);
      try {
        const imgPayload = await viewport2DRef.current?.captureImagePayload();
        if (imgPayload?.dataBase64 && imgPayload?.mimeType) {
          const resp = await fetch('/api/locate-2d', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              fileName: asset.name,
              label: target.label,
              keywords: target.keywords,
              filterDescription: target.filterDescription,
              expectedCount: target.expectedCount,
              answerContext,
              mimeType: imgPayload.mimeType,
              dataBase64: imgPayload.dataBase64,
            }),
          });
          const data = await readJsonResponse<{
            boxes2D?: HighlightTarget['boxes2D'];
          }>(resp);
          if (resp.ok && Array.isArray(data.boxes2D) && data.boxes2D.length > 0) {
            setActiveHighlight((curr) =>
              curr && curr.id === target.id
                ? { ...curr, boxes2D: data.boxes2D }
                : curr,
            );
          }
        }
      } catch {
        // Non-blocking
      } finally {
        setLocatingHighlightId(null);
      }
    }
  };

  const stopSpeaking = useCallback(() => {
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
      audioPlayerRef.current.currentTime = 0;
      audioPlayerRef.current = null;
    }
    if (audioObjectUrlRef.current) {
      URL.revokeObjectURL(audioObjectUrlRef.current);
      audioObjectUrlRef.current = null;
    }
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    setSpeakingMsgId(null);
    setLoadingTtsMsgId(null);
  }, []);

  const handleToggleSpeak = useCallback(
    async (msgId: string, rawText: string) => {
      if (speakingMsgId === msgId || loadingTtsMsgId === msgId) {
        stopSpeaking();
        return;
      }
      stopSpeaking();
      const cleanText = formatCleanAnswer(rawText);
      if (!cleanText) return;

      const speakWithBrowserFallback = () => {
        if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
          setSpeakingMsgId(null);
          return;
        }
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(cleanText);
        utter.lang = 'en-US';
        utter.rate = 1.02;
        utter.onend = () => setSpeakingMsgId(null);
        utter.onerror = () => setSpeakingMsgId(null);
        setSpeakingMsgId(msgId);
        window.speechSynthesis.speak(utter);
      };

      setLoadingTtsMsgId(msgId);
      try {
        const resp = await fetch('/api/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: cleanText }),
        });
        const data = await readJsonResponse<{
          audioBase64?: string | null;
          mimeType?: string;
          fallback?: string;
          error?: string;
        }>(resp);
        if (!resp.ok || !data.audioBase64) {
          setLoadingTtsMsgId(null);
          speakWithBrowserFallback();
          return;
        }
        const byteChars = atob(data.audioBase64);
        const byteNums = new Uint8Array(byteChars.length);
        for (let i = 0; i < byteChars.length; i++) {
          byteNums[i] = byteChars.charCodeAt(i);
        }
        const audioBlob = new Blob([byteNums], {
          type: data.mimeType || 'audio/mpeg',
        });
        const url = URL.createObjectURL(audioBlob);
        audioObjectUrlRef.current = url;
        const audio = new Audio(url);
        audioPlayerRef.current = audio;
        audio.onended = () => {
          stopSpeaking();
        };
        audio.onerror = () => {
          stopSpeaking();
          speakWithBrowserFallback();
        };
        setLoadingTtsMsgId(null);
        setSpeakingMsgId(msgId);
        await audio.play();
      } catch {
        setLoadingTtsMsgId(null);
        speakWithBrowserFallback();
      }
    },
    [loadingTtsMsgId, speakingMsgId, stopSpeaking],
  );

  const stopVoiceInput = useCallback(() => {
    voiceSessionActiveRef.current = false;
    if (chunkTimerRef.current !== null) {
      window.clearInterval(chunkTimerRef.current);
      chunkTimerRef.current = null;
    }
    if (speechRecRef.current) {
      try {
        speechRecRef.current.stop();
      } catch {
        // ignore
      }
      speechRecRef.current = null;
    }
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state === 'recording'
    ) {
      mediaRecorderRef.current.stop();
    }
    setIsListening(false);
  }, []);

  const transcribeBlobWithElevenLabs = useCallback(
    async (blob: Blob): Promise<string> => {
      const buffer = await blob.arrayBuffer();
      const bytes = new Uint8Array(buffer);
      let binary = '';
      const chunkSize = 0x8000;
      for (let i = 0; i < bytes.length; i += chunkSize) {
        binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
      }
      const resp = await fetch('/api/transcribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audioBase64: btoa(binary),
          mimeType: blob.type || 'audio/webm',
        }),
      });
      const data = await readJsonResponse<{ text?: string; error?: string }>(
        resp,
      );
      if (!resp.ok) {
        throw new Error(data.error || 'Voice transcription failed.');
      }
      return String(data.text ?? '').trim();
    },
    [],
  );

  const startVoiceInput = useCallback(async () => {
    if (!asset || asking) return;
    if (isListening) {
      stopVoiceInput();
      return;
    }
    setErrorMsg(null);
    baseDictationTextRef.current = chatInput.trim();
    voiceSessionActiveRef.current = true;
    let liveBrowserTranscript = '';

    // Start live browser SpeechRecognition (en-US) when supported for instant word-by-word feedback
    const SpeechRecCtor =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    let speechRecStarted = false;

    if (SpeechRecCtor) {
      try {
        const rec = new SpeechRecCtor();
        rec.continuous = true;
        rec.interimResults = true;
        rec.lang = 'en-US';

        let committedTranscript = '';
        rec.onresult = (event: any) => {
          let interim = '';
          let nextCommitted = '';
          for (let i = 0; i < event.results.length; i++) {
            const res = event.results[i];
            const piece = res?.[0]?.transcript ?? '';
            if (res.isFinal) {
              nextCommitted += piece + ' ';
            } else {
              interim += piece;
            }
          }
          committedTranscript = nextCommitted;
          liveBrowserTranscript = [committedTranscript.trim(), interim.trim()]
            .filter(Boolean)
            .join(' ')
            .replace(/\s+/g, ' ');
          if (liveBrowserTranscript) {
            const combined = [
              baseDictationTextRef.current,
              liveBrowserTranscript,
            ]
              .filter(Boolean)
              .join(' ')
              .replace(/\s+/g, ' ');
            setChatInput(combined);
          }
        };

        rec.onerror = () => {
          speechRecRef.current = null;
        };

        rec.onend = () => {
          speechRecRef.current = null;
          if (!mediaRecorderRef.current) {
            setIsListening(false);
            voiceSessionActiveRef.current = false;
          }
        };

        speechRecRef.current = rec;
        rec.start();
        speechRecStarted = true;
      } catch {
        speechRecStarted = false;
      }
    }

    const startMediaRecorderPipeline = async (): Promise<boolean> => {
      if (
        typeof MediaRecorder === 'undefined' ||
        !navigator.mediaDevices?.getUserMedia
      ) {
        if (!speechRecStarted) {
          setErrorMsg('Your browser does not support microphone recording.');
          voiceSessionActiveRef.current = false;
        }
        return speechRecStarted;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaStreamRef.current = stream;
        audioChunksRef.current = [];

        const mimeCandidate = MediaRecorder.isTypeSupported('audio/webm')
          ? 'audio/webm'
          : MediaRecorder.isTypeSupported('audio/mp4')
            ? 'audio/mp4'
            : '';
        const recorder = mimeCandidate
          ? new MediaRecorder(stream, { mimeType: mimeCandidate })
          : new MediaRecorder(stream);

        recorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) {
            audioChunksRef.current.push(e.data);
          }
        };

        // Stream progressive chunks to /api/transcribe when browser SpeechRecognition hasn't produced live text
        chunkTimerRef.current = window.setInterval(async () => {
          if (
            !voiceSessionActiveRef.current ||
            chunkInFlightRef.current ||
            audioChunksRef.current.length === 0 ||
            Boolean(liveBrowserTranscript)
          ) {
            return;
          }
          chunkInFlightRef.current = true;
          try {
            const snapshot = new Blob(audioChunksRef.current, {
              type: recorder.mimeType || 'audio/webm',
            });
            if (snapshot.size > 1200) {
              const transcript = await transcribeBlobWithElevenLabs(snapshot);
              if (transcript && voiceSessionActiveRef.current) {
                const combined = [baseDictationTextRef.current, transcript]
                  .filter(Boolean)
                  .join(' ')
                  .replace(/\s+/g, ' ');
                setChatInput(combined);
              }
            }
          } catch {
            // Final transcription runs on stop
          } finally {
            chunkInFlightRef.current = false;
          }
        }, 2500);

        recorder.onstop = async () => {
          if (chunkTimerRef.current !== null) {
            window.clearInterval(chunkTimerRef.current);
            chunkTimerRef.current = null;
          }
          voiceSessionActiveRef.current = false;
          stream.getTracks().forEach((t) => t.stop());
          mediaStreamRef.current = null;
          mediaRecorderRef.current = null;
          setIsListening(false);

          if (audioChunksRef.current.length === 0) return;
          const fullBlob = new Blob(audioChunksRef.current, {
            type: recorder.mimeType || 'audio/webm',
          });
          if (fullBlob.size < 400) return;
          setTranscribingVoice(true);
          try {
            const finalText = await transcribeBlobWithElevenLabs(fullBlob);
            if (finalText) {
              const combined = [baseDictationTextRef.current, finalText]
                .filter(Boolean)
                .join(' ')
                .replace(/\s+/g, ' ');
              setChatInput(combined);
            }
          } catch (err) {
            if (!liveBrowserTranscript) {
              setErrorMsg(
                err instanceof Error
                  ? err.message
                  : 'Could not transcribe voice message.',
              );
            }
          } finally {
            setTranscribingVoice(false);
          }
        };

        mediaRecorderRef.current = recorder;
        recorder.start(1000);
        setIsListening(true);
        return true;
      } catch {
        if (!speechRecStarted) {
          voiceSessionActiveRef.current = false;
          setErrorMsg(
            'Unable to access microphone. Please check browser permissions.',
          );
          return false;
        }
        setIsListening(true);
        return true;
      }
    };

    if (elevenLabsReady || !speechRecStarted) {
      await startMediaRecorderPipeline();
    } else {
      setIsListening(true);
    }
  }, [
    asking,
    asset,
    chatInput,
    elevenLabsReady,
    isListening,
    stopVoiceInput,
    transcribeBlobWithElevenLabs,
  ]);

  const handleAskQuestion = async (questionText?: string) => {
    const q = (questionText ?? chatInput).trim();
    if (!q || !activeEntry || asking) return;
    if (isListening) stopVoiceInput();
    const currentId = activeEntry.asset.id;
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

    updateEntry(currentId, (prev) => ({
      ...prev,
      chatMessages: [...prev.chatMessages, userMsg],
    }));
    setAsking(true);

    try {
      const payload = await buildAnalyzePayload(
        activeEntry.asset,
        activeEntry.parsedSummary,
        activeEntry.calibrationNote,
      );

      const otherFiles = projectFilesRef.current
        .filter((f) => f.asset.id !== currentId)
        .map((f) => {
          const areaVal = f.analysis?.dimensions?.totalAreaSqM;
          const areaStr = isFiniteNum(areaVal)
            ? `${areaVal} m²`
            : 'not yet analyzed';
          return `- ${f.asset.name} (${shortFormatLabel(f.asset.fileKind)}, area: ${areaStr})`;
        })
        .join('\n');

      const resp = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...payload,
          question: q,
          currentAnalysis: activeEntry.analysis,
          projectContextSummary: otherFiles || undefined,
          history: [...activeEntry.chatMessages, userMsg].map((m) => ({
            role: m.role,
            content: m.content,
          })),
        }),
      });
      const data = await readJsonResponse<{
        answer?: string;
        highlights?: Array<Partial<HighlightTarget>>;
        error?: string;
      }>(resp);
      if (!resp.ok) {
        throw new Error(data.error || 'Failed to answer question.');
      }

      const rawHighlights: HighlightTarget[] = Array.isArray(data.highlights)
        ? data.highlights.map((h: Partial<HighlightTarget>, idx: number) => ({
            id: h.id || `hl-${Date.now()}-${idx}`,
            label: h.label || 'Mentioned Items',
            category: h.category || 'All',
            keywords: Array.isArray(h.keywords) ? h.keywords : [],
            ifcTypes: Array.isArray(h.ifcTypes) ? h.ifcTypes : undefined,
            exactNames: Array.isArray(h.exactNames) ? h.exactNames : undefined,
            ifcIds: Array.isArray(h.ifcIds) ? h.ifcIds : undefined,
            storey: typeof h.storey === 'string' && h.storey ? h.storey : undefined,
            excludeKeywords: Array.isArray(h.excludeKeywords)
              ? h.excludeKeywords
              : undefined,
            expectedCount:
              typeof h.expectedCount === 'number' ? h.expectedCount : undefined,
            filterDescription:
              typeof h.filterDescription === 'string'
                ? h.filterDescription
                : undefined,
            boxes2D: Array.isArray(h.boxes2D) ? h.boxes2D : undefined,
          }))
        : [];

      const formattedAnswer = formatCleanAnswer(String(data.answer ?? ''));

      const effectiveHighlights =
        rawHighlights.length > 0
          ? rawHighlights
          : buildFallbackHighlights(formattedAnswer, activeEntry.analysis);

      updateEntry(currentId, (prev) => ({
        ...prev,
        chatMessages: [
          ...prev.chatMessages,
          {
            id: `a-${Date.now()}`,
            role: 'assistant',
            content: formattedAnswer,
            highlights: effectiveHighlights,
            timestamp: new Date().toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            }),
          },
        ],
      }));
    } catch (err) {
      updateEntry(currentId, (prev) => ({
        ...prev,
        chatMessages: [
          ...prev.chatMessages,
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
        ],
      }));
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
      ['Section', 'Category', 'Item', 'Count / Area', 'Unit', 'Evidence', 'Notes'],
      ...(analysis.rooms ?? []).map((r) => [
        r.isExteriorOrPorch ? 'Exterior / Porch' : 'Interior Room',
        'Space',
        r.name,
        areaUnit === 'm2'
          ? String(r.estimatedAreaSqM ?? '')
          : String(r.estimatedAreaSqFt ?? ''),
        areaUnit === 'm2' ? 'm²' : 'ft²',
        r.areaEvidence ?? 'Estimated',
        `${r.dimensionsText} (${r.dimensionsEvidence ?? 'Unknown'}) ${r.notes}`.trim(),
      ]),
      ...(analysis.detectedObjects ?? []).map((o) => [
        'Object',
        o.category,
        o.name,
        String(o.count),
        o.unit,
        o.evidence ?? 'Estimated',
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
    return (analysis.detectedObjects ?? [])
      .filter((o) => o.category === 'Lighting')
      .reduce((acc, o) => acc + (isFiniteNum(o.count) ? o.count : 0), 0);
  }, [analysis]);

  const totalObjectsCount = useMemo(() => {
    if (!analysis) return 0;
    return (analysis.detectedObjects ?? []).reduce(
      (acc, o) => acc + (isFiniteNum(o.count) ? o.count : 0),
      0,
    );
  }, [analysis]);

  const filteredObjects = useMemo(() => {
    if (!analysis) return [];
    return (analysis.detectedObjects ?? []).filter((obj: DetectedObject) => {
      const catOk =
        objectFilter === 'All' || obj.category === objectFilter;
      const q = objectSearch.trim().toLowerCase();
      const searchOk =
        !q ||
        (obj.name || '').toLowerCase().includes(q) ||
        (obj.locationsOrNotes || '').toLowerCase().includes(q) ||
        (obj.category || '').toLowerCase().includes(q);
      return catOk && searchOk;
    });
  }, [analysis, objectFilter, objectSearch]);

  const is3D = asset?.fileKind === '3d-ifc' || asset?.fileKind === '3d-mesh';

  const makeRoomHighlightTarget = (room: RoomMetric, idx: number): HighlightTarget => ({
    id: `takeoff-room-${idx}-${room.name}`,
    label: room.name,
    category: 'Space',
    keywords: [
      (room.name || '').toLowerCase(),
      (room.labelInDrawing || '').toLowerCase(),
    ].filter(Boolean),
    exactNames: [room.name, room.labelInDrawing].filter(Boolean),
    boxes2D: room.boxes2D,
  });

  const makeObjectHighlightTarget = (
    obj: DetectedObject,
    idx: number,
  ): HighlightTarget => ({
    id: `takeoff-obj-${idx}-${obj.name}`,
    label: `${obj.name} (${obj.count})`,
    category: obj.category === 'Other' ? 'All' : obj.category,
    keywords: [
      obj.name.toLowerCase(),
      obj.name.toLowerCase().split(/[\s:(]+/)[0],
    ].filter(Boolean),
    exactNames: [obj.name],
    expectedCount: obj.count,
    filterDescription: `${obj.name} (${obj.count} ${obj.unit || 'items'})`,
    boxes2D: obj.boxes2D,
  });

  return (
    <div className="min-h-screen flex flex-col bg-[#F7F6F2] text-[#161614]">
      {/* Hidden shared multi-file inputs */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".png,.jpg,.jpeg,.webp,.svg,.pdf,.dxf,.ifc,.glb,.gltf,.obj,.stl,.ply"
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0) {
            void loadLocalFiles(e.target.files);
          }
          e.target.value = '';
        }}
      />

      {/* Top Bar Contract: clean, crisp, elevated Stoa header with visible segmented sections */}
      <header className="sticky top-0 z-30 bg-white/95 backdrop-blur-sm border-b border-[#DCD7CD] shadow-[0_1px_3px_rgba(22,22,20,0.04)] px-4 sm:px-6 py-2.5 flex items-center justify-between gap-4 sm:gap-8">
        <div className="flex items-center gap-3 shrink-0">
          <a
            href="#top"
            onClick={(e) => {
              e.preventDefault();
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            className="whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1F4E3D] rounded-md"
          >
            <StoaLogo />
          </a>
        </div>

        <div className="shrink-0">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center gap-2 px-3 sm:px-4 py-2 text-xs font-semibold text-white bg-[#1F4E3D] rounded-lg hover:bg-[#183E30] shadow-xs transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Upload className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Add Project Files</span>
            <span className="sm:hidden">Upload</span>
          </button>
        </div>
      </header>

      {/* Main Spatial Workbench */}
      <main id="workbench" className="flex-1 flex flex-col">
        {/* Context Action Strip */}
        <div className="bg-white border-b border-[#E5E2DC] px-6 py-2.5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-2 text-xs text-[#5C5A54]">
            <span className="font-semibold text-[#161614]">
              {asset ? asset.name : 'Project Workspace'}
            </span>
            <span aria-hidden="true">·</span>
            <span>
              {asset
                ? `${shortFormatLabel(asset.fileKind)} · ${formatBytes(asset.sizeBytes)}`
                : 'Upload 2D plans, CAD DXF, BIM IFC, or 3D meshes'}
            </span>
            {projectFiles.length > 0 && (
              <>
                <span aria-hidden="true">·</span>
                <span className="font-mono tabular-nums text-[#3B3934]">
                  {projectFiles.length}{' '}
                  {projectFiles.length === 1 ? 'file' : 'files'} in project
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
                className="inline-flex items-center gap-2 px-4 py-1.5 text-xs font-semibold text-white bg-[#161614] rounded-lg hover:bg-[#2C2B27] disabled:opacity-50 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                {analyzing ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Analyzing File…
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

        {/* Primary 3-Pane Studio Workbench: Left Project Files Rail | Center 2D/3D Viewport | Right AI & Takeoff Inspector */}
        <section className="flex flex-col lg:flex-row flex-1 min-h-[640px] border-b border-[#E5E2DC]">
          {/* Pane 1: Project Files Binder */}
          <aside
            aria-label="Project Files Binder"
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragActive(false);
              if (e.dataTransfer.files?.length > 0) {
                void loadLocalFiles(e.dataTransfer.files);
              }
            }}
            className="w-full lg:w-[264px] shrink-0 bg-[#FAF9F5] border-b lg:border-b-0 lg:border-r border-[#E5E2DC] flex flex-col"
          >
            <div className="px-4 py-3 border-b border-[#E5E2DC] flex items-center justify-between gap-2">
              <div>
                <h2 className="text-xs font-semibold text-[#161614] font-display">
                  Project Files
                </h2>
                <p className="text-[11px] text-[#5C5A54] tabular-nums">
                  {projectFiles.length === 0
                    ? 'Empty binder'
                    : `${projectFiles.length} ${
                        projectFiles.length === 1 ? 'document' : 'documents'
                      }`}
                </p>
              </div>
              <div className="flex items-center gap-1">
                {projectFiles.length > 0 && (
                  <button
                    type="button"
                    onClick={clearProjectFiles}
                    title="Clear all project files"
                    className="p-1.5 rounded-md text-[#5C5A54] hover:text-[#161614] hover:bg-[#EAE7DF] transition-colors cursor-pointer"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-semibold text-[#1F4E3D] bg-[#E8EFEA] hover:bg-[#DCE7E0] rounded-md transition-colors cursor-pointer whitespace-nowrap shrink-0"
                >
                  <Plus className="w-3 h-3" />
                  Add
                </button>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-2.5 space-y-1.5 max-h-[240px] lg:max-h-[720px]">
              {projectFiles.length === 0 ? (
                <div className="border border-dashed border-[#D6D2C8] rounded-lg p-4 text-center space-y-2.5 bg-white/60">
                  <FolderOpen className="w-5 h-5 text-[#5C5A54] mx-auto" />
                  <p className="text-xs text-[#5C5A54] leading-relaxed">
                    Drop all project drawings, BIM/CAD models, and photos here to
                    inspect each file individually.
                  </p>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full py-1.5 px-3 text-xs font-medium text-[#161614] bg-white border border-[#DEDAD0] rounded-md hover:bg-[#F3F1EB] transition-colors cursor-pointer"
                  >
                    Select Multiple Files
                  </button>
                </div>
              ) : (
                projectFiles.map((entry) => {
                  const isActive = entry.asset.id === activeFileId;
                  const areaVal = entry.analysis?.dimensions?.totalAreaSqM;
                  const lampCount = entry.analysis
                    ? (entry.analysis.detectedObjects ?? [])
                        .filter((o) => o.category === 'Lighting')
                        .reduce(
                          (a, b) => a + (isFiniteNum(b.count) ? b.count : 0),
                          0,
                        )
                    : null;

                  return (
                    <div
                      key={entry.asset.id}
                      onClick={() => selectProjectFile(entry.asset.id)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          selectProjectFile(entry.asset.id);
                        }
                      }}
                      className={`group rounded-lg p-2.5 border transition-colors cursor-pointer flex items-start gap-2.5 ${
                        isActive
                          ? 'bg-white border-[#1F4E3D] shadow-sm'
                          : 'bg-transparent border-transparent hover:bg-white/70 hover:border-[#E5E2DC]'
                      }`}
                    >
                      <div className="mt-0.5">
                        <FileKindIcon kind={entry.asset.fileKind} />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="text-xs font-medium text-[#161614] truncate">
                          {entry.asset.name}
                        </div>
                        <div className="text-[11px] text-[#5C5A54] mt-0.5 tabular-nums flex items-center gap-1.5 truncate">
                          <span>{shortFormatLabel(entry.asset.fileKind)}</span>
                          <span aria-hidden="true">·</span>
                          <span className="font-mono">
                            {formatBytes(entry.asset.sizeBytes)}
                          </span>
                        </div>
                        {entry.analysis ? (
                          <div className="text-[10px] font-mono text-[#1F4E3D] mt-1 tabular-nums truncate">
                            {isFiniteNum(areaVal)
                              ? `${areaVal.toFixed(1)} m²`
                              : 'Analyzed'}
                            {lampCount !== null ? ` · ${lampCount} lamps` : ''}
                          </div>
                        ) : entry.chatMessages.length > 0 ? (
                          <div className="text-[10px] font-mono text-[#5C5A54] mt-1 tabular-nums">
                            {entry.chatMessages.length} Q&amp;A msgs
                          </div>
                        ) : null}
                      </div>
                      <button
                        type="button"
                        aria-label={`Remove ${entry.asset.name}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          removeProjectFile(entry.asset.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 focus:opacity-100 p-1 rounded text-[#5C5A54] hover:text-red-700 hover:bg-red-50 transition-opacity cursor-pointer shrink-0"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          </aside>

          {/* Pane 2: Spatial 2D/3D Viewport */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragActive(false);
              if (e.dataTransfer.files?.length > 0) {
                void loadLocalFiles(e.dataTransfer.files);
              }
            }}
            className={`flex-1 min-w-0 relative border-b lg:border-b-0 lg:border-r border-[#E5E2DC] flex flex-col ${
              dragActive ? 'ring-2 ring-inset ring-[#1F4E3D] bg-[#E8EFEA]/40' : ''
            }`}
          >
            {!asset ? (
              <div className="flex-1 flex flex-col items-center justify-center p-8 drafting-grid">
                <div className="max-w-lg w-full bg-white border border-[#E5E2DC] rounded-xl p-8 text-center shadow-sm">
                  <div className="w-11 h-11 rounded-lg bg-[#E8EFEA] text-[#1F4E3D] flex items-center justify-center mx-auto">
                    <FolderOpen className="w-5 h-5" />
                  </div>
                  <h1 className="text-xl font-bold text-[#161614] mt-4 font-display text-balance">
                    Project Intelligence for 2D Drawings &amp; 3D BIM
                  </h1>
                  <p className="text-sm text-[#5C5A54] mt-2 leading-relaxed">
                    Upload all floor plans, CAD DXF sheets, photos, and 3D
                    IFC/OBJ models at once. Switch between any file in your
                    binder and highlight mentioned objects live.
                  </p>

                  <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-white bg-[#1F4E3D] rounded-lg hover:bg-[#183E30] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <Upload className="w-4 h-4" />
                      Upload Project Files
                    </button>
                  </div>

                  <p className="text-xs text-[#7A776F] mt-5">
                    PNG · JPG · WEBP · SVG · PDF · DXF · IFC · GLB · GLTF · OBJ ·
                    STL · PLY
                  </p>
                </div>
              </div>
            ) : is3D ? (
              <Viewport3D
                ref={viewport3DRef}
                asset={asset}
                activeHighlight={activeHighlight}
                onClearHighlight={() => setActiveHighlight(null)}
                onParsedSummary={(s) =>
                  updateEntry(asset.id, (prev) => ({ ...prev, parsedSummary: s }))
                }
              />
            ) : (
              <Viewport2D
                ref={viewport2DRef}
                asset={asset}
                activeHighlight={activeHighlight}
                onClearHighlight={() => setActiveHighlight(null)}
                onParsedSummary={(s) =>
                  updateEntry(asset.id, (prev) => ({ ...prev, parsedSummary: s }))
                }
                onCalibrationChange={(note) =>
                  updateEntry(asset.id, (prev) => ({
                    ...prev,
                    calibrationNote: note,
                  }))
                }
              />
            )}
          </div>

          {/* Pane 3: Quantity Takeoff + AI Chat + Ground-Truth */}
          <div
            id="inspector-panel"
            className="w-full lg:w-[440px] xl:w-[470px] shrink-0 bg-white flex flex-col min-h-[540px] max-h-[820px]"
          >
            {/* Segmented Tab Bar */}
            <div className="px-4 py-2.5 border-b border-[#E5E2DC] flex items-center justify-between gap-2 bg-[#FAF9F5]">
              <div className="flex items-center gap-1 p-1 bg-[#EAE7DF] rounded-lg">
                <button
                  type="button"
                  onClick={() => setActiveTab('takeoff')}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    activeTab === 'takeoff'
                      ? 'bg-white text-[#161614] shadow-sm'
                      : 'text-[#5C5A54] hover:text-[#161614]'
                  }`}
                >
                  <FileSpreadsheet className="w-3.5 h-3.5" />
                  Takeoff
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('assistant')}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    activeTab === 'assistant'
                      ? 'bg-white text-[#161614] shadow-sm'
                      : 'text-[#5C5A54] hover:text-[#161614]'
                  }`}
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                  Assistant
                </button>
              </div>

              {analysis && activeTab === 'takeoff' && (
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={handleExportCsv}
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-[#3B3934] border border-[#E5E2DC] rounded-md hover:bg-[#FAF9F5] whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <Download className="w-3 h-3" />
                    CSV
                  </button>
                  <button
                    type="button"
                    onClick={handleExportJson}
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-[#3B3934] border border-[#E5E2DC] rounded-md hover:bg-[#FAF9F5] whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <Download className="w-3 h-3" />
                    JSON
                  </button>
                </div>
              )}
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto">
              <AnimatePresence mode="wait">
                {activeTab === 'takeoff' && (
                  <motion.div
                    key="tab-takeoff"
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
                    className="p-5 space-y-6"
                  >
                    {!asset ? (
                      <div className="py-16 text-center max-w-sm mx-auto">
                        <Maximize2 className="w-8 h-8 text-[#C7C3B8] mx-auto" />
                        <h2 className="text-sm font-semibold text-[#161614] mt-3">
                          Ready for Spatial Inspection
                        </h2>
                        <p className="text-xs text-[#5C5A54] mt-1 leading-relaxed">
                          Add files to the Project Binder on the left, then select
                          any file to measure rooms, count fixtures, and
                          highlight objects.
                        </p>
                      </div>
                    ) : analyzing ? (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-3">
                          {[0, 1, 2, 3].map((i) => (
                            <div
                              key={i}
                              className="h-16 bg-[#F3F1EB] rounded-lg animate-pulse"
                            />
                          ))}
                        </div>
                        <div className="h-24 bg-[#F3F1EB] rounded-lg animate-pulse" />
                        <div className="h-44 bg-[#F3F1EB] rounded-lg animate-pulse" />
                      </div>
                    ) : !analysis ? (
                      <div className="py-8 text-center max-w-sm mx-auto space-y-4">
                        <div className="w-10 h-10 rounded-lg bg-[#E8EFEA] text-[#1F4E3D] flex items-center justify-center mx-auto">
                          <Sparkles className="w-5 h-5" />
                        </div>
                        <div>
                          <h2 className="text-sm font-semibold text-[#161614]">
                            {asset.name} Ready
                          </h2>
                          <p className="text-xs text-[#5C5A54] mt-1 leading-relaxed">
                            {is3D
                              ? 'Inspect the 3D geometry in the viewport, then run takeoff to extract BIM quantities and object locations.'
                              : 'Inspect the drawing or calibrate scale with the ruler, then run takeoff to detect rooms and fixtures.'}
                          </p>
                        </div>
                        <button
                          type="button"
                          onClick={() => void runAnalysis()}
                          className="inline-flex items-center gap-2 px-5 py-2.5 text-xs font-semibold text-white bg-[#1F4E3D] rounded-lg hover:bg-[#183E30] transition-colors cursor-pointer"
                        >
                          <Sparkles className="w-3.5 h-3.5" />
                          Analyze Space &amp; Objects
                        </button>

                        {Object.keys(parsedSummary).length > 0 && (
                          <div className="text-left mt-6 pt-5 border-t border-[#E5E2DC]">
                            <h3 className="text-xs font-semibold text-[#161614]">
                              Pre-Parsed File Telemetry
                            </h3>
                            <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                              {'projectName' in parsedSummary && (
                                <div>
                                  <dt className="text-[#5C5A54]">Project</dt>
                                  <dd className="font-medium text-[#161614] truncate">
                                    {String(parsedSummary.projectName)}
                                  </dd>
                                </div>
                              )}
                              {'totalMeshCount' in parsedSummary && (
                                <div>
                                  <dt className="text-[#5C5A54]">3D Meshes</dt>
                                  <dd className="font-mono tabular-nums text-[#161614]">
                                    {String(parsedSummary.totalMeshCount)}
                                  </dd>
                                </div>
                              )}
                              {'storeys' in parsedSummary &&
                                Array.isArray(parsedSummary.storeys) && (
                                  <div>
                                    <dt className="text-[#5C5A54]">Storeys</dt>
                                    <dd className="font-mono tabular-nums text-[#161614]">
                                      {parsedSummary.storeys.length}
                                    </dd>
                                  </div>
                                )}
                              {'spaces' in parsedSummary &&
                                Array.isArray(parsedSummary.spaces) && (
                                  <div>
                                    <dt className="text-[#5C5A54]">
                                      IFC Spaces
                                    </dt>
                                    <dd className="font-mono tabular-nums text-[#161614]">
                                      {parsedSummary.spaces.length}
                                    </dd>
                                  </div>
                                )}
                              {'lightingFixtures' in parsedSummary &&
                                Array.isArray(parsedSummary.lightingFixtures) && (
                                  <div>
                                    <dt className="text-[#5C5A54]">
                                      Luminaires / Lamps
                                    </dt>
                                    <dd className="font-mono tabular-nums text-[#161614]">
                                      {(
                                        parsedSummary.lightingFixtures as Array<{
                                          count: number;
                                        }>
                                      ).reduce((a, b) => a + (b.count || 0), 0)}
                                    </dd>
                                  </div>
                                )}
                              {'detectedRooms' in parsedSummary &&
                                Array.isArray(parsedSummary.detectedRooms) && (
                                  <div>
                                    <dt className="text-[#5C5A54]">
                                      CAD Room Polygons
                                    </dt>
                                    <dd className="font-mono tabular-nums text-[#161614]">
                                      {parsedSummary.detectedRooms.length}
                                    </dd>
                                  </div>
                                )}
                              {'sumIndoorRoomsSqM' in parsedSummary &&
                                typeof parsedSummary.sumIndoorRoomsSqM ===
                                  'number' && (
                                  <div>
                                    <dt className="text-[#5C5A54]">
                                      Indoor CAD Area
                                    </dt>
                                    <dd className="font-mono tabular-nums text-[#161614]">
                                      {parsedSummary.sumIndoorRoomsSqM.toFixed(
                                        2,
                                      )}{' '}
                                      m²
                                    </dd>
                                  </div>
                                )}
                            </dl>
                          </div>
                        )}
                      </div>
                    ) : (
                      <>
                        {/* Headline Spatial KPIs */}
                        <div className="grid grid-cols-2 border border-[#E5E2DC] rounded-lg divide-x divide-y divide-[#E5E2DC]">
                          <div className="p-3">
                            <div className="text-[11px] text-[#5C5A54]">
                              {isFiniteNum(
                                analysis.dimensions?.printedTotalAreaSqM,
                              )
                                ? 'Printed Total Area'
                                : 'Reported Area'}
                            </div>
                            <div className="text-base font-bold font-mono tabular-nums text-[#161614] mt-0.5">
                              {areaUnit === 'm2'
                                ? isFiniteNum(analysis.dimensions?.totalAreaSqM)
                                  ? `${analysis.dimensions.totalAreaSqM.toFixed(2)} m²`
                                  : 'Unknown'
                                : isFiniteNum(analysis.dimensions?.totalAreaSqFt)
                                  ? `${analysis.dimensions.totalAreaSqFt.toFixed(0)} ft²`
                                  : isFiniteNum(
                                        analysis.dimensions?.totalAreaSqM,
                                      )
                                    ? `${(analysis.dimensions.totalAreaSqM * 10.7639).toFixed(0)} ft²`
                                    : 'Unknown'}
                            </div>
                            <div className="text-[10px] font-mono text-[#7A776F] mt-0.5">
                              {analysis.dimensions?.totalAreaEvidence ??
                                'Estimated'}
                            </div>
                          </div>
                          <div className="p-3">
                            <div className="text-[11px] text-[#5C5A54]">
                              Gross Footprint
                            </div>
                            <div className="text-base font-bold font-mono tabular-nums text-[#161614] mt-0.5">
                              {isFiniteNum(
                                analysis.dimensions?.grossFootprintAreaSqM,
                              )
                                ? areaUnit === 'm2'
                                  ? `${analysis.dimensions.grossFootprintAreaSqM.toFixed(2)} m²`
                                  : `${(analysis.dimensions.grossFootprintAreaSqM * 10.7639).toFixed(0)} ft²`
                                : 'Unknown'}
                            </div>
                            <div className="text-[10px] font-mono text-[#7A776F] mt-0.5">
                              {analysis.dimensions?.grossFootprintEvidence ??
                                'Unknown'}
                            </div>
                          </div>
                          <div className="p-3">
                            <div className="text-[11px] text-[#5C5A54] flex items-center gap-1">
                              <Lightbulb className="w-3 h-3 text-[#C26D2D]" />
                              Lamps
                            </div>
                            <div className="text-base font-bold font-mono tabular-nums text-[#161614] mt-0.5">
                              {lightingTotal}
                            </div>
                            <div className="text-[10px] font-mono text-[#7A776F] mt-0.5">
                              {(analysis.rooms ?? []).length} spaces
                            </div>
                          </div>
                          <div className="p-3">
                            <div className="text-[11px] text-[#5C5A54]">
                              Total Objects
                            </div>
                            <div className="text-base font-bold font-mono tabular-nums text-[#161614] mt-0.5">
                              {totalObjectsCount}
                            </div>
                            <div className="text-[10px] font-mono text-[#7A776F] mt-0.5">
                              All categories
                            </div>
                          </div>
                        </div>

                        {/* Discrepancies Banner */}
                        {analysis.discrepanciesAndUncertainties &&
                          analysis.discrepanciesAndUncertainties.length > 0 && (
                            <div className="bg-[#FDF6EC] border border-[#EED6B5] rounded-lg p-3.5 space-y-1.5">
                              <div className="flex items-center gap-1.5 text-xs font-semibold text-[#7C4416]">
                                <AlertTriangle className="w-3.5 h-3.5 text-[#C26D2D] shrink-0" />
                                Discrepancies &amp; Uncertainties
                              </div>
                              <ul className="space-y-1 text-xs text-[#7C4416]/90 leading-relaxed list-disc pl-4">
                                {analysis.discrepanciesAndUncertainties.map(
                                  (item, idx) => (
                                    <li key={idx}>{item}</li>
                                  ),
                                )}
                              </ul>
                            </div>
                          )}

                        {/* Area Reconciliation */}
                        <div className="border border-[#E5E2DC] rounded-lg p-3.5 bg-[#FAF9F5] space-y-2.5">
                          <div className="text-xs font-semibold text-[#161614]">
                            Area Reconciliation &amp; Evidence
                          </div>
                          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                            <div>
                              <dt className="text-[#5C5A54]">
                                Printed Living / Total
                              </dt>
                              <dd className="font-mono tabular-nums text-[#161614] mt-0.5">
                                {isFiniteNum(
                                  analysis.dimensions?.printedLivingAreaSqM,
                                )
                                  ? `${analysis.dimensions.printedLivingAreaSqM.toFixed(2)} m²`
                                  : '—'}{' '}
                                /{' '}
                                {isFiniteNum(
                                  analysis.dimensions?.printedTotalAreaSqM,
                                )
                                  ? `${analysis.dimensions.printedTotalAreaSqM.toFixed(2)} m²`
                                  : '—'}
                              </dd>
                            </div>
                            <div>
                              <dt className="text-[#5C5A54]">
                                Summed Rooms + Exterior
                              </dt>
                              <dd className="font-mono tabular-nums text-[#161614] mt-0.5">
                                {isFiniteNum(
                                  analysis.dimensions?.sumInteriorRoomsSqM,
                                )
                                  ? `${analysis.dimensions.sumInteriorRoomsSqM.toFixed(2)} m²`
                                  : '—'}
                                {isFiniteNum(
                                  analysis.dimensions?.sumExteriorSpacesSqM,
                                )
                                  ? ` + ${analysis.dimensions.sumExteriorSpacesSqM.toFixed(2)} m²`
                                  : ''}
                              </dd>
                            </div>
                            <div>
                              <dt className="text-[#5C5A54]">Ceiling Height</dt>
                              <dd className="font-mono tabular-nums text-[#161614] mt-0.5">
                                {isFiniteNum(
                                  analysis.dimensions?.estimatedCeilingHeightM,
                                )
                                  ? `${analysis.dimensions.estimatedCeilingHeightM.toFixed(2)} m`
                                  : 'Unknown'}{' '}
                                <span className="text-[10px] text-[#7A776F]">
                                  ·{' '}
                                  {analysis.dimensions?.ceilingHeightEvidence ??
                                    'Unknown'}
                                </span>
                              </dd>
                            </div>
                            <div>
                              <dt className="text-[#5C5A54]">
                                Interior Volume
                              </dt>
                              <dd className="font-mono tabular-nums text-[#161614] mt-0.5">
                                {isFiniteNum(
                                  analysis.dimensions?.estimatedVolumeCuM,
                                )
                                  ? `${analysis.dimensions.estimatedVolumeCuM.toFixed(2)} m³`
                                  : 'Unknown'}{' '}
                                <span className="text-[10px] text-[#7A776F]">
                                  ·{' '}
                                  {analysis.dimensions?.volumeEvidence ??
                                    'Unknown'}
                                </span>
                              </dd>
                            </div>
                          </dl>
                        </div>

                        {/* Summary */}
                        <div className="space-y-1.5">
                          <div className="flex items-center justify-between gap-2 text-xs text-[#5C5A54]">
                            <span>{analysis.sceneType}</span>
                            <span className="font-mono">
                              {analysis.dimensions?.footprintDimensionsText ||
                                analysis.dimensions?.footprintSummary}
                            </span>
                          </div>
                          <p className="text-xs text-[#161614] leading-relaxed">
                            {analysis.summary}
                          </p>
                        </div>

                        {/* Room Sizes & Areas Table with Highlight button */}
                        <div>
                          <div className="flex items-center justify-between gap-2 mb-2.5">
                            <h3 className="text-xs font-semibold text-[#161614]">
                              Room Sizes &amp; Areas
                            </h3>
                            <div className="flex items-center gap-1 p-0.5 bg-[#F3F1EB] rounded-md">
                              <button
                                type="button"
                                onClick={() => setAreaUnit('m2')}
                                className={`px-2 py-0.5 text-[11px] font-mono rounded cursor-pointer ${
                                  areaUnit === 'm2'
                                    ? 'bg-white text-[#161614] shadow-sm'
                                    : 'text-[#5C5A54]'
                                }`}
                              >
                                m²
                              </button>
                              <button
                                type="button"
                                onClick={() => setAreaUnit('ft2')}
                                className={`px-2 py-0.5 text-[11px] font-mono rounded cursor-pointer ${
                                  areaUnit === 'ft2'
                                    ? 'bg-white text-[#161614] shadow-sm'
                                    : 'text-[#5C5A54]'
                                }`}
                              >
                                ft²
                              </button>
                            </div>
                          </div>

                          {(analysis.rooms ?? []).length === 0 ? (
                            <p className="text-xs text-[#5C5A54] py-3 border-y border-[#F0EEE8]">
                              No distinct room boundaries identified.
                            </p>
                          ) : (
                            <div className="border border-[#E5E2DC] rounded-lg overflow-hidden">
                              <table className="w-full text-left text-xs">
                                <thead className="bg-[#FAF9F5] border-b border-[#E5E2DC] text-[#5C5A54]">
                                  <tr>
                                    <th className="py-2 px-3 font-medium">
                                      Room
                                    </th>
                                    <th className="py-2 px-2 font-medium text-right">
                                      Area ({areaUnit === 'm2' ? 'm²' : 'ft²'})
                                    </th>
                                    <th className="py-2 px-2 font-medium text-right">
                                      View
                                    </th>
                                  </tr>
                                </thead>
                                <tbody className="divide-y divide-[#F0EEE8]">
                                  {(analysis.rooms ?? []).map((room, idx) => {
                                    const target = makeRoomHighlightTarget(
                                      room,
                                      idx,
                                    );
                                    const isHl =
                                      activeHighlight?.id === target.id;
                                    const areaVal =
                                      areaUnit === 'm2'
                                        ? room.estimatedAreaSqM
                                        : isFiniteNum(room.estimatedAreaSqFt)
                                          ? room.estimatedAreaSqFt
                                          : isFiniteNum(room.estimatedAreaSqM)
                                            ? room.estimatedAreaSqM * 10.7639
                                            : null;
                                    return (
                                      <tr
                                        key={`${room.name}-${idx}`}
                                        className={
                                          isHl
                                            ? 'bg-[#FDF6EC]'
                                            : 'hover:bg-[#FAF9F5]'
                                        }
                                      >
                                        <td className="py-2 px-3">
                                          <div className="font-medium text-[#161614]">
                                            {room.name}
                                          </div>
                                          <div className="text-[11px] text-[#5C5A54]">
                                            {room.dimensionsText} ·{' '}
                                            <span className="font-mono">
                                              {room.areaEvidence ?? 'Estimated'}
                                            </span>
                                          </div>
                                        </td>
                                        <td className="py-2 px-2 text-right font-mono tabular-nums font-medium text-[#161614]">
                                          {isFiniteNum(areaVal)
                                            ? areaVal.toFixed(2)
                                            : 'Unknown'}
                                        </td>
                                        <td className="py-2 px-2 text-right">
                                          <button
                                            type="button"
                                            onClick={() =>
                                              void handleToggleHighlight(target)
                                            }
                                            title="Highlight space in viewport"
                                            className={`inline-flex items-center gap-1 px-2 py-1 text-[10px] font-medium rounded border transition-colors cursor-pointer whitespace-nowrap ${
                                              isHl
                                                ? 'bg-[#C26D2D] text-white border-[#C26D2D]'
                                                : 'bg-white text-[#3B3934] border-[#E5E2DC] hover:border-[#C26D2D] hover:text-[#C26D2D]'
                                            }`}
                                          >
                                            {locatingHighlightId === target.id ? (
                                              <Loader2 className="w-3 h-3 animate-spin" />
                                            ) : (
                                              <Crosshair className="w-3 h-3" />
                                            )}
                                            {isHl ? 'Active' : 'Highlight'}
                                          </button>
                                        </td>
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                            </div>
                          )}
                        </div>

                        {/* Detected Objects Schedule with Highlight Button */}
                        <div>
                          <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
                            <h3 className="text-xs font-semibold text-[#161614]">
                              Lamps, Fixtures &amp; Object Schedule
                            </h3>
                            <div className="relative">
                              <Search className="w-3.5 h-3.5 text-[#7A776F] absolute left-2.5 top-1/2 -translate-y-1/2" />
                              <input
                                type="text"
                                value={objectSearch}
                                onChange={(e) =>
                                  setObjectSearch(e.target.value)
                                }
                                placeholder="Filter objects…"
                                className="pl-7 pr-2.5 py-1 text-xs border border-[#E5E2DC] rounded-md focus:outline-none focus:ring-2 focus:ring-[#1F4E3D] w-36"
                              />
                            </div>
                          </div>

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
                                    ? 'bg-[#161614] text-white'
                                    : 'bg-[#F3F1EB] text-[#5C5A54] hover:text-[#161614]'
                                }`}
                              >
                                {cat}
                              </button>
                            ))}
                          </div>

                          <div className="border border-[#E5E2DC] rounded-lg overflow-hidden">
                            <table className="w-full text-left text-xs">
                              <thead className="bg-[#FAF9F5] border-b border-[#E5E2DC] text-[#5C5A54]">
                                <tr>
                                  <th className="py-2 px-3 font-medium">Item</th>
                                  <th className="py-2 px-2 font-medium text-right">
                                    Qty
                                  </th>
                                  <th className="py-2 px-2 font-medium text-right">
                                    View
                                  </th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-[#F0EEE8]">
                                {filteredObjects.length === 0 ? (
                                  <tr>
                                    <td
                                      colSpan={3}
                                      className="py-4 px-3 text-center text-[#5C5A54]"
                                    >
                                      No matching items in this filter.
                                    </td>
                                  </tr>
                                ) : (
                                  filteredObjects.map((obj, idx) => {
                                    const target = makeObjectHighlightTarget(
                                      obj,
                                      idx,
                                    );
                                    const isHl =
                                      activeHighlight?.id === target.id;
                                    return (
                                      <tr
                                        key={`${obj.name}-${idx}`}
                                        className={
                                          isHl
                                            ? 'bg-[#FDF6EC]'
                                            : 'hover:bg-[#FAF9F5]'
                                        }
                                      >
                                        <td className="py-2 px-3">
                                          <div className="font-medium text-[#161614]">
                                            {obj.name}
                                          </div>
                                          <div className="text-[11px] text-[#5C5A54]">
                                            {obj.category} ·{' '}
                                            <span className="font-mono">
                                              {obj.evidence ?? obj.confidence}
                                            </span>
                                          </div>
                                        </td>
                                        <td className="py-2 px-2 text-right font-mono tabular-nums font-semibold text-[#161614]">
                                          {obj.count}
                                        </td>
                                        <td className="py-2 px-2 text-right">
                                          <button
                                            type="button"
                                            onClick={() =>
                                              void handleToggleHighlight(target)
                                            }
                                            title="Highlight matching objects in viewport"
                                            className={`inline-flex items-center gap-1 px-2 py-1 text-[10px] font-medium rounded border transition-colors cursor-pointer whitespace-nowrap ${
                                              isHl
                                                ? 'bg-[#C26D2D] text-white border-[#C26D2D]'
                                                : 'bg-white text-[#3B3934] border-[#E5E2DC] hover:border-[#C26D2D] hover:text-[#C26D2D]'
                                            }`}
                                          >
                                            {locatingHighlightId === target.id ? (
                                              <Loader2 className="w-3 h-3 animate-spin" />
                                            ) : (
                                              <Crosshair className="w-3 h-3" />
                                            )}
                                            {isHl ? 'Active' : 'Highlight'}
                                          </button>
                                        </td>
                                      </tr>
                                    );
                                  })
                                )}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </>
                    )}
                  </motion.div>
                )}

                {/* Tab 2: Multi-Turn AI Spatial Assistant with Mention Highlight Buttons */}
                {activeTab === 'assistant' && (
                  <motion.div
                    key="tab-assistant"
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
                    className="flex flex-col h-full min-h-[480px]"
                  >
                    <div className="flex-1 p-5 space-y-4 overflow-y-auto">
                      {!asset ? (
                        <p className="text-xs text-[#5C5A54] text-center py-12">
                          Add or select a file from the Project Binder to ask
                          questions and highlight mentioned objects.
                        </p>
                      ) : chatMessages.length === 0 ? (
                        <div className="space-y-4">
                          <p className="text-xs text-[#5C5A54] leading-relaxed">
                            Ask about{' '}
                            <span className="font-semibold text-[#161614]">
                              {asset.name}
                            </span>
                            . Whenever Stoa mentions objects, lamps, doors, or
                            rooms, a small{' '}
                            <span className="font-medium text-[#C26D2D]">
                              Highlight
                            </span>{' '}
                            button lets you isolate them right in the drawing or
                            3D view.
                          </p>
                          <div className="space-y-2">
                            <div className="text-[11px] font-medium text-[#5C5A54]">
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
                                  className="text-left px-3 py-1.5 text-xs bg-[#F3F1EB] hover:bg-[#EAE7DF] text-[#161614] rounded-md transition-colors cursor-pointer"
                                >
                                  {q}
                                </button>
                              ))}
                            </div>
                          </div>
                        </div>
                      ) : (
                        chatMessages.map((msg) => {
                          const msgHighlights = msg.highlights ?? [];
                          const mergedAll =
                            msgHighlights.length > 1
                              ? mergeHighlights(msg.id, msgHighlights)
                              : null;
                          const displayContent =
                            msg.role === 'assistant'
                              ? formatCleanAnswer(msg.content)
                              : msg.content;

                          return (
                            <div
                              key={msg.id}
                              className={`flex flex-col ${
                                msg.role === 'user' ? 'items-end' : 'items-start'
                              }`}
                            >
                              <div
                                className={`max-w-[94%] rounded-lg px-3.5 py-2.5 text-xs leading-[1.55] whitespace-pre-wrap ${
                                  msg.role === 'user'
                                    ? 'bg-[#1F4E3D] text-white'
                                    : 'bg-[#FAF9F5] text-[#161614] border border-[#E5E2DC]'
                                }`}
                              >
                                {displayContent}

                                {/* Small Highlight Buttons for every mentioned object/room */}
                                {msg.role === 'assistant' &&
                                  msgHighlights.length > 0 && (
                                    <div className="mt-2.5 pt-2 border-t border-[#E5E2DC] flex flex-wrap items-center gap-1.5">
                                      {mergedAll && (
                                        <button
                                          type="button"
                                          onClick={() =>
                                            void handleToggleHighlight(
                                              mergedAll,
                                              displayContent,
                                            )
                                          }
                                          className={`inline-flex items-center gap-1 px-2 py-1 text-[10px] font-semibold rounded border transition-colors cursor-pointer ${
                                            activeHighlight?.id === mergedAll.id
                                              ? 'bg-[#C26D2D] text-white border-[#C26D2D]'
                                              : 'bg-[#FDF6EC] text-[#9A4E15] border-[#EED6B5] hover:bg-[#F9EBD6]'
                                          }`}
                                        >
                                          <Crosshair className="w-3 h-3" />
                                          {activeHighlight?.id === mergedAll.id
                                            ? 'Highlighting All'
                                            : mergedAll.label}
                                        </button>
                                      )}
                                      {msgHighlights.map((hl) => {
                                        const isActive =
                                          activeHighlight?.id === hl.id;
                                        return (
                                          <button
                                            key={hl.id}
                                            type="button"
                                            onClick={() =>
                                              void handleToggleHighlight(
                                                hl,
                                                displayContent,
                                              )
                                            }
                                            className={`inline-flex items-center gap-1 px-2 py-1 text-[10px] font-medium rounded border transition-colors cursor-pointer ${
                                              isActive
                                                ? 'bg-[#C26D2D] text-white border-[#C26D2D]'
                                                : 'bg-white text-[#3B3934] border-[#DEDAD0] hover:border-[#C26D2D] hover:text-[#C26D2D]'
                                            }`}
                                          >
                                            {locatingHighlightId === hl.id ? (
                                              <Loader2 className="w-3 h-3 animate-spin" />
                                            ) : (
                                              <Crosshair className="w-3 h-3" />
                                            )}
                                            {hl.label}
                                          </button>
                                        );
                                      })}
                                    </div>
                                  )}
                              </div>

                              <div className="flex items-center gap-2.5 mt-1.5">
                                <span className="text-[10px] text-[#7A776F] font-mono tabular-nums">
                                  {msg.timestamp}
                                </span>
                                {msg.role === 'assistant' && displayContent && (
                                  <button
                                    type="button"
                                    onClick={() =>
                                      void handleToggleSpeak(
                                        msg.id,
                                        displayContent,
                                      )
                                    }
                                    title={
                                      speakingMsgId === msg.id
                                        ? 'Stop playback'
                                        : 'Read AI response aloud'
                                    }
                                    className={`inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-medium rounded border transition-colors cursor-pointer ${
                                      speakingMsgId === msg.id ||
                                      loadingTtsMsgId === msg.id
                                        ? 'bg-[#E8EFEA] text-[#1F4E3D] border-[#1F4E3D]/40'
                                        : 'bg-white text-[#4A4842] border-[#E2DED4] hover:border-[#1F4E3D] hover:text-[#1F4E3D]'
                                    }`}
                                  >
                                    {loadingTtsMsgId === msg.id ? (
                                      <>
                                        <Loader2 className="w-3 h-3 animate-spin" />
                                        Loading…
                                      </>
                                    ) : speakingMsgId === msg.id ? (
                                      <>
                                        <Square className="w-2.5 h-2.5 fill-current" />
                                        Stop
                                      </>
                                    ) : (
                                      <>
                                        <Volume2 className="w-3 h-3" />
                                        Speak
                                      </>
                                    )}
                                  </button>
                                )}
                              </div>
                            </div>
                          );
                        })
                      )}

                      {asking && (
                        <div className="flex items-center gap-2 text-xs text-[#5C5A54]">
                          <Loader2 className="w-3.5 h-3.5 animate-spin text-[#1F4E3D]" />
                          Inspecting geometry &amp; locating objects…
                        </div>
                      )}
                    </div>

                    <form
                      onSubmit={(e) => {
                        e.preventDefault();
                        void handleAskQuestion();
                      }}
                      className="p-3 border-t border-[#E5E2DC] bg-[#FAF9F5] space-y-2"
                    >
                      {(isListening || transcribingVoice) && (
                        <div className="flex items-center justify-between px-2.5 py-1 rounded-md bg-[#E8EFEA] border border-[#C6D9CC] text-[11px] text-[#183E30]">
                          <span className="flex items-center gap-2 font-medium">
                            <span className="w-2 h-2 rounded-full bg-[#C26D2D] animate-pulse" />
                            {transcribingVoice
                              ? 'Transcribing English audio…'
                              : 'Listening (English) — typing live into message box…'}
                          </span>
                          <button
                            type="button"
                            onClick={stopVoiceInput}
                            className="font-semibold underline cursor-pointer"
                          >
                            Stop
                          </button>
                        </div>
                      )}

                      <div className="flex items-end gap-1.5">
                        <textarea
                          rows={2}
                          value={chatInput}
                          onChange={(e) => setChatInput(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                              e.preventDefault();
                              void handleAskQuestion();
                            }
                          }}
                          disabled={!asset || asking}
                          placeholder={
                            asset
                              ? 'Ask a question or click the mic to speak in English…'
                              : 'Select a file to start asking…'
                          }
                          className="flex-1 px-3 py-2 text-xs bg-white border border-[#E5E2DC] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#1F4E3D] disabled:opacity-50 resize-none leading-relaxed"
                        />
                        <div className="flex flex-col gap-1 shrink-0">
                          <button
                            type="button"
                            onClick={() => void startVoiceInput()}
                            disabled={!asset || asking || transcribingVoice}
                            title={
                              isListening
                                ? 'Stop voice dictation'
                                : 'Voice input (English)'
                            }
                            className={`inline-flex items-center justify-center p-2 rounded-lg border transition-colors cursor-pointer disabled:opacity-40 ${
                              isListening
                                ? 'bg-[#C26D2D] text-white border-[#C26D2D]'
                                : 'bg-white text-[#161614] border-[#DEDAD0] hover:border-[#1F4E3D] hover:text-[#1F4E3D]'
                            }`}
                          >
                            {isListening ? (
                              <MicOff className="w-3.5 h-3.5" />
                            ) : (
                              <Mic className="w-3.5 h-3.5" />
                            )}
                          </button>
                          <button
                            type="submit"
                            disabled={!asset || asking || !chatInput.trim()}
                            title="Send message"
                            className="inline-flex items-center justify-center p-2 bg-[#1F4E3D] text-white rounded-lg hover:bg-[#183E30] disabled:opacity-40 transition-colors cursor-pointer"
                          >
                            <Send className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    </form>
                  </motion.div>
                )}

              </AnimatePresence>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
