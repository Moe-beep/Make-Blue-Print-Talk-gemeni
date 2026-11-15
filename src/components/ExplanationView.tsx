// Full explanation of one pipeline answer (or of the document only), shown in the right panel.
// Tabs: Belege (answer + evidence + how it came about), Unsicherheit (model uncertainty), Agent (what it looked at).
// The plan / 3D model in the centre follows the active tab via onVisual.
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { AlertTriangle, ArrowLeft, ChevronLeft, ChevronRight, Pause, Play } from 'lucide-react';
import type { HighlightTarget } from '../types/analysis';
import type {
  ExplainMark,
  ExplainOverlay,
  ExplAnswer,
  ExplElement,
  ExplEvidence,
  ExplHotspot,
  ExplStep,
  PlanqaAnswerBundle,
  PlanqaDocInfo,
} from '../types/explanation';
import { bboxOf, fmtNumber, markFor, planqaAssetUrl, ROLE_STYLE } from '../utils/planqa';

export type ExplainTab = 'belege' | 'unsicherheit' | 'agent';

export interface ExplainVisual {
  overlay: ExplainOverlay | null;
  highlight3D: HighlightTarget | null;
}

// ------------------------------------------------------------------------------- wording

const INTENT_NOUN: Record<string, string> = {
  doors: 'doors',
  windows: 'windows',
  toilets: 'toilets',
  sinks: 'sinks',
  showers_bathtubs: 'showers/bathtubs',
  saunas: 'saunas',
  bathrooms: 'bathrooms',
  wc_rooms: 'WC rooms',
  kitchens: 'kitchens',
  bedrooms: 'bedrooms',
  balconies: 'balconies',
  floors: 'storeys',
};
export const KIND_DE: Record<string, string> = {
  door: 'Doors',
  window: 'Windows',
  toilet: 'Toilets',
  sink: 'Sinks',
  bathtub: 'Bathtubs',
  sauna_bench: 'Sauna benches',
  closet: 'Closets',
  room: 'Rooms',
  text: 'Texts',
  space: 'Rooms',
  wall: 'Walls',
  slab: 'Slabs',
  stair: 'Stairs',
  column: 'Columns',
  beam: 'Beams',
  sanitary: 'Sanitary',
  furniture: 'Furniture',
  electrical_appliance: 'Electrical appliances',
};
const CHANNEL_DE: Record<string, string> = { combined: 'combined', rooms: 'rooms', symbols: 'symbols', text: 'OCR text' };
const SHARE_DE: Record<string, string> = { rooms: 'Room types', symbols: 'Symbols', text: 'OCR texts' };
const STATUS_DE: Record<string, string> = {
  kept: 'detected (above threshold)',
  below_threshold: 'below threshold - not used',
  rejected: 'rejected',
  confirmed: 'confirmed',
};
const CONF_EN: Record<string, string> = { hoch: 'high', mittel: 'medium', niedrig: 'low' };

export function answerValueText(a: ExplAnswer): { value: string | null; noun: string } {
  if (a.value === undefined || a.value === null || a.abstained) return { value: null, noun: '' };
  const noun = a.intent && INTENT_NOUN[a.intent] ? INTENT_NOUN[a.intent] : a.unit && a.unit !== 'count' ? a.unit : '';
  return { value: fmtNumber(a.value), noun };
}

export function ConfidencePill({ a }: { a: ExplAnswer }) {
  const cls =
    a.confidence_label === 'hoch'
      ? 'bg-[#E7F6EC] text-[#15803D]'
      : a.confidence_label === 'mittel'
        ? 'bg-[#FEF3C7] text-[#B45309]'
        : 'bg-[#FDECEC] text-[#B42318]';
  const dot =
    a.confidence_label === 'hoch' ? 'bg-[#16A34A]' : a.confidence_label === 'mittel' ? 'bg-[#F59E0B]' : 'bg-[#DC2626]';
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold ${cls}`}>
      <span className={`w-2 h-2 rounded-full ${dot}`} />
      {CONF_EN[a.confidence_label] ?? a.confidence_label} confidence · {Math.round(a.confidence * 100)}%
    </span>
  );
}

export const AGREEMENT_DE: Record<string, string> = {
  agree: 'Rules and agent agree',
  partial: 'Rules and agent partly agree',
  conflict: 'Rules and agent disagree',
  single: 'single source',
};

export function SourceTiles({ a }: { a: ExplAnswer }) {
  const entries = Object.entries(a.sources);
  return (
    <div className={`grid gap-2 ${entries.length > 1 ? 'grid-cols-2' : 'grid-cols-1'}`}>
      {entries.map(([name, s]) => {
        const v = s.value;
        const same = v !== undefined && v !== null && a.value !== undefined && a.value !== null && v === a.value;
        const cls =
          v === undefined || v === null
            ? 'border-[#E5E2DC] bg-white'
            : same
              ? 'border-[#BBE5C8] bg-[#F0FAF3]'
              : 'border-[#F5D9A8] bg-[#FFF8EB]';
        return (
          <div key={name} className={`rounded-lg border px-2.5 py-1.5 ${cls}`}>
            <div className="text-[10px] text-[#5C5A54]">
              {name === 'rules' ? 'Rules' : name === 'agent' ? 'LLM agent' : name}
            </div>
            <div className={`text-[18px] font-semibold tabular-nums ${same ? 'text-[#15803D]' : v === null || v === undefined ? 'text-[#9A978F]' : 'text-[#B45309]'}`}>
              {v === undefined || v === null ? '–' : fmtNumber(v)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function agentQuote(a: ExplAnswer): string | null {
  const t = a.sources.agent?.text;
  if (!t) return null;
  return t.length > 220 ? `${t.slice(0, 217)}…` : t;
}

// ------------------------------------------------------------------------------- small parts

function Card({ n, title, children, right }: { n?: number; title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="rounded-xl border border-[#E5E2DC] bg-white px-3.5 py-3">
      <div className="flex items-center gap-2 mb-2">
        {n !== undefined && (
          <span className="w-4 h-4 rounded-full bg-[#EEF0FF] text-[#4F46E5] text-[9.5px] font-bold flex items-center justify-center">
            {n}
          </span>
        )}
        <h3 className="text-[10.5px] font-semibold uppercase tracking-wide text-[#5C5A54]">{title}</h3>
        {right && <div className="ml-auto">{right}</div>}
      </div>
      {children}
    </section>
  );
}

function IdChips({ ids, max = 10, onClick }: { ids: string[]; max?: number; onClick?: (id: string) => void }) {
  if (!ids.length) return null;
  return (
    <div className="flex flex-wrap gap-1 mt-1">
      {ids.slice(0, max).map((id) => (
        <button
          key={id}
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onClick?.(id);
          }}
          className="px-1 py-[1px] rounded border border-[#C7D2FE] bg-[#EEF2FF] text-[#3730A3] font-mono text-[9.5px] cursor-pointer"
        >
          {id}
        </button>
      ))}
      {ids.length > max && <span className="text-[9.5px] text-[#7A776F] self-center">+{ids.length - max}</span>}
    </div>
  );
}

const STEP_COLOR: Record<ExplStep['kind'], string> = {
  route: '#64748B',
  read: '#64748B',
  tool: '#2563EB',
  view: '#EA580C',
  check: '#7C3AED',
  final: '#16A34A',
};

function StepRow({
  s,
  active,
  dim,
  onClick,
  onId,
}: {
  s: ExplStep;
  active: boolean;
  dim?: boolean;
  onClick: () => void;
  onId?: (id: string) => void;
}) {
  return (
    <li>
      <div
        role="button"
        tabIndex={0}
        onClick={onClick}
        onKeyDown={(e) => e.key === 'Enter' && onClick()}
        className={`flex gap-2 rounded-lg px-1.5 py-1.5 cursor-pointer border ${
          active ? 'border-[#FDBA74] bg-[#FFF7ED]' : 'border-transparent hover:bg-[#F7F6F2]'
        } ${dim ? 'opacity-45' : ''}`}
      >
        <span
          className="shrink-0 w-5 h-5 rounded-md text-white text-[10px] font-bold flex items-center justify-center mt-[1px]"
          style={{ background: STEP_COLOR[s.kind] }}
        >
          {s.n}
        </span>
        <div className="min-w-0 flex-1 text-[11px] leading-snug">
          <span className="font-semibold text-[#161614]">{s.title}</span>
          {s.result && <span className="text-[#5C5A54]"> → {s.result}</span>}
          <IdChips ids={s.ids ?? []} onClick={onId} />
        </div>
      </div>
    </li>
  );
}

/** Crop of the drawing around a pixel bbox (images only: needs the original as previewUrl). */
function Thumb({ url, imageSize, bbox }: { url?: string; imageSize?: number[] | null; bbox?: number[] }) {
  const S = 52;
  if (!url || !imageSize || !bbox) return <div className="w-[52px] h-[52px] rounded-md bg-[#F3F1EB] shrink-0" />;
  const pad = 0.6;
  const w = bbox[2] - bbox[0];
  const h = bbox[3] - bbox[1];
  const side = Math.max(w, h) * (1 + 2 * pad) || 40;
  const cx = (bbox[0] + bbox[2]) / 2;
  const cy = (bbox[1] + bbox[3]) / 2;
  const scale = S / side;
  return (
    <div
      className="w-[52px] h-[52px] rounded-md border border-[#E5E2DC] shrink-0 relative overflow-hidden bg-white"
      style={{
        backgroundImage: `url(${url})`,
        backgroundSize: `${imageSize[0] * scale}px ${imageSize[1] * scale}px`,
        backgroundPosition: `${-(cx - side / 2) * scale}px ${-(cy - side / 2) * scale}px`,
        backgroundRepeat: 'no-repeat',
      }}
    >
      <div
        className="absolute border-2 border-[#7C3AED] rounded-sm"
        style={{ left: (S - w * scale) / 2, top: (S - h * scale) / 2, width: w * scale, height: h * scale }}
      />
    </div>
  );
}

function Legend({ items }: { items: Array<{ color: string; dashed?: boolean; label: string }> }) {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-[#5C5A54]">
      {items.map((it) => (
        <span key={it.label} className="inline-flex items-center gap-1">
          <span
            className="inline-block w-2.5 h-2.5 rounded-full"
            style={{
              border: `2px ${it.dashed ? 'dashed' : 'solid'} ${it.color}`,
              background: it.dashed ? 'transparent' : it.color,
            }}
          />
          {it.label}
        </span>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------------------- main view

interface Props {
  bundle: PlanqaAnswerBundle | null;
  doc?: PlanqaDocInfo;
  previewUrl?: string;
  initialTab?: ExplainTab;
  onClose: () => void;
  onVisual: (v: ExplainVisual) => void;
}

type Focus = { key: string; ids: string[]; regions?: number[][] } | null;

export function ExplanationView({ bundle, doc, previewUrl, initialTab = 'belege', onClose, onVisual }: Props) {
  const a = bundle?.answer ?? null;
  const kind = bundle?.kind ?? doc?.kind ?? 'plan';
  const is3D = kind === 'ifc';
  const docExpl = doc?.explanation;
  const els: Record<string, ExplElement> = useMemo(
    () => ({ ...(docExpl?.elements ?? {}), ...(bundle?.elements ?? {}) }),
    [docExpl, bundle],
  );
  const imageSize = bundle?.image_size ?? docExpl?.source.image_size ?? null;
  const viewerIds = bundle?.viewer_ids ?? {};
  const hasAgent = Boolean(a?.steps.some((s) => s.kind === 'tool' || s.kind === 'view'));
  const tabs: Array<[ExplainTab, string]> = [
    ...(a ? ([['belege', 'Evidence']] as Array<[ExplainTab, string]>) : []),
    ...(!is3D && docExpl ? ([['unsicherheit', 'Uncertainty']] as Array<[ExplainTab, string]>) : []),
    ...(hasAgent ? ([['agent', 'Agent']] as Array<[ExplainTab, string]>) : []),
  ];
  const [tab, setTab] = useState<ExplainTab>(
    tabs.some(([t]) => t === initialTab) ? initialTab : (tabs[0]?.[0] ?? 'belege'),
  );
  const [focus, setFocus] = useState<Focus>(null);
  const [openEl, setOpenEl] = useState<string | null>(null);
  const [channel, setChannel] = useState('combined');
  const [opacity, setOpacity] = useState(0.7);
  const [showAllEv, setShowAllEv] = useState(false);
  const lastStep = a ? a.steps.length : 0;
  const [step, setStep] = useState(lastStep);
  const [playing, setPlaying] = useState(false);

  useEffect(() => setFocus(null), [tab]);
  useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => {
      setStep((s) => {
        if (s >= lastStep) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 900);
    return () => window.clearInterval(t);
  }, [playing, lastStep]);

  const groups = useMemo(() => {
    const g: Record<string, ExplEvidence[]> = {};
    for (const e of a?.evidence ?? []) (g[e.role] ??= []).push(e);
    return g;
  }, [a]);
  const answerIds = useMemo(() => new Set((a?.evidence ?? []).map((e) => e.element)), [a]);
  const answerKinds = useMemo(() => {
    const k = new Set<string>();
    for (const e of a?.evidence ?? []) if (e.role !== 'context' || is3D) k.add(els[e.element]?.kind ?? '');
    k.delete('');
    return k;
  }, [a, els, is3D]);

  const focusIds = (key: string, ids: string[], regions?: number[][]) =>
    setFocus((f) => (f?.key === key ? null : { key, ids, regions }));

  // ---------------------------------------------------------------- agent coverage up to `step`
  const agentState = useMemo(() => {
    const seen = new Set<string>();
    const views: number[][] = [];
    for (const s of (a?.steps ?? []).slice(0, step)) {
      for (const id of s.ids ?? []) seen.add(id);
      if (s.region) views.push(s.region);
    }
    const byKind: Array<{ kind: string; seen: number; total: number | null; star: boolean }> = [];
    if (!is3D && docExpl) {
      const totals: Record<string, string[]> = {};
      for (const e of Object.values(docExpl.elements)) if (e.status === 'kept') (totals[e.kind] ??= []).push(e.id);
      for (const [k, ids] of Object.entries(totals))
        byKind.push({ kind: k, seen: ids.filter((i) => seen.has(i)).length, total: ids.length, star: answerKinds.has(k) });
    } else {
      const cnt: Record<string, number> = {};
      for (const id of seen) if (els[id]) cnt[els[id].kind] = (cnt[els[id].kind] ?? 0) + 1;
      for (const [k, n] of Object.entries(cnt)) byKind.push({ kind: k, seen: n, total: null, star: answerKinds.has(k) });
    }
    byKind.sort((x, y) => Number(y.star) - Number(x.star) || (y.total ?? y.seen) - (x.total ?? x.seen));
    return { seen, views, byKind };
  }, [a, step, is3D, docExpl, els, answerKinds]);

  // ---------------------------------------------------------------- what the plan / 3D view shows
  const visual = useMemo<ExplainVisual>(() => {
    const marks: ExplainMark[] = [];
    const push = (m: ExplainMark | null) => m && marks.push(m);
    let heatmapUrl: string | null = null;
    let ids3D: string[] = [];
    const focusMarks = () => {
      for (const id of focus?.ids ?? []) push(markFor(bboxOf(els[id]), imageSize, ROLE_STYLE.step, { thick: true, fillOpacity: 0.18, label: els[id]?.label }));
      for (const r of focus?.regions ?? []) push(markFor(r, imageSize, ROLE_STYLE.step, { thick: true, fillOpacity: 0.05 }));
    };
    if (tab === 'belege' && a) {
      for (const e of a.evidence) {
        const st = ROLE_STYLE[e.role];
        push(markFor(bboxOf(els[e.element]), imageSize, st, { order: e.role === 'counted' ? e.order : undefined, label: els[e.element]?.label }));
      }
      focusMarks();
      ids3D = focus?.ids.length ? focus.ids : (groups.counted ?? groups.context ?? []).map((e) => e.element);
    } else if (tab === 'unsicherheit' && docExpl) {
      heatmapUrl = planqaAssetUrl(doc?.assets?.[channel]);
      for (const h of docExpl.hotspots ?? [])
        push(markFor(h.geometry.bbox, imageSize, ROLE_STYLE.hotspot, { order: h.rank, fillOpacity: 0 }));
      for (const b of docExpl.blind_spots ?? [])
        push(markFor(b.geometry.bbox, imageSize, ROLE_STYLE.blind, { fillOpacity: 0.05, label: b.reason }));
      focusMarks();
    } else if (tab === 'agent' && a) {
      const relevant = Object.values(els).filter(
        (e) => e.status === 'kept' && (answerKinds.size ? answerKinds.has(e.kind) : agentState.seen.has(e.id)),
      );
      for (const e of relevant)
        if (!agentState.seen.has(e.id)) push(markFor(bboxOf(e), imageSize, ROLE_STYLE.unseen, { fillOpacity: 0 }));
      for (const id of agentState.seen)
        if (els[id] && (els[id].kind !== 'room' || answerKinds.has('room')))
          push(markFor(bboxOf(els[id]), imageSize, ROLE_STYLE.seen, { fillOpacity: 0.25 }));
      for (const r of agentState.views) push(markFor(r, imageSize, ROLE_STYLE.view, { fillOpacity: 0, thick: true }));
      focusMarks();
      ids3D = [...agentState.seen];
    }
    const overlay = heatmapUrl || marks.length ? { heatmapUrl, heatmapOpacity: opacity, marks } : null;
    const vIds = ids3D.map((id) => viewerIds[id]).filter((x): x is string => Boolean(x));
    const highlight3D: HighlightTarget | null =
      is3D && a
        ? { id: `expl-${tab}-${focus?.key ?? 'all'}-${step}`, label: tab === 'agent' ? `Seen by the agent (${vIds.length})` : `Evidence (${vIds.length})`, category: 'All', keywords: [], ifcIds: vIds.length ? vIds : ['#none'] }
        : null;
    return { overlay, highlight3D };
  }, [tab, a, els, imageSize, focus, groups, docExpl, doc, channel, opacity, agentState, answerKinds, is3D, viewerIds, step]);

  const onVisualRef = useRef(onVisual);
  onVisualRef.current = onVisual;
  useEffect(() => onVisualRef.current(visual), [visual]);
  useEffect(() => () => onVisualRef.current({ overlay: null, highlight3D: null }), []);

  // ---------------------------------------------------------------- render
  const v = a ? answerValueText(a) : null;
  const counted = groups.counted ?? [];
  const considered = groups.considered ?? [];
  const excluded = groups.excluded ?? [];
  const context = groups.context ?? [];
  const evRows = [...counted, ...excluded, ...considered, ...context];
  const hotspots = docExpl?.hotspots ?? [];
  const blind = docExpl?.blind_spots ?? [];
  const relHot = hotspots.filter((h) => (h.elements ?? []).some((id) => answerIds.has(id)));

  return (
    <div className="flex flex-col h-full bg-[#F7F6F2]">
      {/* header */}
      <div className="px-4 pt-3 pb-2.5 bg-white border-b border-[#E5E2DC] sticky top-0 z-10">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onClose}
            className="inline-flex items-center gap-1 text-[11px] font-medium text-[#1F4E3D] hover:underline cursor-pointer"
          >
            <ArrowLeft className="w-3.5 h-3.5" /> Back to chat
          </button>
          <span className="ml-auto text-[10px] text-[#7A776F] font-mono truncate max-w-[55%]">
            {doc?.name}
            {docExpl?.run.thresholds &&
              ` · thresholds T ${docExpl.run.thresholds.texts} / S ${docExpl.run.thresholds.objects} / R ${docExpl.run.thresholds.rooms}`}
          </span>
        </div>
        {a && (
          <div className="mt-2 rounded-lg border border-[#C7D2FE] bg-[#F8F9FF] px-2.5 py-1.5 text-[11.5px]">
            <span className="font-semibold text-[#4338CA] mr-1.5">Question</span>
            {a.question}
          </div>
        )}
        {tabs.length > 1 && (
          <div className="mt-2 flex gap-1">
            {tabs.map(([t, label]) => (
              <button
                key={t}
                type="button"
                onClick={() => setTab(t)}
                className={`px-3 py-1 rounded-md text-[11.5px] font-medium cursor-pointer ${
                  tab === t ? 'bg-[#161614] text-white' : 'bg-[#F3F1EB] text-[#3B3934] hover:bg-[#EAE7DF]'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {/* ============================================================ BELEGE */}
        {tab === 'belege' && a && (
          <>
            <Card n={1} title="Answer">
              {a.abstained ? (
                <div className="flex gap-2 rounded-lg border border-[#F5D9A8] bg-[#FFF8EB] px-3 py-2 text-[12px] text-[#7A3E10]">
                  <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                  <div>
                    <div className="font-semibold">No reliable answer</div>
                    <div>{a.abstain_reason || 'The data is not sufficient for a reliable answer.'}</div>
                  </div>
                </div>
              ) : (
                <div className="flex items-end justify-between gap-2">
                  <div className="leading-none">
                    {v?.value !== null ? (
                      <>
                        <span className="text-[44px] font-semibold tabular-nums text-[#161614]">{v?.value}</span>
                        {v?.noun && <span className="ml-2 text-[20px] font-medium text-[#161614]">{v.noun}</span>}
                      </>
                    ) : (
                      <span className="text-[15px] font-semibold">{a.text}</span>
                    )}
                  </div>
                  <div className="text-right space-y-1">
                    <ConfidencePill a={a} />
                    <div className="text-[10px] text-[#7A776F]">{AGREEMENT_DE[a.agreement]}</div>
                  </div>
                </div>
              )}
              {a.summary && (
                <p className="mt-2.5 text-[12px] leading-relaxed text-[#161614]">
                  <span className="font-semibold">Why? </span>
                  {a.summary}
                </p>
              )}
              <div className="mt-2.5">
                <SourceTiles a={a} />
              </div>
              {agentQuote(a) && <p className="mt-2 text-[10.5px] text-[#5C5A54] italic">Agent: "{agentQuote(a)}"</p>}
            </Card>

            {evRows.length > 0 && (
              <Card n={2} title="Evidence on the plan" right={<span className="text-[10px] text-[#7A776F]">click → details</span>}>
                <Legend
                  items={[
                    ...(counted.length ? [ROLE_STYLE.counted] : []),
                    ...(considered.length ? [ROLE_STYLE.considered] : []),
                    ...(excluded.length ? [ROLE_STYLE.excluded] : []),
                    ...(context.length ? [ROLE_STYLE.context] : []),
                    ...(focus ? [ROLE_STYLE.step] : []),
                  ]}
                />
                <ol className="mt-2 space-y-0.5">
                  {(showAllEv ? evRows : evRows.slice(0, 12)).map((e) => {
                    const el = els[e.element];
                    const open = openEl === e.element;
                    const st = ROLE_STYLE[e.role];
                    return (
                      <li key={`${e.role}-${e.element}`}>
                        <button
                          type="button"
                          onClick={() => {
                            setOpenEl(open ? null : e.element);
                            focusIds(`el:${e.element}`, [e.element]);
                          }}
                          className={`w-full text-left flex items-center gap-2 rounded-md px-1.5 py-1 cursor-pointer ${
                            open ? 'bg-[#EFF6FF]' : 'hover:bg-[#F7F6F2]'
                          }`}
                        >
                          <span
                            className="shrink-0 min-w-[20px] h-[20px] px-1 rounded-full text-[10px] font-bold flex items-center justify-center"
                            style={{
                              background: st.dashed ? '#fff' : st.color,
                              color: st.dashed ? st.color : '#fff',
                              border: `2px ${st.dashed ? 'dashed' : 'solid'} ${st.color}`,
                            }}
                          >
                            {e.order ?? ''}
                          </span>
                          <span className="flex-1 min-w-0 text-[11px] truncate">{el?.label || e.reason || e.element}</span>
                          <span className="font-mono text-[9.5px] text-[#9A978F]">{e.element}</span>
                          {el?.conf !== undefined && (
                            <span className="font-mono text-[10px] text-[#5C5A54] tabular-nums">{el.conf.toFixed(2)}</span>
                          )}
                        </button>
                        {open && (
                          <div className="ml-8 mb-1 rounded-md border border-[#DBEAFE] bg-white px-2.5 py-1.5 text-[10.5px] text-[#3B3934] space-y-0.5">
                            {e.reason && <div>{e.reason}</div>}
                            {el && <div>Status: {STATUS_DE[el.status] ?? el.status}</div>}
                            {el?.conf !== undefined && <div>Model confidence: {el.conf.toFixed(2)}</div>}
                            {el?.class_probs && el.class_probs.length > 0 && (
                              <div>
                                Classes: {el.class_probs.slice(0, 3).map((c) => `${c.label} ${Math.round(c.p * 100)} %`).join(' · ')}
                              </div>
                            )}
                            {el?.room && <div>Room: {els[el.room]?.label ?? el.room}</div>}
                            {el?.geometry.storey && <div>Storey: {el.geometry.storey}</div>}
                            {is3D && !viewerIds[e.element] && <div className="text-[#B45309]">Not found in the 3D viewer.</div>}
                          </div>
                        )}
                      </li>
                    );
                  })}
                </ol>
                {evRows.length > 12 && (
                  <button type="button" onClick={() => setShowAllEv((s) => !s)} className="mt-1 text-[10.5px] font-medium text-[#1F4E3D] cursor-pointer">
                    {showAllEv ? 'show less' : `show all ${evRows.length}`}
                  </button>
                )}
                <div className="mt-2 pt-2 border-t border-[#EFEDE7] text-[10.5px] text-[#3B3934]">
                  <b>{counted.length || context.length}</b> {counted.length ? 'counted' : 'evidence'}
                  {!is3D && (
                    <>
                      {' · '}
                      <b>{considered.length}</b> dropped below threshold · <b>{excluded.length}</b> excluded by the agent
                    </>
                  )}
                </div>
              </Card>
            )}

            {a.steps.length > 0 && (
              <Card n={3} title="How the answer came about" right={<span className="text-[10px] text-[#7A776F]">click a step → elements</span>}>
                <ol className="space-y-0.5">
                  {a.steps.map((s) => (
                    <StepRow
                      key={s.n}
                      s={s}
                      active={focus?.key === `step:${s.n}`}
                      onClick={() => focusIds(`step:${s.n}`, s.ids ?? [], s.region ? [s.region] : undefined)}
                      onId={(id) => focusIds(`el:${id}`, [id])}
                    />
                  ))}
                </ol>
              </Card>
            )}

            {((a.limitations?.length ?? 0) > 0 || (a.references?.length ?? 0) > 0) && (
              <div className="rounded-xl border border-[#F5D9A8] bg-[#FFF8EB] px-3.5 py-2.5 text-[11px] text-[#7A3E10]">
                <div className="flex items-center gap-1.5 font-semibold mb-1">
                  <AlertTriangle className="w-3.5 h-3.5" /> Limitations & assumptions
                </div>
                <ul className="list-disc ml-4 space-y-0.5">
                  {(a.limitations ?? []).map((l, i) => (
                    <li key={i}>{l}</li>
                  ))}
                </ul>
                {(a.references?.length ?? 0) > 0 && (
                  <div className="mt-1 text-[10px] text-[#9A6B3F]">
                    Further sources in the data: <span className="font-mono">{a.references!.join(', ')}</span>
                  </div>
                )}
              </div>
            )}
          </>
        )}

        {/* ============================================================ UNSICHERHEIT */}
        {tab === 'unsicherheit' && docExpl && (
          <>
            <Card title="What does this view show?">
              <p className="text-[11.5px] leading-relaxed text-[#3B3934]">
                For every pixel the layout model gives probabilities (bath, WC, hallway, door, window …) and the OCR a
                reading confidence. Where the models <b>cannot decide</b>, the map is dark - that is where most errors happen.
              </p>
              <div className="mt-2 flex flex-wrap items-center gap-1">
                {['combined', 'rooms', 'symbols', 'text']
                  .filter((c) => doc?.assets?.[c])
                  .map((c) => (
                    <button
                      key={c}
                      type="button"
                      onClick={() => setChannel(c)}
                      className={`px-2.5 py-0.5 rounded-md text-[11px] cursor-pointer ${
                        channel === c ? 'bg-[#4338CA] text-white' : 'bg-[#EEF0FF] text-[#4338CA] hover:bg-[#E0E4FF]'
                      }`}
                    >
                      {CHANNEL_DE[c]}
                    </button>
                  ))}
              </div>
              <div className="mt-2 flex items-center gap-2 text-[10px] text-[#5C5A54]">
                <span className="w-24 h-2 rounded-full" style={{ background: 'linear-gradient(90deg,#FDE68A,#F97316,#7E22CE,#1E0B3A)' }} />
                certain → uncertain
                <span className="ml-auto">Opacity</span>
                <input
                  type="range"
                  min={0.2}
                  max={1}
                  step={0.05}
                  value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                  className="w-20 accent-[#F97316]"
                />
              </div>
              {typeof docExpl.assets?.uncertain_area === 'number' && (
                <div className="mt-1 text-[10px] text-[#7A776F] text-right">
                  {fmtNumber(docExpl.assets.uncertain_area * 100)} % of the plan area uncertain
                </div>
              )}
            </Card>

            <Card title={`Most uncertain spots - found automatically · ${hotspots.length}`}>
              {hotspots.length === 0 ? (
                <p className="text-[11px] text-[#7A776F]">No region above the uncertainty threshold.</p>
              ) : (
                <ul className="space-y-1">
                  {hotspots.map((h: ExplHotspot) => {
                    const key = `hot:${h.id}`;
                    const resolved = h.resolved_by && h.resolved_by !== 'unresolved';
                    const dropped = /dropped|not linked|missed/i.test(h.resolution ?? '');
                    return (
                      <li key={h.id}>
                        <button
                          type="button"
                          onClick={() => focusIds(key, h.elements ?? [], h.geometry.bbox ? [h.geometry.bbox] : undefined)}
                          className={`w-full text-left flex gap-2 rounded-lg p-1.5 cursor-pointer ${focus?.key === key ? 'bg-[#F3EEFF]' : 'hover:bg-[#F7F6F2]'}`}
                        >
                          <span className="shrink-0 w-5 h-5 rounded-full bg-[#161614] text-white text-[10px] font-bold flex items-center justify-center mt-3">
                            {h.rank}
                          </span>
                          <Thumb url={previewUrl} imageSize={imageSize} bbox={h.geometry.bbox} />
                          <div className="min-w-0 flex-1 text-[11px] leading-snug">
                            <div>
                              <b>{h.channel === 'symbols' ? 'Unclear symbol' : h.channel === 'rooms' ? 'Unclear room type' : 'Unclear text'}</b>
                              <span className="text-[#5C5A54]"> · entropy {fmtNumber(h.entropy)}</span>
                            </div>
                            <div className="text-[10.5px] text-[#5C5A54]">
                              {h.top_classes.slice(0, 3).map((c) => `${c.label} ${Math.round(c.p * 100)} %`).join(' · ')}
                            </div>
                            {h.resolution && (
                              <div className={`text-[10.5px] font-medium ${dropped ? 'text-[#DC2626]' : resolved ? 'text-[#15803D]' : 'text-[#7A776F]'}`}>
                                → {h.resolution}
                              </div>
                            )}
                            <IdChips ids={h.elements ?? []} max={6} />
                          </div>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </Card>

            {docExpl.assets?.uncertainty_share && (
              <Card title="Where does the uncertainty come from?">
                <div className="space-y-1.5">
                  {Object.entries(docExpl.assets.uncertainty_share).map(([c, val]) => (
                    <div key={c} className="flex items-center gap-2 text-[11px]">
                      <span className="w-20 text-[#3B3934]">{SHARE_DE[c] ?? c}</span>
                      <div className="flex-1 h-2 rounded-full bg-[#F1EFEA] overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${Math.round(val * 100)}%`, background: c === 'text' ? '#F59E0B' : c === 'rooms' ? '#C026D3' : '#7C3AED' }}
                        />
                      </div>
                      <span className="w-9 text-right font-semibold tabular-nums">{Math.round(val * 100)} %</span>
                    </div>
                  ))}
                </div>
                <p className="mt-1.5 text-[10px] text-[#7A776F]">
                  Share of the uncertain area - shows which model should be improved.
                </p>
              </Card>
            )}

            {a && hotspots.length > 0 && (
              <Card title="Relation to this question">
                <p className="text-[11.5px] text-[#3B3934]">
                  {relHot.length === 0
                    ? `None of the ${hotspots.length} most uncertain spots affects elements of this answer.`
                    : `${relHot.length} of the ${hotspots.length} most uncertain spots affect elements of this answer (spot ${relHot
                        .map((h) => h.rank)
                        .join(', ')}). Click a spot to mark its elements.`}
                </p>
              </Card>
            )}

            {blind.length > 0 && (
              <Card title={`Blind spots · ${blind.length}`}>
                <ul className="space-y-0.5">
                  {blind.slice(0, 30).map((b) => {
                    const key = `blind:${b.id}`;
                    return (
                      <li key={b.id}>
                        <button
                          type="button"
                          onClick={() => b.geometry.bbox && focusIds(key, [], [b.geometry.bbox])}
                          className={`w-full text-left rounded-md px-1.5 py-1 text-[11px] cursor-pointer ${focus?.key === key ? 'bg-[#F3EEFF]' : 'hover:bg-[#F7F6F2]'}`}
                        >
                          <span className="text-[#9333EA] font-semibold mr-1">?</span>
                          {b.reason}
                          {b.suggestion && <span className="block text-[10px] text-[#7A776F] ml-3">{b.suggestion}</span>}
                        </button>
                      </li>
                    );
                  })}
                </ul>
                <p className="mt-1 text-[10px] text-[#7A776F]">Things the pipeline cannot see - no answer can take them into account.</p>
              </Card>
            )}
          </>
        )}

        {/* ============================================================ AGENT */}
        {tab === 'agent' && a && (
          <>
            <Card title={`What did the agent look at? (up to step ${step})`}>
              <Legend items={[ROLE_STYLE.seen, ...(is3D ? [] : [ROLE_STYLE.unseen, ROLE_STYLE.view])]} />
              <div className="mt-2 space-y-1">
                {agentState.byKind.map((k) => (
                  <div key={k.kind} className="flex items-center gap-2 text-[11px]">
                    <span className="w-24 truncate">
                      {KIND_DE[k.kind] ?? k.kind}
                      {k.star && ' ★'}
                    </span>
                    {k.total !== null ? (
                      <>
                        <div className="flex-1 h-2 rounded-full bg-[#F1EFEA] overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{ width: `${k.total ? (k.seen / k.total) * 100 : 0}%`, background: k.star ? '#2563EB' : '#93C5FD' }}
                          />
                        </div>
                        <span className={`w-12 text-right tabular-nums ${k.seen ? 'font-semibold' : 'text-[#9A978F]'}`}>
                          {k.seen} / {k.total}
                        </span>
                      </>
                    ) : (
                      <span className="font-semibold tabular-nums">{k.seen}</span>
                    )}
                  </div>
                ))}
              </div>
              <p className="mt-1.5 text-[10px] text-[#7A776F]">
                ★ = element type of the answer.
                {agentState.views.length > 0 && ` ${agentState.views.length} image crop(s) viewed.`}
              </p>
            </Card>

            <Card title="Agent run - replayable">
              <div className="flex items-center gap-2 mb-2">
                <button
                  type="button"
                  onClick={() => setStep((s) => Math.max(1, s - 1))}
                  className="w-7 h-7 rounded-md bg-[#161614] text-white flex items-center justify-center cursor-pointer"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (!playing && step >= lastStep) setStep(1);
                    setPlaying((p) => !p);
                  }}
                  className="w-7 h-7 rounded-md bg-[#161614] text-white flex items-center justify-center cursor-pointer"
                >
                  {playing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                </button>
                <button
                  type="button"
                  onClick={() => setStep((s) => Math.min(lastStep, s + 1))}
                  className="w-7 h-7 rounded-md bg-[#161614] text-white flex items-center justify-center cursor-pointer"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
                <input
                  type="range"
                  min={1}
                  max={Math.max(1, lastStep)}
                  value={step}
                  onChange={(e) => {
                    setPlaying(false);
                    setStep(Number(e.target.value));
                  }}
                  className="flex-1 accent-[#F97316]"
                />
                <span className="text-[11px] tabular-nums w-[70px] text-right">
                  Step {step} / {lastStep}
                </span>
              </div>
              <ol className="space-y-0.5">
                {a.steps.map((s) => (
                  <StepRow
                    key={s.n}
                    s={s}
                    active={s.n === step}
                    dim={s.n > step}
                    onClick={() => {
                      setPlaying(false);
                      setStep(s.n);
                    }}
                  />
                ))}
              </ol>
              <p className="mt-2 text-[10px] text-[#7A776F]">
                Coverage at step {step} of {lastStep}: everything that appeared in a tool result up to here is blue
                {is3D ? '.' : ' - dashed grey was not read (yet).'}
              </p>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}
