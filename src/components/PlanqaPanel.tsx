// Pipeline status of the active file (one line), with the way into the document-level explanation.
import { AlertTriangle, ChevronRight, Cpu, Loader2, RefreshCw } from 'lucide-react';
import type { PlanqaEntryState } from '../types/explanation';

interface Props {
  state: PlanqaEntryState | undefined;
  supported: boolean;
  onOpenUncertainty: () => void;
  onRetry: () => void;
}

export function PlanqaPanel({ state, supported, onOpenUncertainty, onRetry }: Props) {
  if (!supported) {
    return (
      <div className="rounded-lg border border-[#E5E2DC] bg-[#FAF9F5] px-3 py-2 text-[11px] text-[#5C5A54]">
        <span className="font-semibold text-[#3B3934]">Pipeline:</span> the pipeline cannot read this format yet
        (PNG/JPG/PDF/IFC). Answers come directly from Gemini, without verifiable evidence.
      </div>
    );
  }
  if (!state) return null;
  const busy = state.status === 'uploading' || state.status === 'queued' || state.status === 'processing';
  const info = state.info;
  const expl = info?.explanation;

  return (
    <div className="rounded-lg border border-[#E5E2DC] bg-[#FAF9F5] px-3 py-2 text-[11px] text-[#3B3934]">
      <div className="flex items-center gap-1.5">
        <Cpu className="w-3.5 h-3.5 text-[#1F4E3D]" />
        <span className="font-semibold">Pipeline</span>
        <span className="ml-auto flex items-center gap-1 text-[10.5px]">
          {busy && <Loader2 className="w-3 h-3 animate-spin text-[#1F4E3D]" />}
          {state.status === 'ready' && <span className="w-2 h-2 rounded-full bg-[#16A34A]" />}
          {(state.status === 'error' || state.status === 'offline') && <span className="w-2 h-2 rounded-full bg-[#DC2626]" />}
          {state.status === 'uploading' && 'uploading'}
          {state.status === 'queued' && 'queued'}
          {state.status === 'processing' && (state.stage || 'running')}
          {state.status === 'ready' && `ready${state.seconds ? ` · ${state.seconds} s` : ''}`}
          {state.status === 'error' && 'error'}
          {state.status === 'offline' && 'service unreachable'}
        </span>
      </div>

      {busy && (
        <p className="mt-1 text-[10.5px] text-[#5C5A54]">
          Running OCR and the layout model and building the graph. Questions wait for this and are then answered with evidence.
        </p>
      )}

      {(state.status === 'error' || state.status === 'offline') && (
        <div className="mt-1 flex items-start gap-1.5 text-[10.5px] text-[#B42318]">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
          <div className="flex-1 break-words">
            {state.error}
            <div className="text-[#7A776F]">Until then, questions are answered directly by Gemini (without evidence).</div>
          </div>
          <button
            type="button"
            onClick={onRetry}
            className="shrink-0 inline-flex items-center gap-1 px-1.5 py-0.5 rounded border border-[#DEDAD0] bg-white text-[#3B3934] hover:border-[#1F4E3D] cursor-pointer"
          >
            <RefreshCw className="w-3 h-3" /> retry
          </button>
        </div>
      )}

      {state.status === 'ready' && info?.kind === 'plan' && expl && (
        <button
          type="button"
          onClick={onOpenUncertainty}
          className="mt-1.5 w-full flex items-center gap-2 text-left text-[10.5px] text-[#3B3934] hover:text-[#1F4E3D] cursor-pointer"
        >
          <span>
            {Object.values(expl.elements).filter((e) => e.status === 'kept').length} elements detected ·{' '}
            {(expl.hotspots ?? []).length} uncertain spots · {(expl.blind_spots ?? []).length} blind spots
          </span>
          <span className="ml-auto inline-flex items-center font-medium text-[#1F4E3D]">
            View uncertainty <ChevronRight className="w-3 h-3" />
          </span>
        </button>
      )}

      {state.status === 'ready' && info?.kind === 'ifc' && info.summary && (
        <p className="mt-1 text-[10.5px] text-[#5C5A54]">
          {info.summary.storeys} storeys · {info.summary.spaces} spaces ·{' '}
          {Object.entries(info.summary.elements)
            .filter(([, n]) => n > 0)
            .map(([g, n]) => `${n} ${g}`)
            .join(' · ')}
        </p>
      )}
    </div>
  );
}
