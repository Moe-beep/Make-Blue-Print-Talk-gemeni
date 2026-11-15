// Compact answer card in the chat: the final answer first, then why, the sources and the way into the explanation.
import { AlertTriangle, ChevronRight, Crosshair } from 'lucide-react';
import type { PlanqaAnswerBundle } from '../types/explanation';
import { AGREEMENT_DE, answerValueText, ConfidencePill, SourceTiles } from './ExplanationView';

interface Props {
  bundle: PlanqaAnswerBundle;
  onOpen: () => void;
}

export function ExplanationCard({ bundle, onOpen }: Props) {
  const a = bundle.answer;
  const v = answerValueText(a);
  const counted = a.evidence.filter((e) => e.role === 'counted' || e.role === 'context').length;
  const below = a.evidence.filter((e) => e.role === 'considered').length;

  return (
    <div className="whitespace-normal text-[#161614]">
      <div className="text-[9.5px] font-semibold uppercase tracking-wider text-[#7A776F]">Answer</div>

      {a.abstained ? (
        <div className="mt-1 flex gap-2 rounded-lg border border-[#F5D9A8] bg-[#FFF8EB] px-2.5 py-2 text-[11.5px] text-[#7A3E10]">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold">No reliable answer</div>
            <div>{a.abstain_reason || 'The data is not sufficient for a reliable answer.'}</div>
          </div>
        </div>
      ) : (
        <div className="mt-0.5 flex items-end justify-between gap-2 flex-wrap">
          <div className="leading-none">
            {v.value !== null ? (
              <>
                <span className="text-[34px] font-semibold tabular-nums">{v.value}</span>
                {v.noun && <span className="ml-1.5 text-[16px] font-medium">{v.noun}</span>}
              </>
            ) : (
              <span className="text-[13px] font-semibold leading-snug">{a.text}</span>
            )}
          </div>
          <div className="text-right space-y-0.5">
            <ConfidencePill a={a} />
            <div className="text-[9.5px] text-[#7A776F]">{AGREEMENT_DE[a.agreement]}</div>
          </div>
        </div>
      )}

      {a.summary && (
        <p className="mt-2 text-[11.5px] leading-relaxed">
          <span className="font-semibold">Why? </span>
          {a.summary}
        </p>
      )}

      <div className="mt-2">
        <SourceTiles a={a} />
      </div>

      <button
        type="button"
        onClick={onOpen}
        className="mt-2.5 w-full inline-flex items-center gap-2 rounded-lg bg-[#161614] hover:bg-[#2A2925] text-white px-3 py-2 text-[11.5px] font-medium cursor-pointer"
      >
        <Crosshair className="w-3.5 h-3.5" />
        View explanation & evidence
        <span className="ml-auto text-[10px] text-white/70">
          {counted > 0 && `${counted} evidence`}
          {below > 0 && ` · ${below} below threshold`}
        </span>
        <ChevronRight className="w-3.5 h-3.5" />
      </button>
      {(a.limitations?.length ?? 0) > 0 && (
        <p className="mt-1.5 text-[10px] text-[#9A6B3F]">⚠ {a.limitations![0]}</p>
      )}
    </div>
  );
}
