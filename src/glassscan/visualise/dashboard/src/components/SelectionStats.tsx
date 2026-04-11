import type { Building } from "../types";

interface Props {
  buildings: Building[];
  onClear: () => void;
}

export default function SelectionStats({ buildings, onClear }: Props) {
  const measured = buildings.filter((b) => b.source === "measured").length;
  const predicted = buildings.length - measured;
  const meanWwr =
    buildings.length > 0
      ? buildings.reduce((s, b) => s + b.wwr, 0) / buildings.length
      : 0;

  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-[1000] animate-fade-up">
      <div className="glass glass-glow rounded-xl px-5 py-3 flex items-center gap-6">
        <div className="text-[9px] text-slate-500 uppercase tracking-[0.15em]">
          Selection
        </div>
        <Stat label="Buildings" value={String(buildings.length)} />
        <Stat
          label="Mean WWR"
          value={`${(meanWwr * 100).toFixed(1)}%`}
          accent
        />
        <Stat label="Measured" value={String(measured)} />
        {predicted > 0 && (
          <Stat label="Predicted" value={String(predicted)} />
        )}
        <button
          onClick={onClear}
          className="ml-2 text-slate-500 hover:text-white transition-colors"
          title="Clear selection"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          >
            <path d="M3 3l8 8M11 3l-8 8" />
          </svg>
        </button>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="text-center">
      <div className="text-[9px] text-slate-500 uppercase tracking-[0.15em] leading-none">
        {label}
      </div>
      <div
        className={`text-[14px] font-mono font-semibold tabular-nums mt-1 leading-none ${
          accent ? "text-cyan-400" : "text-slate-200"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
