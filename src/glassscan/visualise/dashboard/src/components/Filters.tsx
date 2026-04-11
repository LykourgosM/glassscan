import { useState } from "react";

export interface FilterState {
  source: "all" | "measured" | "predicted";
  wwrMin: number;
  wwrMax: number;
}

export const DEFAULT_FILTERS: FilterState = {
  source: "all",
  wwrMin: 0,
  wwrMax: 1,
};

interface Props {
  filters: FilterState;
  onChange: (f: FilterState) => void;
  total: number;
  filtered: number;
  drawing: boolean;
  onToggleDraw: () => void;
}

const SOURCE_OPTIONS: { value: FilterState["source"]; label: string }[] = [
  { value: "all", label: "All" },
  { value: "measured", label: "CV" },
  { value: "predicted", label: "ML" },
];

export default function Filters({
  filters,
  onChange,
  total,
  filtered,
  drawing,
  onToggleDraw,
}: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="absolute top-[88px] left-4 z-[1000] pointer-events-none animate-fade-up-d1">
      <div className="flex gap-2">
        {/* Filter button */}
        <button
          onClick={() => setOpen(!open)}
          className="glass rounded-xl px-3.5 py-2 flex items-center gap-2 pointer-events-auto text-slate-400 hover:text-white transition-colors"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          >
            <path d="M1 3h12M3 7h8M5 11h4" />
          </svg>
          <span className="text-xs font-medium">
            Filters
            {filtered < total && (
              <span className="text-cyan-400 ml-1.5 font-mono">
                {filtered}/{total}
              </span>
            )}
          </span>
        </button>

        {/* Draw select button */}
        <button
          onClick={onToggleDraw}
          className={`glass rounded-xl px-3.5 py-2 flex items-center gap-2 pointer-events-auto transition-colors ${
            drawing
              ? "text-cyan-400 glass-glow"
              : "text-slate-400 hover:text-white"
          }`}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M2 10l3-8 4 5 3-3" />
            <path d="M2 10l10-6" strokeDasharray="2 2" />
          </svg>
          <span className="text-xs font-medium">
            {drawing ? "Drawing..." : "Select Area"}
          </span>
        </button>
      </div>

      {open && (
        <div className="glass glass-glow rounded-xl mt-2 p-4 pointer-events-auto w-[240px] space-y-4">
          {/* Source toggle */}
          <div>
            <div className="text-[9px] text-slate-500 uppercase tracking-[0.15em] mb-2">
              Source
            </div>
            <div className="flex gap-1">
              {SOURCE_OPTIONS.map(({ value, label }) => (
                <button
                  key={value}
                  onClick={() => onChange({ ...filters, source: value })}
                  className={`flex-1 px-2 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    filters.source === value
                      ? "bg-cyan-500/15 text-cyan-400 border border-cyan-500/20"
                      : "bg-white/[0.03] text-slate-500 border border-transparent hover:text-slate-300"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* WWR range */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="text-[9px] text-slate-500 uppercase tracking-[0.15em]">
                WWR Range
              </div>
              <div className="text-[10px] font-mono text-slate-400">
                {(filters.wwrMin * 100).toFixed(0)}% &ndash;{" "}
                {(filters.wwrMax * 100).toFixed(0)}%
              </div>
            </div>
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-xs text-slate-400">
                <span className="w-6">Min</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.wwrMin * 100}
                  onChange={(e) =>
                    onChange({
                      ...filters,
                      wwrMin: Number(e.target.value) / 100,
                    })
                  }
                  className="flex-1 accent-cyan-500 h-1"
                />
              </label>
              <label className="flex items-center gap-2 text-xs text-slate-400">
                <span className="w-6">Max</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.wwrMax * 100}
                  onChange={(e) =>
                    onChange({
                      ...filters,
                      wwrMax: Number(e.target.value) / 100,
                    })
                  }
                  className="flex-1 accent-cyan-500 h-1"
                />
              </label>
            </div>
          </div>

          {/* Reset */}
          {(filters.source !== "all" ||
            filters.wwrMin > 0 ||
            filters.wwrMax < 1) && (
            <button
              onClick={() => onChange(DEFAULT_FILTERS)}
              className="text-[10px] text-slate-500 hover:text-slate-300 transition-colors"
            >
              Reset filters
            </button>
          )}
        </div>
      )}
    </div>
  );
}
