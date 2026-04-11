import type { Stats } from "../types";

interface Props {
  stats: Stats;
}

export default function Header({ stats }: Props) {
  return (
    <div className="absolute top-4 left-4 right-4 z-[1000] pointer-events-none">
      <div className="glass glass-glow rounded-2xl px-5 py-3.5 flex items-center justify-between pointer-events-auto animate-fade-up">
        {/* Brand */}
        <div className="flex items-center gap-3.5">
          <svg
            width="30"
            height="30"
            viewBox="0 0 30 30"
            fill="none"
            className="text-cyan-400 shrink-0"
          >
            <path
              d="M15 2L27 8.5v13L15 28 3 21.5v-13L15 2z"
              stroke="currentColor"
              strokeWidth="1.5"
              fill="rgba(6,182,212,0.08)"
            />
            <path
              d="M15 2v26M3 8.5l24 13M27 8.5L3 21.5"
              stroke="currentColor"
              strokeWidth="0.7"
              opacity="0.4"
            />
          </svg>
          <div>
            <h1 className="font-display font-bold text-[1.15rem] tracking-tight leading-none text-white">
              GlassScan
            </h1>
            <p className="text-[9px] text-slate-500 tracking-[0.2em] uppercase mt-1">
              Window-to-Wall Ratio
            </p>
          </div>
        </div>

        {/* Divider */}
        <div className="h-8 w-px bg-slate-700/50 mx-4 hidden sm:block" />

        {/* Stats */}
        <div className="flex items-center gap-7">
          <Stat label="Buildings" value={stats.total.toLocaleString()} />
          <Stat
            label="Mean WWR"
            value={`${(stats.mean_wwr * 100).toFixed(1)}%`}
            accent
          />
          <Stat label="CV Measured" value={stats.measured.toLocaleString()} />
          {stats.predicted > 0 && (
            <Stat
              label="ML Predicted"
              value={stats.predicted.toLocaleString()}
            />
          )}
        </div>
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
    <div className="text-right">
      <div className="text-[9px] text-slate-500 uppercase tracking-[0.15em] leading-none">
        {label}
      </div>
      <div
        className={`text-[15px] font-mono font-semibold tabular-nums mt-1 leading-none ${
          accent ? "text-cyan-400" : "text-slate-200"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
