interface Props {
  wwr: number;
  size?: number;
}

function gaugeColor(wwr: number): string {
  if (wwr < 0.08) return "#06b6d4";
  if (wwr < 0.15) return "#10b981";
  if (wwr < 0.22) return "#22c55e";
  if (wwr < 0.30) return "#84cc16";
  if (wwr < 0.40) return "#eab308";
  if (wwr < 0.50) return "#f97316";
  return "#ef4444";
}

export default function WWRGauge({ wwr, size = 130 }: Props) {
  const strokeW = 5;
  const r = (size - strokeW * 2) / 2;
  const circ = 2 * Math.PI * r;
  const pct = Math.min(wwr / 0.6, 1); // visual cap at 60% WWR
  const offset = circ * (1 - pct);
  const color = gaugeColor(wwr);

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="rgba(255,255,255,0.03)"
          strokeWidth={strokeW}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={strokeW}
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          className="transition-all duration-700 ease-out"
          style={{ filter: `drop-shadow(0 0 8px ${color}30)` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span
          className="font-mono text-[1.65rem] font-bold tabular-nums leading-none"
          style={{ color }}
        >
          {(wwr * 100).toFixed(1)}
        </span>
        <span className="text-[9px] text-slate-500 uppercase tracking-[0.15em] mt-1.5">
          % wwr
        </span>
      </div>
    </div>
  );
}
