import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { Building } from "../types";

interface Props {
  buildings: Building[];
}

const BUCKETS = [
  { min: 0, max: 0.05, label: "0-5", color: "#06b6d4" },
  { min: 0.05, max: 0.1, label: "5-10", color: "#10b981" },
  { min: 0.1, max: 0.15, label: "10-15", color: "#22c55e" },
  { min: 0.15, max: 0.2, label: "15-20", color: "#84cc16" },
  { min: 0.2, max: 0.3, label: "20-30", color: "#eab308" },
  { min: 0.3, max: 0.5, label: "30-50", color: "#f97316" },
  { min: 0.5, max: 1.0, label: "50+", color: "#ef4444" },
];

export default function DistributionChart({ buildings }: Props) {
  const data = useMemo(
    () =>
      BUCKETS.map(({ min, max, label, color }) => ({
        name: label,
        count: buildings.filter((b) => b.wwr >= min && b.wwr < max).length,
        color,
      })),
    [buildings],
  );

  return (
    <div className="absolute bottom-4 left-4 z-[1000] pointer-events-none animate-fade-up-d2">
      <div
        className="glass rounded-2xl p-4 pointer-events-auto"
        style={{ width: 270 }}
      >
        <div className="font-display text-[9px] uppercase tracking-[0.15em] text-slate-500 mb-3 font-bold">
          WWR Distribution
        </div>
        <ResponsiveContainer width="100%" height={90}>
          <BarChart
            data={data}
            margin={{ top: 0, right: 0, bottom: 0, left: 0 }}
          >
            <XAxis
              dataKey="name"
              tick={{
                fill: "#475569",
                fontSize: 8,
                fontFamily: "JetBrains Mono",
              }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis hide />
            <Tooltip
              contentStyle={{
                background: "rgba(10,16,30,0.92)",
                border: "1px solid rgba(148,163,184,0.1)",
                borderRadius: 10,
                fontSize: 11,
                fontFamily: "JetBrains Mono",
                padding: "6px 12px",
                boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
              }}
              itemStyle={{ color: "#e2e8f0" }}
              labelStyle={{ color: "#94a3b8", fontSize: 10 }}
              formatter={(v: number) => [`${v} buildings`, ""]}
              labelFormatter={(l) => `${l}% WWR`}
              cursor={false}
            />
            <Bar dataKey="count" radius={[3, 3, 0, 0]}>
              {data.map((d, i) => (
                <Cell key={i} fill={d.color} fillOpacity={0.7} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
