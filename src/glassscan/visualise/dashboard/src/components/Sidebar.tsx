import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { Stats, Building } from "../types";

interface Props {
  stats: Stats;
  selected: Building | null;
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-slate-700/50 rounded-lg p-3">
      <div className="text-slate-400 text-xs uppercase tracking-wide">
        {label}
      </div>
      <div className="text-xl font-bold mt-1">{value}</div>
    </div>
  );
}

function ChartSection({
  title,
  data,
  color,
  formatValue,
}: {
  title: string;
  data: { name: string; value: number }[];
  color: string;
  formatValue?: (v: number) => string;
}) {
  const fmt = formatValue ?? ((v: number) => v.toFixed(2));

  return (
    <div>
      <h3 className="text-xs uppercase tracking-wide text-slate-400 mb-2">
        {title}
      </h3>
      <ResponsiveContainer width="100%" height={data.length * 32 + 20}>
        <BarChart data={data} layout="vertical" margin={{ left: 0, right: 8 }}>
          <XAxis type="number" hide />
          <YAxis
            type="category"
            dataKey="name"
            width={90}
            tick={{ fill: "#94a3b8", fontSize: 11 }}
          />
          <Tooltip
            formatter={(v: number) => fmt(v)}
            contentStyle={{
              backgroundColor: "#1e293b",
              border: "1px solid #334155",
              borderRadius: 6,
              fontSize: 12,
            }}
            itemStyle={{ color: "#e2e8f0" }}
            labelStyle={{ color: "#94a3b8" }}
          />
          <Bar dataKey="value" fill={color} radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function toChartData(obj: Record<string, number>) {
  return Object.entries(obj)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);
}

export default function Sidebar({ stats, selected }: Props) {
  return (
    <div className="w-[350px] h-full bg-slate-800 text-white overflow-y-auto flex flex-col">
      {/* Header */}
      <div className="p-5 border-b border-slate-700">
        <h1 className="text-xl font-bold tracking-tight">GlassScan</h1>
        <p className="text-slate-400 text-sm mt-1">
          Window-to-Wall Ratio Analysis
        </p>
      </div>

      {/* Stats */}
      <div className="p-5 border-b border-slate-700">
        <div className="grid grid-cols-2 gap-3">
          <StatCard label="Buildings" value={stats.total.toLocaleString()} />
          <StatCard label="Mean WWR" value={`${(stats.mean_wwr * 100).toFixed(1)}%`} />
          <StatCard label="Measured" value={stats.measured.toLocaleString()} />
          <StatCard label="Predicted" value={stats.predicted.toLocaleString()} />
        </div>
      </div>

      {/* Selected building */}
      {selected && (
        <div className="p-5 border-b border-slate-700">
          <h3 className="text-xs uppercase tracking-wide text-slate-400 mb-2">
            Selected Building
          </h3>
          <div className="bg-slate-700/50 rounded-lg p-3">
            <div className="text-lg font-bold">
              WWR: {(selected.wwr * 100).toFixed(1)}%
            </div>
            <div className="text-slate-400 text-xs mt-1">
              {selected.egid} -- {selected.source}
            </div>
          </div>
        </div>
      )}

      {/* Charts */}
      <div className="p-5 space-y-6 flex-1">
        {Object.keys(stats.wwr_by_era).length > 0 && (
          <ChartSection
            title="WWR by Construction Era"
            data={toChartData(stats.wwr_by_era)}
            color="#3b82f6"
            formatValue={(v) => `${(v * 100).toFixed(1)}%`}
          />
        )}

        {Object.keys(stats.wwr_by_type).length > 0 && (
          <ChartSection
            title="WWR by Building Type"
            data={toChartData(stats.wwr_by_type)}
            color="#8b5cf6"
            formatValue={(v) => `${(v * 100).toFixed(1)}%`}
          />
        )}

        {Object.keys(stats.feature_importance).length > 0 && (
          <ChartSection
            title="Feature Importance"
            data={toChartData(stats.feature_importance)}
            color="#f59e0b"
          />
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-slate-700 text-slate-500 text-xs">
        Energy Data Hackdays 2026
      </div>
    </div>
  );
}
