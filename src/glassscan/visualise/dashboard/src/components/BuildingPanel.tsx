import { useState } from "react";
import type { Building, ViewData } from "../types";
import WWRGauge from "./WWRGauge";

interface Props {
  building: Building;
  onClose: () => void;
}

const LABEL: Record<string, string> = {
  construction_year: "Built",
  storeys: "Storeys",
  area_m2: "Area",
  building_category: "Category",
};

function fmtVal(k: string, v: string | number): string {
  if (k === "area_m2") return `${v} m\u00B2`;
  if (k === "construction_year" && typeof v === "number")
    return Math.round(v).toString();
  return String(v);
}

// Expand icon (arrows pointing outward)
function ExpandIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
      <path d="M8.5 1.5h4v4M5.5 12.5h-4v-4M12.5 1.5L8 6M1.5 12.5L6 8" />
    </svg>
  );
}

// Collapse icon (arrows pointing inward)
function CollapseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
      <path d="M9 5h4M1 9h4M13 5l-4 4M1 9l4-4" />
    </svg>
  );
}

export default function BuildingPanel({ building: b, onClose }: Props) {
  const [expanded, setExpanded] = useState(false);

  const details = (
    <>
      {/* Gauge */}
      <div className="flex justify-center py-1">
        <WWRGauge wwr={b.wwr} size={expanded ? 150 : 130} />
      </div>

      {/* Source badge */}
      <div>
        <span
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium ${
            b.source === "measured"
              ? "bg-cyan-500/8 text-cyan-400 border border-cyan-500/15"
              : "bg-violet-500/8 text-violet-400 border border-violet-500/15"
          }`}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              b.source === "measured" ? "bg-cyan-400" : "bg-violet-400"
            }`}
          />
          {b.source === "measured" ? "CV Measured" : "ML Predicted"}
        </span>
      </div>

      {/* Measurement stats */}
      {b.source === "measured" && (
        <div className="grid grid-cols-2 gap-3">
          {b.n_windows != null && (
            <MiniCard label="Windows" value={String(b.n_windows)} />
          )}
          {b.confidence != null && (
            <MiniCard
              label="Confidence"
              value={`${(b.confidence * 100).toFixed(0)}%`}
            />
          )}
        </div>
      )}

      {/* Prediction CI */}
      {b.source === "predicted" && b.prediction_interval && (
        <div className="bg-white/[0.02] rounded-xl p-3.5">
          <div className="text-[9px] text-slate-500 uppercase tracking-[0.15em] mb-1">
            90% Confidence Interval
          </div>
          <div className="font-mono text-sm text-slate-300">
            {(b.prediction_interval[0] * 100).toFixed(1)}% &ndash;{" "}
            {(b.prediction_interval[1] * 100).toFixed(1)}%
          </div>
        </div>
      )}

      {/* Metadata */}
      {Object.keys(b.metadata).length > 0 && (
        <div>
          <div className="text-[9px] text-slate-500 uppercase tracking-[0.15em] mb-2.5">
            Metadata
          </div>
          <div className="space-y-0">
            {Object.entries(b.metadata).map(([k, v]) => (
              <div
                key={k}
                className="flex items-center justify-between py-2 border-b border-white/[0.03] last:border-0"
              >
                <span className="text-xs text-slate-500">
                  {LABEL[k] || k.replace(/_/g, " ")}
                </span>
                <span className="text-xs font-mono text-slate-300">
                  {fmtVal(k, v)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );

  return (
    <div
      className={`absolute top-4 right-4 bottom-4 z-[1000] transition-all duration-300 ease-out ${
        expanded ? "left-[30%]" : "w-[370px]"
      }`}
    >
      <div className="glass glass-glow rounded-2xl h-full flex flex-col overflow-hidden animate-slide-in">
        {/* Header */}
        <div className="flex items-center justify-between px-5 pt-4 pb-3 border-b border-white/[0.04] shrink-0">
          <div>
            <div className="text-[9px] text-slate-500 uppercase tracking-[0.15em]">
              Building
            </div>
            <div className="font-mono text-sm text-slate-300 mt-0.5">
              {b.egid}
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setExpanded(!expanded)}
              className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-500 hover:text-white hover:bg-white/5 transition-colors"
              title={expanded ? "Collapse panel" : "Expand panel"}
            >
              {expanded ? <CollapseIcon /> : <ExpandIcon />}
            </button>
            <button
              onClick={onClose}
              className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-500 hover:text-white hover:bg-white/5 transition-colors"
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M3 3l8 8M11 3l-8 8" />
              </svg>
            </button>
          </div>
        </div>

        {/* Body */}
        {expanded ? (
          /* Expanded: two-column — pipeline stages left, details right */
          <div className="flex-1 flex overflow-hidden min-h-0">
            {/* Left: pipeline stages */}
            <div className="flex-1 overflow-y-auto custom-scroll p-5 min-w-0">
              {b.source === "measured" ? (
                <div className="grid grid-cols-5 gap-3 h-full">
                  <PipelineColumn
                    step={1}
                    label="Fetch"
                    sublabel="Street View"
                    egid={b.egid}
                    views={b.views}
                    srcFn={(egid, suffix) => `./raw/${egid}${suffix}.jpg`}
                  />
                  <PipelineColumn
                    step={2}
                    label="Segment"
                    sublabel="Wall / Window / BG"
                    egid={b.egid}
                    views={b.views}
                    srcFn={(egid, suffix) => `./overlays/${egid}${suffix}.jpg`}
                  />
                  <PipelineColumn
                    step={3}
                    label="Rectify"
                    sublabel="Perspective correction"
                    egid={b.egid}
                    views={b.views}
                    srcFn={(egid, suffix) =>
                      `./rectified/${egid}${suffix}_rectified.jpg`
                    }
                  />
                  <PipelineColumn
                    step={4}
                    label="Measure"
                    sublabel="WWR pixel count"
                    egid={b.egid}
                    views={b.views}
                    srcFn={(egid, suffix) =>
                      `./rectified_overlays/${egid}${suffix}.jpg`
                    }
                  />
                  <WeightColumn views={b.views} aggregatedWwr={b.wwr} />
                </div>
              ) : (
                <div className="h-full flex items-center justify-center text-slate-500 text-sm">
                  No imagery for predicted buildings
                </div>
              )}
            </div>
            {/* Right: details */}
            <div className="w-[300px] shrink-0 overflow-y-auto custom-scroll px-5 py-4 space-y-5 border-l border-white/[0.04]">
              {details}
            </div>
          </div>
        ) : (
          /* Collapsed: single column */
          <div className="flex-1 overflow-y-auto custom-scroll px-5 py-4 space-y-5">
            {/* Building card image */}
            {b.source === "measured" && (
              <div className="rounded-xl overflow-hidden border border-white/[0.04]">
                <img
                  src={`./images/${b.egid}.jpg`}
                  alt={`Building ${b.egid}`}
                  className="w-full block"
                  onError={(e) => {
                    (e.target as HTMLImageElement).parentElement!.style.display =
                      "none";
                  }}
                />
              </div>
            )}
            {details}
          </div>
        )}

        {/* Footer */}
        <div className="px-5 py-3 border-t border-white/[0.04] text-[9px] text-slate-600 tracking-[0.15em] uppercase shrink-0">
          Energy Data Hackdays 2026
        </div>
      </div>
    </div>
  );
}

function MiniCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white/[0.02] rounded-xl p-3.5">
      <div className="text-[9px] text-slate-500 uppercase tracking-[0.15em]">
        {label}
      </div>
      <div className="font-mono text-lg font-semibold text-slate-200 mt-1 leading-none">
        {value}
      </div>
    </div>
  );
}

function PipelineColumn({
  step,
  label,
  sublabel,
  egid,
  views,
  srcFn,
}: {
  step: number;
  label: string;
  sublabel: string;
  egid: string;
  views: ViewData[] | null;
  srcFn: (egid: string, suffix: string) => string;
}) {
  const viewCount = views ? views.length : 1;

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center gap-2 mb-2.5 shrink-0">
        <span className="w-5 h-5 rounded-full bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-[10px] font-mono font-semibold text-cyan-400">
          {step}
        </span>
        <div>
          <div className="text-xs font-medium text-slate-300 leading-none">
            {label}
          </div>
          <div className="text-[9px] text-slate-500 mt-0.5 leading-none">
            {sublabel}
          </div>
        </div>
      </div>
      <div className="flex-1 flex flex-col gap-1.5 min-h-0">
        {Array.from({ length: viewCount }, (_, i) => {
          const suffix = i === 0 ? "" : `_v${i}`;
          return (
            <div
              key={i}
              className="flex-1 rounded-xl overflow-hidden border border-white/[0.04] bg-black/20 min-h-0"
            >
              <img
                src={srcFn(egid, suffix)}
                alt={`${label} view ${i}`}
                className="w-full h-full object-contain"
                onError={(e) => {
                  const el = e.target as HTMLImageElement;
                  el.parentElement!.innerHTML =
                    '<div class="w-full h-full flex items-center justify-center text-slate-600 text-[10px]">N/A</div>';
                }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function WeightColumn({
  views,
  aggregatedWwr,
}: {
  views: ViewData[] | null;
  aggregatedWwr: number;
}) {
  const viewList = views ?? [{ wwr: aggregatedWwr, weight: 1.0, n_windows: 0, confidence: 0 }];

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center gap-2 mb-2.5 shrink-0">
        <span className="w-5 h-5 rounded-full bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-[10px] font-mono font-semibold text-cyan-400">
          5
        </span>
        <div>
          <div className="text-xs font-medium text-slate-300 leading-none">
            Weight
          </div>
          <div className="text-[9px] text-slate-500 mt-0.5 leading-none">
            View contribution
          </div>
        </div>
      </div>
      <div className="flex-1 flex flex-col gap-1.5 min-h-0">
        {viewList.map((v, i) => {
          const maxWeight = Math.max(...viewList.map((vv) => vv.weight));
          const barWidth = maxWeight > 0 ? (v.weight / maxWeight) * 100 : 0;

          return (
            <div
              key={i}
              className="flex-1 rounded-xl border border-white/[0.04] bg-black/20 flex flex-col items-center justify-center gap-2 px-3"
            >
              <div className="text-[9px] text-slate-500 uppercase tracking-wider">
                {i === 0 ? "Primary" : `View ${i + 1}`}
              </div>
              <div className="font-mono text-lg font-semibold text-cyan-400 leading-none">
                {v.weight.toFixed(1)}
              </div>
              <div className="w-full h-1.5 rounded-full bg-white/[0.04] overflow-hidden">
                <div
                  className="h-full rounded-full bg-cyan-500/50 transition-all"
                  style={{ width: `${barWidth}%` }}
                />
              </div>
              <div className="text-[10px] font-mono text-slate-400">
                WWR {(v.wwr * 100).toFixed(1)}%
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
