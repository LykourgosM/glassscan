import { useEffect, useMemo, useState } from "react";
import type { DashboardData, Building } from "./types";
import MapView from "./components/MapView";
import Header from "./components/Header";
import BuildingPanel from "./components/BuildingPanel";
import DistributionChart from "./components/DistributionChart";
import Filters, {
  DEFAULT_FILTERS,
  type FilterState,
} from "./components/Filters";
import { filterByPolygon } from "./components/DrawTool";
import SelectionStats from "./components/SelectionStats";

function applyFilters(buildings: Building[], f: FilterState): Building[] {
  return buildings.filter((b) => {
    if (f.source !== "all" && b.source !== f.source) return false;
    if (b.wwr < f.wwrMin || b.wwr > f.wwrMax) return false;
    return true;
  });
}

export default function App() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Building | null>(null);
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);

  // Draw tool state
  const [drawing, setDrawing] = useState(false);
  const [drawVertices, setDrawVertices] = useState<[number, number][]>([]);
  const [polygonClosed, setPolygonClosed] = useState(false);

  useEffect(() => {
    fetch("./buildings.json")
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load data: ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  const filtered = useMemo(
    () => (data ? applyFilters(data.buildings, filters) : []),
    [data, filters],
  );

  const selectedByPolygon = useMemo(
    () => (polygonClosed ? filterByPolygon(filtered, drawVertices) : null),
    [filtered, drawVertices, polygonClosed],
  );

  const chartBuildings = selectedByPolygon ?? filtered;

  function handleToggleDraw() {
    if (drawing) {
      // Cancel drawing
      setDrawing(false);
      setDrawVertices([]);
      setPolygonClosed(false);
    } else {
      // Start drawing (clear previous)
      setDrawing(true);
      setDrawVertices([]);
      setPolygonClosed(false);
      setSelected(null);
    }
  }

  function handleClearSelection() {
    setDrawVertices([]);
    setPolygonClosed(false);
    setDrawing(false);
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center bg-slate-950 text-white font-sans">
        <div className="text-center">
          <h1 className="text-2xl font-display font-bold mb-2">
            Failed to load data
          </h1>
          <p className="text-slate-400">{error}</p>
          <p className="text-slate-500 mt-4 text-sm">
            Make sure buildings.json is in the same directory.
          </p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="h-full flex items-center justify-center bg-slate-950">
        <div className="text-center">
          <div className="inline-block w-8 h-8 border-2 border-cyan-500/20 border-t-cyan-500 rounded-full animate-spin mb-4" />
          <p className="text-sm text-slate-500 font-sans">
            Loading buildings...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full relative font-sans text-white">
      <MapView
        buildings={filtered}
        onSelect={setSelected}
        selected={selected}
        drawing={drawing}
        drawVertices={drawVertices}
        onAddVertex={(v) => setDrawVertices((prev) => [...prev, v])}
        onClosePolygon={() => {
          setDrawing(false);
          setPolygonClosed(true);
        }}
      />

      <Header stats={data.stats} />

      <Filters
        filters={filters}
        onChange={setFilters}
        total={data.buildings.length}
        filtered={filtered.length}
        drawing={drawing}
        onToggleDraw={handleToggleDraw}
      />

      <DistributionChart buildings={chartBuildings} />

      {polygonClosed && selectedByPolygon && (
        <SelectionStats
          buildings={selectedByPolygon}
          onClear={handleClearSelection}
        />
      )}

      {selected && (
        <BuildingPanel
          building={selected}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
