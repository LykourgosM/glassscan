import { useEffect, useState } from "react";
import type { DashboardData, Building } from "./types";
import MapView from "./components/MapView";
import Header from "./components/Header";
import BuildingPanel from "./components/BuildingPanel";
import DistributionChart from "./components/DistributionChart";

export default function App() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Building | null>(null);

  useEffect(() => {
    fetch("./buildings.json")
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load data: ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

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
        buildings={data.buildings}
        onSelect={setSelected}
        selected={selected}
      />

      <Header stats={data.stats} />

      <DistributionChart buildings={data.buildings} />

      {selected && (
        <BuildingPanel
          building={selected}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
