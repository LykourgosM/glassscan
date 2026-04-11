import { useEffect, useState } from "react";
import type { DashboardData, Building } from "./types";
import Map from "./components/Map";
import Sidebar from "./components/Sidebar";

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
      <div className="h-full flex items-center justify-center bg-slate-900 text-white">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-2">Failed to load data</h1>
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
      <div className="h-full flex items-center justify-center bg-slate-900 text-white">
        <p className="text-xl">Loading...</p>
      </div>
    );
  }

  return (
    <div className="h-full flex">
      <Sidebar stats={data.stats} selected={selected} />
      <div className="flex-1 relative">
        <Map
          buildings={data.buildings}
          onSelect={setSelected}
          selected={selected}
        />
      </div>
    </div>
  );
}
