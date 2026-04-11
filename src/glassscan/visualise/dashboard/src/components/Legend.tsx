import { useMap } from "react-leaflet";
import { useEffect } from "react";
import L from "leaflet";

const STOPS = [
  { color: "#06b6d4", pct: 0 },
  { color: "#10b981", pct: 16 },
  { color: "#22c55e", pct: 28 },
  { color: "#84cc16", pct: 42 },
  { color: "#eab308", pct: 58 },
  { color: "#f97316", pct: 78 },
  { color: "#ef4444", pct: 100 },
];

export default function Legend() {
  const map = useMap();

  useEffect(() => {
    const ctrl = new L.Control({ position: "bottomright" });

    ctrl.onAdd = () => {
      const el = L.DomUtil.create("div");
      el.className = "glass";
      el.style.cssText =
        "border-radius:14px;padding:14px 18px;font-size:11px;color:#e2e8f0;min-width:150px;";

      const grad = STOPS.map((s) => `${s.color} ${s.pct}%`).join(",");

      el.innerHTML = `
        <div style="font-family:Syne,sans-serif;font-weight:700;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#64748b;margin-bottom:10px">
          Window-to-Wall
        </div>
        <div style="height:6px;border-radius:3px;background:linear-gradient(to right,${grad})"></div>
        <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:9px;color:#475569;margin-top:4px">
          <span>0%</span><span>25%</span><span>50%+</span>
        </div>
        <div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(148,163,184,0.06)">
          <div style="display:flex;align-items:center;gap:7px;margin-bottom:5px">
            <span style="width:9px;height:9px;border-radius:50%;background:#10b981;opacity:.8;display:inline-block"></span>
            <span style="font-size:10px;color:#94a3b8">Measured (CV)</span>
          </div>
          <div style="display:flex;align-items:center;gap:7px">
            <span style="width:9px;height:9px;border-radius:50%;border:1.5px solid #10b981;display:inline-block"></span>
            <span style="font-size:10px;color:#94a3b8">Predicted (ML)</span>
          </div>
        </div>
      `;
      return el;
    };

    ctrl.addTo(map);
    return () => void ctrl.remove();
  }, [map]);

  return null;
}
