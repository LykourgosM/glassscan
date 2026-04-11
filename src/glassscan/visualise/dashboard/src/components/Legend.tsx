import { useMap } from "react-leaflet";
import { useEffect } from "react";
import L from "leaflet";

const COLORS = [
  { label: "< 15%", color: "#22c55e" },
  { label: "15-30%", color: "#eab308" },
  { label: "30-45%", color: "#f97316" },
  { label: "> 45%", color: "#ef4444" },
];

export default function Legend() {
  const map = useMap();

  useEffect(() => {
    const legend = new L.Control({ position: "bottomright" });

    legend.onAdd = () => {
      const div = L.DomUtil.create("div");
      div.style.cssText =
        "background:#1e293b; color:#e2e8f0; padding:10px 14px; border-radius:8px; font-size:12px; line-height:1.8;";

      div.innerHTML = `
        <div style="font-weight:600; margin-bottom:4px;">WWR</div>
        ${COLORS.map(
          ({ label, color }) =>
            `<div>
              <span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:${color}; margin-right:6px; vertical-align:middle;"></span>
              ${label}
            </div>`
        ).join("")}
        <div style="margin-top:6px; border-top:1px solid #334155; padding-top:6px;">
          <span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#3b82f6; margin-right:6px; vertical-align:middle;"></span>
          Measured
        </div>
        <div>
          <span style="display:inline-block; width:12px; height:12px; border-radius:50%; border:2px solid #3b82f6; margin-right:6px; vertical-align:middle;"></span>
          Predicted
        </div>
      `;
      return div;
    };

    legend.addTo(map);
    return () => {
      legend.remove();
    };
  }, [map]);

  return null;
}
