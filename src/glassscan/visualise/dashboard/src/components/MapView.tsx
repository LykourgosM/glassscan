import { MapContainer, TileLayer, CircleMarker, useMap } from "react-leaflet";
import { useEffect } from "react";
import L from "leaflet";
import type { Building } from "../types";
import Legend from "./Legend";

function wwrColor(wwr: number): string {
  if (wwr < 0.08) return "#06b6d4";
  if (wwr < 0.15) return "#10b981";
  if (wwr < 0.22) return "#22c55e";
  if (wwr < 0.30) return "#84cc16";
  if (wwr < 0.40) return "#eab308";
  if (wwr < 0.50) return "#f97316";
  return "#ef4444";
}

function FitBounds({ buildings }: { buildings: Building[] }) {
  const map = useMap();
  useEffect(() => {
    if (buildings.length === 0) return;
    const bounds = L.latLngBounds(
      buildings.map((b) => [b.lat, b.lon] as [number, number]),
    );
    map.fitBounds(bounds, { padding: [80, 80], maxZoom: 16 });
  }, [buildings, map]);
  return null;
}

interface Props {
  buildings: Building[];
  onSelect: (b: Building) => void;
  selected: Building | null;
}

export default function MapView({ buildings, onSelect, selected }: Props) {
  return (
    <MapContainer
      center={[47.37, 8.54]}
      zoom={14}
      className="h-full w-full"
      zoomControl={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://carto.com/">CARTO</a>'
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      />

      <FitBounds buildings={buildings} />

      {buildings.map((b) => {
        const color = wwrColor(b.wwr);
        const isMeasured = b.source === "measured";
        const isSelected = selected?.egid === b.egid;

        return (
          <CircleMarker
            key={b.egid}
            center={[b.lat, b.lon]}
            radius={isSelected ? 12 : 6}
            pathOptions={{
              color: isSelected ? "#ffffff" : color,
              fillColor: isMeasured ? color : "transparent",
              fillOpacity: isMeasured ? 0.8 : 0,
              weight: isSelected ? 2.5 : isMeasured ? 1.5 : 2,
              opacity: isSelected ? 1 : 0.85,
            }}
            eventHandlers={{ click: () => onSelect(b) }}
          />
        );
      })}

      <Legend />
    </MapContainer>
  );
}
