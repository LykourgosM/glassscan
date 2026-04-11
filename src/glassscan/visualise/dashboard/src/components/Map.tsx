import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import type { Building } from "../types";
import BuildingPopup from "./BuildingPopup";
import Legend from "./Legend";

function wwrColor(wwr: number): string {
  if (wwr < 0.15) return "#22c55e";
  if (wwr < 0.3) return "#eab308";
  if (wwr < 0.45) return "#f97316";
  return "#ef4444";
}

interface Props {
  buildings: Building[];
  onSelect: (b: Building | null) => void;
  selected: Building | null;
}

export default function Map({ buildings, onSelect, selected }: Props) {
  return (
    <MapContainer
      center={[46.8, 8.2]}
      zoom={8}
      className="h-full w-full"
      zoomControl={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://carto.com/">CARTO</a>'
        url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
      />

      {buildings.map((b) => {
        const color = wwrColor(b.wwr);
        const isMeasured = b.source === "measured";
        const isSelected = selected?.egid === b.egid;

        return (
          <CircleMarker
            key={b.egid}
            center={[b.lat, b.lon]}
            radius={isSelected ? 10 : 7}
            pathOptions={{
              color: color,
              fillColor: isMeasured ? color : "transparent",
              fillOpacity: isMeasured ? 0.7 : 0,
              weight: isSelected ? 3 : 2,
            }}
            eventHandlers={{
              click: () => onSelect(b),
            }}
          >
            <Popup>
              <BuildingPopup building={b} />
            </Popup>
          </CircleMarker>
        );
      })}

      <Legend />
    </MapContainer>
  );
}
