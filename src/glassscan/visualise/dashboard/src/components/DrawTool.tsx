import { useMapEvents, Polygon, Polyline, CircleMarker } from "react-leaflet";
import type { Building } from "../types";

interface Props {
  active: boolean;
  vertices: [number, number][];
  onAddVertex: (latlng: [number, number]) => void;
  onClose: () => void;
}

function pointInPolygon(
  lat: number,
  lon: number,
  poly: [number, number][],
): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [lati, loni] = poly[i];
    const [latj, lonj] = poly[j];
    if (
      lati > lat !== latj > lat &&
      lon < ((lonj - loni) * (lat - lati)) / (latj - lati) + loni
    ) {
      inside = !inside;
    }
  }
  return inside;
}

export function filterByPolygon(
  buildings: Building[],
  poly: [number, number][],
): Building[] {
  if (poly.length < 3) return buildings;
  return buildings.filter((b) => pointInPolygon(b.lat, b.lon, poly));
}

export default function DrawTool({
  active,
  vertices,
  onAddVertex,
  onClose,
}: Props) {
  useMapEvents({
    click(e) {
      if (!active) return;

      // Close polygon if clicking near the first vertex
      if (vertices.length >= 3) {
        const first = vertices[0];
        const dist = Math.sqrt(
          (e.latlng.lat - first[0]) ** 2 + (e.latlng.lng - first[1]) ** 2,
        );
        if (dist < 0.0003) {
          onClose();
          return;
        }
      }

      onAddVertex([e.latlng.lat, e.latlng.lng]);
    },
  });

  if (vertices.length === 0) return null;

  const closed = !active && vertices.length >= 3;

  return (
    <>
      {closed ? (
        <Polygon
          positions={vertices}
          pathOptions={{
            color: "#06b6d4",
            fillColor: "#06b6d4",
            fillOpacity: 0.08,
            weight: 2,
            dashArray: "6 4",
          }}
        />
      ) : (
        <Polyline
          positions={vertices}
          pathOptions={{
            color: "#06b6d4",
            weight: 2,
            dashArray: "6 4",
          }}
        />
      )}

      {/* Vertex dots */}
      {vertices.map((v, i) => (
        <CircleMarker
          key={i}
          center={v}
          radius={i === 0 && active && vertices.length >= 3 ? 6 : 3}
          pathOptions={{
            color: "#06b6d4",
            fillColor: i === 0 && active ? "#06b6d4" : "#0f172a",
            fillOpacity: 1,
            weight: 1.5,
          }}
        />
      ))}
    </>
  );
}
