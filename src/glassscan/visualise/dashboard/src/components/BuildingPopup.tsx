import type { Building } from "../types";

interface Props {
  building: Building;
}

export default function BuildingPopup({ building: b }: Props) {
  const pct = (b.wwr * 100).toFixed(1);

  return (
    <div className="min-w-[220px] text-sm">
      <div className="font-bold text-base mb-1">
        WWR: {pct}%
      </div>

      <table className="w-full text-xs">
        <tbody>
          <tr>
            <td className="text-slate-500 pr-2">EGID</td>
            <td>{b.egid}</td>
          </tr>
          <tr>
            <td className="text-slate-500 pr-2">Source</td>
            <td>
              <span
                className={`inline-block px-1.5 py-0.5 rounded text-white text-[10px] ${
                  b.source === "measured" ? "bg-blue-600" : "bg-purple-600"
                }`}
              >
                {b.source}
              </span>
            </td>
          </tr>

          {b.source === "measured" && b.confidence != null && (
            <tr>
              <td className="text-slate-500 pr-2">Confidence</td>
              <td>{(b.confidence * 100).toFixed(0)}%</td>
            </tr>
          )}
          {b.source === "measured" && b.n_windows != null && (
            <tr>
              <td className="text-slate-500 pr-2">Windows</td>
              <td>{b.n_windows}</td>
            </tr>
          )}
          {b.source === "predicted" && b.prediction_interval && (
            <tr>
              <td className="text-slate-500 pr-2">90% CI</td>
              <td>
                {(b.prediction_interval[0] * 100).toFixed(1)}% --{" "}
                {(b.prediction_interval[1] * 100).toFixed(1)}%
              </td>
            </tr>
          )}

          {Object.entries(b.metadata).map(([key, val]) => (
            <tr key={key}>
              <td className="text-slate-500 pr-2">{key.replace(/_/g, " ")}</td>
              <td>{val}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {b.source === "measured" && (
        <img
          src={`./images/${b.egid}.jpg`}
          alt={`Building ${b.egid}`}
          className="mt-2 w-full rounded"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
      )}
    </div>
  );
}
