"""Export pipeline results for the React dashboard.

Produces:
  output_dir/
    buildings.json          -- building coordinates, WWR, metadata, stats
    images/{egid}.jpg       -- building card images (original + overlay)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from glassscan.types import (
    BuildingImage,
    SegmentationResult,
    WWRResult,
    PredictionResult,
)

logger = logging.getLogger(__name__)

# Wall = blue, Window = green (BGR)
_WALL_COLOR = (255, 150, 50)
_WINDOW_COLOR = (0, 255, 0)


def create_building_card(
    image: np.ndarray,
    mask: np.ndarray,
    wwr: float,
) -> np.ndarray:
    """Create composite image: original | segmentation overlay.

    Args:
        image: BGR image (H x W x 3).
        mask: Segmentation mask (H x W), 0=background, 1=wall, 2=window.
        wwr: WWR value to display.

    Returns:
        Side-by-side composite image (H x 2W x 3).
    """
    overlay = image.copy()
    overlay[mask == 1] = _WALL_COLOR
    overlay[mask == 2] = _WINDOW_COLOR
    blended = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

    # Side by side: original | overlay
    card = np.hstack([image, blended])

    # WWR label at the bottom of the overlay side
    h, w = image.shape[:2]
    label = f"WWR: {wwr:.1%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = h / 500
    thickness = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(card, (w, h - th - 16), (w + tw + 16, h), (0, 0, 0), -1)
    cv2.putText(card, label, (w + 8, h - 8), font, scale, (255, 255, 255), thickness)

    return card


def _create_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend raw image with colored segmentation mask (50% opacity)."""
    overlay = image.copy()
    overlay[mask == 1] = _WALL_COLOR
    overlay[mask == 2] = _WINDOW_COLOR
    return cv2.addWeighted(image, 0.5, overlay, 0.5, 0)


def export_results(
    result: object,
    output_dir: Path | str,
    metadata_df: pd.DataFrame | None = None,
    per_view_wwr: list | None = None,
    weights_file: Path | str | None = None,
) -> None:
    """Export pipeline results for the dashboard.

    Args:
        result: PipelineResult from the pipeline module.
        output_dir: Directory to write buildings.json and images/.
        metadata_df: Optional building metadata with 'egid' column
            plus feature columns. Used to attach metadata to buildings
            and to get coordinates for predicted buildings (needs
            'lat' and 'lon' columns).
        per_view_wwr: Pre-aggregation WWR results (one per image, not
            per building). When provided, per-view data and overlay
            images are exported for multi-view display in the dashboard.
        weights_file: Path to weights.json (LLM-scored view weights).
            If provided, per-view weights in buildings.json use these
            instead of the default 1.0/0.5 scheme.
    """
    from glassscan.pipeline import PipelineResult

    if not isinstance(result, PipelineResult):
        raise TypeError(f"Expected PipelineResult, got {type(result)}")

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Build lookup dicts from pipeline stages (lists, not single items)
    from collections import defaultdict

    images_by_egid: defaultdict[str, list[BuildingImage]] = defaultdict(list)
    for img in result.images:
        images_by_egid[img.egid].append(img)

    segs_by_egid: defaultdict[str, list[SegmentationResult]] = defaultdict(list)
    for seg in result.segmentations:
        segs_by_egid[seg.egid].append(seg)

    rects_by_egid: defaultdict[str, list] = defaultdict(list)
    for rect in result.rectified:
        rects_by_egid[rect.egid].append(rect)

    # Per-view WWR lookup
    pv_wwr_by_egid: defaultdict[str, list] = defaultdict(list)
    if per_view_wwr is not None:
        for w in per_view_wwr:
            pv_wwr_by_egid[w.egid].append(w)

    # Load LLM weights if available
    file_weights: dict[str, list[float]] = {}
    if weights_file is not None:
        from glassscan.wwr import load_weights
        file_weights = load_weights(weights_file)

    metadata_by_egid: dict[str, dict] = {}
    if metadata_df is not None and "egid" in metadata_df.columns:
        for _, row in metadata_df.iterrows():
            egid = row["egid"]
            metadata_by_egid[egid] = {
                k: _to_json_value(v)
                for k, v in row.items()
                if k not in ("egid", "lat", "lon")
            }

    # Generate per-view images if multi-view data available
    if per_view_wwr is not None:
        raw_dir = output_dir / "raw"
        overlays_dir = output_dir / "overlays"
        rect_overlays_dir = output_dir / "rectified_overlays"
        rectified_dir = output_dir / "rectified"
        for d in (raw_dir, overlays_dir, rect_overlays_dir, rectified_dir):
            d.mkdir(parents=True, exist_ok=True)

        for egid, imgs in images_by_egid.items():
            segs = segs_by_egid.get(egid, [])
            rects = rects_by_egid.get(egid, [])
            for i, img in enumerate(imgs):
                suffix = f"_v{i}" if i > 0 else ""
                # Raw image
                cv2.imwrite(str(raw_dir / f"{egid}{suffix}.jpg"), img.image)
                # Overlay (raw + segmentation)
                if i < len(segs):
                    ov = _create_overlay(img.image, segs[i].mask)
                    cv2.imwrite(str(overlays_dir / f"{egid}{suffix}.jpg"), ov)
                # Rectified image + overlay
                if i < len(rects):
                    cv2.imwrite(
                        str(rectified_dir / f"{egid}{suffix}_rectified.jpg"),
                        rects[i].rectified_image,
                    )
                    rov = _create_overlay(
                        rects[i].rectified_image, rects[i].rectified_mask,
                    )
                    cv2.imwrite(
                        str(rect_overlays_dir / f"{egid}{suffix}.jpg"), rov,
                    )

    buildings: list[dict] = []

    # Measured buildings (from CV pipeline)
    for wwr_result in result.wwr_results:
        egid = wwr_result.egid
        imgs = images_by_egid.get(egid, [])
        segs = segs_by_egid.get(egid, [])
        img = imgs[0] if imgs else None
        seg = segs[0] if segs else None

        entry: dict = {
            "egid": egid,
            "lat": img.lat if img else 0.0,
            "lon": img.lon if img else 0.0,
            "wwr": round(wwr_result.wwr, 4),
            "source": "measured",
            "confidence": round(wwr_result.confidence, 3),
            "n_windows": wwr_result.n_windows,
            "prediction_interval": None,
            "metadata": metadata_by_egid.get(egid, {}),
        }

        # Per-view data
        pv = pv_wwr_by_egid.get(egid, [])
        if len(pv) > 1:
            fw = file_weights.get(egid, [])
            views = []
            for vi, pw in enumerate(pv):
                if fw and vi < len(fw):
                    w = fw[vi]
                elif vi == 0:
                    w = 1.0
                else:
                    w = 0.5
                views.append({
                    "wwr": round(pw.wwr, 4),
                    "weight": round(w, 2),
                    "n_windows": pw.n_windows,
                    "confidence": round(pw.confidence, 3),
                })
            entry["views"] = views
        else:
            entry["views"] = None

        buildings.append(entry)

        # Generate building card image (primary view)
        if img is not None and seg is not None:
            card = create_building_card(img.image, seg.mask, wwr_result.wwr)
            cv2.imwrite(str(images_dir / f"{egid}.jpg"), card)

    # Predicted buildings (from prediction pipeline)
    for pred in result.predictions:
        egid = pred.egid
        lat, lon = _get_coords(egid, metadata_df)

        buildings.append({
            "egid": egid,
            "lat": lat,
            "lon": lon,
            "wwr": round(pred.predicted_wwr, 4),
            "source": "predicted",
            "confidence": None,
            "n_windows": None,
            "prediction_interval": [
                round(pred.prediction_interval[0], 4),
                round(pred.prediction_interval[1], 4),
            ],
            "metadata": metadata_by_egid.get(egid, {}),
        })

    # Stats
    stats = _compute_stats(
        result.wwr_results, result.predictions, result.model, metadata_df,
    )

    with open(output_dir / "buildings.json", "w") as f:
        json.dump({"buildings": buildings, "stats": stats}, f, indent=2)

    logger.info(
        "Exported %d buildings (%d measured, %d predicted) to %s",
        len(buildings), len(result.wwr_results), len(result.predictions), output_dir,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_coords(egid: str, metadata_df: pd.DataFrame | None) -> tuple[float, float]:
    """Get lat/lon for a building from metadata."""
    if metadata_df is not None and "egid" in metadata_df.columns:
        row = metadata_df[metadata_df["egid"] == egid]
        if len(row) > 0 and "lat" in row.columns and "lon" in row.columns:
            return float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])
    return 0.0, 0.0


def _to_json_value(v: object) -> object:
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if pd.isna(v):
        return None
    return v


def _compute_stats(
    wwr_results: list[WWRResult],
    predictions: list[PredictionResult],
    model: object | None,
    metadata_df: pd.DataFrame | None,
) -> dict:
    """Compute summary statistics for the dashboard sidebar."""
    all_wwrs = [r.wwr for r in wwr_results] + [p.predicted_wwr for p in predictions]

    stats: dict = {
        "total": len(all_wwrs),
        "measured": len(wwr_results),
        "predicted": len(predictions),
        "mean_wwr": round(float(np.mean(all_wwrs)), 4) if all_wwrs else 0,
        "wwr_by_era": {},
        "wwr_by_type": {},
        "feature_importance": {},
    }

    if metadata_df is not None and "egid" in metadata_df.columns:
        wwr_by_egid = {r.egid: r.wwr for r in wwr_results}
        wwr_by_egid.update({p.egid: p.predicted_wwr for p in predictions})

        merged = metadata_df[metadata_df["egid"].isin(wwr_by_egid)].copy()
        merged["wwr"] = merged["egid"].map(wwr_by_egid)

        if "construction_year" in merged.columns:
            stats["wwr_by_era"] = _wwr_by_era(merged)

        if "building_category" in merged.columns:
            stats["wwr_by_type"] = {
                str(k): round(float(v), 4)
                for k, v in merged.groupby("building_category")["wwr"].mean().items()
            }

    if model is not None and hasattr(model, "metrics"):
        fi = model.metrics.get("feature_importance", {})
        if fi:
            stats["feature_importance"] = {
                k: round(float(v), 4) for k, v in fi.items()
            }

    return stats


def _wwr_by_era(df: pd.DataFrame) -> dict[str, float]:
    """Bin construction years into eras and compute mean WWR."""
    bins = [0, 1950, 1980, 2000, 9999]
    labels = ["pre-1950", "1950-1980", "1980-2000", "post-2000"]
    df = df.copy()
    df["era"] = pd.cut(df["construction_year"], bins=bins, labels=labels)
    return {
        era: round(float(subset.mean()), 4)
        for era in labels
        if len(subset := df[df["era"] == era]["wwr"]) > 0
    }
