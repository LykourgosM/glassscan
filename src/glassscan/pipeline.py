"""End-to-end GlassScan pipeline orchestrator.

Chains: fetch → segment → rectify → wwr → predict

Two entry points:
  - run_cv_pipeline(): image-based WWR measurement
  - run_prediction_pipeline(): metadata-based WWR prediction
  - run_full_pipeline(): both in sequence

At the hackathon, the main rewiring will be in the callers of these
functions (data loading, feature selection, EGID join logic), not in
the pipeline itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from glassscan.types import (
    BuildingImage,
    SegmentationResult,
    RectifiedResult,
    WWRResult,
    PredictionResult,
)

# Lazy imports: PyTorch (segment) and XGBoost (predict) ship conflicting
# libomp builds on macOS. Importing both at module level causes a segfault.
# Each function imports only the modules it needs at call time.
if TYPE_CHECKING:
    from glassscan.predict import WWRModel

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """All outputs from a pipeline run.

    Keeps intermediate results for debugging and visualisation.
    """

    images: list[BuildingImage] = field(default_factory=list)
    segmentations: list[SegmentationResult] = field(default_factory=list)
    rectified: list[RectifiedResult] = field(default_factory=list)
    wwr_results: list[WWRResult] = field(default_factory=list)
    model: WWRModel | None = None
    predictions: list[PredictionResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CV pipeline: fetch → segment → rectify → wwr
# ---------------------------------------------------------------------------


def run_cv_pipeline(
    buildings: list[dict],
    api_key: str,
    *,
    save_dir: Path | None = None,
    max_views: int = 1,
    max_api_calls: int = 1000,
) -> PipelineResult:
    """Measure WWR from Street View imagery.

    Args:
        buildings: List of dicts with keys 'egid', 'lat', 'lon'.
        api_key: Google Street View Static API key.
        save_dir: Optional root directory for intermediate outputs.
            Creates subdirs raw/, masks/, rectified/.
        max_views: Street View angles per building (1-4).
        max_api_calls: API budget cap across all buildings.

    Returns:
        PipelineResult with images, segmentations, rectified, and wwr_results.
    """
    from glassscan.fetch import fetch_batch
    from glassscan.segment import segment_batch
    from glassscan.rectify import rectify_batch
    from glassscan.wwr import compute_wwr_batch

    result = PipelineResult()

    if not buildings:
        logger.warning("No buildings provided")
        return result

    # Normalise egids to strings
    buildings = [{**b, "egid": str(b["egid"])} for b in buildings]

    # 1. Fetch
    logger.info("Fetching images for %d buildings...", len(buildings))
    result.images = fetch_batch(
        buildings,
        api_key,
        save_dir=save_dir / "raw" if save_dir else None,
        max_calls=max_api_calls,
        max_views=max_views,
    )
    logger.info("Fetched %d images", len(result.images))

    if not result.images:
        logger.warning("No images fetched -- stopping")
        return result

    # 2. Segment (auto-loads models on first call)
    logger.info("Segmenting %d images...", len(result.images))
    result.segmentations = segment_batch(
        result.images,
        save_dir=save_dir / "masks" if save_dir else None,
    )
    logger.info("Segmented %d images", len(result.segmentations))

    if not result.segmentations:
        logger.warning("No segmentations produced -- stopping")
        return result

    # 3. Rectify
    logger.info("Rectifying %d facades...", len(result.segmentations))
    result.rectified = rectify_batch(
        result.segmentations,
        save_dir=save_dir / "rectified" if save_dir else None,
    )
    logger.info("Rectified %d facades", len(result.rectified))

    if not result.rectified:
        logger.warning("No rectified results -- stopping")
        return result

    # 4. WWR
    logger.info("Computing WWR for %d facades...", len(result.rectified))
    result.wwr_results = compute_wwr_batch(result.rectified)
    logger.info(
        "Computed WWR for %d buildings (mean=%.3f)",
        len(result.wwr_results),
        np.mean([r.wwr for r in result.wwr_results]) if result.wwr_results else 0,
    )

    return result


# ---------------------------------------------------------------------------
# Prediction pipeline: join + train + predict
# ---------------------------------------------------------------------------


def run_prediction_pipeline(
    wwr_results: list[WWRResult],
    metadata_df: pd.DataFrame,
    predict_df: pd.DataFrame | None = None,
    *,
    predict_egids: list[str] | None = None,
    feature_columns: list[str] | None = None,
    model_path: Path | None = None,
    cv_folds: int = 5,
) -> PipelineResult:
    """Train on CV results + metadata, predict WWR for new buildings.

    Joins wwr_results with metadata_df on the 'egid' column to build
    training pairs (features, wwr_target).

    Args:
        wwr_results: WWR measurements from the CV pipeline.
        metadata_df: Building metadata. Must have an 'egid' column
            plus feature columns.
        predict_df: Buildings to predict for (same feature columns).
            If None, only trains the model.
        predict_egids: EGIDs for prediction buildings.  If None and
            predict_df has an 'egid' column, uses that.
        feature_columns: Columns to use as features.  If None, uses
            all columns except 'egid'.
        model_path: Save trained model here (joblib).
        cv_folds: Cross-validation folds for evaluation.

    Returns:
        PipelineResult with wwr_results, model, and predictions.
    """
    from glassscan.predict import train_model, predict_wwr as _predict_wwr, save_model

    result = PipelineResult(wwr_results=wwr_results)

    if not wwr_results:
        logger.warning("No WWR results to train on")
        return result

    if "egid" not in metadata_df.columns:
        raise ValueError("metadata_df must have an 'egid' column")

    # Join: match CV results to metadata by EGID
    wwr_by_egid = {r.egid: r.wwr for r in wwr_results}
    mask = metadata_df["egid"].isin(wwr_by_egid)
    train_meta = metadata_df[mask].copy()
    targets = np.array([wwr_by_egid[e] for e in train_meta["egid"]])

    logger.info(
        "Matched %d / %d CV results with metadata for training",
        len(train_meta),
        len(wwr_results),
    )

    if len(train_meta) == 0:
        logger.warning("No training data after EGID join -- skipping prediction")
        return result

    # Select features
    if feature_columns is None:
        feature_columns = [c for c in train_meta.columns if c != "egid"]
    train_features = train_meta[feature_columns]

    # Train
    logger.info("Training model on %d samples, %d features...", len(train_features), len(feature_columns))
    model = train_model(train_features, targets, cv_folds=cv_folds)
    result.model = model

    if model_path:
        save_model(model, model_path)
        logger.info("Saved model to %s", model_path)

    # Predict (optional)
    if predict_df is not None and len(predict_df) > 0:
        pred_features = predict_df[feature_columns]

        if predict_egids is None and "egid" in predict_df.columns:
            predict_egids = predict_df["egid"].tolist()

        result.predictions = _predict_wwr(model, pred_features, egids=predict_egids)
        logger.info("Predicted WWR for %d buildings", len(result.predictions))

    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_full_pipeline(
    buildings: list[dict],
    metadata_df: pd.DataFrame,
    api_key: str,
    *,
    predict_df: pd.DataFrame | None = None,
    predict_egids: list[str] | None = None,
    feature_columns: list[str] | None = None,
    save_dir: Path | None = None,
    model_path: Path | None = None,
    max_views: int = 1,
    max_api_calls: int = 1000,
    cv_folds: int = 5,
) -> PipelineResult:
    """Run the complete pipeline: CV measurement + metadata prediction.

    Args:
        buildings: Buildings with Street View coverage.
            List of dicts with 'egid', 'lat', 'lon'.
        metadata_df: Building metadata with 'egid' + feature columns.
            Must cover both the CV buildings and prediction buildings.
        api_key: Google Street View Static API key.
        predict_df: Buildings without imagery to predict for.
        predict_egids: EGIDs for prediction buildings.
        feature_columns: Feature columns to use.  If None, auto-detected.
        save_dir: Root directory for intermediate outputs.
        model_path: Path to save the trained model.
        max_views: Street View angles per building.
        max_api_calls: API budget cap.
        cv_folds: Cross-validation folds.

    Returns:
        PipelineResult with all outputs from both phases.
    """
    logger.info("=== GlassScan full pipeline ===")

    # Phase 1: CV pipeline
    logger.info("--- Phase 1: CV pipeline (%d buildings) ---", len(buildings))
    result = run_cv_pipeline(
        buildings,
        api_key,
        save_dir=save_dir,
        max_views=max_views,
        max_api_calls=max_api_calls,
    )

    if not result.wwr_results:
        logger.warning("CV pipeline produced no results -- skipping prediction")
        return result

    # Phase 2: Prediction pipeline
    logger.info("--- Phase 2: Prediction pipeline ---")
    pred_result = run_prediction_pipeline(
        result.wwr_results,
        metadata_df,
        predict_df,
        predict_egids=predict_egids,
        feature_columns=feature_columns,
        model_path=model_path,
        cv_folds=cv_folds,
    )

    result.model = pred_result.model
    result.predictions = pred_result.predictions

    logger.info(
        "=== Pipeline complete: %d measured, %d predicted ===",
        len(result.wwr_results),
        len(result.predictions),
    )
    return result
