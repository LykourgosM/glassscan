"""Tests for the pipeline orchestrator.

Prediction pipeline tests use real XGBoost (fast with synthetic data).
CV pipeline mock tests are in test_pipeline_cv.py (separate file to
avoid the PyTorch + XGBoost libomp conflict on macOS).

Full pipeline tests mock run_cv_pipeline() and use real prediction,
so they also avoid importing the segment module.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from glassscan.types import WWRResult
from glassscan.pipeline import (
    PipelineResult,
    run_prediction_pipeline,
    run_full_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_wwr(egid: str, wwr: float = 0.25) -> WWRResult:
    return WWRResult(
        egid=egid,
        wwr=wwr,
        window_area_px=400,
        wall_area_px=1200,
        n_windows=2,
        confidence=0.85,
    )


def _make_metadata(egids: list[str], seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(egids)
    return pd.DataFrame({
        "egid": egids,
        "construction_year": rng.integers(1900, 2024, size=n),
        "floor_count": rng.integers(1, 8, size=n),
        "building_category": rng.choice(["residential", "commercial"], size=n),
    })


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

class TestPipelineResult:
    def test_defaults_empty(self):
        r = PipelineResult()
        assert r.images == []
        assert r.segmentations == []
        assert r.rectified == []
        assert r.wwr_results == []
        assert r.model is None
        assert r.predictions == []

    def test_fields_assignable(self):
        wwr = [_fake_wwr("E0")]
        r = PipelineResult(wwr_results=wwr)
        assert r.wwr_results == wwr


# ---------------------------------------------------------------------------
# run_prediction_pipeline
# ---------------------------------------------------------------------------

class TestRunPredictionPipeline:
    def test_trains_and_predicts(self):
        egids = [f"E{i}" for i in range(50)]
        wwr_results = [_fake_wwr(e, wwr=0.15 + i * 0.01) for i, e in enumerate(egids)]
        metadata = _make_metadata(egids)

        new_egids = [f"N{i}" for i in range(10)]
        predict_df = _make_metadata(new_egids, seed=99)

        result = run_prediction_pipeline(
            wwr_results, metadata, predict_df, cv_folds=2,
        )

        assert result.model is not None
        assert len(result.predictions) == 10
        for p in result.predictions:
            assert 0.0 <= p.predicted_wwr <= 1.0

    def test_egid_join_filters_correctly(self):
        wwr_results = [_fake_wwr(f"E{i}") for i in range(5)]
        metadata = _make_metadata(["E0", "E1", "E2"])

        result = run_prediction_pipeline(wwr_results, metadata, cv_folds=2)

        assert result.model is not None
        assert result.model.metrics["n_train"] == 3

    def test_no_match_returns_empty(self):
        wwr_results = [_fake_wwr("E0")]
        metadata = _make_metadata(["X1", "X2"])

        result = run_prediction_pipeline(wwr_results, metadata)

        assert result.model is None
        assert result.predictions == []

    def test_empty_wwr_results(self):
        metadata = _make_metadata(["E0"])
        result = run_prediction_pipeline([], metadata)
        assert result.model is None

    def test_missing_egid_column_raises(self):
        wwr_results = [_fake_wwr("E0")]
        bad_df = pd.DataFrame({"year": [2000]})
        with pytest.raises(ValueError, match="egid"):
            run_prediction_pipeline(wwr_results, bad_df)

    def test_feature_columns_selection(self):
        egids = [f"E{i}" for i in range(30)]
        wwr_results = [_fake_wwr(e, wwr=0.2 + i * 0.01) for i, e in enumerate(egids)]
        metadata = _make_metadata(egids)

        result = run_prediction_pipeline(
            wwr_results, metadata,
            feature_columns=["construction_year"],
            cv_folds=2,
        )

        assert result.model is not None
        assert result.model.feature_names == ["construction_year"]

    def test_predict_egids_from_column(self):
        egids = [f"E{i}" for i in range(30)]
        wwr_results = [_fake_wwr(e, wwr=0.2 + i * 0.01) for i, e in enumerate(egids)]
        metadata = _make_metadata(egids)

        predict_df = _make_metadata(["N0", "N1"], seed=99)
        result = run_prediction_pipeline(
            wwr_results, metadata, predict_df, cv_folds=2,
        )

        assert result.predictions[0].egid == "N0"
        assert result.predictions[1].egid == "N1"

    def test_predict_egids_explicit(self):
        egids = [f"E{i}" for i in range(30)]
        wwr_results = [_fake_wwr(e, wwr=0.2 + i * 0.01) for i, e in enumerate(egids)]
        metadata = _make_metadata(egids)

        predict_df = _make_metadata(["N0"], seed=99)
        result = run_prediction_pipeline(
            wwr_results, metadata, predict_df,
            predict_egids=["CUSTOM_ID"],
            cv_folds=2,
        )

        assert result.predictions[0].egid == "CUSTOM_ID"

    def test_no_predict_df_trains_only(self):
        egids = [f"E{i}" for i in range(30)]
        wwr_results = [_fake_wwr(e) for e in egids]
        metadata = _make_metadata(egids)

        result = run_prediction_pipeline(wwr_results, metadata, cv_folds=2)

        assert result.model is not None
        assert result.predictions == []

    def test_saves_model(self, tmp_path):
        egids = [f"E{i}" for i in range(30)]
        wwr_results = [_fake_wwr(e) for e in egids]
        metadata = _make_metadata(egids)

        path = tmp_path / "model.joblib"
        run_prediction_pipeline(
            wwr_results, metadata, model_path=path, cv_folds=2,
        )

        assert path.exists()


# ---------------------------------------------------------------------------
# run_full_pipeline
# ---------------------------------------------------------------------------

class TestRunFullPipeline:
    @patch("glassscan.pipeline.run_cv_pipeline")
    def test_chains_cv_and_prediction(self, mock_cv):
        egids = [f"E{i}" for i in range(30)]
        mock_cv.return_value = PipelineResult(
            wwr_results=[
                _fake_wwr(e, wwr=0.15 + i * 0.01) for i, e in enumerate(egids)
            ],
        )

        metadata = _make_metadata(egids)
        predict_df = _make_metadata([f"N{i}" for i in range(5)], seed=99)

        result = run_full_pipeline(
            [{"egid": e, "lat": 47.0, "lon": 8.5} for e in egids],
            metadata,
            "fake-key",
            predict_df=predict_df,
            cv_folds=2,
        )

        mock_cv.assert_called_once()
        assert len(result.wwr_results) == 30
        assert result.model is not None
        assert len(result.predictions) == 5

    @patch("glassscan.pipeline.run_cv_pipeline")
    def test_skips_prediction_when_cv_empty(self, mock_cv):
        mock_cv.return_value = PipelineResult()
        metadata = _make_metadata(["E0"])

        result = run_full_pipeline(
            [{"egid": "E0", "lat": 47.0, "lon": 8.5}],
            metadata,
            "fake-key",
        )

        assert result.model is None
        assert result.predictions == []
