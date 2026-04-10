"""Tests for the predict module.

Uses synthetic data with known feature-target relationships so we can
verify the model learns meaningful patterns without real GWR data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glassscan.predict.predict import (
    WWRModel,
    _detect_feature_types,
    _build_preprocessor,
    train_model,
    predict_wwr,
    save_model,
    load_model,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _make_synthetic_data(
    n: int = 200,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate building features with a known WWR relationship.

    Rules baked in:
      - newer buildings -> higher WWR
      - commercial > residential
      - more floors -> slightly higher
    """
    rng = np.random.default_rng(seed)

    years = rng.integers(1900, 2024, size=n)
    floors = rng.integers(1, 8, size=n)
    categories = rng.choice(["residential", "commercial", "industrial"], size=n)
    cantons = rng.choice(["ZH", "BE", "VD", "GE", "BS"], size=n)
    lats = 46.5 + rng.normal(0, 0.5, size=n)
    lons = 7.5 + rng.normal(0, 0.5, size=n)

    # Deterministic relationship for testability
    base = 0.15
    year_effect = (years - 1900) / 124 * 0.15
    cat_map = {"residential": 0.0, "commercial": 0.10, "industrial": -0.05}
    cat_effect = np.array([cat_map[c] for c in categories])
    floor_effect = floors * 0.01
    noise = rng.normal(0, 0.03, size=n)

    targets = np.clip(base + year_effect + cat_effect + floor_effect + noise, 0.05, 0.95)

    df = pd.DataFrame({
        "construction_year": years,
        "floor_count": floors,
        "building_category": categories,
        "canton": cantons,
        "lat": lats,
        "lon": lons,
    })
    return df, targets


# Shared fixtures
@pytest.fixture(scope="module")
def synthetic_data():
    return _make_synthetic_data()


@pytest.fixture(scope="module")
def trained(synthetic_data):
    df, targets = synthetic_data
    return train_model(df, targets, cv_folds=3)


# ---------------------------------------------------------------------------
# _detect_feature_types
# ---------------------------------------------------------------------------

class TestDetectFeatureTypes:
    def test_mixed_df(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.5, 2.5]})
        numeric, categorical = _detect_feature_types(df)
        assert set(numeric) == {"a", "c"}
        assert categorical == ["b"]

    def test_all_numeric(self):
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        numeric, categorical = _detect_feature_types(df)
        assert numeric == ["x", "y"]
        assert categorical == []

    def test_all_categorical(self):
        df = pd.DataFrame({"a": ["x"], "b": ["y"]})
        numeric, categorical = _detect_feature_types(df)
        assert numeric == []
        assert set(categorical) == {"a", "b"}


# ---------------------------------------------------------------------------
# _build_preprocessor
# ---------------------------------------------------------------------------

class TestBuildPreprocessor:
    def test_handles_missing_numeric(self):
        pp = _build_preprocessor(["x"], [])
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        out = pp.fit_transform(df)
        assert not np.isnan(out).any()

    def test_handles_missing_categorical(self):
        pp = _build_preprocessor([], ["c"])
        df = pd.DataFrame({"c": ["a", None, "b"]})
        out = pp.fit_transform(df)
        assert out.shape == (3, 1)

    def test_encodes_unseen_category(self):
        pp = _build_preprocessor([], ["c"])
        df_train = pd.DataFrame({"c": ["a", "b"]})
        pp.fit(df_train)
        df_test = pd.DataFrame({"c": ["a", "UNSEEN"]})
        out = pp.transform(df_test)
        assert out[1, 0] == -1  # unknown_value


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    def test_returns_wwr_model(self, trained):
        assert isinstance(trained, WWRModel)

    def test_feature_names_populated(self, trained):
        assert len(trained.feature_names) > 0
        assert "construction_year" in trained.feature_names

    def test_metrics_present(self, trained):
        assert "n_train" in trained.metrics
        assert trained.metrics["n_train"] == 200
        assert "target_mean" in trained.metrics
        assert "target_std" in trained.metrics

    def test_cv_metrics_present(self, trained):
        assert "cv_mae_mean" in trained.metrics
        assert "cv_r2_mean" in trained.metrics
        assert trained.metrics["cv_mae_mean"] > 0

    def test_feature_importance_present(self, trained):
        fi = trained.metrics["feature_importance"]
        assert isinstance(fi, dict)
        assert len(fi) == len(trained.feature_names)

    def test_skips_cv_when_too_few_samples(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        targets = [0.2, 0.3, 0.4]
        model = train_model(df, targets, cv_folds=5)
        assert "cv_mae_mean" not in model.metrics

    def test_rejects_mismatched_lengths(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="same length"):
            train_model(df, [0.5])

    def test_rejects_empty_data(self):
        df = pd.DataFrame({"x": []})
        with pytest.raises(ValueError, match="empty"):
            train_model(df, [])

    def test_numeric_only_features(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        model = train_model(df, [0.1, 0.2, 0.3, 0.4, 0.5], cv_folds=2)
        assert model.categorical_features == []

    def test_categorical_only_features(self):
        df = pd.DataFrame({"c": ["a", "b", "a", "b", "a"]})
        model = train_model(df, [0.2, 0.4, 0.2, 0.4, 0.2], cv_folds=2)
        assert model.numeric_features == []


# ---------------------------------------------------------------------------
# predict_wwr
# ---------------------------------------------------------------------------

class TestPredictWWR:
    def test_predictions_in_range(self, trained, synthetic_data):
        df, _ = synthetic_data
        results = predict_wwr(trained, df.head(10))
        for r in results:
            assert 0.0 <= r.predicted_wwr <= 1.0

    def test_prediction_interval_valid(self, trained, synthetic_data):
        df, _ = synthetic_data
        results = predict_wwr(trained, df.head(10))
        for r in results:
            lo, hi = r.prediction_interval
            assert lo <= r.predicted_wwr <= hi
            assert 0.0 <= lo <= 1.0
            assert 0.0 <= hi <= 1.0

    def test_egids_from_list(self, trained):
        df = pd.DataFrame({"construction_year": [2000], "floor_count": [3],
                           "building_category": ["residential"], "canton": ["ZH"],
                           "lat": [47.0], "lon": [8.5]})
        results = predict_wwr(trained, df, egids=["EGID_42"])
        assert results[0].egid == "EGID_42"

    def test_egids_default_to_index(self, trained):
        df = pd.DataFrame({"construction_year": [2000], "floor_count": [3],
                           "building_category": ["residential"], "canton": ["ZH"],
                           "lat": [47.0], "lon": [8.5]})
        results = predict_wwr(trained, df)
        assert results[0].egid == "0"

    def test_features_used_populated(self, trained, synthetic_data):
        df, _ = synthetic_data
        results = predict_wwr(trained, df.head(1))
        assert results[0].features_used == trained.feature_names

    def test_empty_df(self, trained):
        results = predict_wwr(trained, pd.DataFrame())
        assert results == []


# ---------------------------------------------------------------------------
# Model learns directional patterns
# ---------------------------------------------------------------------------

class TestModelLearns:
    def test_newer_buildings_higher_wwr(self, trained):
        old = pd.DataFrame({"construction_year": [1920], "floor_count": [3],
                            "building_category": ["residential"], "canton": ["ZH"],
                            "lat": [47.0], "lon": [8.5]})
        new = pd.DataFrame({"construction_year": [2020], "floor_count": [3],
                            "building_category": ["residential"], "canton": ["ZH"],
                            "lat": [47.0], "lon": [8.5]})
        r_old = predict_wwr(trained, old)[0]
        r_new = predict_wwr(trained, new)[0]
        assert r_new.predicted_wwr > r_old.predicted_wwr

    def test_commercial_higher_than_residential(self, trained):
        res = pd.DataFrame({"construction_year": [1980], "floor_count": [3],
                            "building_category": ["residential"], "canton": ["ZH"],
                            "lat": [47.0], "lon": [8.5]})
        com = pd.DataFrame({"construction_year": [1980], "floor_count": [3],
                            "building_category": ["commercial"], "canton": ["ZH"],
                            "lat": [47.0], "lon": [8.5]})
        r_res = predict_wwr(trained, res)[0]
        r_com = predict_wwr(trained, com)[0]
        assert r_com.predicted_wwr > r_res.predicted_wwr

    def test_predictions_not_constant(self, trained, synthetic_data):
        df, _ = synthetic_data
        results = predict_wwr(trained, df)
        wwrs = [r.predicted_wwr for r in results]
        assert max(wwrs) - min(wwrs) > 0.05


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_round_trip(self, trained, synthetic_data, tmp_path):
        path = tmp_path / "model.joblib"
        save_model(trained, path)
        loaded = load_model(path)
        assert isinstance(loaded, WWRModel)
        assert loaded.feature_names == trained.feature_names

    def test_predictions_match_after_load(self, trained, synthetic_data, tmp_path):
        df, _ = synthetic_data
        path = tmp_path / "model.joblib"
        save_model(trained, path)
        loaded = load_model(path)
        orig = predict_wwr(trained, df.head(5))
        reloaded = predict_wwr(loaded, df.head(5))
        for a, b in zip(orig, reloaded):
            assert a.predicted_wwr == pytest.approx(b.predicted_wwr)

    def test_file_created(self, trained, tmp_path):
        path = tmp_path / "sub" / "model.joblib"
        save_model(trained, path)
        assert path.exists()

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nope.joblib")
