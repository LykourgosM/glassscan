"""WWR prediction from building metadata using XGBoost regression.

Trains on (features DataFrame, WWR targets) from the CV pipeline,
then predicts WWR for buildings without Street View imagery.
Prediction intervals via quantile regression (5th / 95th percentile).

Feature-agnostic: auto-detects numeric vs categorical columns from
the DataFrame, so the exact feature set can be decided at runtime
once we see the hackathon dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

from glassscan.types import PredictionResult

logger = logging.getLogger(__name__)


@dataclass
class WWRModel:
    """Trained WWR prediction model."""

    pipeline: Pipeline
    pipeline_lower: Pipeline  # 5th percentile
    pipeline_upper: Pipeline  # 95th percentile
    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_feature_types(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Split columns into numeric and categorical lists."""
    numeric = []
    categorical = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def _build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Preprocessing: median-impute numerics, ordinal-encode categoricals."""
    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1,
                )),
            ]),
            categorical_features,
        ))
    return ColumnTransformer(transformers)


def _build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    quantile: float | None = None,
) -> Pipeline:
    """Full sklearn pipeline: preprocessing + XGBoost."""
    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    if quantile is not None:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantile,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    targets: np.ndarray | list[float],
    *,
    cv_folds: int = 5,
) -> WWRModel:
    """Train WWR prediction model on building metadata.

    Args:
        df: Feature DataFrame -- any columns, auto-detected as
            numeric or categorical.  Must NOT contain the target or
            an EGID/ID column (drop those before calling).
        targets: Ground-truth WWR values (0.0-1.0), same length as df.
        cv_folds: Number of cross-validation folds for evaluation.

    Returns:
        Trained WWRModel with median and quantile pipelines.
    """
    y = np.asarray(targets, dtype=np.float64)
    if len(df) != len(y):
        raise ValueError(
            f"df ({len(df)} rows) and targets ({len(y)}) must have same length"
        )
    if len(df) == 0:
        raise ValueError("Cannot train on empty data")

    numeric_features, categorical_features = _detect_feature_types(df)
    all_features = numeric_features + categorical_features

    # Train median + quantile models
    pipeline = _build_pipeline(numeric_features, categorical_features)
    pipeline.fit(df, y)

    pipeline_lower = _build_pipeline(numeric_features, categorical_features, quantile=0.05)
    pipeline_lower.fit(df, y)

    pipeline_upper = _build_pipeline(numeric_features, categorical_features, quantile=0.95)
    pipeline_upper.fit(df, y)

    # Metrics
    metrics: dict = {
        "n_train": len(df),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
    }

    if len(df) >= cv_folds:
        fresh = _build_pipeline(numeric_features, categorical_features)
        cv_mae = -cross_val_score(fresh, df, y, cv=cv_folds, scoring="neg_mean_absolute_error")
        fresh = _build_pipeline(numeric_features, categorical_features)
        cv_r2 = cross_val_score(fresh, df, y, cv=cv_folds, scoring="r2")
        metrics["cv_mae_mean"] = float(np.mean(cv_mae))
        metrics["cv_mae_std"] = float(np.std(cv_mae))
        metrics["cv_r2_mean"] = float(np.mean(cv_r2))
        metrics["cv_r2_std"] = float(np.std(cv_r2))
        logger.info(
            "CV results: MAE=%.4f+-%.4f, R2=%.4f+-%.4f",
            metrics["cv_mae_mean"], metrics["cv_mae_std"],
            metrics["cv_r2_mean"], metrics["cv_r2_std"],
        )

    # Feature importance
    booster = pipeline.named_steps["model"]
    importances = booster.feature_importances_
    metrics["feature_importance"] = dict(zip(all_features, importances.tolist()))

    logger.info("Trained WWR model on %d samples (mean WWR=%.3f)", len(df), np.mean(y))
    return WWRModel(
        pipeline=pipeline,
        pipeline_lower=pipeline_lower,
        pipeline_upper=pipeline_upper,
        feature_names=all_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        metrics=metrics,
    )


def predict_wwr(
    model: WWRModel,
    df: pd.DataFrame,
    egids: list[str] | None = None,
) -> list[PredictionResult]:
    """Predict WWR for buildings.

    Args:
        model: Trained WWRModel.
        df: Feature DataFrame with the same columns used during training.
        egids: Optional building IDs (one per row).  If None, uses
            row index as string.

    Returns:
        List of PredictionResult, one per row.
    """
    if len(df) == 0:
        return []

    if egids is None:
        egids = [str(i) for i in df.index]

    preds = np.clip(model.pipeline.predict(df), 0.0, 1.0)
    preds_lower = np.clip(model.pipeline_lower.predict(df), 0.0, 1.0)
    preds_upper = np.clip(model.pipeline_upper.predict(df), 0.0, 1.0)

    # Enforce monotonicity: lower <= pred <= upper
    preds_lower = np.minimum(preds_lower, preds)
    preds_upper = np.maximum(preds_upper, preds)

    results = []
    for i in range(len(df)):
        results.append(PredictionResult(
            egid=egids[i],
            predicted_wwr=float(preds[i]),
            prediction_interval=(float(preds_lower[i]), float(preds_upper[i])),
            features_used=model.feature_names,
        ))

    logger.info("Predicted WWR for %d buildings", len(results))
    return results


def save_model(model: WWRModel, path: str | Path) -> None:
    """Save trained model to disk via joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(path: str | Path) -> WWRModel:
    """Load trained model from disk."""
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model
