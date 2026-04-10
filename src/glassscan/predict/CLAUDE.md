# predict -- WWR Regression Model

## Purpose
Predict WWR for buildings without Street View imagery using building metadata.

## Contract
- **Input:** `pd.DataFrame` of features (any columns, auto-detected as numeric/categorical) + target WWR values for training
- **Output:** `PredictionResult` (from `glassscan.types`)

## Public API
- `train_model(df, targets, cv_folds=5)` -- train XGBoost on features + ground-truth WWR
- `predict_wwr(model, df, egids=None)` -- predict for a batch of buildings
- `save_model(model, path)` / `load_model(path)` -- joblib persistence
- `WWRModel` -- trained model dataclass (pipeline + quantile models + metrics)

## How it works
1. Auto-detect numeric vs categorical columns from the DataFrame
2. Preprocessing: median-impute numerics, ordinal-encode categoricals (unknown -> -1)
3. Train three XGBoost models: median, 5th percentile, 95th percentile (quantile regression)
4. Cross-validation for MAE and R² metrics (skipped if fewer samples than folds)
5. Feature importance extracted from the median model
6. Predictions clamped to [0, 1] with monotonicity enforced (lower <= pred <= upper)

## Feature-agnostic design
The module does NOT hardcode specific feature columns. It accepts any DataFrame and
auto-detects types. This lets us plug in whatever GWR/metadata fields the hackathon
provides without changing the model code. The `BuildingFeatures` dataclass in types.py
is a hint of expected features, not a hard requirement.

## Typical usage
```python
# Training (join WWR pipeline output with GWR metadata)
df = gwr_data[["construction_year", "building_category", "canton", ...]]
targets = wwr_results["wwr"].values
model = train_model(df, targets)

# Prediction
new_df = gwr_data_no_imagery[["construction_year", "building_category", "canton", ...]]
egids = gwr_data_no_imagery["egid"].tolist()
results = predict_wwr(model, new_df, egids=egids)

# Persistence
save_model(model, "models/wwr_predictor.joblib")
model = load_model("models/wwr_predictor.joblib")
```

## Hackathon rewiring needed
This module is the training/prediction plumbing only. At the hackathon we'll need to:
- **Data loading:** Fetch GWR metadata (or whatever dataset SDSC provides) and build the feature DataFrame
- **Feature selection:** Decide which columns to use once we see what's available
- **Join logic:** Match GWR records to WWR pipeline output by EGID to assemble training pairs
- **Hyperparameter tuning:** Current defaults (200 trees, depth 5) are reasonable but may need adjustment based on dataset size and signal strength
- **Evaluation criteria:** Adapt metrics to whatever the judges care about

The module's feature-agnostic design means none of this requires changing predict.py itself -- it's all in the caller (pipeline.py or a notebook).

## Model details
- XGBoost: 200 trees, max_depth=5, lr=0.1, random_state=42
- 90% prediction interval via `reg:quantileerror` objective (alpha=0.05 and 0.95)
- Metrics: n_train, target_mean, target_std, cv_mae, cv_r2, feature_importance
