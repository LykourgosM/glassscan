# predict — WWR Regression Model

## Purpose
Predict WWR for buildings without Street View imagery using building metadata.

## Contract
- **Input:** `BuildingFeatures` (from `glassscan.types`) + trained model
- **Output:** `PredictionResult` (from `glassscan.types`)

## Key details
- Training data: WWR values from the wwr module + building metadata from GWR (opendata.swiss)
- Features: construction_year, building_category, canton, floor_count, heating_type, lat, lon
- Model: XGBoost regressor (primary), linear regression + random forest as baselines
- Evaluation: MAE, R², cross-validation
- Prediction intervals: 90% CI via quantile regression or bootstrap
- Saved model: `models/wwr_predictor.joblib`
- Data sources: GWR via opendata.swiss, swissBUILDINGS3D from swisstopo
