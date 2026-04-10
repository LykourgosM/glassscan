# GlassScan — Swiss Window-to-Wall Ratio Pipeline

## What this is
CV + ML pipeline for the Energy Data Hackdays 2026 (May 7-8, Lausanne).
Estimates window-to-wall ratios (WWR) of Swiss buildings from Google Street View imagery,
then predicts WWR for buildings without imagery using building metadata.

## Architecture
```
fetch → segment → rectify → wwr → predict → visualise
```
Each module is an independent sub-package under `src/glassscan/`.
All inter-module data flows through dataclasses defined in `src/glassscan/types.py`.
`pipeline.py` orchestrates the end-to-end chain.

## Key commands
```bash
source .venv/bin/activate
make test          # run all tests
make test-fetch    # run tests for one module
make pipeline      # run end-to-end pipeline
```

## Project structure
- `src/glassscan/types.py` — shared dataclasses (BuildingImage, SegmentationResult, etc.)
- `src/glassscan/pipeline.py` — end-to-end orchestrator
- `src/glassscan/{fetch,segment,rectify,wwr,predict,visualise}/` — independent modules
- `data/` — gitignored, raw images + masks + processed data
- `models/` — gitignored, saved model weights
- `.env` — gitignored, contains GOOGLE_API_KEY

## Module status
- **fetch** — DONE (24 tests). Street View image fetcher with multi-view support.
- **segment** — DONE (21 tests). Two-stage: ADE20K building mask + CMP Facades wall/window.
- **rectify** — DONE (25 tests). Perspective correction via quadrilateral fit + homography warp.
- **wwr** — DONE (29 tests). Pixel counting + connected components for window detection.
- **predict** — DONE (29 tests). Feature-agnostic XGBoost regression with quantile prediction intervals.
- **pipeline.py** — DONE (21 tests). Orchestrator chaining all modules with lazy imports.
- **visualise** — stub

## Conventions
- Python 3.11, PyTorch + HuggingFace Transformers
- Each module has its own CLAUDE.md with input/output contracts
- Tests sit next to the code: `module/test_module.py`
- Images are BGR (OpenCV default) throughout the pipeline
- Building IDs use the Swiss federal EGID system

## GCP setup
- API key lives in `.env` (GOOGLE_API_KEY), belongs to GCP project "My First Project"
- Street View Static API enabled on "My First Project"
- Budget alert "GlassScan spending alert" at CHF 1
- User has blocked Google as merchant on their card (hard stop)
- Street View Static API: ~$7/1000 image requests, metadata endpoint is free
- ~28,000 free image calls/month ($200 Maps credit)

## Pipeline architecture
`pipeline.py` orchestrates: fetch -> segment -> rectify -> wwr -> predict.
Three entry points:
- `run_cv_pipeline(buildings, api_key)` -- image-based WWR measurement
- `run_prediction_pipeline(wwr_results, metadata_df)` -- metadata-based prediction
- `run_full_pipeline(...)` -- both in sequence

Uses lazy imports because PyTorch (segment) and XGBoost (predict) ship
conflicting libomp on macOS. Tests are split: `test_pipeline.py` (prediction,
14 tests) and `test_pipeline_cv.py` (CV mocks, 7 tests).

At the hackathon, the pipeline functions stay stable. What changes is the
caller code: data loading, EGID matching, feature column selection. If a
module API changes (e.g., new param on segment_batch), update the call in
run_cv_pipeline.

## Known limitations / future improvements
- Street View images capture multiple buildings per frame. Accept for now; future fix
  via swissBUILDINGS3D (project building into image to mask neighbors). See `fetch/CLAUDE.md`.
- Road direction estimated as perpendicular to building→panorama bearing. Future fix:
  OSM Overpass API for actual road geometry.
- Swiss data source details (swissBUILDINGS3D, GWR, swissTLM3D) documented in `fetch/CLAUDE.md`.
- GWR needed for predict module features (construction year, storeys, heating). Points only, no polygons.
- Rectification with swissBUILDINGS3D: current approach fits a quad to the segmentation
  contour (approximate). With 3D building geometry + known camera position from Street View
  metadata, we can project exact facade corners into the image and compute a geometrically
  exact homography. Current module serves as fallback when 3D data is unavailable.
- VLM validation: use a vision-language model (Claude, GPT-4V) to spot-check segmentation
  outputs. Could estimate WWR directly from images, flag disagreements with SegFormer masks,
  or filter bad images before segmentation. Not a primary approach (no pixel masks, API cost
  at scale, less reproducible) but useful as a quality signal alongside the CV pipeline.
