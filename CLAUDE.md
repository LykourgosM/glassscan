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
- **visualise** — DONE (13 tests). Python export + React/TypeScript dashboard.

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
- **Swiss building geometry integration — shared prerequisite for three wins below.**
  Data access chain: for any building coordinate, the nearest GWR point gives EGID +
  attributes (garea, gvol, storeys, year). That EGID then indexes into swissBUILDINGS3D
  for the 2D footprint + building height. For non-EGID cantons (e.g. VD, where the
  hackathon is held), the link from GWR POINT to swissBUILDINGS3D polygon is a spatial
  point-in-polygon join — still works, just one extra step. This single integration
  unlocks: (a) neighbor masking at fetch time, (b) adaptive FOV per pano/facade at
  fetch time, (c) geometrically exact rectify homography. All three are detailed in
  the bullets below.
- Road direction estimated as perpendicular to building→panorama bearing. Future fix:
  OSM Overpass API for actual road geometry.
- Swiss data source details (swissBUILDINGS3D, GWR, swissTLM3D) documented in `fetch/CLAUDE.md`.
- GWR needed for predict module features (construction year, storeys, heating). Points only, no polygons.
- Rectification with swissBUILDINGS3D: current approach fits a quad to the segmentation
  contour (approximate). With 3D building geometry + known camera position from Street View
  metadata, we can project exact facade corners into the image and compute a geometrically
  exact homography. Current module serves as fallback when 3D data is unavailable.
- Adaptive FOV from building polygon (fetch-time): current fetch uses fixed fov=70, which
  crops close buildings and over-zooms out distant ones. With the 2D footprint + building
  height (from swissBUILDINGS3D, or GWR `gvol/garea` as a proxy), identify the visible
  facade(s) from each pano — polygon edges whose outward normal faces the camera — then
  project their 4 corners into image space and set FOV to span them with margin. The same
  corner projections feed directly into the rectify improvement above (geometrically exact
  homography from known-good corners), so one integration unlocks both wins. For non-EGID
  cantons (e.g. VD, where the hackathon is held), join GWR POINT → swissBUILDINGS3D polygon
  via point-in-polygon to get the same data.
- Multi-view aggregation: `aggregate_wwr()` merges multiple views per building.
  Default weights: primary view=1.0, secondary views=0.5. Accepts a flat list of
  custom weights to override defaults.
- Claude-weighted aggregation: use Claude Code's vision (included in Max sub)
  to score segmentation quality per view (0-1). Save to `weights.json` keyed
  by EGID (e.g. `{"140040": [0.9, 0.3, 0.7]}`). `aggregate_wwr` loads from
  this file if it exists, falls back to hardcoded weights (primary=1.0,
  secondary=0.5). Run scoring in a separate Claude Code session after pipeline.
- Camera position visualisation: when a building is selected, show panorama
  positions on the map with view cone lines projecting toward the building.
  Data already available in BuildingImage (pano lat/lon, heading, fov).
  Simple 2D version: Leaflet markers + polylines on existing map.
  Advanced 3D version: CesiumJS or deck.gl with building heights from
  swissBUILDINGS3D. 2D version is feasible for hackathon, 3D is stretch goal.
