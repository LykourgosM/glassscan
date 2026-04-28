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
- **Dashboard "debug pipeline" UI for any building.**
  Add a route / button to the existing visualise dashboard that lets the
  user pick a single EGID and see EVERY step of the geometry pipeline
  for that building, the way `notebooks/geometry_single_building.ipynb`
  does today: footprint on a map, edge decomposition with outward
  normals, pano discovery with each filter category, per-pano FOV
  cones, projected facade quads on the captured Street View images,
  rectified per-facade outputs, segmentation overlays, per-facade WWR
  table, and final score×area-weighted building WWR.
  Two purposes:
  1. **Demo / explainability** for the hackathon judges — energy data
     audiences care a lot about "how did you arrive at this number",
     and being able to walk through one building visually is much more
     convincing than just a number on a map.
  2. **Debug** — once the pipeline runs in production batch mode and
     someone notices "this building's WWR looks wrong", being able to
     pull up the full pipeline for that single EGID without re-running
     the notebook by hand is a huge time-saver.
  Build only after the pipeline is fully running in production. The
  notebook's cells are the natural source for what each panel should
  show; most are already self-contained HTML / folium / matplotlib
  outputs that could be served directly.

- **Production aggregation: NaN + warning instead of `raise`.**
  The notebook's cell 16 raises a `RuntimeError` when ALL facades for a
  building have zero valid pixels (catastrophic failure of segmentation
  or rectification). That's right for single-building debugging - forces
  investigation. When we extract the area-weighted aggregation into
  production / `aggregate_wwr`, switch to setting building WWR = `NaN`
  with a logger warning, so a single broken building doesn't abort a
  batch run over thousands. Downstream code should treat NaN as
  "estimate unavailable" and either drop the building from the dashboard
  or fall back to the predict-module's metadata-only WWR.

- **3D Wall-mesh rectification (instead of 4-corner cube approximation).**
  Cell 13 of `notebooks/geometry_single_building.ipynb` currently projects 4
  corners per facade — footprint endpoints at z=0 (ground) and
  z=GESAMTHOEHE (roof peak). This overshoots for pitched-roof buildings:
  GESAMTHOEHE is foundation-to-peak, but the peak sits *inward* from the
  outer edge (near the roof centerline). At the outer edge the building
  only reaches the *eave* (DACH_MIN), not the peak — so the projected top
  corner is "in mid-air above the eave" by 3-7 m worth of image pixels for
  typical Zurich pitched roofs. Rectified output above the eave is
  approximate. Roof / dormer windows are misrectified.
  Proper fix: use the Wall layer's 3D MultiPatch mesh that swissBUILDINGS3D
  already provides. For each visible footprint edge, find the matching
  Wall feature, project all its 3D vertices, use the convex hull (or
  outer-boundary polyline) as the rectification target. More accurate per
  facade (correct eave heights, handles bay windows / irregular walls);
  more complex for the homography step (need to pick 4 corner
  correspondences from a hull rather than from a clean cube). Worth
  tackling after the simpler approach is end-to-end working.

- **EXPERIMENTAL — pano-centric fetch architecture (NOT committed, future
  thinking only, only explore AFTER the hackathon).**
  Current fetch is building-centric: per building, find panos, fetch one narrow
  image per pano. This re-fetches the same pano N times when N buildings share
  a street. Speculative redesign: switch to pano-centric — fetch each unique
  pano once as a full equirectangular 360° (3-4 stitched 120° Static API calls,
  OR via the Street View Tiles API), cache it, then for any building visible
  from that pano, crop a rectilinear view at the heading + FOV computed from
  its 3D corners. At Lausanne scale (~50k buildings) with ~5 buildings/pano,
  this could drop fetch cost from ~250k images to ~10-40k. Tradeoffs: 3-4× per-
  pano cost, equirectangular→rectilinear reprojection code (~30 lines cv2),
  bigger cache (~30 GB for a city). Geometry helpers we build now (e.g.
  `compute_fov_for_facade`) are forward-compatible with this architecture, so
  no rework is wasted.
