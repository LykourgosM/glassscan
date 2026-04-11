# visualise -- Dashboard & Export

## Purpose
Export pipeline results and display them in an interactive React dashboard.

## Contract
- **Input:** `PipelineResult` (from `glassscan.pipeline`) + optional `pd.DataFrame` of metadata
- **Output:** `buildings.json` + building card images + React dashboard

## Public API (Python)
- `export_results(result, output_dir, metadata_df=None)` -- export pipeline results for the dashboard
- `create_building_card(image, mask, wwr)` -- composite image: original | segmentation overlay

## How it works
1. Python `export.py` takes a PipelineResult and writes:
   - `buildings.json` -- building coordinates, WWR values, source (measured/predicted), metadata, summary stats
   - `images/{egid}.jpg` -- side-by-side building cards (original + colored overlay)
2. React dashboard (`dashboard/`) reads these files and displays:
   - Interactive Leaflet map with colored markers (green=low WWR, red=high)
   - Solid circles = measured from CV pipeline, rings = predicted from metadata
   - Click any building for details + segmentation image
   - Sidebar with summary stats + charts (WWR by era, by type, feature importance)

## Dashboard tech stack
- Vite + React + TypeScript
- react-leaflet with CartoDB positron tiles
- Tailwind CSS
- Recharts for charts

## Running the dashboard
```bash
cd src/glassscan/visualise/dashboard
npm install        # first time only
npm run dev        # dev server at localhost:5173
npm run build      # production build to dist/
npx serve dist     # serve the built dashboard
```

## Typical usage
```python
from glassscan.pipeline import run_cv_pipeline
from glassscan.visualise import export_results

result = run_cv_pipeline(buildings, api_key)
export_results(result, "dashboard/public", metadata_df=metadata)
# Then: cd dashboard && npm run dev
```

## Multi-view support
When the pipeline runs with `max_views > 1`, buildings have multiple images.
The export should include per-view data in `buildings.json` so the dashboard
can display all views in the building panel with their weights. Structure:
```json
{
  "egid": "140040",
  "wwr": 0.15,
  "views": [
    {"view_index": 0, "wwr": 0.14, "weight": 1.0},
    {"view_index": 1, "wwr": 0.17, "weight": 0.5}
  ]
}
```
Images saved as `raw/{egid}.jpg` (primary), `raw/{egid}_v1.jpg`, `raw/{egid}_v2.jpg`.

## Hackathon notes
- Pre-compute a batch of buildings before the hackathon, export to JSON
- The dashboard reads static files -- no server needed during the demo
- Modify `Sidebar.tsx` to add custom charts for the judges
- Building cards can be dropped into presentation slides as PNGs
