# wwr — Window-to-Wall Ratio Computation

## Purpose
Compute the window-to-wall ratio from a rectified segmentation mask.

## Contract
- **Input:** `RectifiedResult` (from `glassscan.types`)
- **Output:** `WWRResult` (from `glassscan.types`)

## Key details
- Formula: `WWR = window_pixels / (window_pixels + wall_pixels)`
- Typical range: 0.10–0.50 for Swiss buildings
- n_windows: count of distinct connected components in the window mask
- Confidence: derived from segmentation confidence + rectification quality
- Multi-facade buildings: handle by averaging or taking max across views
