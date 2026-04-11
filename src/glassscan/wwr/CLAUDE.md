# wwr -- Window-to-Wall Ratio Computation

## Purpose
Compute the window-to-wall ratio from a rectified segmentation mask.

## Contract
- **Input:** `RectifiedResult` (from `glassscan.types`)
- **Output:** `WWRResult` (from `glassscan.types`)

## Public API
- `compute_wwr(rectified)` -- single image
- `compute_wwr_batch(rectified_results)` -- list of images
- `aggregate_wwr(results, weights=None)` -- merge multiple views per EGID into one result

## How it works
1. Count wall pixels (class 1) and window pixels (class 2) in the rectified mask
2. Compute `WWR = window_pixels / (window_pixels + wall_pixels)`
3. Detect distinct window regions via `cv2.connectedComponentsWithStats` (8-connectivity)
4. Filter noise: components < 25 pixels (`_MIN_WINDOW_COMPONENT_PX`) are ignored
5. Compute confidence from facade pixel coverage (fraction of image that is wall+window; >=50% coverage gives full confidence, scales linearly below)

## Edge cases
- **No facade pixels:** returns wwr=0.0, confidence=0.0
- **No windows:** returns wwr=0.0 (valid -- some facades genuinely have no windows)
- **All window, no wall:** returns wwr=1.0
- **Small noise components:** filtered out of window count

## Multi-view aggregation
`aggregate_wwr(results, weights=None)` groups WWRResult by EGID and produces one
result per building. If `weights` is None, uses default scheme: primary view
(first per EGID) gets weight 1.0, secondary views get 0.5. Pass a flat list of
floats parallel to `results` to override (e.g. with LLM-assigned quality scores).

## Typical values
- Swiss residential buildings: WWR 0.15-0.40
- Tested on 5 views of test_001: WWR 0.18-0.26 (consistent across angles)
