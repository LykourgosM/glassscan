# wwr -- Window-to-Wall Ratio Computation

## Purpose
Compute the window-to-wall ratio from a rectified segmentation mask.

## Contract
- **Input:** `RectifiedResult` (from `glassscan.types`)
- **Output:** `WWRResult` (from `glassscan.types`)

## Public API
- `compute_wwr(rectified)` -- single image
- `compute_wwr_batch(rectified_results)` -- list of images

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

## Typical values
- Swiss residential buildings: WWR 0.15-0.40
- Tested on 5 views of test_001: WWR 0.18-0.26 (consistent across angles)
