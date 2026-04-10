# rectify -- Perspective Correction

## Purpose
Correct perspective distortion in facade images so that area ratios are accurate for WWR computation.

## Contract
- **Input:** `SegmentationResult` (from `glassscan.types`)
- **Output:** `RectifiedResult` (from `glassscan.types`)

## Public API
- `rectify_image(seg_result, save_dir=None)` -- single image
- `rectify_batch(seg_results, save_dir=None)` -- list of images

## How it works
1. Extract facade pixels from the segmentation mask (wall + window = class 1 + 2)
2. Find the largest contour of the facade region
3. Fit a 4-point quadrilateral using `cv2.approxPolyDP` with iterative epsilon relaxation; falls back to `cv2.minAreaRect` if approximation doesn't converge to 4 points
4. Order corners as TL, TR, BR, BL (by coordinate sum/difference)
5. Compute destination rectangle dimensions from edge lengths
6. `cv2.getPerspectiveTransform(src, dst)` for the 3x3 homography
7. `cv2.warpPerspective()` on both image (INTER_LINEAR) and mask (INTER_NEAREST)

## Edge cases
- **No facade / too small (<500px):** returns original image+mask with identity homography
- **Mask uses INTER_NEAREST** to avoid interpolating between class values

## Saves as
- `{egid}_rectified.jpg` (image) and `{egid}_rectified_mask.png` (mask) in save_dir
