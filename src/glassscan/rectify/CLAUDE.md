# rectify — Perspective Correction

## Purpose
Correct perspective distortion in facade images so that area ratios are accurate.

## Contract
- **Input:** `SegmentationResult` (from `glassscan.types`)
- **Output:** `RectifiedResult` (from `glassscan.types`)

## Key details
- Approach: 4-point perspective transform via OpenCV
- Corner detection: bounding quadrilateral of the wall segmentation mask
- Uses `cv2.getPerspectiveTransform()` + `cv2.warpPerspective()`
- Both image and mask are rectified using the same homography
- The 3x3 homography matrix is stored for traceability
- No full camera calibration needed — only correcting oblique angle
