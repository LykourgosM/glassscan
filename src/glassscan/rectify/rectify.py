"""Perspective correction of facade images for accurate area ratios.

Finds the facade region from the segmentation mask, fits a bounding
quadrilateral, and warps both image and mask to a front-parallel rectangle.
This ensures every pixel represents roughly the same real-world area,
so window_pixels / (window + wall pixels) gives an unbiased WWR.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from glassscan.types import RectifiedResult, SegmentationResult

logger = logging.getLogger(__name__)

# Minimum facade area (in pixels) to attempt rectification.
# Below this the contour is too small / noisy for a reliable quad fit.
_MIN_FACADE_PX = 500


def _facade_mask(mask: np.ndarray) -> np.ndarray:
    """Return binary mask where facade (wall + window) pixels are 1."""
    return ((mask == 1) | (mask == 2)).astype(np.uint8)


def _largest_contour(binary: np.ndarray) -> np.ndarray | None:
    """Find the largest contour by area. Returns None if none found."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _fit_quad(contour: np.ndarray) -> np.ndarray:
    """Approximate a contour with a 4-point quadrilateral.

    Uses iterative epsilon relaxation on cv2.approxPolyDP until we get
    exactly 4 vertices. Falls back to the minimum-area rotated rectangle
    if approximation doesn't converge to 4 points.

    Returns shape (4, 2) float32 array of corner points.
    """
    peri = cv2.arcLength(contour, closed=True)

    # Try progressively looser approximations
    for factor in [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        approx = cv2.approxPolyDP(contour, factor * peri, closed=True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    # Fallback: minimum-area rotated rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left.

    Uses the sum and difference of coordinates:
    - top-left has smallest x+y
    - bottom-right has largest x+y
    - top-right has smallest y-x (i.e. largest x-y)
    - bottom-left has largest y-x (i.e. smallest x-y)
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    ordered[1] = pts[np.argmin(d)]   # top-right
    ordered[3] = pts[np.argmax(d)]   # bottom-left

    return ordered


def _destination_rect(src: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Compute destination rectangle dimensions from ordered source corners.

    Returns (dst_pts, width, height).
    """
    tl, tr, br, bl = src

    # Width: max of top edge and bottom edge
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    width = int(max(w_top, w_bot))

    # Height: max of left edge and right edge
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    height = int(max(h_left, h_right))

    # Clamp to reasonable bounds
    width = max(width, 1)
    height = max(height, 1)

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    return dst, width, height


def rectify_image(
    seg_result: SegmentationResult,
    *,
    save_dir: Path | None = None,
) -> RectifiedResult:
    """Rectify a single segmented facade image.

    Finds the facade boundary from the segmentation mask, fits a
    quadrilateral, and warps both image and mask to a front-parallel view.

    If the facade region is too small (< _MIN_FACADE_PX pixels), returns
    the original image/mask unchanged with an identity homography.
    """
    mask = seg_result.mask
    image = seg_result.original_image
    h, w = mask.shape[:2]

    binary = _facade_mask(mask)
    contour = _largest_contour(binary)

    vi = getattr(seg_result, "view_index", 0)

    # If no facade or too small, return as-is
    if contour is None or cv2.contourArea(contour) < _MIN_FACADE_PX:
        logger.warning("EGID %s: facade too small for rectification, passing through", seg_result.egid)
        return RectifiedResult(
            egid=seg_result.egid,
            rectified_image=image.copy(),
            rectified_mask=mask.copy(),
            homography=np.eye(3, dtype=np.float64),
            view_index=vi,
        )

    # Fit quadrilateral and order corners
    quad = _fit_quad(contour)
    src = _order_corners(quad)
    dst, out_w, out_h = _destination_rect(src)

    # Compute homography and warp
    H = cv2.getPerspectiveTransform(src, dst)

    rectified_image = cv2.warpPerspective(
        image, H, (out_w, out_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    rectified_mask = cv2.warpPerspective(
        mask, H, (out_w, out_h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    result = RectifiedResult(
        egid=seg_result.egid,
        rectified_image=rectified_image,
        rectified_mask=rectified_mask,
        homography=H,
        view_index=vi,
    )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_v{vi}" if vi > 0 else ""
        cv2.imwrite(str(save_dir / f"{seg_result.egid}{suffix}_rectified.jpg"), rectified_image)
        cv2.imwrite(str(save_dir / f"{seg_result.egid}{suffix}_rectified_mask.png"), rectified_mask)

    return result


def rectify_batch(
    seg_results: list[SegmentationResult],
    *,
    save_dir: Path | None = None,
) -> list[RectifiedResult]:
    """Rectify a batch of segmented facade images."""
    results = [rectify_image(sr, save_dir=save_dir) for sr in seg_results]
    logger.info("Rectified %d images", len(results))
    return results
