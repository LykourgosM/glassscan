"""Window-to-wall ratio computation from rectified facade masks.

Counts wall and window pixels in the perspective-corrected mask,
detects distinct window regions via connected components, and
returns a WWR value between 0.0 and 1.0.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from glassscan.types import RectifiedResult, WWRResult

logger = logging.getLogger(__name__)

# Connected components smaller than this are treated as noise, not windows.
_MIN_WINDOW_COMPONENT_PX = 25


def _count_pixels(mask: np.ndarray) -> tuple[int, int]:
    """Count wall and window pixels in a segmentation mask.

    Returns (wall_px, window_px).
    """
    wall_px = int(np.sum(mask == 1))
    window_px = int(np.sum(mask == 2))
    return wall_px, window_px


def _count_windows(mask: np.ndarray, min_size: int = _MIN_WINDOW_COMPONENT_PX) -> int:
    """Count distinct window regions using connected components.

    Small components below `min_size` pixels are filtered out as noise.
    """
    window_binary = (mask == 2).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        window_binary, connectivity=8,
    )
    # Label 0 is background; count labels 1..n that are large enough
    count = 0
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            count += 1
    return count


def _confidence(wall_px: int, window_px: int, mask_shape: tuple[int, ...]) -> float:
    """Estimate measurement confidence from facade pixel coverage.

    Higher confidence when more of the rectified image is facade.
    Returns 0.0-1.0.
    """
    total = mask_shape[0] * mask_shape[1]
    if total == 0:
        return 0.0
    facade_px = wall_px + window_px
    if facade_px == 0:
        return 0.0
    # Fraction of image that is facade (wall + window).
    # In a well-rectified image this is typically 0.3-0.9.
    # Scale so that >=50% coverage gives full confidence.
    facade_fraction = facade_px / total
    return float(min(facade_fraction / 0.5, 1.0))


def compute_wwr(rectified: RectifiedResult) -> WWRResult:
    """Compute window-to-wall ratio for a single rectified facade.

    WWR = window_pixels / (window_pixels + wall_pixels)

    Returns WWRResult with wwr=0.0 if no facade pixels are present.
    """
    mask = rectified.rectified_mask
    wall_px, window_px = _count_pixels(mask)
    facade_px = wall_px + window_px

    if facade_px == 0:
        logger.warning("EGID %s: no facade pixels, WWR=0", rectified.egid)
        wwr = 0.0
    else:
        wwr = window_px / facade_px

    n_windows = _count_windows(mask)
    conf = _confidence(wall_px, window_px, mask.shape)

    logger.info(
        "EGID %s: WWR=%.3f (%d window px / %d facade px), %d windows, conf=%.2f",
        rectified.egid, wwr, window_px, facade_px, n_windows, conf,
    )

    return WWRResult(
        egid=rectified.egid,
        wwr=wwr,
        window_area_px=window_px,
        wall_area_px=wall_px,
        n_windows=n_windows,
        confidence=conf,
        view_index=getattr(rectified, "view_index", 0),
    )


def compute_wwr_batch(
    rectified_results: list[RectifiedResult],
) -> list[WWRResult]:
    """Compute WWR for a batch of rectified facades."""
    results = [compute_wwr(r) for r in rectified_results]
    logger.info("Computed WWR for %d images", len(results))
    return results


def aggregate_wwr(
    results: list[WWRResult],
    weights: list[float] | None = None,
) -> list[WWRResult]:
    """Merge multiple views per building into one WWRResult per EGID.

    Parameters
    ----------
    results
        WWR results, potentially multiple per EGID (from multi-view).
    weights
        Flat list parallel to *results*. If None, uses default scheme:
        first occurrence per EGID gets weight 1.0, subsequent views 0.5.
        Pass custom weights (e.g. from LLM quality scores) to override.

    Returns
    -------
    list[WWRResult]
        One result per unique EGID, with weighted-average WWR.
    """
    from collections import OrderedDict

    # Group by EGID, preserving insertion order
    groups: OrderedDict[str, list[tuple[WWRResult, float]]] = OrderedDict()

    for i, r in enumerate(results):
        if weights is not None:
            w = weights[i]
        elif r.view_index == 0:
            w = 1.0  # primary view
        else:
            w = 0.5  # secondary view

        groups.setdefault(r.egid, []).append((r, w))

    aggregated: list[WWRResult] = []
    for egid, views in groups.items():
        if len(views) == 1:
            aggregated.append(views[0][0])
            continue

        total_w = sum(w for _, w in views)
        wwr = sum(r.wwr * w for r, w in views) / total_w
        window_px = sum(r.window_area_px for r, _ in views)
        wall_px = sum(r.wall_area_px for r, _ in views)
        n_windows = max(r.n_windows for r, _ in views)
        conf = max(r.confidence for r, _ in views)

        aggregated.append(WWRResult(
            egid=egid,
            wwr=wwr,
            window_area_px=window_px,
            wall_area_px=wall_px,
            n_windows=n_windows,
            confidence=conf,
        ))

        logger.info(
            "EGID %s: aggregated %d views -> WWR=%.3f (weights=%s)",
            egid, len(views), wwr, [f"{w:.1f}" for _, w in views],
        )

    logger.info(
        "Aggregated %d results -> %d buildings", len(results), len(aggregated),
    )
    return aggregated
