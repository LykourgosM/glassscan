"""Tests for the window-to-wall ratio module.

Uses synthetic masks -- no real images or models needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from glassscan.wwr.wwr import (
    _count_pixels,
    _count_windows,
    _confidence,
    _MIN_WINDOW_COMPONENT_PX,
    compute_wwr,
    compute_wwr_batch,
)
from glassscan.types import RectifiedResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rectified(
    mask: np.ndarray,
    egid: str = "1234",
) -> RectifiedResult:
    """Create a RectifiedResult with a given mask and matching dummy image."""
    h, w = mask.shape[:2]
    image = np.zeros((h, w, 3), dtype=np.uint8)
    return RectifiedResult(
        egid=egid,
        rectified_image=image,
        rectified_mask=mask,
        homography=np.eye(3),
    )


def _facade_mask(
    h: int = 200,
    w: int = 200,
    wall_rect: tuple[int, int, int, int] = (0, 0, 200, 200),
    window_rects: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """Create a mask with wall (1) and optional window (2) regions.

    Coordinates are (top, left, bottom, right).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    t, l, b, r = wall_rect
    mask[t:b, l:r] = 1
    if window_rects:
        for wt, wl, wb, wr in window_rects:
            mask[wt:wb, wl:wr] = 2
    return mask


# ---------------------------------------------------------------------------
# _count_pixels
# ---------------------------------------------------------------------------

class TestCountPixels:
    def test_all_wall(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        wall, window = _count_pixels(mask)
        assert wall == 100
        assert window == 0

    def test_all_window(self):
        mask = np.full((10, 10), 2, dtype=np.uint8)
        wall, window = _count_pixels(mask)
        assert wall == 0
        assert window == 100

    def test_mixed(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        mask[0:5, :] = 2  # top half is window
        wall, window = _count_pixels(mask)
        assert wall == 50
        assert window == 50

    def test_with_background(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:5, 0:5] = 1  # 25 wall
        mask[5:10, 5:10] = 2  # 25 window
        wall, window = _count_pixels(mask)
        assert wall == 25
        assert window == 25

    def test_empty(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        wall, window = _count_pixels(mask)
        assert wall == 0
        assert window == 0


# ---------------------------------------------------------------------------
# _count_windows
# ---------------------------------------------------------------------------

class TestCountWindows:
    def test_single_window(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[20:60, 20:60] = 2  # one 40x40 window
        assert _count_windows(mask) == 1

    def test_two_separate_windows(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[10:40, 10:40] = 2  # window 1
        mask[60:90, 60:90] = 2  # window 2
        assert _count_windows(mask) == 2

    def test_no_windows(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        assert _count_windows(mask) == 0

    def test_noise_filtered(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[10:40, 10:40] = 2  # real window (900 px)
        mask[80, 80] = 2  # single pixel noise
        mask[90, 90:93] = 2  # 3px noise
        assert _count_windows(mask) == 1

    def test_custom_min_size(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 2  # 100 px window
        # With min_size=200, this should be filtered
        assert _count_windows(mask, min_size=200) == 0
        # With min_size=50, it should count
        assert _count_windows(mask, min_size=50) == 1

    def test_adjacent_windows_merged(self):
        """Touching windows are one connected component."""
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[20:40, 20:40] = 2
        mask[40:60, 20:40] = 2  # directly below, touching
        assert _count_windows(mask) == 1

    def test_diagonal_windows_merged(self):
        """Diagonally touching windows merge with 8-connectivity."""
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[20:30, 20:30] = 2
        mask[30:40, 30:40] = 2  # diagonal touch
        assert _count_windows(mask) == 1


# ---------------------------------------------------------------------------
# _confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_full_facade(self):
        # 100% facade -> confidence 1.0
        assert _confidence(80, 20, (10, 10)) == 1.0

    def test_half_facade(self):
        # 50% facade -> confidence 1.0
        assert _confidence(40, 10, (10, 10)) == 1.0

    def test_quarter_facade(self):
        # 25% facade -> confidence 0.5
        assert _confidence(20, 5, (10, 10)) == pytest.approx(0.5)

    def test_no_facade(self):
        assert _confidence(0, 0, (10, 10)) == 0.0

    def test_zero_image(self):
        assert _confidence(0, 0, (0, 0)) == 0.0

    def test_scales_linearly(self):
        c1 = _confidence(10, 0, (10, 10))  # 10% facade
        c2 = _confidence(20, 0, (10, 10))  # 20% facade
        assert c2 == pytest.approx(2 * c1)


# ---------------------------------------------------------------------------
# compute_wwr
# ---------------------------------------------------------------------------

class TestComputeWWR:
    def test_basic_ratio(self):
        # 50% wall, 50% window -> WWR = 0.5
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[:50, :] = 2
        r = _make_rectified(mask)
        result = compute_wwr(r)
        assert result.wwr == pytest.approx(0.5)
        assert result.window_area_px == 5000
        assert result.wall_area_px == 5000

    def test_no_windows(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        r = _make_rectified(mask)
        result = compute_wwr(r)
        assert result.wwr == 0.0
        assert result.n_windows == 0

    def test_no_facade(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        r = _make_rectified(mask)
        result = compute_wwr(r)
        assert result.wwr == 0.0
        assert result.confidence == 0.0

    def test_realistic_facade(self):
        # Wall with two distinct windows
        mask = _facade_mask(
            h=200, w=200,
            wall_rect=(0, 0, 200, 200),
            window_rects=[(30, 20, 80, 80), (30, 120, 80, 180)],
        )
        r = _make_rectified(mask)
        result = compute_wwr(r)
        window_px = 50 * 60 * 2  # two 50x60 windows
        total = 200 * 200
        expected_wwr = window_px / total
        assert result.wwr == pytest.approx(expected_wwr)
        assert result.n_windows == 2

    def test_egid_preserved(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        r = _make_rectified(mask, egid="EGID_42")
        result = compute_wwr(r)
        assert result.egid == "EGID_42"

    def test_wwr_range(self):
        """WWR should always be in [0, 1]."""
        for window_frac in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            mask = np.ones((100, 100), dtype=np.uint8)
            cutoff = int(100 * window_frac)
            mask[:cutoff, :] = 2
            r = _make_rectified(mask)
            result = compute_wwr(r)
            assert 0.0 <= result.wwr <= 1.0

    def test_all_window_no_wall(self):
        mask = np.full((100, 100), 2, dtype=np.uint8)
        r = _make_rectified(mask)
        result = compute_wwr(r)
        assert result.wwr == 1.0

    def test_confidence_higher_with_more_facade(self):
        # Mostly facade
        mask_full = np.ones((100, 100), dtype=np.uint8)
        mask_full[0:20, :] = 2
        r_full = _make_rectified(mask_full)

        # Mostly background with small facade
        mask_small = np.zeros((100, 100), dtype=np.uint8)
        mask_small[40:60, 40:60] = 1
        mask_small[45:55, 45:55] = 2
        r_small = _make_rectified(mask_small)

        result_full = compute_wwr(r_full)
        result_small = compute_wwr(r_small)
        assert result_full.confidence > result_small.confidence


# ---------------------------------------------------------------------------
# compute_wwr_batch
# ---------------------------------------------------------------------------

class TestComputeWWRBatch:
    def test_returns_list(self):
        masks = [np.ones((50, 50), dtype=np.uint8) for _ in range(3)]
        rects = [_make_rectified(m, egid=str(i)) for i, m in enumerate(masks)]
        results = compute_wwr_batch(rects)
        assert len(results) == 3
        assert [r.egid for r in results] == ["0", "1", "2"]

    def test_empty_batch(self):
        assert compute_wwr_batch([]) == []

    def test_mixed_batch(self):
        # One with windows, one without
        mask_with = _facade_mask(
            window_rects=[(20, 20, 80, 80)],
        )
        mask_without = np.ones((100, 100), dtype=np.uint8)

        results = compute_wwr_batch([
            _make_rectified(mask_with, egid="with"),
            _make_rectified(mask_without, egid="without"),
        ])
        assert results[0].wwr > 0
        assert results[1].wwr == 0.0
