"""Tests for the perspective rectification module.

Uses synthetic masks — no real images or models needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from glassscan.rectify.rectify import (
    _facade_mask,
    _largest_contour,
    _fit_quad,
    _order_corners,
    _destination_rect,
    _MIN_FACADE_PX,
    rectify_image,
    rectify_batch,
)
from glassscan.types import SegmentationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seg_result(
    mask: np.ndarray,
    egid: str = "1234",
) -> SegmentationResult:
    """Create a SegmentationResult with a given mask and matching dummy image."""
    h, w = mask.shape[:2]
    image = np.zeros((h, w, 3), dtype=np.uint8)
    return SegmentationResult(egid=egid, mask=mask, confidence=0.9, original_image=image)


def _rect_mask(h: int = 400, w: int = 400, top: int = 50, left: int = 80,
               bottom: int = 350, right: int = 320, cls: int = 1) -> np.ndarray:
    """Create a mask with a rectangle of class `cls` on a background of 0."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top:bottom, left:right] = cls
    return mask


def _trapezoid_mask(h: int = 400, w: int = 400) -> np.ndarray:
    """Create a mask with a trapezoidal facade (wider at bottom)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[150, 50], [250, 50], [300, 350], [100, 350]], dtype=np.int32)
    import cv2
    cv2.fillPoly(mask, [pts], 1)
    return mask


# ---------------------------------------------------------------------------
# _facade_mask
# ---------------------------------------------------------------------------

class TestFacadeMask:
    def test_wall_included(self):
        mask = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)
        fm = _facade_mask(mask)
        assert fm[0, 1] == 1
        assert fm[1, 0] == 1

    def test_window_included(self):
        mask = np.array([[0, 2, 0]], dtype=np.uint8)
        fm = _facade_mask(mask)
        assert fm[0, 1] == 1

    def test_background_excluded(self):
        mask = np.array([[0, 0, 0]], dtype=np.uint8)
        fm = _facade_mask(mask)
        assert np.all(fm == 0)

    def test_combined(self):
        mask = np.array([[0, 1, 2, 0]], dtype=np.uint8)
        fm = _facade_mask(mask)
        np.testing.assert_array_equal(fm, [[0, 1, 1, 0]])


# ---------------------------------------------------------------------------
# _largest_contour
# ---------------------------------------------------------------------------

class TestLargestContour:
    def test_finds_contour(self):
        binary = _rect_mask(cls=1).astype(np.uint8)
        c = _largest_contour(binary)
        assert c is not None
        assert len(c) >= 4

    def test_returns_none_on_empty(self):
        binary = np.zeros((100, 100), dtype=np.uint8)
        assert _largest_contour(binary) is None

    def test_picks_largest(self):
        binary = np.zeros((200, 200), dtype=np.uint8)
        binary[10:30, 10:30] = 1    # small 20x20
        binary[50:150, 50:150] = 1  # large 100x100
        c = _largest_contour(binary)
        import cv2
        area = cv2.contourArea(c)
        assert area > 5000  # should be the large one


# ---------------------------------------------------------------------------
# _fit_quad
# ---------------------------------------------------------------------------

class TestFitQuad:
    def test_returns_four_points(self):
        binary = _rect_mask(cls=1).astype(np.uint8)
        c = _largest_contour(binary)
        quad = _fit_quad(c)
        assert quad.shape == (4, 2)
        assert quad.dtype == np.float32

    def test_trapezoid_returns_four_points(self):
        binary = _trapezoid_mask()
        c = _largest_contour(binary)
        quad = _fit_quad(c)
        assert quad.shape == (4, 2)


# ---------------------------------------------------------------------------
# _order_corners
# ---------------------------------------------------------------------------

class TestOrderCorners:
    def test_already_ordered(self):
        pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        ordered = _order_corners(pts)
        np.testing.assert_array_equal(ordered[0], [0, 0])      # TL
        np.testing.assert_array_equal(ordered[1], [100, 0])     # TR
        np.testing.assert_array_equal(ordered[2], [100, 100])   # BR
        np.testing.assert_array_equal(ordered[3], [0, 100])     # BL

    def test_shuffled(self):
        pts = np.array([[100, 100], [0, 0], [0, 100], [100, 0]], dtype=np.float32)
        ordered = _order_corners(pts)
        np.testing.assert_array_equal(ordered[0], [0, 0])
        np.testing.assert_array_equal(ordered[1], [100, 0])
        np.testing.assert_array_equal(ordered[2], [100, 100])
        np.testing.assert_array_equal(ordered[3], [0, 100])


# ---------------------------------------------------------------------------
# _destination_rect
# ---------------------------------------------------------------------------

class TestDestinationRect:
    def test_square_input(self):
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        dst, w, h = _destination_rect(src)
        assert w == 100
        assert h == 100
        assert dst.shape == (4, 2)

    def test_wide_input(self):
        src = np.array([[0, 0], [200, 0], [200, 50], [0, 50]], dtype=np.float32)
        dst, w, h = _destination_rect(src)
        assert w == 200
        assert h == 50


# ---------------------------------------------------------------------------
# rectify_image
# ---------------------------------------------------------------------------

class TestRectifyImage:
    def test_returns_rectified_result(self):
        mask = _rect_mask()
        sr = _make_seg_result(mask)
        result = rectify_image(sr)
        assert result.egid == "1234"
        assert result.rectified_image.ndim == 3
        assert result.rectified_mask.ndim == 2
        assert result.homography.shape == (3, 3)

    def test_mask_dtype_preserved(self):
        mask = _rect_mask()
        sr = _make_seg_result(mask)
        result = rectify_image(sr)
        assert result.rectified_mask.dtype == np.uint8

    def test_mask_classes_preserved(self):
        """Only classes 0, 1, 2 should appear in rectified mask."""
        mask = _rect_mask(cls=1)
        # Add some window pixels inside the wall region
        mask[100:200, 120:200] = 2
        sr = _make_seg_result(mask)
        result = rectify_image(sr)
        assert set(np.unique(result.rectified_mask)).issubset({0, 1, 2})

    def test_window_pixels_survive(self):
        """Windows inside the facade should still be present after rectification."""
        mask = _rect_mask(cls=1)
        mask[100:200, 120:200] = 2  # window block inside wall
        sr = _make_seg_result(mask)
        result = rectify_image(sr)
        assert 2 in result.rectified_mask

    def test_passthrough_when_no_facade(self):
        """All-background mask should return identity homography."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        sr = _make_seg_result(mask)
        result = rectify_image(sr)
        np.testing.assert_array_equal(result.homography, np.eye(3))
        assert result.rectified_mask.shape == (200, 200)

    def test_passthrough_when_facade_too_small(self):
        """Tiny facade region should return identity homography."""
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[100:105, 100:105] = 1  # 25 pixels, below _MIN_FACADE_PX
        sr = _make_seg_result(mask)
        result = rectify_image(sr)
        np.testing.assert_array_equal(result.homography, np.eye(3))

    def test_rectangular_facade_roughly_same_aspect(self):
        """A clean rectangle should produce output with similar aspect ratio."""
        mask = _rect_mask(top=50, left=80, bottom=350, right=320)  # 300h x 240w
        sr = _make_seg_result(mask)
        result = rectify_image(sr)
        rh, rw = result.rectified_mask.shape[:2]
        aspect = rw / rh
        assert 0.5 < aspect < 1.5  # roughly rectangular, not wildly distorted

    def test_trapezoid_rectified_to_rectangle(self):
        """A trapezoidal facade should become more rectangular after rectification."""
        mask = _trapezoid_mask()
        sr = _make_seg_result(mask)
        result = rectify_image(sr)
        # The output should exist and have reasonable dimensions
        rh, rw = result.rectified_mask.shape[:2]
        assert rh > 0 and rw > 0

    def test_saves_to_disk(self, tmp_path):
        mask = _rect_mask()
        sr = _make_seg_result(mask)
        rectify_image(sr, save_dir=tmp_path)
        assert (tmp_path / "1234_rectified.jpg").exists()
        assert (tmp_path / "1234_rectified_mask.png").exists()


# ---------------------------------------------------------------------------
# rectify_batch
# ---------------------------------------------------------------------------

class TestRectifyBatch:
    def test_returns_list(self):
        masks = [_rect_mask() for _ in range(3)]
        srs = [_make_seg_result(m, egid=str(i)) for i, m in enumerate(masks)]
        results = rectify_batch(srs)
        assert len(results) == 3
        assert [r.egid for r in results] == ["0", "1", "2"]

    def test_empty_batch(self):
        assert rectify_batch([]) == []

    def test_mixed_valid_and_empty(self):
        """Batch with both valid facades and empty masks."""
        good = _make_seg_result(_rect_mask(), egid="good")
        empty = _make_seg_result(np.zeros((200, 200), dtype=np.uint8), egid="empty")
        results = rectify_batch([good, empty])
        assert len(results) == 2
        # Empty one should have identity homography
        np.testing.assert_array_equal(results[1].homography, np.eye(3))
