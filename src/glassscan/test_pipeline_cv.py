"""CV pipeline orchestration tests (mock all stages).

Separated from test_pipeline.py because the segment module imports
PyTorch, which conflicts with XGBoost's libomp on macOS. These tests
must not run in the same process as XGBoost training.

Run separately: pytest src/glassscan/test_pipeline_cv.py
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from glassscan.types import (
    BuildingImage,
    SegmentationResult,
    RectifiedResult,
    WWRResult,
)
from glassscan.pipeline import run_cv_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_image(egid: str) -> BuildingImage:
    return BuildingImage(
        egid=egid,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        lat=47.0, lon=8.5,
        heading=180.0, pitch=20.0, fov=70.0,
    )


def _fake_segmentation(egid: str) -> SegmentationResult:
    mask = np.ones((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 2
    return SegmentationResult(
        egid=egid, mask=mask, confidence=0.9,
        original_image=np.zeros((64, 64, 3), dtype=np.uint8),
    )


def _fake_rectified(egid: str) -> RectifiedResult:
    return RectifiedResult(
        egid=egid,
        rectified_image=np.zeros((64, 64, 3), dtype=np.uint8),
        rectified_mask=np.ones((64, 64), dtype=np.uint8),
        homography=np.eye(3),
    )


def _fake_wwr(egid: str) -> WWRResult:
    return WWRResult(
        egid=egid, wwr=0.25, window_area_px=400,
        wall_area_px=1200, n_windows=2, confidence=0.85,
    )


def _make_buildings(n: int = 5) -> list[dict]:
    return [
        {"egid": f"E{i}", "lat": 47.0 + i * 0.01, "lon": 8.5 + i * 0.01}
        for i in range(n)
    ]


# Patch targets (source modules)
_FETCH = "glassscan.fetch.fetch_batch"
_SEGMENT = "glassscan.segment.segment_batch"
_RECTIFY = "glassscan.rectify.rectify_batch"
_WWR = "glassscan.wwr.compute_wwr_batch"


class TestRunCVPipeline:
    @patch(_WWR)
    @patch(_RECTIFY)
    @patch(_SEGMENT)
    @patch(_FETCH)
    def test_chains_all_stages(self, mock_fetch, mock_seg, mock_rect, mock_wwr):
        buildings = _make_buildings(3)
        egids = [b["egid"] for b in buildings]

        mock_fetch.return_value = [_fake_image(e) for e in egids]
        mock_seg.return_value = [_fake_segmentation(e) for e in egids]
        mock_rect.return_value = [_fake_rectified(e) for e in egids]
        mock_wwr.return_value = [_fake_wwr(e) for e in egids]

        result = run_cv_pipeline(buildings, "fake-key")

        assert len(result.images) == 3
        assert len(result.segmentations) == 3
        assert len(result.rectified) == 3
        assert len(result.wwr_results) == 3
        mock_fetch.assert_called_once()
        mock_seg.assert_called_once()
        mock_rect.assert_called_once()
        mock_wwr.assert_called_once()

    @patch(_FETCH)
    def test_stops_on_no_images(self, mock_fetch):
        mock_fetch.return_value = []
        result = run_cv_pipeline(_make_buildings(1), "fake-key")
        assert result.images == []
        assert result.segmentations == []

    @patch(_SEGMENT)
    @patch(_FETCH)
    def test_stops_on_no_segmentations(self, mock_fetch, mock_seg):
        mock_fetch.return_value = [_fake_image("E0")]
        mock_seg.return_value = []
        result = run_cv_pipeline(_make_buildings(1), "fake-key")
        assert len(result.images) == 1
        assert result.rectified == []

    @patch(_RECTIFY)
    @patch(_SEGMENT)
    @patch(_FETCH)
    def test_stops_on_no_rectified(self, mock_fetch, mock_seg, mock_rect):
        mock_fetch.return_value = [_fake_image("E0")]
        mock_seg.return_value = [_fake_segmentation("E0")]
        mock_rect.return_value = []
        result = run_cv_pipeline(_make_buildings(1), "fake-key")
        assert len(result.segmentations) == 1
        assert result.wwr_results == []

    def test_empty_buildings(self):
        result = run_cv_pipeline([], "fake-key")
        assert result.images == []

    @patch(_WWR)
    @patch(_RECTIFY)
    @patch(_SEGMENT)
    @patch(_FETCH)
    def test_passes_save_dir(self, mock_fetch, mock_seg, mock_rect, mock_wwr, tmp_path):
        mock_fetch.return_value = [_fake_image("E0")]
        mock_seg.return_value = [_fake_segmentation("E0")]
        mock_rect.return_value = [_fake_rectified("E0")]
        mock_wwr.return_value = [_fake_wwr("E0")]

        run_cv_pipeline(_make_buildings(1), "fake-key", save_dir=tmp_path)

        _, kwargs = mock_fetch.call_args
        assert kwargs["save_dir"] == tmp_path / "raw"
        _, kwargs = mock_seg.call_args
        assert kwargs["save_dir"] == tmp_path / "masks"
        _, kwargs = mock_rect.call_args
        assert kwargs["save_dir"] == tmp_path / "rectified"

    @patch(_WWR)
    @patch(_RECTIFY)
    @patch(_SEGMENT)
    @patch(_FETCH)
    def test_passes_api_params(self, mock_fetch, mock_seg, mock_rect, mock_wwr):
        mock_fetch.return_value = [_fake_image("E0")]
        mock_seg.return_value = [_fake_segmentation("E0")]
        mock_rect.return_value = [_fake_rectified("E0")]
        mock_wwr.return_value = [_fake_wwr("E0")]

        run_cv_pipeline(_make_buildings(1), "my-key", max_views=3, max_api_calls=500)

        _, kwargs = mock_fetch.call_args
        assert kwargs["max_views"] == 3
        assert kwargs["max_calls"] == 500
