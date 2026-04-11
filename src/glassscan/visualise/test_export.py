"""Tests for the visualise export module."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from glassscan.types import (
    BuildingImage,
    SegmentationResult,
    RectifiedResult,
    WWRResult,
    PredictionResult,
)
from glassscan.pipeline import PipelineResult
from glassscan.visualise.export import create_building_card, export_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_image(egid: str, h: int = 64, w: int = 64) -> BuildingImage:
    return BuildingImage(
        egid=egid,
        image=np.full((h, w, 3), 128, dtype=np.uint8),
        lat=47.37, lon=8.54,
        heading=180.0, pitch=20.0, fov=70.0,
    )


def _fake_seg(egid: str, h: int = 64, w: int = 64) -> SegmentationResult:
    mask = np.ones((h, w), dtype=np.uint8)  # all wall
    mask[10:30, 10:30] = 2  # window region
    return SegmentationResult(
        egid=egid, mask=mask, confidence=0.9,
        original_image=np.full((h, w, 3), 128, dtype=np.uint8),
    )


def _fake_rectified(egid: str) -> RectifiedResult:
    return RectifiedResult(
        egid=egid,
        rectified_image=np.zeros((64, 64, 3), dtype=np.uint8),
        rectified_mask=np.ones((64, 64), dtype=np.uint8),
        homography=np.eye(3),
    )


def _fake_wwr(egid: str, wwr: float = 0.25) -> WWRResult:
    return WWRResult(
        egid=egid, wwr=wwr, window_area_px=400,
        wall_area_px=1200, n_windows=5, confidence=0.9,
    )


def _fake_prediction(egid: str, wwr: float = 0.30) -> PredictionResult:
    return PredictionResult(
        egid=egid, predicted_wwr=wwr,
        prediction_interval=(wwr - 0.05, wwr + 0.05),
    )


def _make_metadata(egids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "egid": egids,
        "lat": [47.37 + i * 0.01 for i in range(len(egids))],
        "lon": [8.54 + i * 0.01 for i in range(len(egids))],
        "construction_year": [1920, 1965, 1990, 2010, 2020][:len(egids)],
        "building_category": ["residential", "commercial", "residential",
                              "commercial", "industrial"][:len(egids)],
    })


# ---------------------------------------------------------------------------
# create_building_card
# ---------------------------------------------------------------------------

class TestCreateBuildingCard:
    def test_output_shape(self):
        img = np.zeros((100, 80, 3), dtype=np.uint8)
        mask = np.zeros((100, 80), dtype=np.uint8)
        card = create_building_card(img, mask, 0.25)
        assert card.shape == (100, 160, 3)  # H x 2W x 3

    def test_overlay_has_colors(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.uint8)  # all wall
        mask[20:40, 20:40] = 2  # window
        card = create_building_card(img, mask, 0.3)

        # Right half should have overlay colors blended in
        right_half = card[:, 64:, :]
        # Window area should be greener than the original gray
        window_region = right_half[20:40, 20:40, :]
        assert window_region[:, :, 1].mean() > 128  # green channel boosted

    def test_handles_empty_mask(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)  # all background
        card = create_building_card(img, mask, 0.0)
        assert card.shape == (50, 100, 3)


# ---------------------------------------------------------------------------
# export_results
# ---------------------------------------------------------------------------

class TestExportResults:
    def test_creates_json_and_images(self, tmp_path):
        result = PipelineResult(
            images=[_fake_image("E1")],
            segmentations=[_fake_seg("E1")],
            rectified=[_fake_rectified("E1")],
            wwr_results=[_fake_wwr("E1", 0.25)],
        )

        export_results(result, tmp_path)

        assert (tmp_path / "buildings.json").exists()
        assert (tmp_path / "images" / "E1.jpg").exists()

    def test_json_structure(self, tmp_path):
        result = PipelineResult(
            images=[_fake_image("E1")],
            segmentations=[_fake_seg("E1")],
            rectified=[_fake_rectified("E1")],
            wwr_results=[_fake_wwr("E1", 0.25)],
        )

        export_results(result, tmp_path)

        with open(tmp_path / "buildings.json") as f:
            data = json.load(f)

        assert "buildings" in data
        assert "stats" in data
        assert len(data["buildings"]) == 1

        b = data["buildings"][0]
        assert b["egid"] == "E1"
        assert b["source"] == "measured"
        assert b["wwr"] == 0.25
        assert b["confidence"] == 0.9
        assert b["n_windows"] == 5

    def test_measured_and_predicted(self, tmp_path):
        meta = _make_metadata(["E1", "P1"])
        result = PipelineResult(
            images=[_fake_image("E1")],
            segmentations=[_fake_seg("E1")],
            rectified=[_fake_rectified("E1")],
            wwr_results=[_fake_wwr("E1", 0.20)],
            predictions=[_fake_prediction("P1", 0.35)],
        )

        export_results(result, tmp_path, metadata_df=meta)

        with open(tmp_path / "buildings.json") as f:
            data = json.load(f)

        assert len(data["buildings"]) == 2
        sources = {b["source"] for b in data["buildings"]}
        assert sources == {"measured", "predicted"}

        pred = [b for b in data["buildings"] if b["source"] == "predicted"][0]
        assert pred["prediction_interval"] is not None
        assert len(pred["prediction_interval"]) == 2

    def test_stats_computed(self, tmp_path):
        result = PipelineResult(
            images=[_fake_image("E1")],
            segmentations=[_fake_seg("E1")],
            rectified=[_fake_rectified("E1")],
            wwr_results=[_fake_wwr("E1", 0.25)],
        )

        export_results(result, tmp_path)

        with open(tmp_path / "buildings.json") as f:
            data = json.load(f)

        stats = data["stats"]
        assert stats["total"] == 1
        assert stats["measured"] == 1
        assert stats["predicted"] == 0
        assert stats["mean_wwr"] == 0.25

    def test_metadata_attached(self, tmp_path):
        meta = _make_metadata(["E1"])
        result = PipelineResult(
            images=[_fake_image("E1")],
            segmentations=[_fake_seg("E1")],
            rectified=[_fake_rectified("E1")],
            wwr_results=[_fake_wwr("E1")],
        )

        export_results(result, tmp_path, metadata_df=meta)

        with open(tmp_path / "buildings.json") as f:
            data = json.load(f)

        b = data["buildings"][0]
        assert b["metadata"]["construction_year"] == 1920
        assert b["metadata"]["building_category"] == "residential"

    def test_wwr_by_era_stats(self, tmp_path):
        egids = ["E1", "E2"]
        meta = _make_metadata(egids)
        result = PipelineResult(
            images=[_fake_image(e) for e in egids],
            segmentations=[_fake_seg(e) for e in egids],
            rectified=[_fake_rectified(e) for e in egids],
            wwr_results=[_fake_wwr("E1", 0.15), _fake_wwr("E2", 0.30)],
        )

        export_results(result, tmp_path, metadata_df=meta)

        with open(tmp_path / "buildings.json") as f:
            data = json.load(f)

        assert "pre-1950" in data["stats"]["wwr_by_era"]
        assert "1950-1980" in data["stats"]["wwr_by_era"]

    def test_empty_results(self, tmp_path):
        result = PipelineResult()
        export_results(result, tmp_path)

        with open(tmp_path / "buildings.json") as f:
            data = json.load(f)

        assert data["buildings"] == []
        assert data["stats"]["total"] == 0

    def test_predicted_coords_from_metadata(self, tmp_path):
        meta = pd.DataFrame({
            "egid": ["P1"],
            "lat": [46.5],
            "lon": [7.0],
            "construction_year": [2000],
        })
        result = PipelineResult(
            predictions=[_fake_prediction("P1")],
        )

        export_results(result, tmp_path, metadata_df=meta)

        with open(tmp_path / "buildings.json") as f:
            data = json.load(f)

        b = data["buildings"][0]
        assert b["lat"] == 46.5
        assert b["lon"] == 7.0

    def test_rejects_wrong_type(self, tmp_path):
        with pytest.raises(TypeError):
            export_results({"not": "a PipelineResult"}, tmp_path)

    def test_card_image_readable(self, tmp_path):
        result = PipelineResult(
            images=[_fake_image("E1")],
            segmentations=[_fake_seg("E1")],
            rectified=[_fake_rectified("E1")],
            wwr_results=[_fake_wwr("E1")],
        )

        export_results(result, tmp_path)

        card = cv2.imread(str(tmp_path / "images" / "E1.jpg"))
        assert card is not None
        assert card.shape[1] == 128  # 2 * 64
