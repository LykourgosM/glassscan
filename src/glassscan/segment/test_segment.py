"""Tests for the two-stage facade segmentation module.

All tests use fake models — no downloads or GPU required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch

from glassscan.segment.segment import (
    _CMP_REMAP,
    _preprocess,
    _get_device,
    segment_image,
    segment_batch,
    CMP_CLASSES,
    SegmentModels,
)
from glassscan.types import BuildingImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_building_image(
    egid: str = "1234",
    h: int = 640,
    w: int = 640,
) -> BuildingImage:
    """Create a dummy BuildingImage with a random BGR image."""
    rng = np.random.RandomState(42)
    image = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return BuildingImage(
        egid=egid, image=image,
        lat=46.5, lon=6.6, heading=180.0, pitch=20.0, fov=70.0,
    )


def _make_fake_ade_model(predicted_class: int = 1):
    """Fake ADE20K model. Default predicts class 1 (building) everywhere."""
    def forward(pixel_values):
        B = pixel_values.shape[0]
        logits = torch.zeros(B, 150, 160, 160)
        logits[:, predicted_class, :, :] = 10.0
        result = MagicMock()
        result.logits = logits
        return result

    model = MagicMock()
    model.side_effect = forward
    model.__call__ = forward
    return model


def _make_fake_cmp_model(predicted_class: int = 2):
    """Fake CMP model. Default predicts class 2 (facade) everywhere."""
    def forward(pixel_values):
        B = pixel_values.shape[0]
        logits = torch.zeros(B, len(CMP_CLASSES), 160, 160)
        logits[:, predicted_class, :, :] = 10.0
        result = MagicMock()
        result.logits = logits
        return result

    model = MagicMock()
    model.side_effect = forward
    model.__call__ = forward
    return model


def _make_models(ade_class: int = 1, cmp_class: int = 2) -> SegmentModels:
    """Create a SegmentModels bundle with fake models."""
    return SegmentModels(
        ade_model=_make_fake_ade_model(ade_class),
        cmp_model=_make_fake_cmp_model(cmp_class),
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_output_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out = _preprocess(img)
        assert out.shape == (3, 640, 640)

    def test_output_range(self):
        img = np.full((640, 640, 3), 128, dtype=np.uint8)
        out = _preprocess(img)
        assert out.min() >= 0.0
        assert out.max() <= 1.0
        assert abs(out.mean() - 128.0 / 255.0) < 0.01

    def test_dtype(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        out = _preprocess(img)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Remap table
# ---------------------------------------------------------------------------

class TestRemap:
    def test_length(self):
        assert len(_CMP_REMAP) == len(CMP_CLASSES)

    def test_background_classes(self):
        assert _CMP_REMAP[0] == 0  # unknown
        assert _CMP_REMAP[1] == 0  # background

    def test_wall_classes(self):
        for i in [2, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            assert _CMP_REMAP[i] == 1, f"CMP class {i} ({CMP_CLASSES[i]}) should map to wall (1)"

    def test_window_class(self):
        assert _CMP_REMAP[3] == 2  # window

    def test_only_three_output_classes(self):
        assert set(_CMP_REMAP) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Two-stage segmentation
# ---------------------------------------------------------------------------

class TestSegmentImage:
    def test_returns_segmentation_result(self):
        bi = _make_building_image()
        models = _make_models(ade_class=1, cmp_class=2)  # building + facade
        result = segment_image(bi, models)
        assert result.egid == "1234"
        assert result.mask.shape == (640, 640)
        assert result.mask.dtype == np.uint8
        assert result.original_image is bi.image

    def test_building_with_facade_maps_to_wall(self):
        bi = _make_building_image()
        models = _make_models(ade_class=1, cmp_class=2)  # ADE=building, CMP=facade→wall
        result = segment_image(bi, models)
        assert np.all(result.mask == 1)

    def test_building_with_window_maps_to_window(self):
        bi = _make_building_image()
        models = _make_models(ade_class=1, cmp_class=3)  # ADE=building, CMP=window
        result = segment_image(bi, models)
        assert np.all(result.mask == 2)

    def test_non_building_always_background(self):
        """When ADE says not-building, CMP result is ignored."""
        bi = _make_building_image()
        models = _make_models(ade_class=2, cmp_class=3)  # ADE=sky, CMP=window
        result = segment_image(bi, models)
        # Sky is not in _ADE_BUILDING_IDS, so everything should be background
        assert np.all(result.mask == 0)

    def test_confidence_is_float(self):
        bi = _make_building_image()
        models = _make_models(ade_class=1, cmp_class=3)
        result = segment_image(bi, models)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_zero_when_no_building(self):
        bi = _make_building_image()
        models = _make_models(ade_class=2, cmp_class=3)  # ADE=sky (no building)
        result = segment_image(bi, models)
        assert result.confidence == 0.0

    def test_non_square_image_resized(self):
        bi = _make_building_image(h=480, w=320)
        models = _make_models(ade_class=1, cmp_class=2)
        result = segment_image(bi, models)
        assert result.mask.shape == (480, 320)

    def test_saves_mask_to_disk(self, tmp_path):
        bi = _make_building_image()
        models = _make_models(ade_class=1, cmp_class=3)
        segment_image(bi, models, save_dir=tmp_path)
        assert (tmp_path / "1234_mask.png").exists()


# ---------------------------------------------------------------------------
# Batch segmentation
# ---------------------------------------------------------------------------

class TestSegmentBatch:
    def test_batch_returns_list(self):
        images = [_make_building_image(egid=str(i)) for i in range(3)]
        models = _make_models(ade_class=1, cmp_class=2)
        results = segment_batch(images, models)
        assert len(results) == 3
        assert [r.egid for r in results] == ["0", "1", "2"]

    def test_batch_size_chunking(self):
        images = [_make_building_image(egid=str(i)) for i in range(5)]
        models = _make_models(ade_class=1, cmp_class=3)
        results = segment_batch(images, models, batch_size=2)
        assert len(results) == 5

    def test_empty_batch(self):
        models = _make_models()
        results = segment_batch([], models)
        assert results == []

    def test_masking_applied_in_batch(self):
        """Non-building ADE prediction zeroes out CMP window prediction."""
        images = [_make_building_image(egid="A")]
        models = _make_models(ade_class=2, cmp_class=3)  # ADE=sky, CMP=window
        results = segment_batch(images, models)
        assert np.all(results[0].mask == 0)


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_second_run_skips_inference(self, tmp_path):
        """Re-running with a populated cache should not invoke the models."""
        bi = _make_building_image(egid="9001")
        models = _make_models(ade_class=1, cmp_class=3)
        first = segment_image(bi, models, save_dir=tmp_path)

        # New models instance; if the cache works, forward() is never called.
        cold_models = _make_models(ade_class=1, cmp_class=3)
        cold_models.ade_model.side_effect = RuntimeError("cache miss — ADE called")
        cold_models.cmp_model.side_effect = RuntimeError("cache miss — CMP called")

        second = segment_image(bi, cold_models, save_dir=tmp_path)
        assert np.array_equal(second.mask, first.mask)
        assert second.confidence == pytest.approx(first.confidence)

    def test_batch_mixes_cached_and_fresh(self, tmp_path):
        """A batch with one cached image + two new ones returns all three in order."""
        a = _make_building_image(egid="A")
        models = _make_models(ade_class=1, cmp_class=3)
        segment_image(a, models, save_dir=tmp_path)  # warm cache for A only

        b = _make_building_image(egid="B")
        c = _make_building_image(egid="C")
        results = segment_batch([a, b, c], models, save_dir=tmp_path)
        assert [r.egid for r in results] == ["A", "B", "C"]
        # All three masks now cached; re-running returns identical values.
        again = segment_batch([a, b, c], models, save_dir=tmp_path)
        for r1, r2 in zip(results, again):
            assert np.array_equal(r1.mask, r2.mask)
            assert r1.confidence == pytest.approx(r2.confidence)

    def test_cache_key_respects_view_index(self, tmp_path):
        """v0 and v1 of the same egid are cached separately."""
        bi0 = _make_building_image(egid="42")
        bi1 = BuildingImage(
            egid="42", image=bi0.image, lat=bi0.lat, lon=bi0.lon,
            heading=bi0.heading, pitch=bi0.pitch, fov=bi0.fov, view_index=1,
        )
        models = _make_models(ade_class=1, cmp_class=3)
        segment_image(bi0, models, save_dir=tmp_path)
        segment_image(bi1, models, save_dir=tmp_path)
        assert (tmp_path / "42_mask.png").exists()
        assert (tmp_path / "42_v1_mask.png").exists()
        assert (tmp_path / "metadata.json").exists()

    def test_missing_metadata_triggers_reinference(self, tmp_path):
        """Mask PNG without a metadata entry should not be used as cache."""
        bi = _make_building_image(egid="7")
        models = _make_models(ade_class=1, cmp_class=3)
        segment_image(bi, models, save_dir=tmp_path)
        (tmp_path / "metadata.json").unlink()  # mask survives, metadata gone
        # Should re-run inference rather than load a cached result silently.
        models2 = _make_models(ade_class=1, cmp_class=3)
        segment_image(bi, models2, save_dir=tmp_path)
        models2.ade_model.assert_called()


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_returns_torch_device(self):
        d = _get_device()
        assert isinstance(d, torch.device)
