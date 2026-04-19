"""Tests for the Street View fetcher.

All tests mock the API — no real network calls or API charges.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest

from glassscan.fetch.fetch import (
    _bearing,
    _offset_point,
    get_panorama_location,
    find_nearby_panoramas,
    fetch_image,
    fetch_multi_view,
    fetch_batch,
)


# ---------------------------------------------------------------------------
# Bearing
# ---------------------------------------------------------------------------

class TestBearing:
    def test_north(self):
        h = _bearing(46.0, 6.0, 47.0, 6.0)
        assert abs(h) < 1 or abs(h - 360) < 1

    def test_east(self):
        h = _bearing(46.0, 6.0, 46.0, 7.0)
        assert 85 < h < 95

    def test_south(self):
        h = _bearing(47.0, 6.0, 46.0, 6.0)
        assert 175 < h < 185

    def test_west(self):
        h = _bearing(46.0, 7.0, 46.0, 6.0)
        assert 265 < h < 275


# ---------------------------------------------------------------------------
# Offset point
# ---------------------------------------------------------------------------

class TestOffsetPoint:
    def test_north_offset(self):
        lat, lon = _offset_point(46.0, 6.0, 0, 100)
        assert lat > 46.0
        assert abs(lon - 6.0) < 0.001

    def test_east_offset(self):
        lat, lon = _offset_point(46.0, 6.0, 90, 100)
        assert abs(lat - 46.0) < 0.001
        assert lon > 6.0

    def test_distance_reasonable(self):
        lat, _ = _offset_point(46.0, 6.0, 0, 50)
        assert 0.0003 < (lat - 46.0) < 0.0006


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_metadata(status: str, lat: float = 0, lon: float = 0, pano_id: str = "abc"):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    body = {"status": status}
    if status == "OK":
        body["location"] = {"lat": lat, "lng": lon}
        body["pano_id"] = pano_id
    resp.json.return_value = body
    return resp


def _make_jpeg_bytes() -> bytes:
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _mock_image_response() -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.content = _make_jpeg_bytes()
    return resp


# ---------------------------------------------------------------------------
# Panorama metadata
# ---------------------------------------------------------------------------

class TestGetPanoramaLocation:
    @patch("glassscan.fetch.fetch.requests.get")
    def test_found(self, mock_get):
        mock_get.return_value = _mock_metadata("OK", 46.5, 6.6, "pano_1")
        result = get_panorama_location(46.5, 6.6, "fake-key")
        assert result == (46.5, 6.6, "pano_1")

    @patch("glassscan.fetch.fetch.requests.get")
    def test_not_found(self, mock_get):
        mock_get.return_value = _mock_metadata("ZERO_RESULTS")
        assert get_panorama_location(46.5, 6.6, "fake-key") is None


# ---------------------------------------------------------------------------
# Find nearby panoramas (road-following)
# ---------------------------------------------------------------------------

class TestFindNearbyPanoramas:
    @patch("glassscan.fetch.fetch.requests.get")
    def test_returns_nearest_first(self, mock_get):
        """First result should always be the nearest panorama."""
        mock_get.return_value = _mock_metadata("OK", 46.501, 6.601, "nearest")
        results = find_nearby_panoramas(46.5, 6.6, "fake-key", n_steps=2)
        assert results[0] == (46.501, 6.601, "nearest")

    @patch("glassscan.fetch.fetch.requests.get")
    def test_no_panorama(self, mock_get):
        mock_get.return_value = _mock_metadata("ZERO_RESULTS")
        assert find_nearby_panoramas(46.5, 6.6, "fake-key") == []

    @patch("glassscan.fetch.fetch.requests.get")
    def test_deduplicates_same_pano(self, mock_get):
        """All probes returning the same pano_id → single result."""
        mock_get.return_value = _mock_metadata("OK", 46.501, 6.601, "same")
        results = find_nearby_panoramas(46.5, 6.6, "fake-key", n_steps=3)
        assert len(results) == 1

    @patch("glassscan.fetch.fetch.requests.get")
    def test_finds_multiple_along_road(self, mock_get):
        """Different pano_ids at road offsets → multiple results."""
        call_count = iter(range(100))
        def side_effect(*args, **kwargs):
            i = next(call_count)
            return _mock_metadata("OK", 46.5 + i * 0.0002, 6.6, f"pano_{i}")
        mock_get.side_effect = side_effect

        results = find_nearby_panoramas(46.5, 6.6, "fake-key", n_steps=2)
        # 1 (nearest) + up to 2 each direction = up to 5, but depends on dedup
        assert len(results) >= 3
        # All unique pano_ids
        ids = [r[2] for r in results]
        assert len(ids) == len(set(ids))

    @patch("glassscan.fetch.fetch.requests.get")
    def test_stops_direction_when_no_pano(self, mock_get):
        """When a probe finds nothing, stop searching that direction."""
        calls = [
            _mock_metadata("OK", 46.501, 6.601, "first"),   # nearest
            _mock_metadata("OK", 46.502, 6.601, "road_1"),  # road +1
            _mock_metadata("ZERO_RESULTS"),                   # road +2 → stop this direction
            _mock_metadata("OK", 46.500, 6.601, "road_r1"), # road -1
            _mock_metadata("OK", 46.499, 6.601, "road_r2"), # road -2
            _mock_metadata("OK", 46.498, 6.601, "road_r3"), # road -3
        ]
        mock_get.side_effect = calls
        results = find_nearby_panoramas(46.5, 6.6, "fake-key", n_steps=3)
        ids = {r[2] for r in results}
        # first + road_1 + (stopped) + road_r1 + road_r2 + road_r3 = 5
        assert ids == {"first", "road_1", "road_r1", "road_r2", "road_r3"}


# ---------------------------------------------------------------------------
# Single image fetch
# ---------------------------------------------------------------------------

class TestFetchImage:
    @patch("glassscan.fetch.fetch.requests.get")
    def test_success(self, mock_get):
        mock_get.side_effect = [
            _mock_metadata("OK", 46.51, 6.61, "pano_x"),
            _mock_image_response(),
        ]
        result = fetch_image("1234", 46.5, 6.6, "fake-key")
        assert result is not None
        assert result.egid == "1234"
        assert result.image.shape == (640, 640, 3)
        assert result.pano_id == "pano_x"
        assert result.view_index == 0

    @patch("glassscan.fetch.fetch.requests.get")
    def test_no_panorama(self, mock_get):
        mock_get.return_value = _mock_metadata("ZERO_RESULTS")
        assert fetch_image("1234", 46.5, 6.6, "fake-key") is None

    @patch("glassscan.fetch.fetch.requests.get")
    def test_saves_to_disk(self, mock_get, tmp_path):
        mock_get.side_effect = [
            _mock_metadata("OK", 46.51, 6.61, "pano_y"),
            _mock_image_response(),
        ]
        result = fetch_image("5678", 46.5, 6.6, "fake-key", save_dir=tmp_path)
        assert result is not None
        assert (tmp_path / "5678.jpg").exists()


# ---------------------------------------------------------------------------
# Multi-view fetch
# ---------------------------------------------------------------------------

class TestFetchMultiView:
    @patch("glassscan.fetch.fetch.requests.get")
    def test_multiple_views(self, mock_get):
        call_count = iter(range(100))
        def side_effect(*args, **kwargs):
            i = next(call_count)
            params = kwargs.get("params", {})
            if "radius" in params:
                return _mock_metadata("OK", 46.5 + i * 0.001, 6.6, f"pano_{i}")
            return _mock_image_response()
        mock_get.side_effect = side_effect

        results = fetch_multi_view("1234", 46.5, 6.6, "fake-key", max_views=3)
        assert len(results) == 3
        assert results[0].view_index == 0
        assert results[1].view_index == 1
        assert results[2].view_index == 2
        pano_ids = {r.pano_id for r in results}
        assert len(pano_ids) == 3

    @patch("glassscan.fetch.fetch.requests.get")
    def test_no_panoramas(self, mock_get):
        mock_get.return_value = _mock_metadata("ZERO_RESULTS")
        assert fetch_multi_view("1234", 46.5, 6.6, "fake-key") == []

    @patch("glassscan.fetch.fetch.requests.get")
    def test_saves_with_view_suffix(self, mock_get, tmp_path):
        call_count = iter(range(100))
        def side_effect(*args, **kwargs):
            i = next(call_count)
            params = kwargs.get("params", {})
            if "radius" in params:
                return _mock_metadata("OK", 46.5 + i * 0.001, 6.6, f"pano_{i}")
            return _mock_image_response()
        mock_get.side_effect = side_effect

        fetch_multi_view("9999", 46.5, 6.6, "fake-key", max_views=2, save_dir=tmp_path)
        assert (tmp_path / "9999.jpg").exists()
        assert (tmp_path / "9999_v1.jpg").exists()


# ---------------------------------------------------------------------------
# Batch fetch
# ---------------------------------------------------------------------------

class TestFetchBatch:
    @patch("glassscan.fetch.fetch.requests.get")
    def test_respects_max_calls(self, mock_get):
        mock_get.side_effect = lambda *a, **kw: (
            _mock_metadata("OK", 46.51, 6.61, "p") if "radius" in kw.get("params", {})
            else _mock_image_response()
        )
        buildings = [{"egid": str(i), "lat": 46.5, "lon": 6.6} for i in range(10)]
        results = fetch_batch(buildings, "fake-key", max_calls=3, delay=0)
        assert len(results) == 3

    @patch("glassscan.fetch.fetch.requests.get")
    def test_cached_building_loaded_from_disk(self, mock_get, tmp_path):
        """Building with image on disk is returned from cache without API calls."""
        import cv2
        import numpy as np
        # Write a valid JPEG for egid "1"
        cached_img = np.full((100, 100, 3), 42, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "1.jpg"), cached_img)

        mock_get.side_effect = lambda *a, **kw: (
            _mock_metadata("OK", 46.51, 6.61, "p") if "radius" in kw.get("params", {})
            else _mock_image_response()
        )
        buildings = [{"egid": str(i), "lat": 46.5, "lon": 6.6} for i in range(3)]
        results = fetch_batch(buildings, "fake-key", save_dir=tmp_path, delay=0)
        # Cached building appears in results alongside the 2 freshly fetched.
        assert len(results) == 3
        assert sorted(r.egid for r in results) == ["0", "1", "2"]
        cached_result = next(r for r in results if r.egid == "1")
        assert cached_result.image.shape == (100, 100, 3)
        assert cached_result.pano_id == ""  # placeholder, since API was skipped

    @patch("glassscan.fetch.fetch.requests.get")
    def test_multi_view_full_cache_skips_api(self, mock_get, tmp_path):
        """If max_views images are on disk, no metadata or image API calls fire."""
        import cv2
        import numpy as np
        img = np.full((100, 100, 3), 7, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "A.jpg"), img)
        cv2.imwrite(str(tmp_path / "A_v1.jpg"), img)
        cv2.imwrite(str(tmp_path / "A_v2.jpg"), img)

        buildings = [{"egid": "A", "lat": 46.5, "lon": 6.6}]
        results = fetch_batch(buildings, "fake-key", save_dir=tmp_path,
                              max_views=3, delay=0)
        assert len(results) == 3
        assert [r.view_index for r in results] == [0, 1, 2]
        mock_get.assert_not_called()

    @patch("glassscan.fetch.fetch.requests.get")
    def test_multi_view_batch(self, mock_get):
        call_count = iter(range(1000))
        def side_effect(*args, **kwargs):
            i = next(call_count)
            params = kwargs.get("params", {})
            if "radius" in params:
                return _mock_metadata("OK", 46.5 + i * 0.001, 6.6, f"pano_{i}")
            return _mock_image_response()
        mock_get.side_effect = side_effect

        buildings = [{"egid": "A", "lat": 46.5, "lon": 6.6}]
        results = fetch_batch(buildings, "fake-key", max_views=3, max_calls=100, delay=0)
        assert len(results) == 3
        assert all(r.egid == "A" for r in results)

    @patch("glassscan.fetch.fetch.requests.get")
    def test_max_calls_caps_multi_view(self, mock_get):
        call_count = iter(range(1000))
        def side_effect(*args, **kwargs):
            i = next(call_count)
            params = kwargs.get("params", {})
            if "radius" in params:
                return _mock_metadata("OK", 46.5 + i * 0.001, 6.6, f"pano_{i}")
            return _mock_image_response()
        mock_get.side_effect = side_effect

        buildings = [
            {"egid": "A", "lat": 46.5, "lon": 6.6},
            {"egid": "B", "lat": 46.6, "lon": 6.7},
        ]
        results = fetch_batch(buildings, "fake-key", max_views=3, max_calls=4, delay=0)
        assert len(results) == 4
