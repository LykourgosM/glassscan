"""Street View image fetcher for Swiss buildings.

Fetches facade images from the Google Street View Static API,
computing optimal camera heading from panorama position to building centroid.
Supports multi-view fetching: multiple panoramas around a building to handle
occlusion (trees, other buildings, vehicles) and capture different angles.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from glassscan.types import BuildingImage

load_dotenv()

logger = logging.getLogger(__name__)

# API endpoints
_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
_STREETVIEW_URL = "https://maps.googleapis.com/maps/api/streetview"

# Defaults
DEFAULT_SIZE = "640x640"
DEFAULT_FOV = 70
DEFAULT_PITCH = 20
DEFAULT_RADIUS = 50  # metres — search radius for nearest panorama


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Geodesic bearing (degrees, 0=N, clockwise) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360


def _offset_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    """Offset a lat/lon point by a given bearing and distance (metres)."""
    R = 6_371_000  # Earth radius in metres
    d = distance_m / R
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _query_metadata(
    lat: float, lon: float, api_key: str, radius: int,
    source: str = "outdoor",
) -> dict | None:
    """Query the Street View metadata API. Returns the JSON body or None."""
    resp = requests.get(
        _METADATA_URL,
        params={
            "location": f"{lat},{lon}",
            "radius": radius,
            "source": source,
            "key": api_key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK":
        return None
    return data


def get_panorama_location(
    lat: float,
    lon: float,
    api_key: str,
    radius: int = DEFAULT_RADIUS,
) -> tuple[float, float, str] | None:
    """Find the nearest Street View panorama.

    Returns (lat, lon, pano_id) or None if nothing within *radius*.
    Uses the free metadata endpoint (no charge).
    """
    data = _query_metadata(lat, lon, api_key, radius)
    if data is None:
        return None
    loc = data["location"]
    return loc["lat"], loc["lng"], data["pano_id"]


def find_nearby_panoramas(
    lat: float,
    lon: float,
    api_key: str,
    *,
    radius: int = DEFAULT_RADIUS,
    step_m: int = 15,
    n_steps: int = 3,
) -> list[tuple[float, float, str]]:
    """Find multiple unique panoramas along the road nearest to a building.

    Strategy:
    1. Find the nearest panorama to the building (free metadata call).
    2. Infer the road direction as perpendicular to the building→panorama line.
    3. Search at offsets along the road in both directions (*n_steps* each way,
       *step_m* metres apart) to find adjacent panoramas.

    This keeps all viewpoints on the same street, giving different angles
    of the same facade rather than views from unrelated roads.

    All calls use the free metadata endpoint — no billed charges.

    Returns list of (lat, lon, pano_id), nearest panorama first.
    """
    # Step 1: find the nearest panorama
    first = _query_metadata(lat, lon, api_key, radius)
    if first is None:
        return []

    first_loc = first["location"]
    first_lat, first_lon = first_loc["lat"], first_loc["lng"]
    first_id = first["pano_id"]

    seen_ids: set[str] = {first_id}
    results: list[tuple[float, float, str]] = [(first_lat, first_lon, first_id)]

    # Step 2: infer road direction (perpendicular to building→panorama bearing)
    to_building = _bearing(first_lat, first_lon, lat, lon)
    road_bearing = (to_building + 90) % 360

    # Step 3: search along the road in both directions
    for direction in (road_bearing, (road_bearing + 180) % 360):
        for i in range(1, n_steps + 1):
            probe_lat, probe_lon = _offset_point(first_lat, first_lon, direction, step_m * i)
            data = _query_metadata(probe_lat, probe_lon, api_key, radius)
            if data is None:
                break  # no more panoramas in this direction
            pano_id = data["pano_id"]
            if pano_id in seen_ids:
                continue
            seen_ids.add(pano_id)
            loc = data["location"]
            results.append((loc["lat"], loc["lng"], pano_id))

    return results


def _fetch_from_panorama(
    egid: str,
    building_lat: float,
    building_lon: float,
    pano_lat: float,
    pano_lon: float,
    pano_id: str,
    api_key: str,
    view_index: int,
    *,
    fov: int = DEFAULT_FOV,
    pitch: int = DEFAULT_PITCH,
    save_dir: Path | None = None,
) -> BuildingImage | None:
    """Fetch a single image from a known panorama, aimed at the building."""
    heading = _bearing(pano_lat, pano_lon, building_lat, building_lon)

    resp = requests.get(
        _STREETVIEW_URL,
        params={
            "pano": pano_id,
            "size": DEFAULT_SIZE,
            "fov": fov,
            "heading": round(heading, 1),
            "pitch": pitch,
            "key": api_key,
        },
        timeout=15,
    )
    resp.raise_for_status()

    arr = np.frombuffer(resp.content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        logger.warning("Failed to decode image for EGID %s (view %d)", egid, view_index)
        return None

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_v{view_index}" if view_index > 0 else ""
        cv2.imwrite(str(save_dir / f"{egid}{suffix}.jpg"), image)

    return BuildingImage(
        egid=egid,
        image=image,
        lat=building_lat,
        lon=building_lon,
        heading=heading,
        pitch=pitch,
        fov=fov,
        pano_id=pano_id,
        view_index=view_index,
    )


def fetch_image(
    egid: str,
    lat: float,
    lon: float,
    api_key: str,
    *,
    fov: int = DEFAULT_FOV,
    pitch: int = DEFAULT_PITCH,
    radius: int = DEFAULT_RADIUS,
    save_dir: Path | None = None,
) -> BuildingImage | None:
    """Fetch a single Street View image for a building (nearest panorama).

    For multiple viewpoints, use ``fetch_multi_view`` instead.
    """
    pano = get_panorama_location(lat, lon, api_key, radius=radius)
    if pano is None:
        logger.info("No panorama for EGID %s at (%.5f, %.5f)", egid, lat, lon)
        return None

    pano_lat, pano_lon, pano_id = pano
    return _fetch_from_panorama(
        egid, lat, lon, pano_lat, pano_lon, pano_id, api_key, 0,
        fov=fov, pitch=pitch, save_dir=save_dir,
    )


def fetch_multi_view(
    egid: str,
    lat: float,
    lon: float,
    api_key: str,
    *,
    max_views: int = 4,
    fov: int = DEFAULT_FOV,
    pitch: int = DEFAULT_PITCH,
    radius: int = DEFAULT_RADIUS,
    save_dir: Path | None = None,
) -> list[BuildingImage]:
    """Fetch up to *max_views* images of a building from different angles.

    Discovers nearby panoramas along the road using free metadata calls,
    then fetches an image from each unique viewpoint (billed calls). This
    gives resilience against occlusion (trees, vehicles, other buildings)
    and lets downstream modules pick the best view or average across views.

    Returns a list of BuildingImage (may be empty if no panoramas found).
    Each image has a unique ``view_index`` (0, 1, 2, ...) and ``pano_id``.
    """
    panoramas = find_nearby_panoramas(
        lat, lon, api_key, radius=radius,
    )

    if not panoramas:
        logger.info("No panoramas for EGID %s at (%.5f, %.5f)", egid, lat, lon)
        return []

    # Limit to max_views
    panoramas = panoramas[:max_views]

    results = []
    for i, (pano_lat, pano_lon, pano_id) in enumerate(panoramas):
        img = _fetch_from_panorama(
            egid, lat, lon, pano_lat, pano_lon, pano_id, api_key, i,
            fov=fov, pitch=pitch, save_dir=save_dir,
        )
        if img is not None:
            results.append(img)

    return results


def fetch_batch(
    buildings: list[dict],
    api_key: str,
    *,
    save_dir: Path | None = None,
    max_calls: int = 1000,
    max_views: int = 1,
    delay: float = 0.1,
) -> list[BuildingImage]:
    """Fetch Street View images for a batch of buildings.

    Parameters
    ----------
    buildings
        List of dicts, each with keys ``egid``, ``lat``, ``lon``.
    api_key
        Google Maps API key.
    save_dir
        Directory to save images. Pass None to skip saving.
    max_calls
        Hard cap on *billed* API calls (image fetches, not metadata).
    max_views
        How many viewpoints per building. 1 = nearest only (cheapest),
        2-4 = multiple angles (more robust to occlusion).
    delay
        Seconds to sleep between billed calls (rate limiting).

    Returns
    -------
    list[BuildingImage]
        Successfully fetched images. Buildings without panoramas are skipped.
    """
    results: list[BuildingImage] = []
    billed = 0

    for b in buildings:
        if billed >= max_calls:
            logger.warning(
                "Hit max_calls cap (%d). Stopping with %d/%d buildings fetched.",
                max_calls, len(results), len(buildings),
            )
            break

        egid, lat, lon = b["egid"], b["lat"], b["lon"]

        # Skip if primary image already on disk
        if save_dir and (save_dir / f"{egid}.jpg").exists():
            logger.debug("EGID %s already on disk, skipping", egid)
            continue

        remaining = max_calls - billed

        if max_views <= 1:
            img = fetch_image(egid, lat, lon, api_key, save_dir=save_dir)
            if img is not None:
                results.append(img)
                billed += 1
        else:
            views = fetch_multi_view(
                egid, lat, lon, api_key,
                max_views=min(max_views, remaining),
                save_dir=save_dir,
            )
            results.extend(views)
            billed += len(views)

        if delay > 0 and billed > 0:
            time.sleep(delay)

    logger.info(
        "Fetched %d images (%d billed calls, %d buildings total)",
        len(results), billed, len(buildings),
    )
    return results
