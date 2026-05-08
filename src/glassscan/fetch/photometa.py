"""Google Street View photometa endpoint helper.

The Static API metadata endpoint reports a snapped/reverse-geocoded position
for each panorama, which can differ from the actual capture position by ~10m
in Zurich Altstadt 2021-vintage panos. The undocumented photometa endpoint
returns the raw capture position (and pano yaw) used by Google's image
renderer, so projections aligned to photometa coords match the rendered
image content.

Schema (JSPB nested arrays):
    data[1][5][0][1][2:4]  -> (lat, lon)
    data[1][5][0][3][0]    -> pano_yaw (degrees, 0=N)
    data[1][6][7]          -> [year, month] imagery date (when present)

Endpoint is undocumented and may change without notice. On any parse or
network failure, returns None and the caller should fall back to the
Static API position.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

_PHOTOMETA_URL = "https://www.google.com/maps/photometa/v1"

# pb= template for the "find panorama by ID" request. Derived from the
# streetlevel package's URL builder (which tracks the schema as Google
# updates it). The {pano_id} placeholder is URL-encoded before substitution.
_PB_TEMPLATE = (
    "!1m4!1smaps_sv.tactile"
    "!11m2!2m1!1b1"
    "!2m2!1sen!2sen"
    "!3m3!1m2!1e2!2s{pano_id}"
    "!4m54!1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12"
    "!2m1!1e1!4m1!1i48"
    "!5m0!6m0"
    "!9m36"
    "!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3"
    "!1m3!1e3!2b1!3e2!1m3!1e3!2b0!3e3"
    "!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3"
    "!1m3!1e4!2b0!3e3"
    "!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3"
    "!11m2!3m1!4b1"
)


def _sanitize_pano_id(pano_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in pano_id)


def _extract_fields(data: list, pano_id: str) -> dict | None:
    """Pull lat/lon/yaw/date out of the JSPB structure. None on shape error.

    Schema (current as of 2026-05):
        data[1][0][5][0][1][0]    -> [null, null, lat, lon]
        data[1][0][5][0][1][2][0] -> pano_yaw (first of yaw/pitch/roll)
        data[1][0][6][7]          -> [year, month]
    """
    try:
        pano = data[1][0]
        if pano is None:
            return None
        position_block = pano[5][0][1]
        position = position_block[0]
        lat, lon = position[2], position[3]
        yaw = position_block[2][0]
        try:
            date = pano[6][7]
            year, month = date[0], date[1]
        except (IndexError, TypeError):
            year, month = None, None
        return {
            "pano_id": pano_id,
            "lat": float(lat),
            "lon": float(lon),
            "pano_yaw": float(yaw),
            "imagery_year": year,
            "imagery_month": month,
        }
    except (IndexError, KeyError, TypeError) as exc:
        logger.warning("photometa parse failed for %s: %s", pano_id, exc)
        return None


def query_photometa(
    pano_id: str,
    cache_dir: Path | str | None = None,
    timeout: int = 10,
) -> dict | None:
    """Query the photometa endpoint for a single pano.

    Returns a dict with keys (pano_id, lat, lon, pano_yaw, imagery_year,
    imagery_month) on success, None on failure. Disk-caches the raw JSPB
    response when ``cache_dir`` is given.
    """
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{_sanitize_pano_id(pano_id)}.json"
        if cache_path.exists() and cache_path.stat().st_size > 100:
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                return _extract_fields(data, pano_id)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "photometa cache read failed for %s: %s; refetching",
                    pano_id, exc,
                )

    pb = _PB_TEMPLATE.format(pano_id=quote(pano_id, safe=""))
    url = f"{_PHOTOMETA_URL}?authuser=0&hl=en&gl=us&pb={pb}"

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 Chrome/124.0"},
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.text
    except requests.RequestException as exc:
        logger.warning("photometa fetch failed for %s: %s", pano_id, exc)
        return None

    if text.startswith(")]}'"):
        text = text.split("\n", 1)[1] if "\n" in text else text[len(")]}'"):]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("photometa decode failed for %s: %s", pano_id, exc)
        return None

    if cache_path is not None:
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except OSError as exc:
            logger.warning("photometa cache write failed for %s: %s", pano_id, exc)

    return _extract_fields(data, pano_id)
