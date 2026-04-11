"""Shared data types for the GlassScan pipeline.

Every module consumes and produces these types, making modules
independently swappable without changing downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BuildingImage:
    """Raw facade image fetched from Street View."""

    egid: str  # Federal Building Identifier
    image: np.ndarray  # H x W x 3, BGR (OpenCV default)
    lat: float
    lon: float
    heading: float  # compass bearing used for the request
    pitch: float
    fov: float
    pano_id: str = ""  # Street View panorama ID (for deduplication)
    view_index: int = 0  # 0 = primary view, 1+ = additional angles


@dataclass
class SegmentationResult:
    """Per-pixel semantic segmentation of a facade image."""

    egid: str
    mask: np.ndarray  # H x W, uint8 — 0=background, 1=wall, 2=window
    confidence: float  # mean model confidence across facade pixels
    original_image: np.ndarray  # the input image, kept for visualisation
    view_index: int = 0  # 0 = primary view, 1+ = additional angles


@dataclass
class RectifiedResult:
    """Perspective-corrected facade with its segmentation mask."""

    egid: str
    rectified_image: np.ndarray  # H x W x 3
    rectified_mask: np.ndarray  # H x W, same classes as SegmentationResult.mask
    homography: np.ndarray  # 3x3 transformation matrix
    view_index: int = 0  # 0 = primary view, 1+ = additional angles


@dataclass
class WWRResult:
    """Computed window-to-wall ratio for one building."""

    egid: str
    wwr: float  # 0.0–1.0
    window_area_px: int
    wall_area_px: int
    n_windows: int  # count of distinct window regions
    confidence: float
    view_index: int = 0  # 0 = primary view, 1+ = additional angles


@dataclass
class BuildingFeatures:
    """Metadata features for a building, used as regression input."""

    egid: str
    construction_year: int | None
    building_category: str | None  # e.g. "residential", "commercial"
    canton: str | None
    floor_count: int | None
    heating_type: str | None
    lat: float
    lon: float


@dataclass
class PredictionResult:
    """Predicted WWR for a building without Street View imagery."""

    egid: str
    predicted_wwr: float
    prediction_interval: tuple[float, float]  # (lower, upper) 90% CI
    features_used: list[str] = field(default_factory=list)
