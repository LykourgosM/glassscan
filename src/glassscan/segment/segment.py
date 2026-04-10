"""Two-stage facade segmentation: ADE20K building mask + CMP wall/window.

Stage 1: SegFormer-B5 (ADE20K) identifies building pixels vs background
         (sky, trees, pavement, people, etc.)
Stage 2: SegFormer-B5 (CMP Facades) segments wall vs window within the
         building region only.

This avoids the CMP model misclassifying non-building objects as wall.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from glassscan.types import BuildingImage, SegmentationResult

logger = logging.getLogger(__name__)

# ── CMP Facades class mapping ───────────────────────────────────────────────

CMP_CLASSES = [
    "unknown",     # 0
    "background",  # 1
    "facade",      # 2
    "window",      # 3
    "door",        # 4
    "cornice",     # 5
    "sill",        # 6
    "balcony",     # 7
    "blind",       # 8
    "molding",     # 9
    "deco",        # 10
    "pillar",      # 11
    "shop",        # 12
]

# CMP → pipeline: 0=background, 1=wall, 2=window
_CMP_REMAP = np.array([
    0,  # unknown    → background
    0,  # background → background
    1,  # facade     → wall
    2,  # window     → window
    1,  # door       → wall
    1,  # cornice    → wall
    1,  # sill       → wall
    1,  # balcony    → wall
    1,  # blind      → wall
    1,  # molding    → wall
    1,  # deco       → wall
    1,  # pillar     → wall
    1,  # shop       → wall
], dtype=np.uint8)

# ADE20K class IDs that count as "building"
_ADE_BUILDING_IDS = np.array([0, 1])  # wall, building/edifice

# HuggingFace model identifiers
_BASE_MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"
_WEIGHTS_REPO = "aycaduran/cmp_segformer"
_WEIGHTS_FILE = "segformer_b5_cmp_best.pt"

_INPUT_SIZE = 640


@dataclass
class SegmentModels:
    """Both models needed for two-stage segmentation."""

    ade_model: SegformerForSemanticSegmentation  # stage 1: building mask
    cmp_model: SegformerForSemanticSegmentation  # stage 2: wall/window
    device: torch.device


def _get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    device: torch.device | str | None = None,
) -> SegmentModels:
    """Load both segmentation models.

    Downloads weights from HuggingFace on first call (~680 MB total).

    Returns a SegmentModels bundle.
    """
    if device is None:
        device = _get_device()
    device = torch.device(device) if isinstance(device, str) else device

    logger.info("Loading segmentation models on %s ...", device)

    # Stage 1: ADE20K model (building detection)
    ade_model = SegformerForSemanticSegmentation.from_pretrained(_BASE_MODEL)
    ade_model = ade_model.to(device).eval()
    logger.info("ADE20K model loaded")

    # Stage 2: CMP model (wall/window within building)
    config = SegformerConfig.from_pretrained(_BASE_MODEL)
    config.num_labels = len(CMP_CLASSES)
    config.id2label = {i: c for i, c in enumerate(CMP_CLASSES)}
    config.label2id = {c: i for i, c in enumerate(CMP_CLASSES)}

    cmp_model = SegformerForSemanticSegmentation.from_pretrained(
        _BASE_MODEL, config=config, ignore_mismatched_sizes=True,
    )

    from huggingface_hub import hf_hub_download

    weights_path = hf_hub_download(repo_id=_WEIGHTS_REPO, filename=_WEIGHTS_FILE)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    cmp_model.load_state_dict(state, strict=False)
    cmp_model = cmp_model.to(device).eval()
    logger.info("CMP Facades model loaded")

    return SegmentModels(ade_model=ade_model, cmp_model=cmp_model, device=device)


def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 image → float32 CHW tensor data, normalised to [0, 1]."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    return chw


def _run_twostage(
    pixel_values: torch.Tensor,
    models: SegmentModels,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run both models on a single preprocessed image tensor.

    Returns (pipeline_mask, building_mask, cmp_probs) all at _INPUT_SIZE.
    """
    # Stage 1: ADE20K building mask
    ade_logits = models.ade_model(pixel_values=pixel_values).logits
    ade_up = F.interpolate(
        ade_logits, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False,
    )
    ade_pred = ade_up.argmax(dim=1)[0].cpu().numpy()
    building_mask = np.isin(ade_pred, _ADE_BUILDING_IDS)

    # Stage 2: CMP wall/window
    cmp_logits = models.cmp_model(pixel_values=pixel_values).logits
    cmp_up = F.interpolate(
        cmp_logits, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False,
    )
    cmp_probs = F.softmax(cmp_up, dim=1)[0].cpu().numpy()  # [13, 640, 640]
    cmp_pred = cmp_up.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    cmp_mask = _CMP_REMAP[cmp_pred]

    # Combine: CMP result only where ADE says building
    pipeline_mask = np.zeros((_INPUT_SIZE, _INPUT_SIZE), dtype=np.uint8)
    pipeline_mask[building_mask] = cmp_mask[building_mask]

    return pipeline_mask, building_mask, cmp_probs


def _compute_confidence(
    cmp_probs: np.ndarray,
    building_mask: np.ndarray,
) -> float:
    """Mean max-class probability over building pixels."""
    max_probs = cmp_probs.max(axis=0)  # [640, 640]
    if building_mask.any():
        return float(max_probs[building_mask].mean())
    return 0.0


@torch.no_grad()
def segment_image(
    building_image: BuildingImage,
    models: SegmentModels,
    *,
    save_dir: Path | None = None,
) -> SegmentationResult:
    """Segment a single facade image into wall / window / background.

    Uses two-stage approach: ADE20K for building mask, CMP for wall/window.
    Returns a SegmentationResult with the 3-class mask (same H x W as input).
    """
    img = building_image.image
    h, w = img.shape[:2]

    pixel_values = torch.from_numpy(_preprocess(img)).unsqueeze(0).to(models.device)
    pipeline_mask, building_mask, cmp_probs = _run_twostage(pixel_values, models)

    # Resize mask to original image dimensions
    if (h, w) != (_INPUT_SIZE, _INPUT_SIZE):
        pipeline_mask = cv2.resize(pipeline_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    confidence = _compute_confidence(cmp_probs, building_mask)

    result = SegmentationResult(
        egid=building_image.egid,
        mask=pipeline_mask,
        confidence=confidence,
        original_image=img,
    )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f"{building_image.egid}_mask.png"), pipeline_mask)

    return result


@torch.no_grad()
def segment_batch(
    images: list[BuildingImage],
    models: SegmentModels | None = None,
    *,
    save_dir: Path | None = None,
    batch_size: int = 4,
) -> list[SegmentationResult]:
    """Segment a batch of facade images.

    If *models* is None, loads automatically (caches for the batch).
    """
    if models is None:
        models = load_model()

    results: list[SegmentationResult] = []

    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        logger.debug("Segmenting batch %d-%d / %d", i, i + len(chunk), len(images))

        # Build batch tensor
        batch_np = np.stack([_preprocess(img.image) for img in chunk])
        pixel_values = torch.from_numpy(batch_np).to(models.device)

        # Stage 1: ADE20K building masks
        ade_logits = models.ade_model(pixel_values=pixel_values).logits
        ade_up = F.interpolate(
            ade_logits, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False,
        )
        ade_preds = ade_up.argmax(dim=1).cpu().numpy()

        # Stage 2: CMP wall/window
        cmp_logits = models.cmp_model(pixel_values=pixel_values).logits
        cmp_up = F.interpolate(
            cmp_logits, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False,
        )
        cmp_probs = F.softmax(cmp_up, dim=1).cpu().numpy()
        cmp_preds = cmp_up.argmax(dim=1).cpu().numpy().astype(np.uint8)

        for j, building_image in enumerate(chunk):
            h, w = building_image.image.shape[:2]

            building_mask = np.isin(ade_preds[j], _ADE_BUILDING_IDS)
            cmp_mask = _CMP_REMAP[cmp_preds[j]]

            pipeline_mask = np.zeros((_INPUT_SIZE, _INPUT_SIZE), dtype=np.uint8)
            pipeline_mask[building_mask] = cmp_mask[building_mask]

            if (h, w) != (_INPUT_SIZE, _INPUT_SIZE):
                pipeline_mask = cv2.resize(pipeline_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            confidence = _compute_confidence(cmp_probs[j], building_mask)

            result = SegmentationResult(
                egid=building_image.egid,
                mask=pipeline_mask,
                confidence=confidence,
                original_image=building_image.image,
            )

            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_dir / f"{building_image.egid}_mask.png"), pipeline_mask)

            results.append(result)

    logger.info("Segmented %d images", len(results))
    return results
