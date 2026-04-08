# segment — Facade/Window Segmentation

## Purpose
Semantic segmentation of facade images to identify wall and window regions.

## Contract
- **Input:** `BuildingImage` (from `glassscan.types`)
- **Output:** `SegmentationResult` (from `glassscan.types`)

## Key details
- Model: SegFormer (HuggingFace Transformers, PyTorch)
- Preferred weights: WWR-specific SegFormer from github.com/zoedesimone/wwr-semantic-segmentation
- Fallback: nvidia/segformer-b5-finetuned-cityscapes-1024-1024 (has building/wall but not window classes)
- Mask classes: 0=background, 1=wall, 2=window (uint8)
- Confidence: mean softmax probability across facade pixels
- Masks saved to `data/masks/{egid}.png`
