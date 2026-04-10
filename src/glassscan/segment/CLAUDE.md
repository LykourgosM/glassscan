# segment -- Two-Stage Facade Segmentation

## Purpose
Semantic segmentation of facade images into wall, window, and background.

## Contract
- **Input:** `BuildingImage` (from `glassscan.types`)
- **Output:** `SegmentationResult` (from `glassscan.types`)

## Public API
- `load_model(device=None)` -- load both models, returns `SegmentModels`
- `segment_image(building_image, models)` -- single image
- `segment_batch(images, models, batch_size=4)` -- batch with chunking

## Two-stage approach
CMP Facades model alone misclassifies pavement, trees, people as wall (it was
trained on cropped facade images, never saw non-building objects). ADE20K model
correctly finds buildings but can't distinguish windows. Solution: combine both.

1. **Stage 1 (ADE20K):** identifies building pixels. ADE20K class 0 (wall) and
   class 1 (building) count as building; everything else (sky, tree, road,
   person, etc.) becomes background.
2. **Stage 2 (CMP Facades):** segments wall vs window, but only within the
   building mask from stage 1. Non-building pixels are forced to background.

## Models
- **ADE20K:** `nvidia/segformer-b5-finetuned-ade-640-640` (~339 MB, 150 classes)
- **CMP Facades:** same base, fine-tuned on 606 rectified facade images (13 classes).
  Weights: `aycaduran/cmp_segformer` on HuggingFace. Window IoU 0.69.
- Both auto-downloaded on first use (~680 MB total).

## CMP class remapping
| CMP ID | CMP class   | Pipeline class |
|--------|-------------|----------------|
| 0      | unknown     | 0 (background) |
| 1      | background  | 0 (background) |
| 2      | facade      | 1 (wall)       |
| 3      | window      | 2 (window)     |
| 4-12   | door, cornice, sill, balcony, blind, molding, deco, pillar, shop | 1 (wall) |

## Performance
- ~2x inference time vs single model (two forward passes)
- ~4 GB peak memory (two models loaded)
- Batch inference supported (default batch_size=4)

## Masks saved as
`{egid}_mask.png` in save_dir (uint8, values 0/1/2)
