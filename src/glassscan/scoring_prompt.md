# GlassScan View Quality Scoring Prompt

You are scoring Street View facade images for a window-to-wall ratio (WWR)
measurement pipeline. Each image has been segmented into wall (blue overlay)
and window (green overlay) regions by a computer vision model. Your job is to
rate how reliable each segmented view is for measuring WWR.

You will be shown overlay images where the segmentation mask is drawn on top
of the original Street View photo. Score each image on the 6 criteria below,
then compute the final weight.

## Scoring Criteria (each 0.0 to 1.0)

### 1. Building Isolation
Is the segmentation mask on the correct/target building only?

- **1.0** -- mask covers exactly one building, clean boundaries
- **0.7** -- mask mostly on target building, minor bleed onto adjacent building
- **0.3** -- mask spans multiple buildings, hard to tell which is the target
- **0.0** -- mask is on the wrong building entirely, or covers mostly non-building

### 2. Window Detection
Are windows correctly identified? Check for missed windows and false positives.

- **1.0** -- all visible windows detected, including shop/retail glazing
- **0.7** -- most windows detected, 1-2 small ones missed
- **0.3** -- many windows missed, or large false positive regions (e.g. shadow marked as window)
- **0.0** -- window detection completely wrong (no windows found on a windowed facade, or all-window on a blank wall)

### 3. Occlusion
Is the facade free from obstructions?

- **1.0** -- facade fully visible, nothing blocking it
- **0.7** -- minor occlusion (small tree branch, street sign, parked bike)
- **0.3** -- significant occlusion (large tree, parked truck, scaffolding covering part of facade)
- **0.0** -- facade mostly hidden (construction wrap, dense foliage, large vehicle)

### 4. View Angle
Is the camera facing the facade roughly head-on?

- **1.0** -- near-frontal view (facade plane roughly perpendicular to camera)
- **0.7** -- slight angle, facade still clearly readable
- **0.3** -- steep oblique angle, significant perspective distortion
- **0.0** -- extreme angle (facade nearly edge-on) or rear view of building

### 5. Segmentation Cleanliness
Are the mask boundaries crisp and accurate?

- **1.0** -- mask edges tightly follow building/window contours
- **0.7** -- mostly clean, minor jagged edges or small misclassified patches
- **0.3** -- noisy boundaries, fragmented mask regions, obvious errors
- **0.0** -- mask is a mess (random patches, no coherent building shape)

### 6. Zoom / Framing
Is the building appropriately framed in the image?

- **1.0** -- full facade visible with some context (ground, roofline, sides)
- **0.7** -- most of facade visible, minor cropping at edges
- **0.3** -- heavily cropped, only a portion of the facade visible (e.g. zoomed into one floor)
- **0.0** -- extreme close-up (e.g. just a wall texture or single window), or building is tiny/distant in frame

## Computing the Final Weight

**Final weight = geometric mean of all 6 scores.**

```
weight = (isolation * windows * occlusion * angle * cleanliness * zoom) ^ (1/6)
```

The geometric mean ensures that one badly failing criterion (e.g. 0.1 occlusion)
pulls the overall weight down significantly, rather than being averaged away.

Round the final weight to 2 decimal places.

## Output Format

Return a JSON object keyed by EGID. Each EGID maps to a list of weights,
one per view in order (view 0, view 1, view 2, ...).

```json
{
  "140040": [0.85, 0.42, 0.71],
  "140041": [0.93, 0.68],
  "140042": [0.77]
}
```

Save this to `weights.json` in the project data directory.

## Workflow

Before scoring, find which buildings still need weights:

```python
from glassscan.wwr import unscored_egids
unscored = unscored_egids(raw_wwr, "weights.json")
print(f"{len(unscored)} buildings to score")
```

Only score buildings in this list. Skip any EGID already in `weights.json`.

The overlay images to evaluate are at:
- `dashboard/public/overlay/{egid}.jpg` (primary view)
- `dashboard/public/overlay/{egid}_v1.jpg`, `{egid}_v2.jpg` (secondary views)

## Batching

There may be hundreds of buildings with multiple views each. Process them in
batches of ~50 buildings at a time. After each batch, write the results to
`weights.json` immediately so progress is saved incrementally. Then compact
your context (use /compact) before starting the next batch to avoid hitting
context limits. You do not need to keep previous batch scores in memory
since they are already saved to the file.

**Before starting, always check progress:**

1. Read the existing `weights.json` (if it exists) to see how many buildings
   are already scored.
2. Run `unscored_egids` to get the remaining list.
3. Report: "X of Y buildings scored. Z remaining. Starting from batch N."

This way, if a session is interrupted and restarted, you pick up where you
left off rather than re-scoring buildings or losing track of progress. Merge
new results into the existing file rather than overwriting.

## Example Scoring

**Good image (weight ~0.9):**
Frontal view, full facade visible, all windows detected, no occlusion,
clean mask edges, only the target building masked.
Scores: 1.0, 0.9, 1.0, 0.9, 0.85, 1.0 -> weight = 0.94

**Medium image (weight ~0.5):**
Slight angle, one tree partially blocking, a few windows missed,
mask bleeds slightly onto neighbor, reasonable framing.
Scores: 0.7, 0.7, 0.5, 0.7, 0.7, 0.7 -> weight = 0.67

**Bad image (weight ~0.2):**
Extreme close-up of one floor, steep angle, scaffolding blocking half
the facade, fragmented mask, windows mostly missed.
Scores: 0.5, 0.3, 0.2, 0.3, 0.3, 0.1 -> weight = 0.26

## IMPORTANT

You MUST visually read and evaluate every single overlay image. Do NOT use
programmatic scoring, deterministic hashing, file metadata, or any shortcut.
Every weight must come from actually looking at the image and applying the
rubric criteria above. There is no way to produce valid scores without
seeing the images.

## Notes

- Be consistent across images. If two images have similar issues, give similar scores.
- A weight of 0.0 means "discard this view entirely" -- use sparingly.
- Every building must have at least one view with a non-zero weight.
  If all views are terrible, give the least-bad one a small weight (e.g. 0.1)
  rather than zeroing them all out. Otherwise the building has no usable WWR.
- Primary views (view 0) are not inherently better; score purely on quality.
- Shop windows / retail glazing should count as windows (green). If the model
  missed them, penalise under Window Detection.
- Swiss buildings often have shutters -- closed shutters hiding windows is an
  occlusion issue, not a window detection issue.
