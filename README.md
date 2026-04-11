# GlassScan

**Automated window-to-wall ratio estimation for Swiss buildings from Street View imagery.**

GlassScan is a computer vision pipeline that measures how much of a building's facade is glazed, a quantity known as the window-to-wall ratio (WWR). It fetches Street View photographs, segments them into wall and window regions using a two-stage semantic segmentation model, corrects for perspective distortion via projective geometry, and computes WWR as a simple area ratio on the rectified facade. For buildings without Street View coverage, it trains an XGBoost regressor on the measured buildings' metadata to predict WWR with calibrated uncertainty intervals.

Built for the [Energy Data Hackdays 2026](https://hack.opendata.ch/event/68) (Lausanne, May 7-8).

## Pipeline

```
fetch --> segment --> rectify --> wwr --> aggregate --> predict
```

Each stage is an independent module under `src/glassscan/`. All inter-module data flows through typed dataclasses in `types.py`. The pipeline orchestrator (`pipeline.py`) chains them with lazy imports to avoid the PyTorch/XGBoost libomp conflict on macOS.

### 1. Fetch

Retrieves facade photographs from the Google Street View Static API. Given a building's coordinates (lat, lon), the fetcher:

1. Queries the free metadata endpoint to find the nearest panorama within 50m.
2. Computes the geodesic bearing from the panorama position to the building centroid using the standard forward azimuth formula:

$$\theta = \text{atan2}\!\bigl(\sin\Delta\lambda\;\cos\phi_2,\;\cos\phi_1\sin\phi_2 - \sin\phi_1\cos\phi_2\cos\Delta\lambda\bigr)$$

3. Fetches a 640x640 image aimed along that bearing with FOV = 70 degrees and pitch = 20 degrees.

**Multi-view mode** captures up to 3 views per building by probing along the estimated road direction at 15m intervals, deduplicating by panorama ID. The road direction is estimated as perpendicular to the panorama-to-building bearing. Probe points are computed via the spherical Earth direct geodesic formula:

$$\phi_2 = \arcsin\!\bigl(\sin\phi_1\cos\delta + \cos\phi_1\sin\delta\cos\theta\bigr)$$

$$\lambda_2 = \lambda_1 + \text{atan2}\!\bigl(\sin\theta\sin\delta\cos\phi_1,\;\cos\delta - \sin\phi_1\sin\phi_2\bigr)$$

where $\delta = d / R$ is the angular distance ($d$ = 15m, $R$ = 6,371 km) and $\theta$ is the probe bearing. This provides redundancy against occlusion (trees, vehicles, scaffolding) and captures different facade angles.

### 2. Segment

Two-stage semantic segmentation:

**Stage 1 (building detection):** SegFormer-B5 fine-tuned on ADE20K (150 scene classes) identifies which pixels belong to buildings vs. background (sky, pavement, vegetation, pedestrians). This prevents the facade model from hallucinating wall structure on trees or parked cars.

**Stage 2 (facade parsing):** SegFormer-B5 fine-tuned on CMP Facades (13 architectural classes) segments the building region into wall, window, door, cornice, balcony, etc. We remap the 13 CMP classes to a 3-class pipeline mask:

| Pipeline class | CMP sources |
|---|---|
| 0 (background) | unknown, background |
| 1 (wall) | facade, door, cornice, sill, balcony, blind, molding, deco, pillar |
| 2 (window) | window, shop |

The combined mask is:

$$M(x, y) = \begin{cases} \text{CMP}_{\text{remapped}}(x,y) & \text{if ADE}(x,y) \in \{\text{wall, building}\} \\ 0 & \text{otherwise} \end{cases}$$

Segmentation confidence is the mean of the per-pixel maximum softmax probability, restricted to building pixels:

$$c_{\text{seg}} = \frac{1}{|\mathcal{B}|}\sum_{(x,y)\in\mathcal{B}} \max_k\; \sigma(\mathbf{z}_{x,y})_k$$

where $\mathcal{B}$ is the set of building pixels from Stage 1, $\mathbf{z}_{x,y}$ is the CMP logit vector at pixel $(x,y)$, and $\sigma$ is softmax. This gives a per-image measure of how certain the model is about its class assignments.

### 3. Rectify

Street View images are taken from ground level at varying distances and angles, so raw pixel counts would give biased area ratios (pixels near the camera subtend more real-world area than distant pixels). Perspective rectification corrects for this.

1. Compute a binary facade mask (wall + window pixels).
2. Extract the largest contour and approximate it with a quadrilateral via iterative `approxPolyDP` relaxation. Falls back to the minimum-area rotated rectangle if no 4-point approximation converges.
3. Order the 4 corners as (top-left, top-right, bottom-right, bottom-left) using the sum/difference heuristic: top-left minimises $x + y$, bottom-right maximises $x + y$, top-right minimises $y - x$, bottom-left maximises $y - x$.
4. Compute the perspective transform $H$ that maps the quadrilateral to a rectangle:

$$H = \text{getPerspectiveTransform}(\mathbf{p}_{\text{src}}, \mathbf{p}_{\text{dst}})$$

5. Compute destination rectangle dimensions from the source quadrilateral. Width is $\max(\|\mathbf{p}_{\text{TR}} - \mathbf{p}_{\text{TL}}\|, \|\mathbf{p}_{\text{BR}} - \mathbf{p}_{\text{BL}}\|)$ and height is $\max(\|\mathbf{p}_{\text{BL}} - \mathbf{p}_{\text{TL}}\|, \|\mathbf{p}_{\text{BR}} - \mathbf{p}_{\text{TR}}\|)$.
6. Warp both the image and the segmentation mask using $H$. The image uses bilinear interpolation; the mask uses nearest-neighbour to preserve discrete class labels.

After rectification, every pixel in the output represents approximately the same real-world area, so pixel counting gives an unbiased WWR.

### 4. WWR Computation

On the rectified mask, the window-to-wall ratio is:

$$\text{WWR} = \frac{|\{(x,y) : M(x,y) = 2\}|}{|\{(x,y) : M(x,y) \in \{1, 2\}\}|}$$

i.e., window pixels divided by total facade pixels (wall + window). Background pixels are excluded entirely.

Individual windows are detected via 8-connected component analysis, with components smaller than 25 pixels filtered as noise. Measurement confidence is derived from facade coverage:

$$c_{\text{wwr}} = \min\!\left(\frac{|\{M = 1\}| + |\{M = 2\}|}{W \times H} \;\bigg/\; 0.5,\;\; 1.0\right)$$

where $W \times H$ is the rectified image size. This scales linearly from 0 at no facade pixels to 1.0 at 50% or greater facade coverage. Well-rectified images typically have 30-90% facade coverage.

### 5. Multi-view Aggregation

When multiple views are available per building, WWR is computed independently for each view and then combined as a weighted average:

$$\text{WWR}_{\text{agg}} = \frac{\sum_{i} w_i \cdot \text{WWR}_i}{\sum_{i} w_i}$$

**Default weights:** primary view $w_0 = 1.0$, secondary views $w_k = 0.5$ for $k \geq 1$.

**LLM-scored weights:** a separate scoring step evaluates each view's overlay image against a 6-criterion rubric (building isolation, window detection, occlusion, view angle, segmentation cleanliness, zoom/framing). Each criterion is scored on [0, 1] and the final weight is the geometric mean:

$$w_i = \left(\prod_{j=1}^{6} s_{ij}\right)^{1/6}$$

The geometric mean ensures a single failing criterion (e.g. heavy occlusion) dominates the weight, rather than being diluted by an arithmetic average. Scores are stored in `weights.json` keyed by building ID. If a building has no LLM scores, the default scheme applies.

### 6. Prediction

For buildings without Street View coverage, an XGBoost regressor predicts WWR from building metadata (construction year, storey count, floor area, building category, heating type, coordinates). The model is feature-agnostic: it auto-detects numeric vs. categorical columns from whatever DataFrame it receives, so the exact feature set adapts to the available data without code changes.

Three models are trained jointly:
- **Median regression** (standard `reg:squarederror`) for the point estimate
- **5th percentile** (`reg:quantileerror`, $\alpha = 0.05$) for the lower bound
- **95th percentile** (`reg:quantileerror`, $\alpha = 0.95$) for the upper bound

This gives a 90% prediction interval for each building's WWR, with monotonicity enforced post-hoc ($\hat{y}_{\text{lower}} \leq \hat{y} \leq \hat{y}_{\text{upper}}$). Model evaluation uses k-fold cross-validation reporting MAE and R-squared.

## Dashboard

An interactive React/TypeScript dashboard visualises the results on a dark Leaflet map with glass-morphism UI panels. Features:

- Colour-coded markers (green = low WWR, red = high) with solid circles for measured buildings and rings for predicted ones
- Click any building to see a slide-out panel with the raw photograph, segmentation overlay, rectified facade, and rectified overlay for each view
- Per-view weight display showing which views contributed most to the aggregated WWR
- WWR distribution histogram and summary statistics
- Legend with continuous colour scale

## Project Structure

```
src/glassscan/
    types.py              shared dataclasses
    pipeline.py           end-to-end orchestrator
    fetch/                Street View image fetcher
    segment/              two-stage semantic segmentation
    rectify/              perspective correction
    wwr/                  pixel counting + aggregation
    predict/              XGBoost regression
    visualise/
        export.py         JSON + image export
        dashboard/        React/TypeScript/Tailwind app
    scoring_prompt.md     LLM scoring rubric for view weighting
notebooks/
    run.ipynb             pipeline runner notebook
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires a Google Street View Static API key in `.env`:
```
GOOGLE_API_KEY=your_key_here
```

## Usage

Run the pipeline from the Jupyter notebook (`notebooks/run.ipynb`) or programmatically:

```python
from glassscan.pipeline import run_cv_pipeline

buildings = [{"egid": "140040", "lat": 47.372, "lon": 8.540}, ...]
result = run_cv_pipeline(buildings, api_key, max_views=3)

# result.wwr_results -> list of WWRResult per building
```

## Tests

```bash
make test          # all 162 tests
make test-fetch    # single module
```

## Tech Stack

- Python 3.11, PyTorch, HuggingFace Transformers (SegFormer-B5)
- OpenCV for image processing and geometric transforms
- XGBoost + scikit-learn for metadata-based prediction
- React, TypeScript, Tailwind CSS, Leaflet, Recharts for the dashboard
