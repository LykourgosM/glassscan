# visualise — Heat Map Generation

## Purpose
Generate interactive maps showing WWR distribution across Swiss buildings.

## Contract
- **Input:** List of `WWRResult` and/or `PredictionResult` (from `glassscan.types`)
- **Output:** HTML file (folium map)

## Key details
- Library: folium (produces standalone HTML, no server needed)
- Choropleth by postcode or municipality, coloured by WWR
- Layer toggle: measured WWR vs predicted WWR
- Additional plots (matplotlib): feature importance, WWR distributions by building type/era
- Output: `data/processed/heatmap.html`
