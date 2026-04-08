# fetch — Street View Image Fetcher

## Purpose
Fetch facade images of Swiss buildings from the Google Street View Static API.

## Contract
- **Input:** Building coordinates (lat, lon) + optional street point for heading computation
- **Output:** `BuildingImage` (from `glassscan.types`)

## Key details
- API: Google Street View Static API (10k free calls from hackathon)
- Image size: 640x640
- Default params: fov=70, pitch=20
- Heading: computed as bearing from nearest street point to building centroid
- Rate limiting required to stay within API quotas
- Images saved to `data/raw/{egid}.jpg`
- API key loaded from `.env` via python-dotenv
