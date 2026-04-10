# fetch — Street View Image Fetcher

## Purpose
Fetch facade images of Swiss buildings from the Google Street View Static API.

## Contract
- **Input:** Building coordinates (lat, lon) + EGID
- **Output:** `BuildingImage` (from `glassscan.types`)

## Public API
- `fetch_image(egid, lat, lon, api_key)` — single image from nearest panorama
- `fetch_multi_view(egid, lat, lon, api_key, max_views=4)` — multiple images along the road
- `fetch_batch(buildings, api_key, max_calls=1000, max_views=1)` — batch with cost cap
- `find_nearby_panoramas(lat, lon, api_key)` — discover panoramas (free metadata)
- `get_panorama_location(lat, lon, api_key)` — nearest panorama (free metadata)

## How multi-view works
1. Find nearest panorama via free metadata API
2. Infer road direction (perpendicular to panorama→building bearing)
3. Probe at 15m, 30m, 45m in both directions along road
4. Deduplicate by pano_id
5. Fetch image from each unique panorama, aimed at building centroid

## Key details
- Image size: 640x640, fov=70, pitch=20
- Metadata calls are free; only image fetches are billed
- `max_calls` cap in fetch_batch counts only billed calls
- Skip-if-exists: won't re-fetch images already saved to disk
- Rate limiting via configurable `delay` (default 0.1s)
- Files saved as `{egid}.jpg` (view 0) and `{egid}_v{i}.jpg` (additional views)

## Design decisions
- Road-following over circular search: keeps viewpoints on same street
- 15m step size: matches Google's ~10-20m panorama spacing
- Downstream modules handle quality filtering (occlusion, bad angles)

## Known limitation: multiple buildings per image
Street View images capture everything in the frame, not just the target building.
In dense rows of terraced houses, neighboring facades appear too. The current
approach accepts this noise (target building usually dominates when close).

### Future fix: swissBUILDINGS3D integration
Use swisstopo 3D building data to isolate the target building in each image:
1. Get building footprint polygon + height from swissBUILDINGS3D
2. Each footprint edge = one facade with known orientation and dimensions
3. Project the building's 3D bounding box into the 2D Street View image
4. Mask out everything outside the projection → only segment the target building
5. Also enables: aiming cameras perpendicular to each facade, proper rectification

### Swiss building data sources (for future integration)

**swissBUILDINGS3D 3.0 Beta (swisstopo):**
- Free, open data. 3D building models for all of Switzerland.
- Format: ESRI File Geodatabase (GDB), tiles via STAC API (~7-16MB each)
- STAC: `https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissbuildings3d_3_0`
- CRS: Swiss LV95 (EPSG:2056), needs pyproj conversion to WGS84
- Total: ~55GB GDB for all of Switzerland
- Contains: 3D MultiPatch geometry, building height, building type (OBJEKTART)
- EGID available for 17 cantons: AG, AI, AR, BE, BL, BS, FR, GL, JU, LU, NE, SG, SH, SO, SZ, TG + Zürich city
- "Separated Elements" variant splits roofs/facades/footprints into separate surfaces
- Parse with: `geopandas` + `fiona` (GDAL OpenFileGDB driver)
- Caveat: 3D MultiPatch geometry is fiddly (not simple 2D polygons)

**GWR (Gebäude- und Wohnungsregister):**
- Free, national building register. POINT geometry only (no polygons).
- API: `https://api3.geo.admin.ch/rest/services/api/MapServer/ch.bfs.gebaeude_wohnungs_register`
- Attributes: egid, garea (m²), gvol (m³), gastw (storeys), gbauj (construction year), gkat (category), heating fields, address
- Essential for predict module. Join to swissBUILDINGS3D via EGID or spatial join.

**swissTLM3D:**
- 2D building footprints (simpler than swissBUILDINGS3D), ~2.2GB national file
- No EGID — needs spatial join
- STAC: `https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swisstlm3d`

### Other future improvements
- Road direction: currently estimated as perpendicular to building→panorama bearing.
  For curved roads/intersections this drifts. Fix: query OSM Overpass API for actual
  road geometry (free).
- Zug (ZG) is not in the EGID-structured canton list. For non-EGID cantons: use
  spatial join with GWR point coordinates (point-in-polygon).
