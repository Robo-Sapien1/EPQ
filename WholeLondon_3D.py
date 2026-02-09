import os
import osmnx as ox
import time
import pandas as pd
import re

OUT_GPKG = "london_buildings_3m__30km.gpkg"
OUT_LAYER = "buildings_3m"

# delete old file so we 100% overwrite
if os.path.exists(OUT_GPKG):
    os.remove(OUT_GPKG)
# ------------------ 1. SETTINGS ------------------

CENTER_POINT = (51.51861, -0.12583)
RADIUS_M = 30000

# geometry simplification tolerance in metres (in EPSG:27700)
# higher = fewer vertices = faster + smaller file
SIMPLIFY_TOL = 3

start_time = time.time()

# ------------------ 2. DOWNLOAD BUILDINGS ------------------

print(f"Downloading buildings within {RADIUS_M} m of {CENTER_POINT} ...")

buildings = ox.features_from_point(
    CENTER_POINT,
    tags={"building": True},
    dist=RADIUS_M
)

# keep only polygon geometries (actual footprints)
buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
print(f"Number of raw building footprints: {len(buildings)}")

# ------------------ 3. PROJECT TO METRES (FOR SIMPLIFY) ------------------

# British National Grid – good for London, units in metres
buildings = buildings.to_crs(epsg=27700)

# ------------------ 4. SIMPLIFY GEOMETRY (OPTIONAL) ------------------

if SIMPLIFY_TOL is not None and SIMPLIFY_TOL > 0:
    print(f"Simplifying geometry (tolerance = {SIMPLIFY_TOL} m) ...")
    buildings["geometry"] = buildings.geometry.simplify(
        SIMPLIFY_TOL, preserve_topology=True
    )

# ------------------ 5. ESTIMATE BUILDING HEIGHTS ------------------

def get_height(row):
    # --- height tag ---
    h = row.get("height")
    if pd.notna(h):
        if isinstance(h, str):
            # extract first number (handles "12", "12 m", "12.5m", etc)
            m = re.search(r"[-+]?\d*\.?\d+", h.replace(",", "."))
            if m:
                return float(m.group())
        elif isinstance(h, (int, float)):
            return float(h)

    # --- building:levels tag ---
    levels = row.get("building:levels")
    if pd.notna(levels):
        if isinstance(levels, str):
            m = re.search(r"[-+]?\d*\.?\d+", levels.replace(",", "."))
            if m:
                return float(m.group()) * 3.0
        elif isinstance(levels, (int, float)):
            return float(levels) * 3.0

    # fallback
    return 10.0

print("Estimating heights ...")
buildings["height_m"] = buildings.apply(get_height, axis=1)

# ------------------ 6. CLEAN FIELDS (IMPORTANT FOR GPKG EXPORT) ------------------
# Some OSM tag columns (e.g., "FIXME") can break GeoPackage export.
# For a clean, small, reliable file, keep only what we need.
buildings = buildings[["geometry", "height_m"]].copy()

# ------------------ 7. EXPORT FOR QGIS (3D EXTRUSION) ------------------

# QGIS works great with big data in GeoPackage.
# For maximum compatibility, export in WGS84 lat/lon.
print("Exporting to GeoPackage for QGIS ...")
# Keep in metres for UK analysis + 3D
buildings_out = buildings  # already EPSG:27700

# Write to GeoPackage
buildings_out.to_file(OUT_GPKG, layer=OUT_LAYER, driver="GPKG")

elapsed = time.time() - start_time
print(f"Saved: {OUT_GPKG} (layer: {OUT_LAYER})")
print(f"Done in {elapsed:.1f} seconds")
