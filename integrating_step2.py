# -*- coding: utf-8 -*-
"""
Integrating step 2: Delivery-demand for the study area defined by step 1.

Pipeline: london_boundary.py (once) -> integrating_step1.py (centre + radius, intersect
with London) -> this script (demand within that space) -> depot_model_demand.py.

If integrating_step1_output.json exists and contains city_polygon_xy (the London-
clipped space), demand is computed for that polygon. Otherwise falls back to building
a circle from CENTER_POINT and RADIUS_M (legacy / any city).

Demand = *package delivery demand* (drone delivery networks in urban space). Real
parcel-order data is not publicly available; we use proxies (OSM buildings, LSOA pop).

Inputs (when using pipeline): space from integrating_step1 (city_polygon_xy, etc.)
Or: centre (lat, lon) + radius (m) for legacy/standalone.
Outputs:
  - city polygon and demand hotspots (x, y, peak) in projected CRS (metres), for depot_model_demand.

Demand sources (set DEMAND_SOURCE):
  - "osm_buildings": OSM building area per grid cell. Works globally; no extra download.
  - "lsoa_population": Small-area population (UK only). LSOA boundaries + population CSV;
    better proxy for households; see LSOA_* config.

Other data you could add later:
  - OSM POIs (amenities, retail): count per cell as delivery-relevant proxy (global).
  - WorldPop / GHS-POP: global population rasters; sample into grid (global).
  - Country-specific: Census tracts (US), NUTS/LAU (EU), etc., with your own boundaries + CSV.

Why Voronoi was removed:
  Voronoi was an alternative way to *aggregate* the same underlying data (e.g. building
  area): assign each building to the nearest of a set of points, then sum per Voronoi
  cell. So it did not add new demand information—only repartitioned the same data into
  different shapes. The depot model needs (x, y, peak) per hotspot; we get that from
  grid-cell centroids anyway. Grid cells are simpler, easier to explain, and avoid an
  extra dependency (scipy). So Voronoi was dropped to keep the pipeline simple.
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon, Point, box
from shapely.prepared import prep

# =============================================================================
# Configuration (same style as dash_drone_sim.py)
# =============================================================================

# Study area: centre (lat, lon) and radius in metres — works for any city
CENTER_POINT = (51.51861, -0.12583)   # (lat, lon) WGS84 — default London; change for any city
RADIUS_M = 1000.0                      # circle radius in metres (match R_max_user=300 in depot_model_demand)
USE_UK_BNG_FOR_UK = True              # if True, use British National Grid (EPSG:27700) when centre is in UK; else UTM worldwide

# Demand = package delivery demand. Source (no public carrier data; we use proxies):
#   "osm_buildings"  = OSM building area per cell (no extra download)
#   "lsoa_population" = LSOA population (better proxy; requires LSOA files - see below)
DEMAND_SOURCE = "osm_buildings"

# Demand detection
DEMAND_GRID_CELL_M = 80.0             # grid cell size for density (metres)
HOTSPOT_TOP_N = 15                    # number of top-density areas to output as hotspots
HOTSPOT_PERCENTILE = 85.0             # or: cells above this density percentile (if USE_PERCENTILE)
USE_PERCENTILE = False                # if True, use HOTSPOT_PERCENTILE; else use HOTSPOT_TOP_N
MIN_BUILDING_AREA_M2 = 10.0           # ignore very small building fragments (OSM only)

# LSOA population (only when DEMAND_SOURCE == "lsoa_population"):
# Download once: (1) LSOA boundaries: data.gov.uk "Lower Layer Super Output Area (LSOA) boundaries"
#    (GeoJSON/Shapefile) or London Datastore "Statistical GIS boundary files London"
# (2) Population CSV: London Datastore "Super Output Area Population (LSOA, MSOA)" -> ons-mye-LSOA11.csv
#    https://data.london.gov.uk/dataset/super-output-area-population-lsoa-msoa-london
# Set paths below; adjust LSOA_CODE_COLUMN / LSOA_POPULATION_COLUMN to match your CSV.
LSOA_BOUNDARIES_PATH: Optional[Path] = None   # e.g. Path("data/lsoa_boundaries.geojson")
LSOA_POPULATION_CSV_PATH: Optional[Path] = None  # e.g. Path("data/ons-mye-LSOA11.csv")
LSOA_CODE_COLUMN = "Geography Code"   # or "LSOA11CD" depending on file
LSOA_POPULATION_COLUMN = "All Ages"   # or "Total" / "Population"

# Output
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = OUTPUT_DIR / "integrating_step1_output.json"
OUTPUT_GEOJSON_HOTSPOTS = OUTPUT_DIR / "integrating_step1_hotspot_areas.geojson"

# When True, prefer loading study area from step1 output (London-clipped polygon)
USE_STEP1_SPACE_IF_AVAILABLE = True


# =============================================================================
# Load step1 space (London-clipped polygon)
# =============================================================================

def load_step1_space(path: Optional[Path] = None) -> Optional[Tuple[Polygon, float, float, float, int]]:
    """
    Load study area from integrating_step1 output (space only).
    Returns (city_polygon, cx, cy, radius_m, epsg) or None if file missing/invalid.
    """
    path = path or OUTPUT_JSON
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if "city_polygon_xy" not in data:
        return None
    xy = data["city_polygon_xy"]
    if not xy:
        return None
    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    cx, cy = data.get("center_projected", (0.0, 0.0))
    radius_m = float(data.get("radius_m", 0))
    crs = data.get("crs", "EPSG:27700")
    epsg = int(crs.replace("EPSG:", "")) if "EPSG:" in crs else 27700
    return (poly, float(cx), float(cy), radius_m, epsg)


# =============================================================================
# Geometry: circle in projected CRS (any city)
# =============================================================================

def _utm_zone_from_lon(lon: float) -> int:
    """UTM zone 1–60 from longitude."""
    return min(60, max(1, int((float(lon) + 180) / 6) + 1))


def get_projected_epsg_for_centre(
    lat: float, lon: float, use_uk_bng_for_uk: bool = True
) -> int:
    """
    Return EPSG code for a projected CRS in metres at (lat, lon).
    For UK (approx 49–61 N, -8–2 E): use British National Grid 27700 if use_uk_bng_for_uk.
    Elsewhere: use UTM (32600+zone N, 32700+zone S).
    """
    lat, lon = float(lat), float(lon)
    if use_uk_bng_for_uk and (49 <= lat <= 61 and -8 <= lon <= 2):
        return 27700
    zone = _utm_zone_from_lon(lon)
    return 32600 + zone if lat >= 0 else 32700 + zone


def center_point_projected(
    center_latlon: Tuple[float, float], target_epsg: int
) -> Tuple[float, float]:
    """Project (lat, lon) WGS84 to target EPSG. Returns (easting, northing)."""
    geom_proj, _crs = ox.projection.project_geometry(
        Point(center_latlon[1], center_latlon[0]),
        crs="EPSG:4326",
        to_crs=f"EPSG:{target_epsg}"
    )
    pt = geom_proj.geoms[0] if hasattr(geom_proj, "geoms") else geom_proj
    return float(pt.x), float(pt.y)


def city_circle_polygon(
    center_latlon: Tuple[float, float],
    radius_m: float,
    use_uk_bng_for_uk: bool = True,
) -> Tuple[Polygon, float, float, int]:
    """
    Build the study area as a circular polygon in a projected CRS (metres).
    Works for any city: UK -> British National Grid; elsewhere -> UTM.
    Returns (polygon, cx, cy, epsg).
    """
    epsg = get_projected_epsg_for_centre(
        center_latlon[0], center_latlon[1], use_uk_bng_for_uk=use_uk_bng_for_uk
    )
    cx, cy = center_point_projected(center_latlon, epsg)
    pt = Point(cx, cy)
    circle = pt.buffer(radius_m)
    if not isinstance(circle, Polygon):
        circle = circle.geoms[0] if hasattr(circle, "geoms") else Polygon(circle.exterior)
    return circle, cx, cy, epsg


# =============================================================================
# OSM: load buildings inside the circle (demand proxy)
# =============================================================================

def load_buildings_in_circle(
    center_latlon: Tuple[float, float],
    radius_m: float,
    city_polygon: Polygon,
    target_epsg: int,
) -> gpd.GeoDataFrame:
    """
    Download OSM buildings within radius of centre, clip to city circle.
    Works for any city (OSM global). Returns geometries in target_epsg with area_m2.
    """
    print(f"Downloading OSM buildings within {radius_m:.0f} m of {center_latlon} ...")
    b = ox.features_from_point(center_latlon, tags={"building": True}, dist=radius_m)
    b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    b = b.to_crs(epsg=target_epsg)

    # Clip to exact circle (in case OSM returns a square bounding box)
    city_prep = prep(city_polygon)
    clipped = []
    for idx, row in b.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.intersects(city_polygon):
            inter = geom.intersection(city_polygon)
            if not inter.is_empty:
                if inter.geom_type == "MultiPolygon":
                    for p in inter.geoms:
                        a = p.area
                        if a >= MIN_BUILDING_AREA_M2:
                            clipped.append({"geometry": p, "area_m2": a})
                else:
                    a = inter.area
                    if a >= MIN_BUILDING_AREA_M2:
                        clipped.append({"geometry": inter, "area_m2": a})
    if not clipped:
        return gpd.GeoDataFrame(columns=["geometry", "area_m2"], crs=f"EPSG:{target_epsg}")

    gdf = gpd.GeoDataFrame(clipped, crs=f"EPSG:{target_epsg}")
    print(f"Buildings (clipped to circle): {len(gdf)}, total area {gdf['area_m2'].sum():.0f} m²")
    return gdf


def load_buildings_in_polygon(
    city_polygon: Polygon,
    target_epsg: int,
) -> gpd.GeoDataFrame:
    """
    Download OSM buildings in the bounding box of the polygon (WGS84), then project
    and clip to city_polygon. Used when study area comes from step1 (London-clipped).
    """
    # Bbox in WGS84 for OSM
    gdf_city = gpd.GeoDataFrame({"geometry": [city_polygon]}, crs=f"EPSG:{target_epsg}")
    gdf_wgs = gdf_city.to_crs(epsg=4326)
    bounds = gdf_wgs.total_bounds  # (minx, miny, maxx, maxy) = (lon_min, lat_min, lon_max, lat_max)
    bbox = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    print(f"Downloading OSM buildings in bbox (study area from step1) ...")
    b = ox.features_from_bbox(bbox, tags={"building": True})
    b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    b = b.to_crs(epsg=target_epsg)

    city_prep = prep(city_polygon)
    clipped = []
    for idx, row in b.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.intersects(city_polygon):
            inter = geom.intersection(city_polygon)
            if not inter.is_empty:
                if inter.geom_type == "MultiPolygon":
                    for p in inter.geoms:
                        a = p.area
                        if a >= MIN_BUILDING_AREA_M2:
                            clipped.append({"geometry": p, "area_m2": a})
                else:
                    a = inter.area
                    if a >= MIN_BUILDING_AREA_M2:
                        clipped.append({"geometry": inter, "area_m2": a})
    if not clipped:
        return gpd.GeoDataFrame(columns=["geometry", "area_m2"], crs=f"EPSG:{target_epsg}")

    gdf = gpd.GeoDataFrame(clipped, crs=f"EPSG:{target_epsg}")
    print(f"Buildings (clipped to study area): {len(gdf)}, total area {gdf['area_m2'].sum():.0f} m²")
    return gdf


# =============================================================================
# Demand density: grid-based (area = grid cell)
# =============================================================================

@dataclass
class HotspotArea:
    """One demand hotspot: an area (polygon) and its weight (peak)."""
    centroid_x: float
    centroid_y: float
    peak: float
    polygon_xy: Optional[List[Tuple[float, float]]] = None  # exterior coords for the area
    cell_ix: Optional[int] = None
    cell_iy: Optional[int] = None

    def to_depot_format(self) -> Tuple[float, float, float]:
        """Format for depot_model_demand: (x, y, peak)."""
        return (self.centroid_x, self.centroid_y, self.peak)


def _grid_cells_in_circle(
    city: Polygon,
    cell_size: float
) -> List[Tuple[Polygon, float, float, int, int]]:
    """Yield (polygon, cx, cy, ix, iy) for each grid cell whose centre lies inside the circle."""
    minx, miny, maxx, maxy = city.bounds
    city_prep = prep(city)
    out = []
    ix = 0
    for x in np.arange(minx, maxx + cell_size, cell_size):
        iy = 0
        for y in np.arange(miny, maxy + cell_size, cell_size):
            cx = x + cell_size / 2.0
            cy = y + cell_size / 2.0
            if city_prep.contains(Point(cx, cy)):
                cell = box(x, y, x + cell_size, y + cell_size)
                cell = cell.intersection(city)
                if not cell.is_empty and not cell.area < 1e-6:
                    out.append((cell, cx, cy, ix, iy))
            iy += 1
        ix += 1
    return out


def demand_density_grid(
    buildings_gdf: gpd.GeoDataFrame,
    city: Polygon,
    cell_size: float
) -> List[Tuple[float, float, float, float, Polygon, int, int]]:
    """
    Compute demand (building area) per grid cell. Returns list of
    (centroid_x, centroid_y, demand, density, cell_polygon, ix, iy).
    """
    cells = _grid_cells_in_circle(city, cell_size)
    if not cells:
        return []

    city_prep = prep(city)
    results = []

    for cell_poly, cx, cy, ix, iy in cells:
        area_in_cell = 0.0
        for _, row in buildings_gdf.iterrows():
            g = row.geometry
            if g is None or g.is_empty:
                continue
            try:
                inter = cell_poly.intersection(g)
                if hasattr(inter, "area"):
                    area_in_cell += inter.area
                elif hasattr(inter, "geoms"):
                    for geom in inter.geoms:
                        area_in_cell += geom.area
            except Exception:
                continue
        cell_area = cell_poly.area
        density = area_in_cell / cell_area if cell_area > 1e-6 else 0.0
        results.append((cx, cy, area_in_cell, density, cell_poly, ix, iy))

    return results


def select_hotspot_areas(
    density_results: List[Tuple[float, float, float, float, Polygon, int, int]],
    top_n: int,
    percentile: float,
    use_percentile: bool
) -> List[HotspotArea]:
    """Convert density results to HotspotArea list: either top N or above percentile."""
    if not density_results:
        return []

    # Sort by demand (total area in cell) descending
    sorted_results = sorted(density_results, key=lambda r: r[2], reverse=True)
    if use_percentile:
        arr = np.array([r[2] for r in sorted_results])
        thresh = np.percentile(arr, percentile)
        selected = [r for r in sorted_results if r[2] >= thresh]
    else:
        selected = sorted_results[: max(1, top_n)]

    hotspots = []
    for (cx, cy, demand, density, cell_poly, ix, iy) in selected:
        # Peak for depot_model: use normalised weight (e.g. demand / mean so scale is reasonable)
        peak = float(demand)
        exterior = list(cell_poly.exterior.coords) if cell_poly and not cell_poly.is_empty else None
        hotspots.append(HotspotArea(
            centroid_x=cx, centroid_y=cy, peak=peak,
            polygon_xy=exterior, cell_ix=ix, cell_iy=iy
        ))
    return hotspots


# =============================================================================
# LSOA population demand (proxy for delivery demand: households / population)
# =============================================================================

def load_lsoa_population_demand(
    city: Polygon,
    cell_size: float,
    boundaries_path: Path,
    csv_path: Path,
    target_epsg: int = 27700,
    lsoa_code_col: str = "Geography Code",
    population_col: str = "All Ages",
) -> List[Tuple[float, float, float, float, Polygon, int, int]]:
    """
    Load small-area boundaries and population CSV, clip to city circle, rasterise to grid.
    UK: use LSOA boundaries + ONS/London Datastore population CSV. Other countries:
    use your own boundaries + CSV with same idea (code column + population column).
    Returns same format as demand_density_grid: (cx, cy, demand, density, cell_poly, ix, iy).

    UK data: London Datastore "Super Output Area Population (LSOA, MSOA)" ->
      LSOA 2011 boundaries + ons-mye-LSOA11.csv. ONS boundaries: data.gov.uk LSOA boundaries.
    """
    import pandas as pd

    # Load boundaries and project to city CRS
    gdf = gpd.read_file(boundaries_path)
    if gdf.crs is None or str(gdf.crs) == "EPSG:4326":
        gdf = gdf.to_crs(epsg=target_epsg)
    elif str(gdf.crs) != f"EPSG:{target_epsg}":
        gdf = gdf.to_crs(epsg=target_epsg)

    # Find LSOA code column (try common names)
    code_col = None
    for c in [lsoa_code_col, "LSOA11CD", "lsoa11cd", "LSOA code", "Geography Code"]:
        if c in gdf.columns:
            code_col = c
            break
    if code_col is None:
        raise ValueError(f"LSOA code column not found in boundaries. Has: {list(gdf.columns)}")

    # Load population CSV
    df = pd.read_csv(csv_path)
    pop_col = None
    for c in [population_col, "All Ages", "All ages", "Total", "Population"]:
        if c in df.columns:
            pop_col = c
            break
    if pop_col is None:
        raise ValueError(f"Population column not found in CSV. Has: {list(df.columns)}")

    geo_col = None
    for c in [lsoa_code_col, "Geography Code", "LSOA11CD", "lsoa11cd"]:
        if c in df.columns:
            geo_col = c
            break
    if geo_col is None:
        raise ValueError(f"LSOA/Geography code column not found in CSV. Has: {list(df.columns)}")

    # Use latest year if multiple; assume one row per LSOA or take first year column
    if "Year" in df.columns:
        df = df.loc[df["Year"] == df["Year"].max()].copy()
    pop_by_lsoa = df.set_index(geo_col)[pop_col].to_dict()

    # Clip LSOAs to city and get population per LSOA (full population if centroid in city, else 0 for simplicity; or area-weighted)
    city_prep = prep(city)
    gdf = gdf[gdf.intersects(city)].copy()
    gdf["_pop"] = gdf[code_col].map(lambda x: pop_by_lsoa.get(x, 0) or 0)
    gdf["_area"] = gdf.geometry.area
    gdf = gdf[gdf["_pop"] > 0].copy()
    if len(gdf) == 0:
        return []

    # Rasterise: for each grid cell, sum (population * (intersection area / LSOA area)) over LSOAs
    cells = _grid_cells_in_circle(city, cell_size)
    if not cells:
        return []

    results = []
    for cell_poly, cx, cy, ix, iy in cells:
        cell_area = cell_poly.area
        pop_in_cell = 0.0
        for _, row in gdf.iterrows():
            g = row.geometry
            if g is None or g.is_empty:
                continue
            try:
                inter = cell_poly.intersection(g)
                if inter.is_empty:
                    continue
                frac = inter.area / (row["_area"] + 1e-20)
                pop_in_cell += row["_pop"] * min(1.0, frac)
            except Exception:
                continue
        density = pop_in_cell / cell_area if cell_area > 1e-6 else 0.0
        results.append((cx, cy, pop_in_cell, density, cell_poly, ix, iy))
    return results


# =============================================================================
# Demand density map (raster for lookup at any point)
# =============================================================================

def build_demand_density_raster(
    city: Polygon,
    cell_size: float,
    density_results: List[Tuple[float, float, float, float, Polygon, int, int]],
) -> Optional[dict]:
    """
    Build a raster (grid) of demand values from density_results so depot_model_demand
    can look up demand at any (x, y). Returns dict: xmin, ymin, cell_size, ncols, nrows, values (flat).
    values[iy * ncols + ix] = demand at cell (ix, iy). Cells not in results = 0.
    """
    if not density_results:
        return None
    minx, miny, maxx, maxy = city.bounds
    ncols = max(r[5] for r in density_results) + 1  # ix
    nrows = max(r[6] for r in density_results) + 1  # iy
    values = [0.0] * (ncols * nrows)
    for (cx, cy, demand, density, cell_poly, ix, iy) in density_results:
        if 0 <= ix < ncols and 0 <= iy < nrows:
            values[iy * ncols + ix] = float(demand)
    return {
        "xmin": float(minx),
        "ymin": float(miny),
        "cell_size": float(cell_size),
        "ncols": int(ncols),
        "nrows": int(nrows),
        "values": values,
    }


# =============================================================================
# Normalise hotspot peaks for depot_model_demand
# =============================================================================

def normalise_hotspot_peaks(
    hotspots: List[HotspotArea],
    mean_peak: float = 2.0
) -> List[HotspotArea]:
    """Scale peaks so their mean is mean_peak (keeps depot_model objective in a nice range)."""
    if not hotspots:
        return []
    peaks = np.array([h.peak for h in hotspots])
    m = float(np.mean(peaks))
    if m <= 0:
        return hotspots
    scale = mean_peak / m
    return [
        HotspotArea(centroid_x=h.centroid_x, centroid_y=h.centroid_y, peak=h.peak * scale,
                    polygon_xy=h.polygon_xy, cell_ix=h.cell_ix, cell_iy=h.cell_iy)
        for h in hotspots
    ]


# =============================================================================
# Main API: run and output
# =============================================================================

@dataclass
class AreaAndDemandResult:
    """Result of step 1: city circle + demand data for depot_model_demand (any city)."""
    city_polygon_xy: List[Tuple[float, float]]   # exterior coords in projected CRS (metres)
    center_projected: Tuple[float, float]      # (cx, cy) in same CRS
    radius_m: float
    demand_hotspots: List[Tuple[float, float, float]]  # (x, y, peak) legacy / fallback
    hotspot_areas: List[dict]  # list of {centroid_x, centroid_y, peak, polygon_xy (optional)}
    demand_density_map: Optional[dict] = None  # raster: xmin, ymin, cell_size, ncols, nrows, values (flat)
    crs: str = "EPSG:27700"  # e.g. "EPSG:27700" or "EPSG:32630"

    def to_dict(self) -> dict:
        out = {
            "city_polygon_xy": self.city_polygon_xy,
            "center_projected": list(self.center_projected),
            "radius_m": self.radius_m,
            "demand_hotspots": self.demand_hotspots,
            "hotspot_areas": self.hotspot_areas,
            "crs": self.crs,
        }
        if self.demand_density_map is not None:
            out["demand_density_map"] = self.demand_density_map
        return out


# Backward compatibility
LondonAreaAndDemandResult = AreaAndDemandResult


def get_area_and_demand(
    center_latlon: Optional[Tuple[float, float]] = None,
    radius_m: Optional[float] = None,
    grid_cell_m: Optional[float] = None,
    top_n: Optional[int] = None,
    use_percentile: Optional[bool] = None,
    percentile: Optional[float] = None,
    demand_source: Optional[str] = None,
    use_uk_bng_for_uk: Optional[bool] = None,
    step1_json_path: Optional[Path] = None,
    use_step1_space: Optional[bool] = None,
) -> AreaAndDemandResult:
    """
    Main entry: compute delivery-demand for study area.
    If use_step1_space and integrating_step1_output.json exists with city_polygon_xy,
    uses that (London-clipped) polygon. Otherwise builds circle from centre + radius.
    Output: city polygon + demand_hotspots in projected CRS (metres), for depot_model_demand.
    """
    center_latlon = center_latlon or CENTER_POINT
    radius_m = radius_m if radius_m is not None else RADIUS_M
    grid_cell_m = grid_cell_m if grid_cell_m is not None else DEMAND_GRID_CELL_M
    top_n = top_n if top_n is not None else HOTSPOT_TOP_N
    use_percentile = use_percentile if use_percentile is not None else USE_PERCENTILE
    percentile = percentile if percentile is not None else HOTSPOT_PERCENTILE
    demand_source = demand_source or DEMAND_SOURCE
    use_uk_bng_for_uk = use_uk_bng_for_uk if use_uk_bng_for_uk is not None else USE_UK_BNG_FOR_UK
    use_step1_space = use_step1_space if use_step1_space is not None else USE_STEP1_SPACE_IF_AVAILABLE

    city = None
    cx, cy, epsg = 0.0, 0.0, 27700

    used_step1_space = False
    if use_step1_space:
        step1 = load_step1_space(path=step1_json_path or OUTPUT_JSON)
        if step1 is not None:
            city, cx, cy, radius_m, epsg = step1
            city_xy = list(city.exterior.coords)
            used_step1_space = True
            print("Using study area from integrating_step1 (London-clipped polygon).")

    if city is None:
        city, cx, cy, epsg = city_circle_polygon(center_latlon, radius_m, use_uk_bng_for_uk=use_uk_bng_for_uk)
        city_xy = list(city.exterior.coords)

    crs_str = f"EPSG:{epsg}"

    if demand_source == "lsoa_population":
        if not LSOA_BOUNDARIES_PATH or not LSOA_POPULATION_CSV_PATH or not LSOA_BOUNDARIES_PATH.exists() or not LSOA_POPULATION_CSV_PATH.exists():
            print("DEMAND_SOURCE is 'lsoa_population' but paths not set or files missing. Fallback to osm_buildings.")
            demand_source = "osm_buildings"
        else:
            density_results = load_lsoa_population_demand(
                city, grid_cell_m,
                LSOA_BOUNDARIES_PATH, LSOA_POPULATION_CSV_PATH,
                target_epsg=epsg,
                lsoa_code_col=LSOA_CODE_COLUMN, population_col=LSOA_POPULATION_COLUMN,
            )
            if not density_results:
                print("No LSOA population in circle; returning empty hotspots.")
                return AreaAndDemandResult(
                    city_polygon_xy=city_xy, center_projected=(cx, cy), radius_m=radius_m,
                    demand_hotspots=[], hotspot_areas=[], crs=crs_str,
                )
            hotspot_areas_list = select_hotspot_areas(
                density_results, top_n=top_n, percentile=percentile, use_percentile=use_percentile
            )
            hotspot_areas_list = normalise_hotspot_peaks(hotspot_areas_list, mean_peak=2.0)
            demand_hotspots = [h.to_depot_format() for h in hotspot_areas_list]
            hotspot_areas_dict = [
                {"centroid_x": h.centroid_x, "centroid_y": h.centroid_y, "peak": h.peak, "polygon_xy": h.polygon_xy}
                for h in hotspot_areas_list
            ]
            demand_density_map = build_demand_density_raster(city, grid_cell_m, density_results)
            return AreaAndDemandResult(
                city_polygon_xy=city_xy, center_projected=(cx, cy), radius_m=radius_m,
                demand_hotspots=demand_hotspots, hotspot_areas=hotspot_areas_dict,
                demand_density_map=demand_density_map, crs=crs_str,
            )

    # OSM buildings (works for any city; use polygon bbox when space from step1)
    if used_step1_space:
        buildings = load_buildings_in_polygon(city, target_epsg=epsg)
    else:
        buildings = load_buildings_in_circle(center_latlon, radius_m, city, target_epsg=epsg)
    if len(buildings) == 0:
        print("No buildings in circle; returning empty hotspots.")
        return AreaAndDemandResult(
            city_polygon_xy=city_xy,
            center_projected=(cx, cy),
            radius_m=radius_m,
            demand_hotspots=[],
            hotspot_areas=[],
            crs=crs_str,
        )

    density_results = demand_density_grid(buildings, city, grid_cell_m)
    hotspot_areas_list = select_hotspot_areas(
        density_results, top_n=top_n, percentile=percentile, use_percentile=use_percentile
    )

    hotspot_areas_list = normalise_hotspot_peaks(hotspot_areas_list, mean_peak=2.0)
    demand_hotspots = [h.to_depot_format() for h in hotspot_areas_list]
    hotspot_areas_dict = [
        {
            "centroid_x": h.centroid_x, "centroid_y": h.centroid_y, "peak": h.peak,
            "polygon_xy": h.polygon_xy,
        }
        for h in hotspot_areas_list
    ]
    demand_density_map = build_demand_density_raster(city, grid_cell_m, density_results)

    return AreaAndDemandResult(
        city_polygon_xy=city_xy,
        center_projected=(cx, cy),
        radius_m=radius_m,
        demand_hotspots=demand_hotspots,
        hotspot_areas=hotspot_areas_dict,
        demand_density_map=demand_density_map,
        crs=crs_str,
    )


# Backward compatibility
def get_london_area_and_demand(
    center_latlon: Optional[Tuple[float, float]] = None,
    radius_m: Optional[float] = None,
    grid_cell_m: Optional[float] = None,
    top_n: Optional[int] = None,
    use_percentile: Optional[bool] = None,
    percentile: Optional[float] = None,
    demand_source: Optional[str] = None,
    use_uk_bng_for_uk: Optional[bool] = None,
) -> AreaAndDemandResult:
    """Alias for get_area_and_demand (same behaviour; works for any city)."""
    return get_area_and_demand(
        center_latlon=center_latlon, radius_m=radius_m, grid_cell_m=grid_cell_m,
        top_n=top_n, use_percentile=use_percentile, percentile=percentile,
        demand_source=demand_source, use_uk_bng_for_uk=use_uk_bng_for_uk,
    )


def load_step1_output(path: Optional[Path] = None) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float]]]:
    """
    Load step 1 output from JSON for use in depot_model_demand.
    Returns (city_coords, demand_hotspots) where city_coords is list of (x,y) for polygon exterior.
    """
    path = path or OUTPUT_JSON
    with open(path) as f:
        data = json.load(f)
    city_coords = [tuple(p) for p in data["city_polygon_xy"]]
    demand_hotspots = [tuple(h) for h in data["demand_hotspots"]]
    return city_coords, demand_hotspots


def save_results(result: AreaAndDemandResult) -> None:
    """Save result to JSON and optional GeoJSON for hotspot areas."""
    out = result.to_dict()
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")

    # GeoJSON for hotspot polygons (if any have polygon_xy)
    features = []
    for i, ha in enumerate(result.hotspot_areas):
        if not ha.get("polygon_xy"):
            continue
        from shapely.geometry import mapping
        poly = Polygon(ha["polygon_xy"])
        features.append({
            "type": "Feature",
            "properties": {"index": i, "peak": ha["peak"], "centroid_x": ha["centroid_x"], "centroid_y": ha["centroid_y"]},
            "geometry": mapping(poly),
        })
    if features:
        geojson = {"type": "FeatureCollection", "features": features}
        with open(OUTPUT_GEOJSON_HOTSPOTS, "w") as f:
            json.dump(geojson, f, indent=2)
        print(f"Saved {OUTPUT_GEOJSON_HOTSPOTS}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("Integrating step 1: City area (centre + radius) + delivery-demand hotspots")
    print(f"Centre (lat, lon): {CENTER_POINT}, radius: {RADIUS_M} m (any city)")
    print(f"Demand source: {DEMAND_SOURCE} | Grid: {DEMAND_GRID_CELL_M} m | Hotspots: top N={HOTSPOT_TOP_N}")

    result = get_area_and_demand()
    print(f"\nCity circle: {len(result.city_polygon_xy)} vertices")
    print(f"Demand hotspots (areas -> centroid + peak): {len(result.demand_hotspots)}")
    for i, (x, y, p) in enumerate(result.demand_hotspots[:5]):
        print(f"  {i+1}: ({x:.1f}, {y:.1f}) peak={p:.2f}")
    if len(result.demand_hotspots) > 5:
        print("  ...")

    save_results(result)
    print(f"\nDone. CRS: {result.crs} (metres). Pass to depot_model_demand:")
    print("  - city_polygon_xy: city boundary | demand_hotspots: (x, y, peak) for compute_demand_weights(hotspots=...)")
    print("  - Load: integrating_step1_output.json or get_area_and_demand(center_latlon=(lat,lon), radius_m=...)")
