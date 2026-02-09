# -*- coding: utf-8 -*-
"""
London boundary: defines the exact polygon for what is considered London
(Greater London, the administrative region used by the GLA and ONS).

This is the starting program for the pipeline: it provides:
- The London boundary polygon (WGS84 and optionally projected)
- A way to check if (lat, long) is inside or outside the city
- Saved boundary for use by integrating_step1 (intersect user circle with London).

Data source: OpenStreetMap / Nominatim — "Greater London" administrative boundary.
"""

from pathlib import Path
from typing import Tuple, Union, List
import json

import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union

# Output path (same directory as this script)
OUTPUT_DIR = Path(__file__).resolve().parent
BOUNDARY_GEOJSON = OUTPUT_DIR / "london_boundary.geojson"
BOUNDARY_JSON = OUTPUT_DIR / "london_boundary.json"

# Greater London: OSM/Nominatim query
# which_result=1 gives the main polygon (administrative boundary)
LONDON_QUERY = "Greater London, UK"
WHICH_RESULT = 1

# Cache: in-memory polygon (WGS84) and prepared for point-in-polygon
_london_wgs84 = None
_london_prepared = None


def fetch_london_boundary(
    query: str = LONDON_QUERY,
    which_result: int = WHICH_RESULT,
) -> Union[Polygon, MultiPolygon]:
    """
    Fetch Greater London boundary from OSM via Nominatim.
    Returns a (Multi)Polygon in WGS84 (EPSG:4326).
    """
    gdf = ox.geocode_to_gdf(query, which_result=which_result)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    geom = gdf.geometry.union_all()
    if geom.geom_type == "MultiPolygon" and len(geom.geoms) == 1:
        geom = geom.geoms[0]
    return geom


def get_london_polygon_wgs84(
    use_cache: bool = True,
    query: str = LONDON_QUERY,
    which_result: int = WHICH_RESULT,
) -> Union[Polygon, MultiPolygon]:
    """
    Return London boundary as (Multi)Polygon in WGS84.
    If use_cache is True and we have already loaded from file or fetched, reuse it.
    """
    global _london_wgs84
    if use_cache and _london_wgs84 is not None:
        return _london_wgs84

    if BOUNDARY_GEOJSON.exists():
        gdf = gpd.read_file(BOUNDARY_GEOJSON)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)
        _london_wgs84 = gdf.geometry.union_all()
        if _london_wgs84.geom_type == "MultiPolygon" and len(_london_wgs84.geoms) == 1:
            _london_wgs84 = _london_wgs84.geoms[0]
        return _london_wgs84

    _london_wgs84 = fetch_london_boundary(query=query, which_result=which_result)
    return _london_wgs84


def _get_prepared():
    """Lazy-prepared geometry for fast point-in-polygon."""
    global _london_prepared
    if _london_prepared is None:
        from shapely.prepared import prep
        _london_prepared = prep(get_london_polygon_wgs84(use_cache=True))
    return _london_prepared


def is_inside_london(lat: float, lon: float) -> bool:
    """
    Return True if (lat, lon) in WGS84 is inside the London boundary (Greater London).
    """
    pt = Point(lon, lat)  # Shapely uses (x,y) = (lon, lat)
    return _get_prepared().contains(pt)


def polygon_to_coords(geom: Union[Polygon, MultiPolygon]) -> List[List[Tuple[float, float]]]:
    """
    Convert (Multi)Polygon to list of rings: each ring is list of (lon, lat) or (x, y).
    For WGS84 we use (lon, lat) per GeoJSON convention.
    """
    if isinstance(geom, Polygon):
        if geom.is_empty:
            return []
        return [list(geom.exterior.coords)]
    if isinstance(geom, MultiPolygon):
        out = []
        for p in geom.geoms:
            if not p.is_empty:
                out.append(list(p.exterior.coords))
        return out
    return []


def save_boundary(
    geojson_path: Path = BOUNDARY_GEOJSON,
    json_path: Path = BOUNDARY_JSON,
    query: str = LONDON_QUERY,
    which_result: int = WHICH_RESULT,
) -> Union[Polygon, MultiPolygon]:
    """
    Fetch London boundary, save as GeoJSON and as JSON (list of rings in lon/lat),
    and return the polygon.
    """
    geom = fetch_london_boundary(query=query, which_result=which_result)
    gdf = gpd.GeoDataFrame({"name": ["Greater London"]}, geometry=[geom], crs="EPSG:4326")
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"Saved {geojson_path}")

    rings = polygon_to_coords(geom)
    with open(json_path, "w") as f:
        json.dump({"type": "MultiPolygon", "coordinates_wgs84_lonlat": rings}, f, indent=2)
    print(f"Saved {json_path}")

    global _london_wgs84, _london_prepared
    _london_wgs84 = geom
    _london_prepared = None
    return geom


def load_boundary_from_file(geojson_path: Path = BOUNDARY_GEOJSON) -> Union[Polygon, MultiPolygon]:
    """Load London boundary from saved GeoJSON."""
    gdf = gpd.read_file(geojson_path)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    geom = gdf.geometry.union_all()
    if geom.geom_type == "MultiPolygon" and len(geom.geoms) == 1:
        geom = geom.geoms[0]
    return geom


def get_london_polygon_projected(epsg: int = 27700) -> Union[Polygon, MultiPolygon]:
    """
    Return London boundary in projected CRS (e.g. EPSG:27700 British National Grid).
    """
    geom = get_london_polygon_wgs84(use_cache=True)
    gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=epsg)
    out = gdf.geometry.iloc[0]
    if out.geom_type == "MultiPolygon" and len(out.geoms) == 1:
        out = out.geoms[0]
    return out


if __name__ == "__main__":
    print("London boundary: defining Greater London polygon (OSM/Nominatim).")
    if BOUNDARY_GEOJSON.exists():
        print(f"Found existing {BOUNDARY_GEOJSON}; re-fetching and overwriting.")
    geom = save_boundary()
    print(f"Boundary type: {geom.geom_type}")

    # Quick test: centre of London (e.g. Trafalgar Square) should be inside
    trafalgar = (51.5074, -0.1278)
    print(f"Trafalgar Square {trafalgar}: inside London = {is_inside_london(*trafalgar)}")
    # Point outside (e.g. Birmingham)
    birmingham = (52.4862, -1.8904)
    print(f"Birmingham {birmingham}: inside London = {is_inside_london(*birmingham)}")
