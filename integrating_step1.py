# -*- coding: utf-8 -*-
"""
Integrating step 1: User-defined space (centre + radius) intersected with London boundary.

Takes centre (lat, lon) and radius (m), builds the user's circular area, then
intersects it with the Greater London boundary from london_boundary.py. The result
is the allowed study area (possibly smaller than the circle if the circle extends
outside London). If the intersection is empty, prints a message and exits without
writing output.

Pipeline: run london_boundary.py first (once) to define/save London polygon.
Then run this script; then integrating_step2.py for demand; then depot_model_demand.py.

Output: integrating_step1_output.json with keys:
  - city_polygon_xy: list of [x, y] in projected CRS (metres)
  - center_projected: [cx, cy]
  - radius_m: user radius
  - crs: e.g. "EPSG:27700"

(integrating_step2.py then adds demand_hotspots, hotspot_areas, demand_density_map.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple, List

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# Same directory as this script
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = OUTPUT_DIR / "integrating_step1_output.json"

# User-defined space: centre (lat, lon) WGS84, radius in metres
CENTER_POINT = (51.51861, -0.12583)
RADIUS_M = 500.0
USE_UK_BNG = True


def _utm_zone_from_lon(lon: float) -> int:
    return min(60, max(1, int((float(lon) + 180) / 6) + 1))


def get_projected_epsg(lat: float, lon: float, use_uk_bng: bool = True) -> int:
    if use_uk_bng and (49 <= lat <= 61 and -8 <= lon <= 2):
        return 27700
    zone = _utm_zone_from_lon(lon)
    return 32600 + zone if lat >= 0 else 32700 + zone


def center_point_projected(center_latlon: Tuple[float, float], target_epsg: int) -> Tuple[float, float]:
    import osmnx as ox
    pt = Point(center_latlon[1], center_latlon[0])
    geom_proj, _ = ox.projection.project_geometry(pt, crs="EPSG:4326", to_crs=f"EPSG:{target_epsg}")
    p = geom_proj.geoms[0] if hasattr(geom_proj, "geoms") else geom_proj
    return float(p.x), float(p.y)


def user_circle_polygon(
    center_latlon: Tuple[float, float],
    radius_m: float,
    use_uk_bng: bool = True,
) -> Tuple[Polygon, float, float, int]:
    """User's circular area in projected CRS. Returns (polygon, cx, cy, epsg)."""
    epsg = get_projected_epsg(center_latlon[0], center_latlon[1], use_uk_bng=use_uk_bng)
    cx, cy = center_point_projected(center_latlon, epsg)
    circle = Point(cx, cy).buffer(radius_m)
    if not isinstance(circle, Polygon):
        circle = Polygon(circle.exterior) if hasattr(circle, "exterior") else circle.geoms[0]
    return circle, cx, cy, epsg


def intersect_user_space_with_london(
    center_latlon: Tuple[float, float],
    radius_m: float,
) -> Tuple[Polygon, float, float, int]:
    """
    Intersect user circle (centre + radius) with London boundary.
    Returns (intersection_polygon, cx, cy, epsg).
    Raises ValueError if intersection is empty.
    """
    from london_boundary import get_london_polygon_projected

    circle, cx, cy, epsg = user_circle_polygon(center_latlon, radius_m, use_uk_bng=USE_UK_BNG)
    london = get_london_polygon_projected(epsg=epsg)

    inter = circle.intersection(london)
    if inter.is_empty:
        raise ValueError("The intersection of the user-defined space with the London boundary is empty. Not possible.")
    if inter.geom_type == "MultiPolygon":
        # Take the largest polygon if multiple
        inter = max(inter.geoms, key=lambda g: g.area)
    if inter.geom_type != "Polygon":
        inter = Polygon(inter.exterior) if hasattr(inter, "exterior") else inter
    if inter.is_empty or inter.area < 1.0:
        raise ValueError("The intersection of the user-defined space with the London boundary is empty or negligible. Not possible.")

    return inter, cx, cy, epsg


def polygon_xy_to_list(poly: Polygon) -> List[List[float]]:
    """Exterior coords as list of [x, y] for JSON."""
    return [[float(x), float(y)] for x, y in poly.exterior.coords]


def run_step1(
    center_latlon: Tuple[float, float] = CENTER_POINT,
    radius_m: float = RADIUS_M,
    output_path: Path = OUTPUT_JSON,
) -> Tuple[Polygon, float, float, int]:
    """
    Run step 1: intersect user space with London, write space-only JSON.
    Returns (intersection_polygon, cx, cy, epsg).
    """
    inter, cx, cy, epsg = intersect_user_space_with_london(center_latlon, radius_m)
    crs_str = f"EPSG:{epsg}"

    out = {
        "city_polygon_xy": polygon_xy_to_list(inter),
        "center_projected": [cx, cy],
        "radius_m": radius_m,
        "crs": crs_str,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Step 1: intersection written to {output_path}")
    return inter, cx, cy, epsg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 1: Intersect user circle with London boundary.")
    parser.add_argument("--lat", type=float, default=CENTER_POINT[0], help="Centre latitude (WGS84)")
    parser.add_argument("--lon", type=float, default=CENTER_POINT[1], help="Centre longitude (WGS84)")
    parser.add_argument("--radius", type=float, default=RADIUS_M, help="Radius in metres")
    args = parser.parse_args()
    center = (args.lat, args.lon)
    try:
        run_step1(center_latlon=center, radius_m=args.radius)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
