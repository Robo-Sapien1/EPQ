# -*- coding: utf-8 -*-
"""
Standalone L0 preview: load pipeline study area + buildings, compute L0 heights,
and show a 3D plot. Lets you tweak clearance, threshold, and smooth iterations
and see the result quickly without running the full Dash app.

Speed: Buildings are cached in cache/l0_preview_buildings.pkl after first run
(until you change the study area). Use --out file.html to skip opening the browser.

Usage:
  python l0_preview.py
  python l0_preview.py --clearance 25 --smooth 5
  python l0_preview.py --no-buildings --out l0_preview.html

Requires: integrating_step1_output.json (run integrating_step1.py first).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import geopandas as gpd
import networkx as nx
import plotly.graph_objects as go
from shapely.geometry import Polygon, Point
from shapely.prepared import prep

from fleet_specs import get_l0_cell_m
from l0_height import compute_l0_node_heights

# -----------------------------------------------------------------------------
# CONFIG (edit these or override via CLI) — same meaning as dash_scene_builder L0_*
# -----------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent
STEP1_JSON = OUTPUT_DIR / "integrating_step1_output.json"
CACHE_DIR = OUTPUT_DIR / "cache"
BUILDINGS_CACHE_PKL = CACHE_DIR / "l0_preview_buildings.pkl"
BUILDINGS_CACHE_META = CACHE_DIR / "l0_preview_buildings_meta.json"
SIMPLIFY_TOL = 3.0

# L0 parameters (mirror of dash_scene_builder.py)
L0_HEIGHT_CLEARANCE_M = 20.0           # metres above building/ground
L0_BUILDING_EXCEED_THRESHOLD = 0.02   # max fraction of buildings exceeding base (0.02 = 2%)
L0_SMOOTH_ITERATIONS = 50             # smooth bumps so neighbors rise (0 = no smoothing)
L0_SMOOTHSTEP_RESHAPE = True          # S-curve roll-off; False = A-style gradient
L0_PLATEAU_SOFTEN_ITERATIONS = 10     # round off flat-to-slope boundary (0 = disable)
L0_PLATEAU_SOFTEN_ALPHA = 0.35        # blend toward local avg (0.2–0.5)
L0_PLATEAU_SOFTEN_RADIUS = 2          # neighborhood radius in hops (2 = curved dome tops; 1 = edge-only)

# Preview-only
SHOW_BUILDINGS = True
MAX_BUILDINGS_PLOTTED = None

# -----------------------------------------------------------------------------
# Helpers (minimal copy so we don't depend on full dash_scene_builder)
# -----------------------------------------------------------------------------
def get_height(row):
    h = row.get("height")
    if isinstance(h, str):
        try:
            return float(h.replace("m", "").strip())
        except ValueError:
            pass
    levels = row.get("building:levels")
    if isinstance(levels, str):
        try:
            return float(levels) * 3.0
        except ValueError:
            pass
    return 10.0


def load_study_area():
    """Load cx, cy, radius_m, city_polygon from integrating_step1_output.json."""
    if not STEP1_JSON.exists():
        raise FileNotFoundError(
            f"Study area not found: {STEP1_JSON}. Run integrating_step1.py first."
        )
    with open(STEP1_JSON) as f:
        data = json.load(f)
    xy = data.get("city_polygon_xy")
    if not xy:
        raise ValueError("city_polygon_xy missing in step1 output")
    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    cx, cy = float(data["center_projected"][0]), float(data["center_projected"][1])
    radius_m = float(data.get("radius_m", 1000.0))
    return cx, cy, radius_m, poly


def load_buildings_fresh(city_polygon: Polygon, target_epsg: int = 27700) -> gpd.GeoDataFrame:
    """OSM buildings in polygon bbox, clipped to polygon (no cache)."""
    import osmnx as ox
    gdf_city = gpd.GeoDataFrame({"geometry": [city_polygon]}, crs=f"EPSG:{target_epsg}")
    gdf_wgs = gdf_city.to_crs(epsg=4326)
    bounds = gdf_wgs.total_bounds
    bbox = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    b = ox.features_from_bbox(bbox, tags={"building": True})
    b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    b = b.to_crs(epsg=target_epsg)
    city_prep = prep(city_polygon)
    clipped = []
    for _, row in b.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or not geom.intersects(city_polygon):
            continue
        inter = geom.intersection(city_polygon)
        if inter.is_empty:
            continue
        h_val, l_val = row.get("height"), row.get("building:levels")
        if inter.geom_type == "MultiPolygon":
            for p in inter.geoms:
                if p.area >= 10:
                    clipped.append({"geometry": p, "height": h_val, "building:levels": l_val})
        else:
            if inter.area >= 10:
                clipped.append({"geometry": inter, "height": h_val, "building:levels": l_val})
    if not clipped:
        return gpd.GeoDataFrame(columns=["geometry", "height_m"], crs=f"EPSG:{target_epsg}")
    gdf = gpd.GeoDataFrame(clipped, crs=f"EPSG:{target_epsg}")
    if SIMPLIFY_TOL and SIMPLIFY_TOL > 0:
        gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    gdf["height_m"] = gdf.apply(get_height, axis=1)
    return gdf


def load_buildings_cached(city_polygon: Polygon, step1_path: Path, target_epsg: int = 27700):
    """Load buildings from cache if step1 unchanged; else fetch from OSM and cache. Returns (gdf, from_cache)."""
    step1_mtime = step1_path.stat().st_mtime
    meta_path = BUILDINGS_CACHE_META
    pkl_path = BUILDINGS_CACHE_PKL
    if CACHE_DIR.exists() and pkl_path.exists() and meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("step1_path") == str(step1_path.resolve()) and meta.get("step1_mtime") == step1_mtime:
                with open(pkl_path, "rb") as f:
                    return pickle.load(f), True
        except Exception:
            pass
    buildings = load_buildings_fresh(city_polygon, target_epsg)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(buildings, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(meta_path, "w") as f:
            json.dump({"step1_path": str(step1_path.resolve()), "step1_mtime": step1_mtime}, f, indent=0)
    except Exception:
        pass
    return buildings, False


def build_grid_in_circle(cx: float, cy: float, radius_m: float, cell_size: float, z_placeholder: float):
    """L0 grid graph inside circle (same logic as dash_scene_builder)."""
    spacing = cell_size
    effective_radius = radius_m + spacing
    xmin, ymin = cx - effective_radius, cy - effective_radius
    xmax, ymax = cx + effective_radius, cy + effective_radius
    nxn = int((xmax - xmin) // spacing) + 1
    nyn = int((ymax - ymin) // spacing) + 1
    G = nx.Graph()
    inside = set()
    for ix in range(nxn):
        x = xmin + ix * spacing
        dx2 = (x - cx) ** 2
        for iy in range(nyn):
            y = ymin + iy * spacing
            if dx2 + (y - cy) ** 2 <= effective_radius ** 2:
                nid = (ix, iy)
                inside.add(nid)
                G.add_node(nid, x=x, y=y, z=z_placeholder)
    for (ix, iy) in inside:
        a = (ix, iy)
        b1, b2 = (ix + 1, iy), (ix, iy + 1)
        if b1 in inside:
            G.add_edge(a, b1, weight=spacing)
        if b2 in inside:
            G.add_edge(a, b2, weight=spacing)
    meta = {
        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
        "nxn": nxn, "nyn": nyn, "cx": cx, "cy": cy,
        "radius": radius_m, "grid_spacing": spacing, "z": z_placeholder,
    }
    return G, meta


def _buildings_traces(buildings: gpd.GeoDataFrame, max_plot=None):
    """Plotly 3D traces for buildings (ground, roof, verticals)."""
    from shapely.geometry import MultiPolygon
    if len(buildings) == 0:
        return []
    bsub = buildings if max_plot is None else buildings.iloc[:max_plot]
    ground_x, ground_y, ground_z = [], [], []
    roof_x, roof_y, roof_z = [], [], []
    vert_x, vert_y, vert_z = [], [], []
    for _, row in bsub.iterrows():
        geom = row.geometry
        h = float(row.get("height_m", 10.0))
        polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms) if hasattr(geom, "geoms") else []
        for poly in polys:
            x, y = poly.exterior.xy
            x, y = np.array(x), np.array(y)
            ground_x.extend(list(x) + [None])
            ground_y.extend(list(y) + [None])
            ground_z.extend([0] * len(x) + [None])
            roof_x.extend(list(x) + [None])
            roof_y.extend(list(y) + [None])
            roof_z.extend([h] * len(x) + [None])
            for xi, yi in zip(x[::5], y[::5]):
                vert_x.extend([xi, xi, None])
                vert_y.extend([yi, yi, None])
                vert_z.extend([0, h, None])
    return [
        go.Scatter3d(x=ground_x, y=ground_y, z=ground_z, mode="lines", line=dict(width=1), name="Buildings (ground)", showlegend=True),
        go.Scatter3d(x=roof_x, y=roof_y, z=roof_z, mode="lines", line=dict(width=1), name="Buildings (roof)", showlegend=True),
        go.Scatter3d(x=vert_x, y=vert_y, z=vert_z, mode="lines", line=dict(width=1), showlegend=False),
    ]


def _grid_trace(G: nx.Graph, color: str = "rgb(0, 90, 255)"):
    """Plotly 3D trace for L0 grid edges."""
    xs, ys, zs = [], [], []
    for a, b in G.edges():
        ax, ay, az = G.nodes[a]["x"], G.nodes[a]["y"], G.nodes[a]["z"]
        bx, by, bz = G.nodes[b]["x"], G.nodes[b]["y"], G.nodes[b]["z"]
        xs += [ax, bx, None]
        ys += [ay, by, None]
        zs += [az, bz, None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(width=2.5, color=color),
        name="L0 grid",
        showlegend=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Preview L0 layer over study area")
    parser.add_argument("--clearance", type=float, default=L0_HEIGHT_CLEARANCE_M, help="Clearance above building/ground (m)")
    parser.add_argument("--threshold", type=float, default=L0_BUILDING_EXCEED_THRESHOLD, help="Max fraction of buildings exceeding base (e.g. 0.02)")
    parser.add_argument("--smooth", type=int, default=L0_SMOOTH_ITERATIONS, help="Smoothing iterations (0 = none)")
    parser.add_argument("--no-buildings", action="store_true", help="Hide buildings")
    parser.add_argument("--out", type=str, default=None, help="Save HTML to file instead of opening browser")
    parser.add_argument("--cell", type=float, default=None, help="L0 cell size (m); default from fleet_specs")
    parser.add_argument("--no-cache", action="store_true", help="Ignore buildings cache and re-download from OSM")
    parser.add_argument("--no-smoothstep", action="store_true", help="Disable S-curve reshape (overrides L0_SMOOTHSTEP_RESHAPE)")
    parser.add_argument("--plateau-iter", type=int, default=L0_PLATEAU_SOFTEN_ITERATIONS, help="Plateau softening iterations (0=disable)")
    parser.add_argument("--plateau-alpha", type=float, default=L0_PLATEAU_SOFTEN_ALPHA, help="Plateau soften blend toward local avg (0.2-0.5)")
    parser.add_argument("--plateau-radius", type=int, default=L0_PLATEAU_SOFTEN_RADIUS, help="Neighborhood radius in hops (2=curved dome tops)")
    args = parser.parse_args()

    # Apply CONFIG: --no-smoothstep overrides L0_SMOOTHSTEP_RESHAPE
    smoothstep_reshape = L0_SMOOTHSTEP_RESHAPE and not args.no_smoothstep

    print("Loading study area ...")
    cx, cy, radius_m, city_polygon = load_study_area()
    print(f"  centre=({cx:.0f}, {cy:.0f}), radius={radius_m:.0f} m")

    print("Loading buildings ...")
    if args.no_cache:
        buildings = load_buildings_fresh(city_polygon, 27700)
        print(f"  {len(buildings)} buildings (from OSM)")
    else:
        buildings, from_cache = load_buildings_cached(city_polygon, STEP1_JSON, 27700)
        print(f"  {len(buildings)} buildings" + (" (cached)" if from_cache else " (from OSM, cached for next run)"))

    cell_size = args.cell if args.cell is not None else get_l0_cell_m()
    print(f"Building L0 grid (cell={cell_size:.1f} m) ...")
    G, meta = build_grid_in_circle(cx, cy, radius_m, cell_size, z_placeholder=50.0)

    print(f"Computing L0 heights (clearance={args.clearance}m, smooth={args.smooth}, S-curve={smoothstep_reshape}, plateau_soften={args.plateau_iter} r={args.plateau_radius}) ...")
    compute_l0_node_heights(
        buildings, G, meta, cx, cy, radius_m,
        clearance_m=args.clearance,
        building_exceed_threshold=args.threshold,
        smooth_iterations=args.smooth,
        smoothstep_reshape=smoothstep_reshape,
        plateau_soften_iterations=args.plateau_iter,
        plateau_soften_alpha=args.plateau_alpha if args.plateau_iter > 0 else 0.0,
        plateau_soften_radius=args.plateau_radius,
    )
    max_z = max(G.nodes[n]["z"] for n in G.nodes)
    print(f"  max L0 z = {max_z:.1f} m")

    traces = []
    if not args.no_buildings:
        traces.extend(_buildings_traces(buildings, MAX_BUILDINGS_PLOTTED))
    traces.append(_grid_trace(G))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"L0 preview — clearance={args.clearance}m, smooth={args.smooth}, plateau={args.plateau_iter} r={args.plateau_radius}",
            x=0.5, xanchor="center",
        ),
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="height (m)",
            aspectmode="data",
        ),
        showlegend=True,
    )

    if args.out:
        outpath = Path(args.out)
        if not outpath.suffix:
            outpath = outpath.with_suffix(".html")
        fig.write_html(str(outpath))
        print(f"Saved to {outpath}")
    else:
        fig.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
