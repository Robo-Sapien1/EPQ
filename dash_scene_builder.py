"""
Pre-builds all static scene data for the Dash drone sim.
Heavy work (OSM, grids, 3D graph, static traces) runs here once;
dash_drone_sim.py only uses the result and handles sim state + UI updates.
Optional disk cache avoids re-downloading OSM and re-building graphs on reruns.

You do NOT run this file. Run dash_drone_sim.py; it imports this module and
calls get_scene() automatically.
"""

import json
import math
import os
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from scipy.spatial import cKDTree

import plotly.graph_objects as go

from layers_with_divisions import compute_layer_plan, optimize_layer_plan
from fleet_specs import get_l0_cell_m
from l0_height import compute_l0_node_heights, get_l0_max_z
from h_function import compute_h_function, region_boundary_segments, _auto_demand_threshold


# ------------------ SETTINGS (mirror of dash_drone_sim where needed) ------------------

OUTPUT_DIR = Path(__file__).resolve().parent
STEP1_JSON = OUTPUT_DIR / "integrating_step1_output.json"
SOLUTION_JSON = OUTPUT_DIR / "depot_solution_for_overlay.json"
CACHE_PATH = OUTPUT_DIR / "cache" / "dash_scene_cache.pkl"

CENTER_POINT = (51.51861, -0.12583)
RADIUS_M = 1000.0
BOTTOM_LAYER_Z_M = 300.0
LAYER_SPACING_M = 100.0
# L0 cell size = 1.5 × max(H&B length, H&B width); from drone_fleet_specs.json when available
BOTTOM_LAYER_CELL_M = get_l0_cell_m()
GRID_SPACING = get_l0_cell_m()
FLAT_LAYER_Z = BOTTOM_LAYER_Z_M
SIMPLIFY_TOL = 3.0
MAX_BUILDINGS_PLOTTED = None
GRID_DRAW_EVERY = 1
MAX_GRID_EDGES_PLOTTED = None
DEPOT_GRID_TOUCH_THRESHOLD_M = 5.0

# L0 (bottom layer) height above buildings/ground. None = flat L0 at BOTTOM_LAYER_Z_M.
# Method A: base height so at most L0_BUILDING_EXCEED_THRESHOLD of buildings exceed it;
# cells intersected by a building get corners + 3 pts/edge >= clearance above building.
# Second layer = max(L0 z) + 100 m; subsequent layers +100 m each.
L0_HEIGHT_CLEARANCE_M = 20.0   # metres above building/ground; None to disable
L0_BUILDING_EXCEED_THRESHOLD = 0.02   # max fraction of buildings that may exceed base height (0.02 = 2%)
L0_SMOOTH_ITERATIONS = 50 # smooth bumps so neighbors rise gradually (0 = no smoothing)
L0_SMOOTHSTEP_RESHAPE = True   # S-curve roll-off (gradient increases then decreases); False = A-style
L0_PLATEAU_SOFTEN_ITERATIONS = 10   # round off flat-to-slope boundary (0 = disable)
L0_PLATEAU_SOFTEN_ALPHA = 0.35      # blend toward local avg each iteration (0.2–0.5)
L0_PLATEAU_SOFTEN_RADIUS = 2        # neighborhood radius in hops (2 = curved dome tops; 1 = edge-only)

GRID_LAYER_COLOURS = [
    "rgb(0, 90, 255)", "rgb(230, 40, 40)", "rgb(0, 180, 80)",
    "rgb(255, 130, 0)", "rgb(140, 0, 200)", "rgb(0, 200, 200)",
]
DEPOT_GRID_LINE_COLOUR = "rgb(180, 30, 30)"


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


def center_point_27700():
    center_27700 = ox.projection.project_geometry(
        Point(CENTER_POINT[1], CENTER_POINT[0]),
        crs="EPSG:4326",
        to_crs="EPSG:27700"
    )[0]
    return float(center_27700.x), float(center_27700.y)


def load_pipeline_study_area():
    if not STEP1_JSON.exists():
        return None
    with open(STEP1_JSON) as f:
        data = json.load(f)
    xy = data.get("city_polygon_xy")
    if not xy:
        return None
    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    center_proj = data.get("center_projected", [0, 0])
    cx, cy = float(center_proj[0]), float(center_proj[1])
    radius_m = float(data.get("radius_m", RADIUS_M))
    return cx, cy, radius_m, poly


def load_pipeline_solution():
    if not SOLUTION_JSON.exists():
        return None
    with open(SOLUTION_JSON) as f:
        data = json.load(f)
    out = {
        "interior_pts": np.array(data["interior_pts"], dtype=float),
        "weights": np.array(data["weights"], dtype=float),
        "chosen_depots": np.array(data["chosen_depots"], dtype=float),
    }
    if "R" in data:
        out["R"] = float(data["R"])
    return out


def load_demand_density_map():
    """Load the demand_density_map dict from integrating_step1_output.json."""
    if not STEP1_JSON.exists():
        return None
    with open(STEP1_JSON) as f:
        data = json.load(f)
    return data.get("demand_density_map")


def load_buildings(center_latlon, radius_m, simplify_tol) -> gpd.GeoDataFrame:
    print(f"[scene] Downloading OSM buildings within {radius_m} m of {center_latlon} ...")
    b = ox.features_from_point(center_latlon, tags={"building": True}, dist=radius_m)
    b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    b = b.to_crs(epsg=27700)
    if simplify_tol and simplify_tol > 0:
        b["geometry"] = b.geometry.simplify(simplify_tol, preserve_topology=True)
    b["height_m"] = b.apply(get_height, axis=1)
    return b


def load_buildings_in_polygon(city_polygon: Polygon, target_epsg: int = 27700) -> gpd.GeoDataFrame:
    from shapely.geometry import MultiPolygon
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


def build_grid_graph_in_circle(cx, cy, radius_m, cell_size=None, z=None):
    spacing = cell_size if cell_size is not None else GRID_SPACING
    layer_z = z if z is not None else FLAT_LAYER_Z
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
                G.add_node(nid, x=x, y=y, z=layer_z)
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
        "radius": radius_m, "grid_spacing": spacing, "z": layer_z,
    }
    return G, meta


def depot_node(G, meta):
    cx, cy = meta["cx"], meta["cy"]
    best = None
    best_x = float("inf")
    best_dy = float("inf")
    for n in G.nodes:
        x, y = G.nodes[n]["x"], G.nodes[n]["y"]
        if x < best_x - 1e-6:
            best_x, best_dy, best = x, abs(y - cy), n
        elif abs(x - best_x) < 1e-6:
            dy = abs(y - cy)
            if dy < best_dy:
                best_dy, best = dy, n
    return best


def closest_point_on_segment(ax, ay, bx, by, px, py):
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby + 1e-20
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx = ax + t * abx
    cy = ay + t * aby
    return (cx, cy), (px - cx) ** 2 + (py - cy) ** 2


def nearest_edge_to_depot(depot_xy, G_top):
    """Return (point_on_edge, node_a, node_b) for the edge nearest to depot."""
    px, py = float(depot_xy[0]), float(depot_xy[1])
    best_point, best_edge, best_d2 = None, None, float("inf")
    for (a, b) in G_top.edges():
        ax, ay = G_top.nodes[a]["x"], G_top.nodes[a]["y"]
        bx, by = G_top.nodes[b]["x"], G_top.nodes[b]["y"]
        (cx, cy), d2 = closest_point_on_segment(ax, ay, bx, by, px, py)
        if d2 < best_d2:
            best_d2, best_point, best_edge = d2, (cx, cy), (a, b)
    if best_point is None:
        return None, None, None
    return best_point, best_edge[0], best_edge[1]


def depot_link_segments(G_top, depots_xy, z, threshold_m):
    xs, ys, zs = [], [], []
    for (dx, dy) in depots_xy:
        point_on_edge, _, _ = nearest_edge_to_depot((dx, dy), G_top)
        if point_on_edge is None:
            continue
        dist = math.hypot(dx - point_on_edge[0], dy - point_on_edge[1])
        if dist <= threshold_m:
            continue
        xs.extend([dx, point_on_edge[0], None])
        ys.extend([dy, point_on_edge[1], None])
        zs.extend([z, z, None])
    return xs, ys, zs


def node_xyz(G_layers, node_3d):
    layer, nid = node_3d
    nd = G_layers[layer].nodes[nid]
    return nd["x"], nd["y"], nd["z"]


def build_G_3d(G_layers, meta_layers, vertical_epsilon_m=1.0):
    G_3d = nx.Graph()
    if not G_layers:
        return G_3d
    for layer, G in enumerate(G_layers):
        for nid in G.nodes:
            node_3d = (layer, nid)
            x, y, z = G.nodes[nid]["x"], G.nodes[nid]["y"], G.nodes[nid]["z"]
            G_3d.add_node(node_3d, x=x, y=y, z=z, layer=layer)
    for layer, G in enumerate(G_layers):
        for a, b in G.edges():
            w = G.edges[a, b].get("weight", meta_layers[layer].get("grid_spacing", GRID_SPACING))
            G_3d.add_edge((layer, a), (layer, b), weight=w)
    # Vertical edges: use KD-tree so we don't do O(n_lo * n_hi) distance checks
    w_vert = LAYER_SPACING_M
    for layer in range(len(G_layers) - 1):
        G_lo, G_hi = G_layers[layer], G_layers[layer + 1]
        # Progress when building vertical links (was the slow part with many nodes)
        if G_lo.number_of_nodes() > 5000:
            print(f"  connecting layer {layer}→{layer + 1} ({G_lo.number_of_nodes()} nodes)...", end=" ", flush=True)
        sp_lo = meta_layers[layer].get("grid_spacing", GRID_SPACING)
        sp_hi = meta_layers[layer + 1].get("grid_spacing", GRID_SPACING)
        layer_eps2 = (min(sp_lo, sp_hi) * 0.55) ** 2
        layer_eps = math.sqrt(layer_eps2)
        # Build tree over higher-layer (x, y)
        hi_nids = list(G_hi.nodes)
        hi_xy = np.array([[G_hi.nodes[n]["x"], G_hi.nodes[n]["y"]] for n in hi_nids])
        tree = cKDTree(hi_xy)
        # For each lower-layer node, find nearest hi node within epsilon
        lo_xy = np.array([[G_lo.nodes[n]["x"], G_lo.nodes[n]["y"]] for n in G_lo.nodes])
        lo_nids = list(G_lo.nodes)
        indices = tree.query_ball_point(lo_xy, r=layer_eps)
        for i, nid_lo in enumerate(lo_nids):
            js = indices[i]
            if not js:
                continue
            x, y = lo_xy[i, 0], lo_xy[i, 1]
            best_j = min(js, key=lambda j: (x - hi_xy[j, 0]) ** 2 + (y - hi_xy[j, 1]) ** 2)
            best_nid_hi = hi_nids[best_j]
            G_3d.add_edge((layer, nid_lo), (layer + 1, best_nid_hi), weight=w_vert)
        if G_lo.number_of_nodes() > 5000:
            print("done", flush=True)
    return G_3d


def vertical_segments_trace(G_3d, G_layers, color="rgba(100,100,100,0.5)"):
    xs, ys, zs = [], [], []
    for layer in range(len(G_layers) - 1):
        for u, v in G_3d.edges():
            if u[0] == layer and v[0] == layer + 1:
                x1, y1, z1 = node_xyz(G_layers, u)
                x2, y2, z2 = node_xyz(G_layers, v)
                xs.extend([x1, x2, None])
                ys.extend([y1, y2, None])
                zs.extend([z1, z2, None])
    if not xs:
        return None
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(width=2, color=color),
        name="Vertical (layer links)", showlegend=True,
    )


def _buildings_traces(buildings: gpd.GeoDataFrame):
    from shapely.geometry import MultiPolygon
    if len(buildings) == 0:
        return []
    bsub = buildings if MAX_BUILDINGS_PLOTTED is None else buildings.iloc[:MAX_BUILDINGS_PLOTTED]
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
        go.Scatter3d(x=ground_x, y=ground_y, z=ground_z, mode="lines", line=dict(width=1), showlegend=False),
        go.Scatter3d(x=roof_x, y=roof_y, z=roof_z, mode="lines", line=dict(width=1), showlegend=False),
        go.Scatter3d(x=vert_x, y=vert_y, z=vert_z, mode="lines", line=dict(width=1), showlegend=False),
    ]


def _grid_trace(G: nx.Graph, every=1, name="Grid", color=None):
    xs, ys, zs = [], [], []
    edges = list(G.edges())
    if MAX_GRID_EDGES_PLOTTED is not None and len(edges) > MAX_GRID_EDGES_PLOTTED:
        step = max(1, len(edges) // MAX_GRID_EDGES_PLOTTED)
        edges = edges[::step]
    for a, b in edges:
        if every > 1:
            if a[0] == b[0] and a[0] % every != 0:
                continue
            if a[0] != b[0] and a[1] % every != 0:
                continue
        ax, ay, az = G.nodes[a]["x"], G.nodes[a]["y"], G.nodes[a]["z"]
        bx, by, bz = G.nodes[b]["x"], G.nodes[b]["y"], G.nodes[b]["z"]
        xs += [ax, bx, None]
        ys += [ay, by, None]
        zs += [az, bz, None]
    line_style = dict(width=3.5, color=color) if color else dict(width=1)
    opacity = 0.95 if color else 0.25
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=line_style, opacity=opacity, name=name, showlegend=True,
    )


@dataclass
class Scene:
    """All data the Dash app needs: pre-built static traces + graph/sim objects."""
    static_traces: list = field(default_factory=list, repr=False)
    G_layers: list = field(default_factory=list, repr=False)
    meta_layers: list = field(default_factory=list, repr=False)
    G_3d: object = None
    G: object = None
    meta: dict = field(default_factory=dict, repr=False)
    overlay_data: object = None
    buildings: object = None
    depot: object = None
    N_GRID_LAYERS: int = 1
    TOP_LAYER_Z: float = FLAT_LAYER_Z
    layer_plan: object = None
    h_result: object = None          # HFunctionResult (demand-driven regions)
    has_overlay: bool = False
    grid_0_idx: int = 3
    base_traces: int = 0
    cx: float = 0.0
    cy: float = 0.0
    radius_m: float = RADIUS_M


def _build_scene_from_data(cx, cy, radius_m, overlay_data, layer_plan,
                           G_layers, meta_layers, G, meta, buildings, G_3d, depot,
                           N_GRID_LAYERS, TOP_LAYER_Z, h_result=None):
    """Build static trace list and trace indices from already-loaded data."""
    static = []
    # 1) Buildings (3 traces)
    static.extend(_buildings_traces(buildings))
    has_overlay = overlay_data is not None and G_3d is not None and N_GRID_LAYERS > 0
    if has_overlay:
        pts = overlay_data["interior_pts"]
        w = overlay_data["weights"]
        wmin, wmax = float(np.min(w)), float(np.max(w))
        w_norm = (w - wmin) / (wmax - wmin) if wmax > wmin else np.ones_like(w) * 0.5
        static.append(go.Scatter3d(
            x=pts[:, 0].tolist(), y=pts[:, 1].tolist(), z=[TOP_LAYER_Z] * len(pts),
            mode="markers", name="Demand",
            marker=dict(
                size=3, color=w_norm,
                colorscale=[[0, "green"], [0.5, "yellow"], [1, "red"]],
                showscale=True,
                colorbar=dict(orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle",
                              len=0.6, thickness=20, title=dict(text="Demand", side="right"), tickfont=dict(size=11)),
            ),
        ))
        depots = overlay_data["chosen_depots"]
        static.append(go.Scatter3d(
            x=depots[:, 0].tolist(), y=depots[:, 1].tolist(), z=[TOP_LAYER_Z] * len(depots),
            mode="markers", name="Depots (solution)",
            marker=dict(size=10, symbol="diamond", color="crimson"),
        ))
        G_top = G_layers[-1]
        lx, ly, lz = depot_link_segments(
            G_top, depots[:, :2].tolist(), TOP_LAYER_Z, DEPOT_GRID_TOUCH_THRESHOLD_M
        )
        if lx:
            static.append(go.Scatter3d(
                x=lx, y=ly, z=lz, mode="lines",
                line=dict(width=5, color=DEPOT_GRID_LINE_COLOUR, dash="dot"),
                name="Depot -> grid", showlegend=True,
            ))
        else:
            static.append(go.Scatter3d(x=[], y=[], z=[], mode="lines", name="Depot -> grid", showlegend=False))

    # 2) h-function region overlay on the curved L0 surface
    if h_result is not None and G_layers:
        l0_meta = meta_layers[0] if meta_layers else {}
        l0_z = l0_meta.get("z", BOTTOM_LAYER_Z_M)
        G_l0 = G_layers[0]  # the L0 graph with per-node z from building clearance
        boundary_data = region_boundary_segments(
            h_result, z_fallback=l0_z, G_l0=G_l0, z_offset=2.0,
        )
        # Combine all region boundaries into one trace
        all_xs, all_ys, all_zs = [], [], []
        for bd in boundary_data:
            all_xs.extend(bd["xs"])
            all_ys.extend(bd["ys"])
            all_zs.extend(bd["zs"])
        if all_xs:
            static.append(go.Scatter3d(
                x=all_xs, y=all_ys, z=all_zs, mode="lines",
                line=dict(width=4, color="rgb(255, 200, 0)"),
                opacity=0.9,
                name="h-function regions",
                showlegend=True,
            ))

    grid_0_idx = len(static)
    for i, G_layer in enumerate(G_layers):
        layer_color = GRID_LAYER_COLOURS[i % len(GRID_LAYER_COLOURS)] if N_GRID_LAYERS > 1 else None
        static.append(_grid_trace(
            G_layer, every=GRID_DRAW_EVERY,
            name=f"Grid L{i}" if N_GRID_LAYERS > 1 else "Grid",
            color=layer_color,
        ))
    vert_offset = 0
    if G_3d is not None and N_GRID_LAYERS > 1:
        vert_tr = vertical_segments_trace(G_3d, G_layers)
        if vert_tr is not None:
            static.append(vert_tr)
            vert_offset = 1
    base_traces = grid_0_idx + N_GRID_LAYERS + vert_offset
    return static, grid_0_idx, base_traces, has_overlay


def get_scene(use_cache: bool = True) -> Scene:
    """Load or build scene (grids, buildings, 3D graph, static traces). Use cache when possible."""
    cache_dir = CACHE_PATH.parent
    if use_cache and CACHE_PATH.exists():
        try:
            import pickle
            with open(CACHE_PATH, "rb") as f:
                data = pickle.load(f)
            static, grid_0_idx, base_traces, has_overlay = _build_scene_from_data(**data)
            print("[scene] Loaded from cache:", CACHE_PATH)
            return Scene(
                static_traces=static,
                grid_0_idx=grid_0_idx,
                base_traces=base_traces,
                has_overlay=has_overlay,
                **data,
            )
        except Exception as e:
            print("[scene] Cache load failed:", e, "- rebuilding.")

    # Build from scratch
    import time
    t0 = time.perf_counter()
    print("[scene] Building from scratch (no cache)...")
    study = load_pipeline_study_area()
    if study is not None:
        print(f"  study area loaded ({time.perf_counter() - t0:.1f}s)")
        cx, cy, radius_m, city_polygon = study
        overlay_data = load_pipeline_solution()
        R = overlay_data.get("R") if overlay_data else None

        # ---- h-function: demand-driven mesh sizing ----
        density_map = load_demand_density_map()
        h_result = None
        h_effective_S0 = BOTTOM_LAYER_CELL_M  # fallback: original L0 cell size
        if density_map is not None and R is not None:
            t_h = time.perf_counter()
            h_threshold = _auto_demand_threshold(density_map, BOTTOM_LAYER_CELL_M, target_n_regions=30)
            h_result = compute_h_function(
                density_map, BOTTOM_LAYER_CELL_M, cx, cy, radius_m, h_threshold,
            )
            h_effective_S0 = h_result.snapped_cell_m
            print(f"  h-function done: {len(h_result.regions)} regions, "
                  f"effective L1 cell = {h_effective_S0:.4g} m "
                  f"({h_result.snap_n} x {BOTTOM_LAYER_CELL_M} m) "
                  f"({time.perf_counter() - t_h:.1f}s)")

        # ---- Layer plan: use h-function effective size for upper layers ----
        layer_plan = None
        if R is not None:
            layer_plan = optimize_layer_plan(
                R_final=R, S0=h_effective_S0,
                layer_spacing_m=LAYER_SPACING_M, city_radius_m=radius_m,
                verbose=True,
            )
        if layer_plan is not None:
            # layer_plan.layer_sizes[0] is the h-function effective size (for upper layers)
            # L0 always uses the original fine BOTTOM_LAYER_CELL_M
            upper_layer_sizes = layer_plan.layer_sizes  # [h_eff, h_eff*d1, ...]

            # Load buildings first so we can set L0 heights from building clearance
            t_b = time.perf_counter()
            buildings = load_buildings_in_polygon(city_polygon, 27700)
            print(f"  buildings: {len(buildings)} ({time.perf_counter() - t_b:.1f}s)")

            # Build L0 with ORIGINAL fine cell size
            t_layer = time.perf_counter()
            z0_placeholder = BOTTOM_LAYER_Z_M
            G_0, meta_0 = build_grid_graph_in_circle(cx, cy, radius_m, cell_size=BOTTOM_LAYER_CELL_M, z=z0_placeholder)
            print(f"  grid layer 0 (cell={BOTTOM_LAYER_CELL_M:.2f}m, original L0): "
                  f"{G_0.number_of_nodes()} nodes ({time.perf_counter() - t_layer:.1f}s)")

            # Set L0 node heights (Method A: base height + per-cell where building intersects)
            if L0_HEIGHT_CLEARANCE_M is not None and L0_HEIGHT_CLEARANCE_M > 0:
                t_l0 = time.perf_counter()
                L0_max_z = compute_l0_node_heights(
                    buildings, G_0, meta_0, cx, cy, radius_m,
                    clearance_m=L0_HEIGHT_CLEARANCE_M,
                    building_exceed_threshold=L0_BUILDING_EXCEED_THRESHOLD,
                    smooth_iterations=L0_SMOOTH_ITERATIONS,
                    smoothstep_reshape=L0_SMOOTHSTEP_RESHAPE,
                    plateau_soften_iterations=L0_PLATEAU_SOFTEN_ITERATIONS,
                    plateau_soften_alpha=L0_PLATEAU_SOFTEN_ALPHA,
                    plateau_soften_radius=L0_PLATEAU_SOFTEN_RADIUS,
                )
                print(f"  L0 heights set (Method A, clearance={L0_HEIGHT_CLEARANCE_M}m, "
                      f"smooth={L0_SMOOTH_ITERATIONS}, "
                      f"plateau_soften={L0_PLATEAU_SOFTEN_ITERATIONS} r={L0_PLATEAU_SOFTEN_RADIUS}, "
                      f"max_z={L0_max_z:.1f}m) ({time.perf_counter() - t_l0:.1f}s)")
            else:
                L0_max_z = get_l0_max_z(G_0)

            G_layers = [G_0]
            meta_layers = [meta_0]

            # Build upper layers from the layer plan (starting from h-function effective size)
            for i, cell_size in enumerate(upper_layer_sizes):
                t_layer = time.perf_counter()
                z_i = L0_max_z + (i + 1) * LAYER_SPACING_M
                G_i, meta_i = build_grid_graph_in_circle(cx, cy, radius_m, cell_size=cell_size, z=z_i)
                G_layers.append(G_i)
                meta_layers.append(meta_i)
                print(f"  grid layer {i + 1} (cell={cell_size:.1f}m): "
                      f"{G_i.number_of_nodes()} nodes, z={z_i:.0f}m "
                      f"({time.perf_counter() - t_layer:.1f}s)")

            # Use second-from-top as the top layer (drop the current top); demand/depots live on this layer.
            if len(G_layers) >= 2:
                G_layers = G_layers[:-1]
                meta_layers = meta_layers[:-1]
            G, meta = G_layers[0], meta_layers[0]
        else:
            h_result = None
            t_b = time.perf_counter()
            buildings = load_buildings_in_polygon(city_polygon, 27700)
            print(f"  buildings: {len(buildings)} ({time.perf_counter() - t_b:.1f}s)")
            G, meta = build_grid_graph_in_circle(cx, cy, radius_m, cell_size=BOTTOM_LAYER_CELL_M, z=BOTTOM_LAYER_Z_M)
            if L0_HEIGHT_CLEARANCE_M is not None and L0_HEIGHT_CLEARANCE_M > 0:
                compute_l0_node_heights(
                    buildings, G, meta, cx, cy, radius_m,
                    clearance_m=L0_HEIGHT_CLEARANCE_M,
                    building_exceed_threshold=L0_BUILDING_EXCEED_THRESHOLD,
                    smooth_iterations=L0_SMOOTH_ITERATIONS,
                    smoothstep_reshape=L0_SMOOTHSTEP_RESHAPE,
                    plateau_soften_iterations=L0_PLATEAU_SOFTEN_ITERATIONS,
                    plateau_soften_alpha=L0_PLATEAU_SOFTEN_ALPHA,
                    plateau_soften_radius=L0_PLATEAU_SOFTEN_RADIUS,
                )
            G_layers, meta_layers = [G], [meta]
    else:
        print("  using default circle (no pipeline JSONs)")
        cx, cy = center_point_27700()
        radius_m = RADIUS_M
        G, meta = build_grid_graph_in_circle(cx, cy, radius_m, cell_size=GRID_SPACING, z=FLAT_LAYER_Z)
        G_layers, meta_layers = [G], [meta]
        overlay_data = None
        layer_plan = None
        h_result = None
        city_polygon = None
        t_b = time.perf_counter()
        buildings = load_buildings(CENTER_POINT, RADIUS_M, SIMPLIFY_TOL)
        print(f"  buildings: {len(buildings)} ({time.perf_counter() - t_b:.1f}s)")

    depot = depot_node(G, meta)
    N_GRID_LAYERS = len(G_layers)
    # Top layer z: from meta when we have layers (so L0-varying case is correct)
    if N_GRID_LAYERS:
        TOP_LAYER_Z = meta_layers[-1].get("z", BOTTOM_LAYER_Z_M + (N_GRID_LAYERS - 1) * LAYER_SPACING_M)
    else:
        TOP_LAYER_Z = BOTTOM_LAYER_Z_M
    t_3d = time.perf_counter()
    G_3d = build_G_3d(G_layers, meta_layers) if N_GRID_LAYERS else None
    print(f"  G_3d built ({time.perf_counter() - t_3d:.1f}s)")
    has_overlay = overlay_data is not None and G_3d is not None

    t_traces = time.perf_counter()
    build_args = {
        "cx": cx, "cy": cy, "radius_m": radius_m,
        "overlay_data": overlay_data, "layer_plan": layer_plan,
        "G_layers": G_layers, "meta_layers": meta_layers,
        "G": G, "meta": meta, "buildings": buildings, "G_3d": G_3d,
        "depot": depot, "N_GRID_LAYERS": N_GRID_LAYERS, "TOP_LAYER_Z": TOP_LAYER_Z,
        "h_result": h_result,
    }
    static, grid_0_idx, base_traces, has_overlay = _build_scene_from_data(**build_args)
    print(f"  static traces built ({time.perf_counter() - t_traces:.1f}s)")
    print(f"[scene] Total build: {time.perf_counter() - t0:.1f}s")

    if use_cache and cache_dir.exists():
        try:
            import pickle
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(build_args, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("[scene] Cached to", CACHE_PATH)
        except Exception as e:
            print("[scene] Cache write failed:", e)

    return Scene(
        static_traces=static,
        G_layers=G_layers,
        meta_layers=meta_layers,
        G_3d=G_3d,
        G=G,
        meta=meta,
        overlay_data=overlay_data,
        buildings=buildings,
        depot=depot,
        N_GRID_LAYERS=N_GRID_LAYERS,
        TOP_LAYER_Z=TOP_LAYER_Z,
        layer_plan=layer_plan,
        h_result=h_result,
        has_overlay=has_overlay,
        grid_0_idx=grid_0_idx,
        base_traces=base_traces,
        cx=cx, cy=cy, radius_m=radius_m,
    )
