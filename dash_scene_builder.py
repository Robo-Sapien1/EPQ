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

from layers_with_divisions import compute_layer_plan, GridManager, Pathfinder, setup_grid
from fleet_specs import get_l0_cell_m, get_atomic_unit_m, get_drone_dims
from l0_height import compute_l0_node_heights, get_l0_max_z


# ------------------ SETTINGS (mirror of dash_drone_sim where needed) ------------------

OUTPUT_DIR = Path(__file__).resolve().parent
STEP1_JSON = OUTPUT_DIR / "integrating_step1_output.json"
SOLUTION_JSON = OUTPUT_DIR / "depot_solution_for_overlay.json"
CACHE_PATH = OUTPUT_DIR / "cache" / "dash_scene_cache.pkl"

CENTER_POINT = (51.51861, -0.12583)
RADIUS_M = 1000.0
BOTTOM_LAYER_Z_M = 300.0
LAYER_SPACING_M = 100.0
# Atomic unit = AirMatrix cell side (largest drone + padding); L0 cell = 4 * atomic
ATOMIC_UNIT_M = get_atomic_unit_m()
BOTTOM_LAYER_CELL_M = get_l0_cell_m()    # = 4 * ATOMIC_UNIT_M
GRID_SPACING = get_l0_cell_m()           # grid spacing for networkx graph = L0 cell
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


def build_grid_graph_in_circle(cx, cy, radius_m, cell_size=None, z=None,
                               full_box_side=None):
    """
    Build a NetworkX grid graph.

    If *full_box_side* is given (metres), the grid covers a square of
    that side length centred on (cx, cy) -- no circular clipping.
    Otherwise falls back to the original circle-based clipping.
    """
    spacing = cell_size if cell_size is not None else GRID_SPACING
    layer_z = z if z is not None else FLAT_LAYER_Z

    if full_box_side is not None:
        half = full_box_side / 2.0
        xmin, ymin = cx - half, cy - half
        xmax, ymax = cx + half, cy + half
    else:
        effective_radius = radius_m + spacing
        xmin, ymin = cx - effective_radius, cy - effective_radius
        xmax, ymax = cx + effective_radius, cy + effective_radius

    nxn = int((xmax - xmin) // spacing) + 1
    nyn = int((ymax - ymin) // spacing) + 1
    G = nx.Graph()
    inside = set()

    if full_box_side is not None:
        # Full rectangular grid -- no circular clipping
        for ix in range(nxn):
            x = xmin + ix * spacing
            for iy in range(nyn):
                y = ymin + iy * spacing
                nid = (ix, iy)
                inside.add(nid)
                G.add_node(nid, x=x, y=y, z=layer_z)
    else:
        effective_radius = radius_m + spacing
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
    has_overlay: bool = False
    grid_0_idx: int = 3
    base_traces: int = 0
    cx: float = 0.0
    cy: float = 0.0
    radius_m: float = RADIUS_M
    grid_manager: object = None
    pathfinder: object = None
    layer_speeds_mps: object = None
    layer_speed_details: object = None


def _build_scene_from_data(cx, cy, radius_m, overlay_data, layer_plan,
                           G_layers, meta_layers, G, meta, buildings, G_3d, depot,
                           N_GRID_LAYERS, TOP_LAYER_Z,
                           grid_manager=None, pathfinder=None,
                           layer_speeds_mps=None, layer_speed_details=None):
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
        # --- Depots at GROUND LEVEL (z=0) ---
        DEPOT_GROUND_Z = 0.0
        static.append(go.Scatter3d(
            x=depots[:, 0].tolist(), y=depots[:, 1].tolist(),
            z=[DEPOT_GROUND_Z] * len(depots),
            mode="markers", name="Depots (ground)",
            marker=dict(size=10, symbol="diamond", color="crimson"),
        ))
        # --- Depot vertical columns (ground -> top layer) + branch lines ---
        # For each depot, draw:
        #   1) A vertical pole from z=0 up to the highest layer z
        #   2) At each layer, a horizontal branch to the nearest grid node
        col_x, col_y, col_z = [], [], []  # vertical columns
        br_x, br_y, br_z = [], [], []     # horizontal branches
        layer_zs = [meta_layers[i].get("z", 0) for i in range(N_GRID_LAYERS)]
        top_z = max(layer_zs) if layer_zs else TOP_LAYER_Z
        for (dx, dy) in depots[:, :2].tolist():
            # Vertical column
            col_x.extend([dx, dx, None])
            col_y.extend([dy, dy, None])
            col_z.extend([DEPOT_GROUND_Z, top_z, None])
            # Branch lines at each layer: connect to nearest grid EDGE.
            # For L0 (blanket layer), use the actual node z-values at
            # the edge endpoints so the branch sits on the blanket
            # surface rather than floating at the flat meta z.
            for li, G_layer in enumerate(G_layers):
                pt, na, nb = nearest_edge_to_depot((dx, dy), G_layer)
                if pt is None or na is None:
                    continue
                dist = math.hypot(dx - pt[0], dy - pt[1])
                if dist <= 0.5:  # skip if depot sits on the edge
                    continue
                # Compute z at the nearest-edge point by interpolating
                # between the two endpoint z-values.
                ax = G_layer.nodes[na]["x"]
                ay = G_layer.nodes[na]["y"]
                az = G_layer.nodes[na]["z"]
                bx = G_layer.nodes[nb]["x"]
                by = G_layer.nodes[nb]["y"]
                bz = G_layer.nodes[nb]["z"]
                seg_len2 = (bx - ax) ** 2 + (by - ay) ** 2
                if seg_len2 > 1e-9:
                    t = ((pt[0] - ax) * (bx - ax) +
                         (pt[1] - ay) * (by - ay)) / seg_len2
                    t = max(0.0, min(1.0, t))
                else:
                    t = 0.0
                edge_z = az + t * (bz - az)
                br_x.extend([dx, pt[0], None])
                br_y.extend([dy, pt[1], None])
                br_z.extend([edge_z, edge_z, None])

        if col_x:
            static.append(go.Scatter3d(
                x=col_x, y=col_y, z=col_z, mode="lines",
                line=dict(width=4, color="rgba(200,30,30,0.8)"),
                name="Depot columns", showlegend=True,
            ))
        else:
            static.append(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                       name="Depot columns", showlegend=False))
        if br_x:
            static.append(go.Scatter3d(
                x=br_x, y=br_y, z=br_z, mode="lines",
                line=dict(width=3, color="red"),
                name="Depot → grid", showlegend=True,
            ))
        else:
            static.append(go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                       name="Depot → grid", showlegend=False))
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
            # Recreate Pathfinder from cached GridManager (not stored in cache)
            cached_gm = data.get("grid_manager")
            if cached_gm is not None and "pathfinder" not in data:
                data["pathfinder"] = Pathfinder(cached_gm)
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
    layer_speeds_mps, layer_speed_details = None, None
    study = load_pipeline_study_area()
    if study is not None:
        print(f"  study area loaded ({time.perf_counter() - t0:.1f}s)")
        cx, cy, radius_m, city_polygon = study
        overlay_data = load_pipeline_solution()
        R = overlay_data.get("R") if overlay_data else None
        layer_plan = None
        if R is not None:
            # R_final = depot service radius, S0 = atomic unit (padded drone footprint)
            # compute_layer_plan internally computes L0 = 4 * S0
            layer_plan = compute_layer_plan(R_final=R, S0=ATOMIC_UNIT_M, verbose=False)
        if layer_plan is not None:
            layer_sizes = layer_plan.layer_sizes
            num_layers = len(layer_sizes)
            # Load buildings first so we can set L0 heights from building clearance
            t_b = time.perf_counter()
            buildings = load_buildings_in_polygon(city_polygon, 27700)
            print(f"  buildings: {len(buildings)} ({time.perf_counter() - t_b:.1f}s)")
            # Compute the full bounding-box side so ALL grids cover the
            # entire simulation area (not just a circle).  This ensures
            # depot branch-lines always reach a grid node at every layer.
            _n_l0_raw = math.ceil((2.0 * R) / layer_sizes[0])
            def _npow2(n):
                n -= 1
                n |= n >> 1; n |= n >> 2; n |= n >> 4
                n |= n >> 8; n |= n >> 16
                return n + 1
            bbox_side = float(_npow2(_n_l0_raw)) * layer_sizes[0]

            # Build L0 with placeholder z
            t_layer = time.perf_counter()
            z0_placeholder = BOTTOM_LAYER_Z_M
            G_0, meta_0 = build_grid_graph_in_circle(
                cx, cy, radius_m, cell_size=layer_sizes[0], z=z0_placeholder,
                full_box_side=bbox_side,
            )
            print(f"  grid layer 0 (cell={layer_sizes[0]:.1f}m): {G_0.number_of_nodes()} nodes, bbox={bbox_side:.0f}m ({time.perf_counter() - t_layer:.1f}s)")
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
                print(f"  L0 heights set (Method A, clearance={L0_HEIGHT_CLEARANCE_M}m, smooth={L0_SMOOTH_ITERATIONS}, plateau_soften={L0_PLATEAU_SOFTEN_ITERATIONS} r={L0_PLATEAU_SOFTEN_RADIUS}, max_z={L0_max_z:.1f}m) ({time.perf_counter() - t_l0:.1f}s)")
            else:
                L0_max_z = get_l0_max_z(G_0)
            # Update L0 meta z to reflect the actual blanket height
            meta_0["z"] = L0_max_z
            G_layers = [G_0]
            meta_layers = [meta_0]
            # Build upper layers: flat at L0_max_z + 100m, then +100m each
            for i in range(1, num_layers):
                t_layer = time.perf_counter()
                z_i = L0_max_z + i * LAYER_SPACING_M
                G_i, meta_i = build_grid_graph_in_circle(
                    cx, cy, radius_m, cell_size=layer_sizes[i], z=z_i,
                    full_box_side=bbox_side,
                )
                G_layers.append(G_i)
                meta_layers.append(meta_i)
                print(f"  grid layer {i} (cell={layer_sizes[i]:.1f}m): {G_i.number_of_nodes()} nodes, z={z_i:.0f}m ({time.perf_counter() - t_layer:.1f}s)")
            # All layers are kept — the actual top layer is the coarsest.
            # Demand/depots live on the top layer.  (Old code dropped the
            # top and used second-from-top; that workaround is removed.)
            G, meta = G_layers[0], meta_layers[0]

            # -- Build GridManager (full_box) for the Pathfinder --
            # CRITICAL: use the SAME altitudes as the NetworkX visual grids
            # so that drone paths align with the drawn layers.
            # L0 is at L0_max_z (the blanket height); upper layers at
            # L0_max_z + i * LAYER_SPACING_M  (same formula as the
            # build_grid_graph_in_circle calls above).
            gm_layer_altitudes = [L0_max_z + i * LAYER_SPACING_M
                                  for i in range(num_layers)]
            t_gm = time.perf_counter()
            gm = setup_grid(
                depot_radius=R,
                atomic_unit_size=ATOMIC_UNIT_M,
                centre_x=cx, centre_y=cy,
                full_box=True,
                layer_altitudes=gm_layer_altitudes,
                verbose=False,
            )
            pf = Pathfinder(gm)
            print(f"  GridManager (full_box) + Pathfinder built "
                  f"({sum(l.total_cells for l in gm.layers):,} cells, "
                  f"altitudes={[f'{z:.0f}' for z in gm_layer_altitudes]}) "
                  f"({time.perf_counter() - t_gm:.1f}s)")

            # --- Compute per-layer speeds (post-construction) ---
            # Uses CORRIDRONE tradeoff: faster -> bigger geofence -> lower packing capacity.
            speed_report = gm.compute_layer_speeds_mps(
                get_drone_dims("H&B"),
                traffic_intensity=0.7,
                edge_traversal_time_low_s=6.0,
                edge_traversal_time_high_s=2.0,
                speed_cap_mps=23.0,
            )
            layer_speeds_mps = speed_report.get("speeds_mps")
            layer_speed_details = speed_report.get("details")
        else:
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
            gm, pf = None, None
            layer_speeds_mps, layer_speed_details = None, None
    else:
        print("  using default circle (no pipeline JSONs)")
        cx, cy = center_point_27700()
        radius_m = RADIUS_M
        G, meta = build_grid_graph_in_circle(cx, cy, radius_m, cell_size=GRID_SPACING, z=FLAT_LAYER_Z)
        G_layers, meta_layers = [G], [meta]
        overlay_data = None
        layer_plan = None
        city_polygon = None
        gm, pf = None, None
        layer_speeds_mps, layer_speed_details = None, None
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
        "grid_manager": gm, "pathfinder": pf,
        "layer_speeds_mps": layer_speeds_mps,
        "layer_speed_details": layer_speed_details,
    }
    static, grid_0_idx, base_traces, has_overlay = _build_scene_from_data(**build_args)
    print(f"  static traces built ({time.perf_counter() - t_traces:.1f}s)")
    print(f"[scene] Total build: {time.perf_counter() - t0:.1f}s")

    if use_cache and cache_dir.exists():
        try:
            import pickle
            # Cache build_args minus non-picklable pathfinder (it's cheap to recreate)
            cache_args = {k: v for k, v in build_args.items() if k not in ("pathfinder",)}
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(cache_args, f, protocol=pickle.HIGHEST_PROTOCOL)
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
        has_overlay=has_overlay,
        grid_0_idx=grid_0_idx,
        base_traces=base_traces,
        cx=cx, cy=cy, radius_m=radius_m,
        grid_manager=gm, pathfinder=pf,
        layer_speeds_mps=layer_speeds_mps,
        layer_speed_details=layer_speed_details,
    )
