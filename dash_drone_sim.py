"""
Dash drone simulation UI.
Heavy scene data (OSM, grids, static traces) is built/cached in dash_scene_builder.
This module only: loads scene, runs sim state, and relays updates to the figure.
"""

import math
import os
import random

import numpy as np
import networkx as nx
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State
from dash import Patch  # Dash >= 2.9

from dash_scene_builder import get_scene
from fleet_specs import get_l0_cell_m, get_drone_dims

# ------------------ SETTINGS (sim + UI only) ------------------

TICK_MS = 300   # tick interval; slightly higher reduces callback load
MAX_DRONE_SLOTS = 80
N_DRONES = 3
BASE_STEP_M = 3.0
RNG_SEED = 123
# Collision avoidance: set False to deactivate SAFE_SEP and segment-crossing checks (drones ignore each other)
COLLISIONS_ENABLED = False
SAFE_SEP_M = 2.5
DELIVERY_SPAWN_TICKS_MIN = 4
# Scale factor for drawn drone box so they stay visible when not zoomed in (sim logic unchanged)
DRONE_DISPLAY_SCALE = 4.0
DELIVERY_SPAWN_TICKS_MAX = 20
# Use Euclidean distance to pick depot (fast, no lag). Set True for path-length-based choice (accurate but laggy on spawn).
USE_PATH_BASED_DEPOT = False
# L0 cell size from drone_fleet_specs.json (1.5 × max H&B L/W); fallback 15.0
GRID_SPACING = get_l0_cell_m(fallback_m=15.0)
LAYER_SPACING_M = 100.0
DRONE_TYPES = ["Standard", "Oversize", "H&B"]

DRONE_COLOURS = [
    "rgb(30, 100, 255)",
    "rgb(220, 0, 160)",
    "rgb(255, 150, 0)",
]
FLAT_LAYER_Z = 300.0  # delivery marker height

# ------------------ SCENE (load once; all static data lives here) ------------------

scene = get_scene()
G = scene.G
meta = scene.meta
G_layers = scene.G_layers
G_3d = scene.G_3d
overlay_data = scene.overlay_data
depot = scene.depot
N_GRID_LAYERS = scene.N_GRID_LAYERS
TOP_LAYER_Z = scene.TOP_LAYER_Z
layer_plan = scene.layer_plan

rng = random.Random(RNG_SEED)

# ------------------ COLLISIONS ------------------


def dist2_xy(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy


def segments_cross(p0, p1, q0, q1):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        return (min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 and
                min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9)

    o1, o2 = orient(p0, p1, q0), orient(p0, p1, q1)
    o3, o4 = orient(q0, q1, p0), orient(q0, q1, p1)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    if abs(o1) < 1e-9 and on_segment(p0, p1, q0): return True
    if abs(o2) < 1e-9 and on_segment(p0, p1, q1): return True
    if abs(o3) < 1e-9 and on_segment(q0, q1, p0): return True
    if abs(o4) < 1e-9 and on_segment(q0, q1, p1): return True
    return False


# ------------------ ROUTING (uses scene) ------------------


def closest_point_on_segment(ax, ay, bx, by, px, py):
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby + 1e-20
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx = ax + t * abx
    cy = ay + t * aby
    return (cx, cy), (px - cx) ** 2 + (py - cy) ** 2


def nearest_edge_to_depot(depot_xy, G_top):
    """Return (point_on_edge, node_a, node_b) for the edge nearest to depot. Both endpoints returned so caller can choose best entry for the goal."""
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


def random_point_in_circle(rng, cx, cy, radius_m):
    ang = rng.uniform(0, 2 * math.pi)
    r = radius_m * math.sqrt(rng.uniform(0, 1))
    return cx + r * math.cos(ang), cy + r * math.sin(ang)


def nearest_corner_candidates(G, meta, x, y):
    xmin, ymin = meta["xmin"], meta["ymin"]
    spacing = meta.get("grid_spacing", GRID_SPACING)
    ix = int((x - xmin) // spacing)
    iy = int((y - ymin) // spacing)
    corners = [(ix, iy), (ix + 1, iy), (ix, iy + 1), (ix + 1, iy + 1)]
    corners = [c for c in corners if c in G.nodes]
    if corners:
        return corners
    best, best_d = None, float("inf")
    for n in G.nodes:
        dx = G.nodes[n]["x"] - x
        dy = G.nodes[n]["y"] - y
        d = dx * dx + dy * dy
        if d < best_d:
            best_d, best = d, n
    return [best]


def astar_path(G, start, goal):
    def h(u, v):
        ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
        vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
        return abs(ux - vx) + abs(uy - vy)
    return nx.astar_path(G, start, goal, heuristic=h, weight="weight")


def node_xyz(G_layers, node_3d):
    layer, nid = node_3d
    nd = G_layers[layer].nodes[nid]
    return nd["x"], nd["y"], nd["z"]


def nearest_node_3d(G_3d, x, y, layer=None):
    best, best_d2 = None, float("inf")
    for n in G_3d.nodes:
        if layer is not None and n[0] != layer:
            continue
        nx_, ny_ = G_3d.nodes[n]["x"], G_3d.nodes[n]["y"]
        d2 = (x - nx_) ** 2 + (y - ny_) ** 2
        if d2 < best_d2:
            best_d2, best = d2, n
    return best, math.sqrt(best_d2) if best is not None else float("inf")


def astar_path_3d(G_3d, start_3d, goal_3d):
    def h(u, v):
        ux, uy, uz = G_3d.nodes[u]["x"], G_3d.nodes[u]["y"], G_3d.nodes[u]["z"]
        vx, vy, vz = G_3d.nodes[v]["x"], G_3d.nodes[v]["y"], G_3d.nodes[v]["z"]
        return math.sqrt((vx - ux) ** 2 + (vy - uy) ** 2 + (vz - uz) ** 2)
    return nx.astar_path(G_3d, start_3d, goal_3d, heuristic=h, weight="weight")


def interpolate_segment_3d(p0, p1, step_m):
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    seg_len = math.hypot(x1 - x0, y1 - y0, z1 - z0)
    if seg_len < 1e-9:
        return []
    steps = max(1, int(seg_len / step_m))
    positions = []
    for s in range(1, steps + 1):
        t = min(1.0, (s * step_m) / seg_len)
        positions.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), z0 + t * (z1 - z0)))
    return positions


def path_positions_xyz_3d(G_3d, path_nodes_3d, step_m):
    coords = [(G_3d.nodes[n]["x"], G_3d.nodes[n]["y"], G_3d.nodes[n]["z"]) for n in path_nodes_3d]
    positions = []
    for i in range(len(coords) - 1):
        x0, y0, z0 = coords[i]
        x1, y1, z1 = coords[i + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0, z1 - z0)
        if seg_len < 1e-9:
            continue
        steps = max(1, int(seg_len // step_m))
        for s in range(steps):
            t = min(1.0, (s * step_m) / seg_len)
            positions.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), z0 + t * (z1 - z0)))
    positions.append(coords[-1])
    return positions


def path_positions_xyz(G, path_nodes, step_m):
    coords = [(G.nodes[n]["x"], G.nodes[n]["y"], G.nodes[n]["z"]) for n in path_nodes]
    positions = []
    for i in range(len(coords) - 1):
        x0, y0, z0 = coords[i]
        x1, y1, z1 = coords[i + 1]
        dx, dy = x1 - x0, y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9:
            continue
        steps = max(1, int(seg_len // step_m))
        for s in range(steps):
            t = min(1.0, (s * step_m) / seg_len)
            positions.append((x0 + t * dx, y0 + t * dy, z0 + t * (z1 - z0)))
    positions.append(coords[-1])
    return positions


# ------------------ SIM STATE ------------------


def nearest_depot_idx(depots_xy, x, y):
    best_i, best_d2 = 0, float("inf")
    for i, (dx, dy) in enumerate(depots_xy):
        d2 = (x - dx) ** 2 + (y - dy) ** 2
        if d2 < best_d2:
            best_d2, best_i = d2, i
    return best_i


def nearest_depot_idx_by_path(depots_xy, delivery_xy):
    if G_3d is None or N_GRID_LAYERS == 0:
        return nearest_depot_idx(depots_xy, delivery_xy[0], delivery_xy[1])
    dx, dy = delivery_xy
    goal_3d, _ = nearest_node_3d(G_3d, dx, dy, layer=0)
    if goal_3d is None:
        return 0
    G_top = G_layers[-1]
    best_i, best_len = 0, float("inf")
    for i, (dep_x, dep_y) in enumerate(depots_xy):
        dep_x, dep_y = float(dep_x), float(dep_y)
        point_on_edge, node_a, node_b = nearest_edge_to_depot((dep_x, dep_y), G_top)
        if point_on_edge is None:
            continue
        leg1 = math.hypot(dep_x - point_on_edge[0], dep_y - point_on_edge[1])
        depot_best = float("inf")
        for start_node in (node_a, node_b):
            start_3d = (N_GRID_LAYERS - 1, start_node)
            try:
                path_3d = astar_path_3d(G_3d, start_3d, goal_3d)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            sx, sy = G_3d.nodes[start_3d]["x"], G_3d.nodes[start_3d]["y"]
            leg2 = math.hypot(point_on_edge[0] - sx, point_on_edge[1] - sy)
            path_weight = sum(
                G_3d.edges[path_3d[k], path_3d[k + 1]].get("weight", LAYER_SPACING_M)
                for k in range(len(path_3d) - 1)
            )
            total = leg1 + leg2 + path_weight
            if total < depot_best:
                depot_best = total
        if depot_best < best_len:
            best_len, best_i = depot_best, i
    return best_i


def new_mission_3d(delivery_xy, depot_xy, start_tick=0):
    if G_3d is None or N_GRID_LAYERS == 0:
        return new_mission(start_tick=start_tick)
    dx, dy = delivery_xy
    dep_x, dep_y = float(depot_xy[0]), float(depot_xy[1])
    G_top = G_layers[-1]
    point_on_edge, node_a, node_b = nearest_edge_to_depot((dep_x, dep_y), G_top)
    if point_on_edge is None:
        return new_mission(start_tick=start_tick)
    goal_3d, _ = nearest_node_3d(G_3d, dx, dy, layer=0)
    if goal_3d is None:
        return new_mission(start_tick=start_tick)
    # Use the edge endpoint that gives the shortest total path to the goal (avoids going to wrong corner then backtracking).
    leg_depot_to_edge = math.hypot(dep_x - point_on_edge[0], dep_y - point_on_edge[1])
    best_start_node = None
    best_total = float("inf")
    for node in (node_a, node_b):
        start_3d = (N_GRID_LAYERS - 1, node)
        leg_edge_to_node = math.hypot(
            point_on_edge[0] - G_3d.nodes[start_3d]["x"],
            point_on_edge[1] - G_3d.nodes[start_3d]["y"],
        )
        try:
            path_3d = astar_path_3d(G_3d, start_3d, goal_3d)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        path_len = sum(
            G_3d.edges[path_3d[k], path_3d[k + 1]].get("weight", LAYER_SPACING_M)
            for k in range(len(path_3d) - 1)
        )
        total = leg_depot_to_edge + leg_edge_to_node + path_len
        if total < best_total:
            best_total, best_start_node = total, node
    if best_start_node is None:
        return new_mission(start_tick=start_tick)
    start_3d = (N_GRID_LAYERS - 1, best_start_node)
    try:
        path_3d = astar_path_3d(G_3d, start_3d, goal_3d)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return new_mission(start_tick=start_tick)
    path_positions = path_positions_xyz_3d(G_3d, path_3d, BASE_STEP_M)
    depot_start = (dep_x, dep_y, TOP_LAYER_Z)
    point_on_edge_3d = (point_on_edge[0], point_on_edge[1], TOP_LAYER_Z)
    straight_leg = interpolate_segment_3d(depot_start, point_on_edge_3d, BASE_STEP_M)
    positions = [depot_start] + straight_leg + path_positions
    # Precompute route lists for fast patching (no list comp every tick)
    route_x = [p[0] for p in positions]
    route_y = [p[1] for p in positions]
    route_z = [p[2] for p in positions]
    return {
        "delivery_xy": (dx, dy),
        "drone_type": rng.choice(DRONE_TYPES),
        "path_nodes": [path_3d[0][1]],
        "path_3d": path_3d,
        "positions": positions,
        "route_x": route_x,
        "route_y": route_y,
        "route_z": route_z,
        "pos_i": 0,
        "start_tick": int(start_tick),
    }


def new_mission(start_tick=0):
    dx, dy = random_point_in_circle(rng, meta["cx"], meta["cy"], meta["radius"])
    candidates = nearest_corner_candidates(G, meta, dx, dy)
    spacing = meta.get("grid_spacing", GRID_SPACING)
    best_path, best_len = None, float("inf")
    for c in candidates:
        p = astar_path(G, depot, c)
        dist = (len(p) - 1) * spacing
        if dist < best_len:
            best_len, best_path = dist, p
    positions = path_positions_xyz(G, best_path, BASE_STEP_M)
    route_x = [G.nodes[n]["x"] for n in best_path]
    route_y = [G.nodes[n]["y"] for n in best_path]
    route_z = [G.nodes[n]["z"] for n in best_path]
    return {
        "delivery_xy": (dx, dy),
        "drone_type": rng.choice(DRONE_TYPES),
        "path_nodes": best_path,
        "path_3d": None,
        "positions": positions,
        "route_x": route_x,
        "route_y": route_y,
        "route_z": route_z,
        "pos_i": 0,
        "start_tick": int(start_tick),
    }


def drone_idle_at_depot(depot_xy):
    x, y = depot_xy[0], depot_xy[1]
    pos = (float(x), float(y), TOP_LAYER_Z)
    return {
        "delivery_xy": None,
        "drone_type": "Standard",
        "depot_xy": (float(x), float(y)),
        "path_nodes": [],
        "path_3d": None,
        "positions": [pos],
        "route_x": [x, x],
        "route_y": [y, y],
        "route_z": [TOP_LAYER_Z, TOP_LAYER_Z],
        "pos_i": 0,
        "start_tick": 0,
    }


def is_drone_idle(d):
    if d.get("delivery_xy") is None:
        return True
    return d["pos_i"] >= len(d["positions"]) - 1


def init_sim_state():
    if overlay_data is not None and G_3d is not None:
        ticks_until_spawn = rng.randint(DELIVERY_SPAWN_TICKS_MIN, DELIVERY_SPAWN_TICKS_MAX)
        return {"tick": 0, "drones": [], "ticks_until_spawn": ticks_until_spawn}
    drones = [new_mission(start_tick=0) for _ in range(N_DRONES)]
    return {"tick": 0, "drones": drones}


# ------------------ TRACE INDEX MAP ------------------

BASE_TRACES = scene.base_traces


def route_idx(i):
    return BASE_TRACES + 3 * i + 0


def delivery_idx(i):
    return BASE_TRACES + 3 * i + 1


def drone_idx(i):
    return BASE_TRACES + 3 * i + 2


_nan = float("nan")
EMPTY_POINT = ([_nan], [_nan], [_nan])
EMPTY_ROUTE = ([_nan, _nan], [_nan, _nan], [_nan, _nan])


def box_edges_xyz(cx, cy, cz, L, W, H):
    """Return (xs, ys, zs) for the 12 edges of a cuboid (L×W×H) centred at (cx, cy, cz)."""
    hL, hW, hH = L / 2, W / 2, H / 2
    v = [
        (cx - hL, cy - hW, cz - hH), (cx + hL, cy - hW, cz - hH), (cx + hL, cy + hW, cz - hH), (cx - hL, cy + hW, cz - hH),
        (cx - hL, cy - hW, cz + hH), (cx + hL, cy - hW, cz + hH), (cx + hL, cy + hW, cz + hH), (cx - hL, cy + hW, cz + hH),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for i, j in edges:
        xs.extend([v[i][0], v[j][0], None])
        ys.extend([v[i][1], v[j][1], None])
        zs.extend([v[i][2], v[j][2], None])
    return xs, ys, zs


# ------------------ BUILD FIGURE (static from scene + drone slots) ------------------


def make_figure(sim_state):
    fig = go.Figure()
    for tr in scene.static_traces:
        fig.add_trace(tr)
    drones_list = sim_state.get("drones", [])
    for i in range(MAX_DRONE_SLOTS):
        if i >= len(drones_list):
            rx, ry, rz = EMPTY_ROUTE
            dx, dy = _nan, _nan
            drone_type = "Standard"
            x0, y0, z0 = _nan, _nan, _nan
        else:
            d = drones_list[i]
            rx = d.get("route_x")
            ry = d.get("route_y")
            rz = d.get("route_z")
            if rx is None and d.get("positions"):
                poss = d["positions"]
                rx = [p[0] for p in poss]
                ry = [p[1] for p in poss]
                rz = [p[2] for p in poss]
            elif rx is None and d.get("depot_xy"):
                dx_, dy_ = d["depot_xy"]
                rx, ry, rz = [dx_, dx_], [dy_, dy_], [TOP_LAYER_Z, TOP_LAYER_Z]
            elif rx is None:
                path_nodes = d["path_nodes"]
                rx = [G.nodes[n]["x"] for n in path_nodes]
                ry = [G.nodes[n]["y"] for n in path_nodes]
                rz = [G.nodes[n]["z"] for n in path_nodes]
            dx, dy = d.get("delivery_xy") or d.get("depot_xy", (0, 0))
            drone_type = d.get("drone_type") or "Standard"
            x0, y0, z0 = d["positions"][min(d["pos_i"], len(d["positions"]) - 1)]
        L, W, H = get_drone_dims(drone_type)
        Ld, Wd, Hd = L * DRONE_DISPLAY_SCALE, W * DRONE_DISPLAY_SCALE, H * DRONE_DISPLAY_SCALE
        box_x, box_y, box_z = box_edges_xyz(x0, y0, z0, Ld, Wd, Hd)
        path_color = DRONE_COLOURS[i % len(DRONE_COLOURS)]
        fig.add_trace(go.Scatter3d(
            x=rx, y=ry, z=rz, mode="lines",
            line=dict(width=6, color=path_color), opacity=0.9, showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[dx], y=[dy], z=[FLAT_LAYER_Z + 2],
            mode="markers", marker=dict(size=9, symbol="x"), showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=box_x, y=box_y, z=box_z,
            mode="lines", line=dict(width=2.5, color=path_color), opacity=0.95, showlegend=False,
        ))
    fig.update_layout(
        title=dict(text="", font=dict(size=1)),
        uirevision="keep-camera",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title="height (m)"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=95, b=0),
        legend=dict(itemsizing="constant", orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5),
    )
    return fig


# ------------------ DASH APP ------------------

app = Dash(__name__)
sim0 = init_sim_state()
fig0 = make_figure(sim0)

app.layout = html.Div([
    html.Div([
        html.Span("Simulation: ", style={"fontWeight": "600", "marginRight": "8px"}),
        html.Button("Play / Pause", id="btn-play", n_clicks=0, style={"marginRight": "16px"}),
        html.Label("Speed", style={"marginRight": "6px"}),
        dcc.Slider(
            id="speed",
            min=1, max=30, step=1, value=6,
            marks={1: "1×", 6: "6×", 12: "12×", 20: "20×", 30: "30×"},
        ),
    ], style={"padding": "12px 16px", "borderBottom": "1px solid #ddd", "backgroundColor": "#fafafa"}),
    html.Div([
        dcc.Graph(id="graph", figure=fig0, style={"height": "85vh", "marginTop": "0"}),
    ], style={"position": "relative"}),
    dcc.Interval(id="tick", interval=TICK_MS, n_intervals=0, disabled=True),
    dcc.Store(id="sim-state", data=sim0),
    dcc.Store(id="is-running", data=False),
])


@app.callback(
    Output("tick", "disabled"),
    Output("is-running", "data"),
    Input("btn-play", "n_clicks"),
    State("is-running", "data"),
    prevent_initial_call=True,
)
def toggle_play(n_clicks, is_running):
    running = not bool(is_running)
    return (not running), running


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("sim-state", "data"),
    Input("tick", "n_intervals"),
    State("sim-state", "data"),
    State("speed", "value"),
    prevent_initial_call=True,
)
def tick(n, sim_state, speed):
    patch = Patch()
    steps_to_advance = max(1, int(speed))
    sim_tick = int(sim_state.get("tick", 0))
    drones_list = sim_state["drones"]
    n_drones = len(drones_list)
    use_overlay = overlay_data is not None and G_3d is not None

    if COLLISIONS_ENABLED and not use_overlay and n_drones <= N_DRONES:
        sep2 = SAFE_SEP_M * SAFE_SEP_M
        for _ in range(steps_to_advance):
            cur_pos = {}
            nxt_pos = {}
            can_move = {}
            for i, d in enumerate(drones_list):
                active = sim_tick >= int(d.get("start_tick", 0))
                pi = min(d["pos_i"], len(d["positions"]) - 1)
                cur = d["positions"][pi]
                cur_pos[i] = cur
                if active and d["pos_i"] < len(d["positions"]) - 1:
                    ni = d["pos_i"] + 1
                    nxt_pos[i] = d["positions"][ni]
                    can_move[i] = True
                else:
                    nxt_pos[i], can_move[i] = cur, False
            order = list(range(n_drones))
            approved_next = {i: cur_pos[i] for i in order}
            moved = {i: False for i in order}
            for i in order:
                if not can_move[i]:
                    continue
                cand = nxt_pos[i]
                seg_i0 = (cur_pos[i][0], cur_pos[i][1])
                seg_i1 = (cand[0], cand[1])
                conflict = False
                for j in order:
                    if j == i:
                        continue
                    pj = approved_next[j]
                    seg_j0 = (cur_pos[j][0], cur_pos[j][1])
                    seg_j1 = (pj[0], pj[1])
                    if dist2_xy((cand[0], cand[1]), (pj[0], pj[1])) < sep2:
                        conflict = True
                        break
                    if moved[j] and segments_cross(seg_i0, seg_i1, seg_j0, seg_j1):
                        conflict = True
                        break
                if not conflict:
                    approved_next[i], moved[i] = cand, True
            for i, d in enumerate(drones_list):
                if moved[i]:
                    d["pos_i"] += 1
    else:
        for _ in range(steps_to_advance):
            for d in drones_list:
                if sim_tick >= int(d.get("start_tick", 0)) and d["pos_i"] < len(d["positions"]) - 1:
                    d["pos_i"] += 1

    if use_overlay:
        ticks_until = sim_state.get("ticks_until_spawn", DELIVERY_SPAWN_TICKS_MIN)
        ticks_until -= 1
        if ticks_until <= 0:
            dx, dy = random_point_in_circle(rng, meta["cx"], meta["cy"], meta["radius"])
            depots_xy = overlay_data["chosen_depots"]
            depot_idx = (
                nearest_depot_idx_by_path(depots_xy, (dx, dy))
                if USE_PATH_BASED_DEPOT
                else nearest_depot_idx(depots_xy, dx, dy)
            )
            depot_xy = (float(depots_xy[depot_idx][0]), float(depots_xy[depot_idx][1]))
            if len(drones_list) < MAX_DRONE_SLOTS:
                drones_list.append(new_mission_3d((float(dx), float(dy)), depot_xy, start_tick=sim_tick))
            ticks_until = rng.randint(DELIVERY_SPAWN_TICKS_MIN, DELIVERY_SPAWN_TICKS_MAX)
        sim_state["ticks_until_spawn"] = ticks_until
        sim_state["drones"] = [d for d in drones_list if d["pos_i"] < len(d["positions"]) - 1]
        drones_list = sim_state["drones"]
    else:
        depots_xy = overlay_data["chosen_depots"] if overlay_data is not None else None
        for i, d in enumerate(drones_list):
            if d["pos_i"] >= len(d["positions"]) - 1:
                sim_state["drones"][i] = new_mission(start_tick=sim_tick)

    n_drones = len(sim_state["drones"])
    for i in range(MAX_DRONE_SLOTS):
        if i >= n_drones:
            patch["data"][route_idx(i)]["x"], patch["data"][route_idx(i)]["y"], patch["data"][route_idx(i)]["z"] = EMPTY_ROUTE
            patch["data"][delivery_idx(i)]["x"], patch["data"][delivery_idx(i)]["y"], patch["data"][delivery_idx(i)]["z"] = EMPTY_POINT
            box_x, box_y, box_z = box_edges_xyz(_nan, _nan, _nan, 0.01, 0.01, 0.01)
            patch["data"][drone_idx(i)]["x"], patch["data"][drone_idx(i)]["y"], patch["data"][drone_idx(i)]["z"] = box_x, box_y, box_z
        else:
            d = sim_state["drones"][i]
            rx = d.get("route_x")
            ry = d.get("route_y")
            rz = d.get("route_z")
            if rx is not None:
                patch["data"][route_idx(i)]["x"] = rx
                patch["data"][route_idx(i)]["y"] = ry
                patch["data"][route_idx(i)]["z"] = rz
            else:
                if d.get("positions"):
                    poss = d["positions"]
                    patch["data"][route_idx(i)]["x"] = [p[0] for p in poss]
                    patch["data"][route_idx(i)]["y"] = [p[1] for p in poss]
                    patch["data"][route_idx(i)]["z"] = [p[2] for p in poss]
                else:
                    path_nodes = d["path_nodes"]
                    patch["data"][route_idx(i)]["x"] = [G.nodes[n]["x"] for n in path_nodes]
                    patch["data"][route_idx(i)]["y"] = [G.nodes[n]["y"] for n in path_nodes]
                    patch["data"][route_idx(i)]["z"] = [G.nodes[n]["z"] for n in path_nodes]
            dx, dy = d.get("delivery_xy") or d.get("depot_xy", (0, 0))
            patch["data"][delivery_idx(i)]["x"] = [dx]
            patch["data"][delivery_idx(i)]["y"] = [dy]
            patch["data"][delivery_idx(i)]["z"] = [FLAT_LAYER_Z + 2]
            pi = min(d["pos_i"], len(d["positions"]) - 1)
            x, y, z = d["positions"][pi]
            L, W, H = get_drone_dims(d.get("drone_type") or "Standard")
            Ld, Wd, Hd = L * DRONE_DISPLAY_SCALE, W * DRONE_DISPLAY_SCALE, H * DRONE_DISPLAY_SCALE
            box_x, box_y, box_z = box_edges_xyz(x, y, z, Ld, Wd, Hd)
            patch["data"][drone_idx(i)]["x"], patch["data"][drone_idx(i)]["y"], patch["data"][drone_idx(i)]["z"] = box_x, box_y, box_z

    sim_state["tick"] = sim_tick + 1
    return patch, sim_state


def _print_init_once(debug=True):
    if debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return
    if layer_plan is not None:
        print(f"Pipeline: {N_GRID_LAYERS} sky layers (bottom 300m, +100m), buildings in polygon")
        print("  Layer plan: number of layers =", N_GRID_LAYERS)
        print("  Cell size (m) per layer:", [round(s, 2) for s in layer_plan.layer_sizes])
    else:
        print("Fallback: center+radius, single grid layer")
    print("Grid nodes (bottom layer):", G.number_of_nodes(), "Grid edges:", G.number_of_edges())


server = app.server
if __name__ == "__main__":
    _print_init_once(debug=True)
    app.run(debug=True)
