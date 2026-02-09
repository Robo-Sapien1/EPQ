import time
import math
import random
import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd
import plotly.graph_objects as go

from shapely.geometry import Polygon, MultiPolygon, Point

from fleet_specs import get_l0_cell_m


# ------------------ SETTINGS ------------------

CENTER_POINT = (51.51861, -0.12583)  # (lat, lon)
RADIUS_M = 5000.0                    # simulation radius circle
# L0 cell size from drone_fleet_specs.json (1.5 × max H&B L/W); fallback 15.0
GRID_SPACING = get_l0_cell_m(fallback_m=15.0)
FLAT_LAYER_Z = 200.0

# Buildings (visual only)
SIMPLIFY_TOL = 3.0
MAX_BUILDINGS_PLOTTED = None  # set e.g. 800 if slow

# Grid draw density (reduces moire/squiggles + speeds rendering)
GRID_DRAW_EVERY = 1  # draw every 4th grid line (try 3, 5, etc.)

# Drones
N_DELIVERIES_AT_ONCE = 3
DRONE_STEP_M = 50
DRONE_STEP_DT = 0.25  # seconds per step (THIS is the base speed: 1.0x)

# Plotly animation: how many batches to prebuild
BATCHES_TO_ANIMATE = 5

RNG_SEED = 123

# Speed slider options (0.5x to 10x)
# Speed slider options (0.5x to 30x)
SPEED_OPTIONS = list(range(1, 31))



# ------------------ BUILDING HEIGHT ESTIMATOR (visual only) ------------------

def get_height(row):
    h = row.get("height")
    if isinstance(h, str):
        h_clean = h.replace("m", "").strip()
        try:
            return float(h_clean)
        except ValueError:
            pass

    levels = row.get("building:levels")
    if isinstance(levels, str):
        try:
            return float(levels) * 3.0
        except ValueError:
            pass

    return 10.0


# ------------------ LOAD BUILDINGS ------------------

def load_buildings(center_latlon, radius_m, simplify_tol) -> gpd.GeoDataFrame:
    print(f"Downloading OSM buildings within {radius_m} m of {center_latlon} ...")
    b = ox.features_from_point(center_latlon, tags={"building": True}, dist=radius_m)
    b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    print("Raw building footprints:", len(b))

    b = b.to_crs(epsg=27700)

    if simplify_tol and simplify_tol > 0:
        b["geometry"] = b.geometry.simplify(simplify_tol, preserve_topology=True)

    b["height_m"] = b.apply(get_height, axis=1)
    return b


def center_point_27700():
    center_27700 = ox.projection.project_geometry(
        Point(CENTER_POINT[1], CENTER_POINT[0]),
        crs="EPSG:4326",
        to_crs="EPSG:27700"
    )[0]
    return float(center_27700.x), float(center_27700.y)


# ------------------ GRID + GRAPH IN CIRCLE ------------------

def build_grid_graph_in_circle(cx, cy, radius_m):
    xmin, ymin = cx - radius_m, cy - radius_m
    xmax, ymax = cx + radius_m, cy + radius_m

    nxn = int((xmax - xmin) // GRID_SPACING) + 1
    nyn = int((ymax - ymin) // GRID_SPACING) + 1

    G = nx.Graph()
    inside = set()

    # nodes inside circle
    for ix in range(nxn):
        x = xmin + ix * GRID_SPACING
        dx2 = (x - cx) ** 2
        for iy in range(nyn):
            y = ymin + iy * GRID_SPACING
            if dx2 + (y - cy) ** 2 <= radius_m ** 2:
                nid = (ix, iy)
                inside.add(nid)
                G.add_node(nid, x=x, y=y, z=FLAT_LAYER_Z)

    # edges
    for (ix, iy) in inside:
        a = (ix, iy)
        b1 = (ix + 1, iy)
        b2 = (ix, iy + 1)
        if b1 in inside:
            G.add_edge(a, b1, weight=GRID_SPACING)
        if b2 in inside:
            G.add_edge(a, b2, weight=GRID_SPACING)

    meta = {
        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
        "nxn": nxn, "nyn": nyn,
        "cx": cx, "cy": cy,
        "radius": radius_m
    }
    return G, meta


def depot_node(G, meta):
    # westmost node, closest to center y
    cx, cy = meta["cx"], meta["cy"]
    best = None
    best_x = float("inf")
    best_dy = float("inf")
    for n in G.nodes:
        x = G.nodes[n]["x"]
        y = G.nodes[n]["y"]
        if x < best_x - 1e-6:
            best_x = x
            best_dy = abs(y - cy)
            best = n
        elif abs(x - best_x) < 1e-6:
            dy = abs(y - cy)
            if dy < best_dy:
                best_dy = dy
                best = n
    return best


# ------------------ DELIVERY + ROUTING ------------------

def random_point_in_circle(rng, cx, cy, radius_m):
    ang = rng.uniform(0, 2 * math.pi)
    r = radius_m * math.sqrt(rng.uniform(0, 1))
    return cx + r * math.cos(ang), cy + r * math.sin(ang)


def nearest_cell_corner_candidates(G, meta, x, y):
    xmin, ymin = meta["xmin"], meta["ymin"]
    ix = int((x - xmin) // GRID_SPACING)
    iy = int((y - ymin) // GRID_SPACING)
    corners = [(ix, iy), (ix + 1, iy), (ix, iy + 1), (ix + 1, iy + 1)]
    corners = [c for c in corners if c in G.nodes]
    if corners:
        return corners

    # fallback nearest node
    best = None
    best_d = float("inf")
    for n in G.nodes:
        dx = G.nodes[n]["x"] - x
        dy = G.nodes[n]["y"] - y
        d = dx * dx + dy * dy
        if d < best_d:
            best_d = d
            best = n
    return [best]


def astar_path(G, start, goal):
    def h(u, v):
        ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
        vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
        return abs(ux - vx) + abs(uy - vy)
    return nx.astar_path(G, start, goal, heuristic=h, weight="weight")


def choose_delivery_and_path(G, meta, rng):
    cx, cy, radius_m = meta["cx"], meta["cy"], meta["radius"]
    start = depot_node(G, meta)

    x, y = random_point_in_circle(rng, cx, cy, radius_m)
    candidates = nearest_cell_corner_candidates(G, meta, x, y)

    best_path = None
    best_len = float("inf")
    best_corner = None
    for c in candidates:
        p = astar_path(G, start, c)
        dist = (len(p) - 1) * GRID_SPACING
        if dist < best_len:
            best_len = dist
            best_path = p
            best_corner = c

    return (x, y), best_corner, best_path


# ------------------ PLOTLY TRACES ------------------

def add_buildings_traces(fig, buildings, max_buildings=None):
    if len(buildings) == 0:
        return

    bsub = buildings if max_buildings is None else buildings.iloc[:max_buildings]

    ground_x, ground_y, ground_z = [], [], []
    roof_x, roof_y, roof_z = [], [], []
    vert_x, vert_y, vert_z = [], [], []

    for _, row in bsub.iterrows():
        geom = row.geometry
        h = float(row.get("height_m", 10.0))

        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:
            continue

        for poly in polys:
            x, y = poly.exterior.xy
            x = np.array(x); y = np.array(y)

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

    fig.add_trace(go.Scatter3d(x=ground_x, y=ground_y, z=ground_z, mode="lines",
                              line=dict(width=1), showlegend=False))
    fig.add_trace(go.Scatter3d(x=roof_x, y=roof_y, z=roof_z, mode="lines",
                              line=dict(width=1), showlegend=False))
    fig.add_trace(go.Scatter3d(x=vert_x, y=vert_y, z=vert_z, mode="lines",
                              line=dict(width=1), showlegend=False))


def add_grid_edges_trace(fig, G, every=4):
    xs, ys, zs = [], [], []

    for a, b in G.edges():
        if a[0] == b[0]:
            if a[0] % every != 0:
                continue
        else:
            if a[1] % every != 0:
                continue

        ax, ay, az = G.nodes[a]["x"], G.nodes[a]["y"], G.nodes[a]["z"]
        bx, by, bz = G.nodes[b]["x"], G.nodes[b]["y"], G.nodes[b]["z"]
        xs += [ax, bx, None]
        ys += [ay, by, None]
        zs += [az, bz, None]

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(width=1),
        opacity=0.25,
        name="Grid",
        showlegend=True
    ))


# ------------------ ANIMATION: update ONLY drone traces ------------------

def path_positions_xyz(G, path_nodes):
    coords = [(G.nodes[n]["x"], G.nodes[n]["y"], G.nodes[n]["z"]) for n in path_nodes]
    positions = []

    for i in range(len(coords) - 1):
        x0, y0, z0 = coords[i]
        x1, y1, z1 = coords[i + 1]
        dx, dy = x1 - x0, y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9:
            continue

        steps = max(1, int(seg_len // DRONE_STEP_M))
        for s in range(steps):
            t = min(1.0, (s * DRONE_STEP_M) / seg_len)
            positions.append((x0 + t * dx, y0 + t * dy, z0 + t * (z1 - z0)))

    positions.append(coords[-1])
    return positions


def build_frames_for_3_drones(pos_lists, drone_trace_indices, start_frame_index):
    max_len = max(len(p) for p in pos_lists)
    frames = []

    for k in range(max_len):
        data = []
        for i, positions in enumerate(pos_lists):
            idx = min(k, len(positions) - 1)
            x, y, z = positions[idx]
            data.append(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode="markers",
                marker=dict(size=10),
                showlegend=False
            ))

        frames.append(go.Frame(
            data=data,
            traces=drone_trace_indices,
            name=str(start_frame_index + k)
        ))

    return frames, start_frame_index + max_len


# ------------------ MAIN ------------------

def main():
    t0 = time.time()
    rng = random.Random(RNG_SEED)

    cx, cy = center_point_27700()
    buildings = load_buildings(CENTER_POINT, RADIUS_M, SIMPLIFY_TOL)

    G, meta = build_grid_graph_in_circle(cx, cy, RADIUS_M)
    print("Grid nodes:", G.number_of_nodes(), "Grid edges:", G.number_of_edges())

    depot = depot_node(G, meta)

    fig = go.Figure()
    add_buildings_traces(fig, buildings, max_buildings=MAX_BUILDINGS_PLOTTED)
    add_grid_edges_trace(fig, G, every=GRID_DRAW_EVERY)

    # Depot marker
    fig.add_trace(go.Scatter3d(
        x=[G.nodes[depot]["x"]], y=[G.nodes[depot]["y"]], z=[G.nodes[depot]["z"]],
        mode="markers",
        marker=dict(size=9),
        name="Depot"
    ))

    # Add 3 drone traces ONCE (these are the only traces animated)
    d0x, d0y, d0z = G.nodes[depot]["x"], G.nodes[depot]["y"], G.nodes[depot]["z"]
    fig.add_trace(go.Scatter3d(x=[d0x], y=[d0y], z=[d0z], mode="markers", marker=dict(size=10), name="Drone 1"))
    fig.add_trace(go.Scatter3d(x=[d0x], y=[d0y], z=[d0z], mode="markers", marker=dict(size=10), name="Drone 2"))
    fig.add_trace(go.Scatter3d(x=[d0x], y=[d0y], z=[d0z], mode="markers", marker=dict(size=10), name="Drone 3"))

    drone_trace_indices = [len(fig.data) - 3, len(fig.data) - 2, len(fig.data) - 1]

    all_frames = []
    frame_index = 0

    for batch in range(BATCHES_TO_ANIMATE):
        deliveries_xy = []
        paths = []

        for _ in range(N_DELIVERIES_AT_ONCE):
            dxy, corner, path = choose_delivery_and_path(G, meta, rng)
            deliveries_xy.append(dxy)
            paths.append(path)

        for (x, y) in deliveries_xy:
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[FLAT_LAYER_Z + 2],
                mode="markers",
                marker=dict(size=7, symbol="x"),
                showlegend=False
            ))

        for path in paths:
            rx = [G.nodes[n]["x"] for n in path]
            ry = [G.nodes[n]["y"] for n in path]
            rz = [G.nodes[n]["z"] for n in path]
            fig.add_trace(go.Scatter3d(
                x=rx, y=ry, z=rz,
                mode="lines",
                line=dict(width=5),
                opacity=0.55,
                showlegend=False
            ))

        pos_lists = [path_positions_xyz(G, p) for p in paths]
        frames, frame_index = build_frames_for_3_drones(pos_lists, drone_trace_indices, frame_index)
        all_frames.extend(frames)

    fig.frames = all_frames

    # ---- SPEED SLIDER (0.5x to 10x) ----
    base_duration_ms = int(DRONE_STEP_DT * 1000)

    def anim_args_for_speed(speed: float):
        # Higher speed => shorter duration
        dur = max(1, int(base_duration_ms / speed))
        return [None, dict(
            frame=dict(duration=dur, redraw=True),
            fromcurrent=True,
            transition=dict(duration=0),
            mode="immediate"
        )]

    speed_steps = []
    for s in SPEED_OPTIONS:
        speed_steps.append(dict(
            label=f"{s}×",
            method="animate",
            args=anim_args_for_speed(float(s))
        ))

    # default to 1x if present, else first
    default_speed = 1.0
    try:
        default_speed_index = SPEED_OPTIONS.index(default_speed)
    except ValueError:
        default_speed_index = 0

    fig.update_layout(
        title="London OSM (radius 1000m) + flat 200m grid + A* + 3 drones (no collisions) — FIXED",
        uirevision="keep-camera",
        scene_uirevision="keep-camera",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title="height (m)"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=45, b=0),

        # Play/Pause buttons (1x base duration)
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.0,
            y=1.05,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=anim_args_for_speed(default_speed)
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0)
                    )]
                ),
            ]
        )],

        # Speed slider
        sliders=[dict(
            active=default_speed_index,
            x=0.2,
            y=0.02,
            len=0.75,
            currentvalue=dict(prefix="Speed: ", suffix="", visible=True),
            pad=dict(t=10, b=10),
            steps=speed_steps
        )]
    )

    fig.show()
    print("Done in", round(time.time() - t0, 2), "seconds")

main()