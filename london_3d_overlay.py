# -*- coding: utf-8 -*-
"""
3D map of London for the pipeline-defined space, with buildings and overlay of
depot_model_demand outputs: depots, Poisson sample points with demand colour grading (green→red).

Pipeline: run london_boundary.py (once), integrating_step1.py, integrating_step2.py,
depot_model_demand.py, then this script. Reads integrating_step1_output.json for the
study area and depot_solution_for_overlay.json for depots/samples/weights (optional).
"""

from pathlib import Path
import json

import numpy as np

from layers_with_divisions import compute_layer_plan, optimize_layer_plan
from fleet_specs import get_l0_cell_m
import geopandas as gpd
import osmnx as ox
import plotly.graph_objects as go
from shapely.geometry import Polygon
from shapely.prepared import prep

OUTPUT_DIR = Path(__file__).resolve().parent
STEP1_JSON = OUTPUT_DIR / "integrating_step1_output.json"
SOLUTION_JSON = OUTPUT_DIR / "depot_solution_for_overlay.json"

# 3D display
MAX_BUILDINGS = None  # None = all
SIMPLIFY_TOL = 3     # metres
OVERLAY_HEIGHT = 2.0  # height above ground for depot/demand layer (metres)
# L0 cell size from drone_fleet_specs.json (1.5 × max H&B L/W); fallback 10.0
BOTTOM_LAYER_CELL_M = get_l0_cell_m()


def get_height(row):
    """Return building height in metres, using OSM tags if possible (same as london_3d.py)."""
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


def load_study_area():
    """Load city polygon (projected) and crs from step1 output."""
    if not STEP1_JSON.exists():
        raise FileNotFoundError(f"Run pipeline first. Missing {STEP1_JSON}")
    with open(STEP1_JSON) as f:
        data = json.load(f)
    xy = data["city_polygon_xy"]
    crs = data.get("crs", "EPSG:27700")
    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly, crs


def load_solution():
    """Load interior_pts, weights, chosen_depots, R from depot model output. Return None if missing."""
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


def load_buildings_in_polygon(city_polygon: Polygon, target_epsg: int = 27700) -> gpd.GeoDataFrame:
    """OSM buildings in polygon bbox, projected and clipped to polygon."""
    gdf_city = gpd.GeoDataFrame({"geometry": [city_polygon]}, crs=f"EPSG:{target_epsg}")
    gdf_wgs = gdf_city.to_crs(epsg=4326)
    bounds = gdf_wgs.total_bounds
    bbox = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    print("Downloading OSM buildings for study area ...")
    b = ox.features_from_bbox(bbox, tags={"building": True})
    b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    b = b.to_crs(epsg=target_epsg)
    # Clip each building to polygon, preserving OSM height/levels from source row so heights vary
    city_prep = prep(city_polygon)
    clipped = []
    for _, row in b.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.intersects(city_polygon):
            inter = geom.intersection(city_polygon)
            if not inter.is_empty:
                # Keep OSM tags so get_height() can use height / building:levels
                height_val = row.get("height")
                levels_val = row.get("building:levels")
                if inter.geom_type == "MultiPolygon":
                    for p in inter.geoms:
                        if p.area >= 10:
                            clipped.append({
                                "geometry": p,
                                "area_m2": p.area,
                                "height": height_val,
                                "building:levels": levels_val,
                            })
                else:
                    if inter.area >= 10:
                        clipped.append({
                            "geometry": inter,
                            "area_m2": inter.area,
                            "height": height_val,
                            "building:levels": levels_val,
                        })
    if not clipped:
        return gpd.GeoDataFrame(columns=["geometry", "area_m2", "height_m"], crs=f"EPSG:{target_epsg}")
    gdf = gpd.GeoDataFrame(clipped, crs=f"EPSG:{target_epsg}")
    if SIMPLIFY_TOL and SIMPLIFY_TOL > 0:
        gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    gdf["height_m"] = gdf.apply(get_height, axis=1)
    return gdf


def build_3d_traces(buildings_gdf, overlay_data=None):
    """Build Plotly 3D traces: buildings + optional overlay (samples coloured by demand, depots)."""
    ground_x, ground_y, ground_z = [], [], []
    roof_x, roof_y, roof_z = [], [], []
    vert_x, vert_y, vert_z = [], [], []

    buildings_subset = buildings_gdf if MAX_BUILDINGS is None else buildings_gdf.iloc[:MAX_BUILDINGS]
    for _, row in buildings_subset.iterrows():
        geom = row.geometry
        h = row["height_m"]
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
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

    traces = [
        go.Scatter3d(x=ground_x, y=ground_y, z=ground_z, mode="lines", line=dict(width=1), name="Buildings (ground)"),
        go.Scatter3d(x=roof_x, y=roof_y, z=roof_z, mode="lines", line=dict(width=1), name="Buildings (roof)"),
        go.Scatter3d(x=vert_x, y=vert_y, z=vert_z, mode="lines", line=dict(width=1), showlegend=False),
    ]

    if overlay_data:
        pts = overlay_data["interior_pts"]
        w = overlay_data["weights"]
        # Normalise weights to [0,1] for colours
        wmin, wmax = float(np.min(w)), float(np.max(w))
        if wmax > wmin:
            w_norm = (w - wmin) / (wmax - wmin)
        else:
            w_norm = np.ones_like(w) * 0.5
        traces.append(go.Scatter3d(
            x=pts[:, 0].tolist(),
            y=pts[:, 1].tolist(),
            z=[OVERLAY_HEIGHT] * len(pts),
            mode="markers",
            name="Demand (samples)",
            marker=dict(
                size=4,
                color=w_norm,
                colorscale=[[0, "green"], [0.5, "yellow"], [1, "red"]],
                showscale=True,
                colorbar=dict(title="Demand"),
            ),
        ))
        depots = overlay_data["chosen_depots"]
        traces.append(go.Scatter3d(
            x=depots[:, 0].tolist(),
            y=depots[:, 1].tolist(),
            z=[OVERLAY_HEIGHT] * len(depots),
            mode="markers",
            name="Depots",
            marker=dict(size=12, symbol="diamond", color="crimson", line=dict(width=2, color="black")),
        ))

    return traces


def main():
    city_polygon, crs = load_study_area()
    epsg = int(crs.replace("EPSG:", "")) if "EPSG:" in crs else 27700

    buildings = load_buildings_in_polygon(city_polygon, target_epsg=epsg)
    print(f"Buildings: {len(buildings)}")

    overlay_data = load_solution()
    if overlay_data is None:
        print("No depot_solution_for_overlay.json; showing buildings only. Run depot_model_demand.py first for overlay.")
    else:
        R = overlay_data.get("R")
        if R is not None:
            plan = optimize_layer_plan(
                R_final=R, S0=BOTTOM_LAYER_CELL_M,
                layer_spacing_m=100.0, city_radius_m=1000.0,
                verbose=True,
            )
            if plan is not None:
                print(f"Layer plan: {plan.k + 1} layers, divisions={plan.divisions}")

    traces = build_3d_traces(buildings, overlay_data)
    fig = go.Figure(data=traces)
    n = len(traces)
    n_building_traces = 3  # ground, roof, verts
    n_overlay_traces = n - n_building_traces  # Demand (samples) + Depots when overlay_data

    # Buttons to show/hide layers: All, Buildings only, Depots & demand only
    layer_buttons = [
        dict(
            label="All",
            method="update",
            args=[{"visible": [True] * n}],
        ),
        dict(
            label="Buildings only",
            method="update",
            args=[{"visible": [True] * n_building_traces + [False] * n_overlay_traces}],
        ),
        dict(
            label="Depots & demand only",
            method="update",
            args=[{"visible": [False] * n_building_traces + [True] * n_overlay_traces}],
        ),
    ]
    # If no overlay data, only show "All" and "Buildings only" (no overlay traces)
    if n_overlay_traces == 0:
        layer_buttons = [layer_buttons[0], layer_buttons[1]]

    fig.update_layout(
        title="London 3D – study area + buildings" + (" + depots & demand" if overlay_data else ""),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title="height (m)"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                active=0,
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=1,
                buttons=layer_buttons,
                font=dict(size=12),
            ),
        ],
        annotations=[
            dict(
                text="Layer",
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="bottom",
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=11, color="gray"),
            ),
        ],
    )
    fig.show()


if __name__ == "__main__":
    main()
