import osmnx as ox
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Polygon, MultiPolygon
import time

# ------------------ 1. SETTINGS ------------------

CENTRE_POINT=(51.51861, -0.12583)


RADIUS_M = 1000

# limit number of buildings drawn (None = all)
MAX_BUILDINGS = None

# geometry simplification tolerance in metres
# smaller SIMPLI FY_TOL = more detail = slower
SIMPLIFY_TOL = 3

start_time = time.time()

# download information

print(f"Downloading buildings within {RADIUS_M} m of {CENTRE_POINT} ...")

buildings = ox.features_from_point(
    CENTRE_POINT,
    tags={"building": True},
    dist=RADIUS_M
)

# keep only polygon geometries (actual footprints)
buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
print(f"Number of raw building footprints: {len(buildings)}")


# change scale to metres

# British National Grid – good for London, units in metres
buildings = buildings.to_crs(epsg=27700)


# simplifying geometry for speed if needed

# reduces the number of vertices per building but keeps the overall shape.
if SIMPLIFY_TOL is not None and SIMPLIFY_TOL > 0:
    buildings["geometry"] = buildings.geometry.simplify(
        SIMPLIFY_TOL, preserve_topology=True
    )


# estimate building heights

def get_height(row):
    """Return building height in metres, using OSM tags if possible."""
    # Try 'height' tag first (e.g. "12" or "12 m")
    h = row.get("height")
    if isinstance(h, str):
        h_clean = h.replace("m", "").strip()
        try:
            return float(h_clean)
        except ValueError:
            pass

    # Try 'building:levels' * 3 m
    levels = row.get("building:levels")
    if isinstance(levels, str):
        try:
            return float(levels) * 3.0
        except ValueError:
            pass

    # Fallback default
    return 10.0


buildings["height_m"] = buildings.apply(get_height, axis=1)


# put in place number of buildings constraint

if MAX_BUILDINGS is None:
    buildings_subset = buildings
else:
    buildings_subset = buildings.iloc[:MAX_BUILDINGS]

print(f"Plotting {len(buildings_subset)} buildings out of {len(buildings)}")


# prepare coordinates 

ground_x, ground_y, ground_z = [], [], []
roof_x, roof_y, roof_z = [], [], []
vert_x, vert_y, vert_z = [], [], []

for _, row in buildings_subset.iterrows():
    geom = row.geometry
    h = row["height_m"]

    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    else:
        continue

    for poly in polys:
        x, y = poly.exterior.xy
        x = np.array(x)
        y = np.array(y)

        # Ground outline (z = 0) with separator None
        ground_x.extend(list(x) + [None])
        ground_y.extend(list(y) + [None])
        ground_z.extend([0] * len(x) + [None])

        # Roof outline (z = h)
        roof_x.extend(list(x) + [None])
        roof_y.extend(list(y) + [None])
        roof_z.extend([h] * len(x) + [None])

        # Vertical edges every 5th vertex
        for xi, yi in zip(x[::5], y[::5]):
            vert_x.extend([xi, xi, None])
            vert_y.extend([yi, yi, None])
            vert_z.extend([0, h, None])


# BUILD 3D MAP

fig = go.Figure()

# ground shape of building
fig.add_trace(go.Scatter3d(
    x=ground_x,
    y=ground_y,
    z=ground_z,
    mode="lines",
    line=dict(width=1),
    showlegend=False
))

# roof shape of building
fig.add_trace(go.Scatter3d(
    x=roof_x,
    y=roof_y,
    z=roof_z,
    mode="lines",
    line=dict(width=1),
    showlegend=False
))

# vertical edges
fig.add_trace(go.Scatter3d(
    x=vert_x,
    y=vert_y,
    z=vert_z,
    mode="lines",
    line=dict(width=1),
    showlegend=False
))

fig.update_layout(
    title=f"3D buildings – {len(buildings_subset)} of {len(buildings)} "
          f"footprints ({RADIUS_M} m radius around Soho)",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(title="height (m)"),
        aspectmode="data"
    ),
    margin=dict(l=0, r=0, t=40, b=0),
)

fig.show()

print(time.time()-start_time)