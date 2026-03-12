# EPQ Code Overview — All Files, Pipeline, and Iterations

This document describes every Python file in the project, what it does, how it fits into the main pipeline, and where files represent earlier iterations that were superseded.

---

## Main Pipeline (in execution order)

The core pipeline runs these files in sequence. Each step produces a JSON output that the next step consumes.

```
london_boundary.py          (run once)
        |
        v
integrating_step1.py        --> integrating_step1_output.json
        |
        v
integrating_step2.py        --> updates integrating_step1_output.json with demand data
        |
        v
depot_model_demand.py       --> depot_solution_for_overlay.json
        |
        v
dash_scene_builder.py       (imports layers_with_divisions, fleet_specs, l0_height)
        |
        v
dash_drone_sim.py           (the final interactive simulation)
```

---

## File-by-File Breakdown

### 1. `london_boundary.py` — London Boundary Definition

**Role in pipeline:** First step. Run once to fetch and cache the Greater London boundary.

**What it does:**
- Fetches the Greater London administrative boundary from OpenStreetMap via osmnx/Nominatim
- Saves the polygon in two formats: `london_boundary.geojson` (for GIS tools) and `london_boundary.json` (simplified lat/lon coordinate list)
- Provides a `is_inside_london(lat, lon)` function that tests whether a point falls inside the boundary using a prepared Shapely polygon for fast repeated queries
- Provides `get_london_polygon_projected(epsg)` to return the boundary in any projected CRS (e.g. EPSG:27700 British National Grid)

**Important implementation details:**
- Uses `shapely.prepared.prep()` to create a pre-processed polygon for O(log n) point-in-polygon tests instead of O(n)
- Caches the WGS84 polygon in a module-level variable `_london_wgs84` so it's only fetched once per Python session
- The boundary query uses `osmnx.geocode_to_gdf("Greater London, England")` which returns the official administrative boundary

**Why it matters:** Every subsequent step needs to know what counts as "London" — the study area is defined as the intersection of a user circle with this boundary.

---

### 2. `integrating_step1.py` — Study Area Definition

**Role in pipeline:** Step 1. Takes a user-defined centre point + radius and intersects it with the London boundary.

**What it does:**
- Takes `CENTER_POINT = (51.51861, -0.12583)` (near Soho, London) and `RADIUS_M = 500.0`
- Projects the centre from WGS84 (lat/lon) to British National Grid (EPSG:27700, metres) using osmnx's projection utility
- Creates a circular polygon around the centre with the given radius
- Intersects this circle with the London boundary from `london_boundary.py`
- Saves the resulting polygon (the study area) to `integrating_step1_output.json`

**Output JSON structure:**
```json
{
  "city_polygon_xy": [[x1,y1], [x2,y2], ...],  // study area polygon in EPSG:27700
  "center_projected": [cx, cy],                  // centre in metres
  "radius_m": 500.0,
  "crs": "EPSG:27700"
}
```

**Important details:**
- The intersection handles edge cases where the circle extends beyond London (e.g. near the Thames estuary) — the polygon is clipped to only include areas that are actually in London
- Uses `USE_UK_BNG = True` to force EPSG:27700 (British National Grid) for all downstream files, ensuring everything is in metres

---

### 3. `integrating_step2.py` — Demand Modelling

**Role in pipeline:** Step 2. Computes where delivery demand is within the study area.

**What it does:**
- Loads the study area from step 1's JSON
- Downloads all OSM building footprints within the study area
- Computes a demand density grid: divides the area into small cells and counts the number of building centroids (delivery destinations) in each cell
- Identifies hotspot areas — clusters of high demand — using a top-N selection
- Saves demand data (hotspot polygons, density raster) back into the step 1 JSON and as a separate GeoJSON

**Important implementation details:**
- Two demand sources are supported: `"osm_buildings"` (default — every building is a potential delivery destination) and `"lsoa_population"` (UK census data — demand proportional to residential population)
- The demand grid uses a configurable cell size `DEMAND_GRID_CELL_M = 100` (100 m cells)
- Hotspot detection selects the `HOTSPOT_TOP_N = 5` densest grid cells and expands them into rectangular areas
- Originally considered using Voronoi tessellations for demand partitioning, but the docstring explicitly notes this was abandoned — Voronoi cells don't respect fixed service radii

---

### 4. `fleet_specs.py` — Drone Fleet Specifications

**Role in pipeline:** Provides physical constants (drone dimensions, atomic unit, L0 cell size) used by nearly every other file.

**What it does:**
- Defines three drone tiers: Standard (11.9 kg payload, 1.0x1.0x0.6 m), Oversize (29.8 kg, 1.8x1.5x1.0 m), H&B (45.0 kg, 2.5x2.0x1.2 m)
- Given a target range in km, computes total takeoff weight, battery weight, and frame/motor weight for each tier using a physics model: `W_total = W_payload / (0.70 - range_km * 0.0006)`
- Computes the **atomic unit** (AirMatrix cell side): `max(H&B length, H&B width) + 2*padding ≈ 2.5 + 0.6 = 3.1 m`
- Computes **L0 cell size**: `4 * atomic_unit ≈ 12.4 m` — the bottom layer cell (a 4x4 bucket of atomic units)
- Writes everything to `drone_fleet_specs.json`

**Key exported functions used by other files:**
- `get_atomic_unit_m()` → ~3.1 m (used by `dash_scene_builder.py`, `london_3d_overlay.py` when calling `compute_layer_plan`)
- `get_l0_cell_m()` → ~12.4 m (used by `dash_scene_builder.py`, `dash_drone_sim.py`, `plotly_drone_sim.py` for grid spacing)
- `get_drone_dims(drone_type)` → (L, W, H) tuple (used by `dash_drone_sim.py` for drawing drone boxes, by `corridrone.py` for geofence sizing, and by `layers_with_divisions.py` for per-layer speed computation)

**Further reading (design justification):**
- `DRONE_DIMENSIONS_FUNCTION_EXPLAINED.md` — explains how `get_drone_dims()`, the atomic unit, and L0 sizing are derived and why padding is used.

---

### 4b. `corridrone.py` — CORRIDRONE Geofence Model (Support Module)

**Role in pipeline:** Support module. Imported by `layers_with_divisions.py` when computing per-layer speeds; provides speed-dependent safety buffer sizing used across the system.

**What it does:**
- Defines a simple, explainable geofence model: given drone speed and dimensions (L, W, H), returns a protective cylinder (horizontal radius + vertical half-spans) that scales with speed (response distance, braking distance, wind drift, position uncertainty).
- `geofence_for_speed(speed_mps, drone_dims_lwh_m, ...)` → `GeofenceCylinder` (radius_m, up_m, down_m)
- `max_speed_for_spacing(spacing, dims, ...)` → max speed such that 2× geofence radius ≤ spacing
- `max_speed_for_edge_occupancy(edge_length_m, max_drones_on_edge, dims, ...)` → max speed for a given edge capacity
- `speed_for_max_edge_flow(dims, ...)` → speed that maximises steady-state flow rate v/(2r(v)) along an edge
- `edge_flow_rate_drones_per_s(v, dims, ...)` and `max_drones_on_edge_at_speed(v, edge_length_m, dims, ...)` for capacity/flow calculations

**Why it matters:** The atomic unit and L0 cell size define *spatial discretisation*; CORRIDRONE defines *operational separation* (safety buffer) that grows with speed. Keeping them separate avoids “double counting” and keeps grid resolution tractable while still supporting defensible per-layer speed caps (see `LAYER_SPEED_MODEL.md` and `GridManager.compute_layer_speeds_mps()` in `layers_with_divisions.py`).

**Further reading:** `CORRIDRONE_MODEL.md` — equations, rationale, and usage.

---

### 5. `layers_with_divisions.py` — Hierarchical Spatial Grid

**Role in pipeline:** Core spatial data structure. Imported by `dash_scene_builder.py` and `london_3d_overlay.py`.

**What it does:**
- Defines the hierarchical grid that partitions the depot service area into cells at multiple resolutions
- Three main classes:
  - `Cell` — one cell at one layer (grid indices, Morton code, world-coordinate bounds, altitude)
  - `Layer` — all cells at one resolution level, stored in a dict keyed by Morton code for O(1) lookup
  - `GridManager` — the full hierarchy: builds layers, runs the cost-function optimizer, provides lookups
- `LayerPlan` — a lightweight dataclass describing the layer hierarchy (backward-compatible with the old API)

**Important algorithms:**
- **Morton codes (Z-order curves):** `morton_encode_2d(ix, iy)` interleaves the bits of the x and y grid indices into a single integer using the "magic bits" method. This enables O(1) cell lookup and O(1) parent-child navigation (`parent_morton = child_morton >> 2`).
- **Cost-function optimizer** (`_calculate_optimal_divisions`): sweeps candidate configurations of division factor k (2 or 4) and pruning levels, computing `Total_Cost = alpha * NodeCount + beta * HeuristicError` for each, and picks the minimum.
- **Power-of-2 constraint:** `cells_per_axis` at L0 is rounded up to the next power of 2 (using bit-twiddling in `_next_power_of_2`), ensuring clean subdivision at every layer.
- **Sparse circular masking:** only cells whose centre falls within the depot circle are instantiated (~21.5% memory savings over a full square grid).

**Key exported functions:**
- `setup_grid(depot_radius, atomic_unit_size, ...)` → `GridManager` (the primary entry point)
- `compute_layer_plan(R_final, S0, verbose)` → `LayerPlan` (backward-compatible wrapper for `dash_scene_builder.py`)
- `GridManager.compute_layer_speeds_mps(drone_dims_lwh_m, traffic_intensity=0.7, ...)` → dict with `speeds_mps` and `details` per layer; uses `corridrone.py` (geofence, max_speed_for_edge_occupancy, speed_for_max_edge_flow) to derive defensible per-layer speeds from capacity and CORRIDRONE tradeoffs

**Iteration history:** The original version used divisions of 2 and 3 (not strict powers of 2), and the top layer cell size was passed as `R_final`. It has been completely rewritten to use strict power-of-2 quadtree divisions, Morton codes, and a cost-function optimizer. `R_final` now means depot service radius, and `S0` now means atomic unit size (internally computing L0 = 4 * S0).

---

### 6. `l0_height.py` — Bottom Layer Height Computation

**Role in pipeline:** Imported by `dash_scene_builder.py` and `l0_preview.py` to set L0 node heights.

**What it does:**
- Takes the L0 grid (a networkx graph with nodes at grid intersections) and the OSM buildings GeoDataFrame
- Computes a base height using Method A: sort all building heights, find the height at which only `building_exceed_threshold` (default 2%) of buildings are taller
- For each L0 cell that intersects a building taller than the base height, raises the cell's z-coordinate to `building_height + clearance_m` (default 20 m)
- Uses 16 sample points per cell (4 corners + 3 points on each of the 4 edges) to detect building intersections

**Smoothing algorithms:**
- **Iterative Laplacian smoothing** (`_smooth_l0_heights`): averages each node's height with its neighbours over multiple iterations, preventing abrupt height jumps between adjacent cells
- **Smoothstep reshaping** (`_smoothstep_reshape`): applies an S-curve to the height profile so that the transition from flat areas to raised areas is gradual (slope increases then decreases, like a smoothstep function)
- **Plateau softening** (`_plateau_soften`): rounds off the flat-to-slope boundary at the edges of raised plateaus, using a k-hop neighbourhood blend

**Output:** modifies the graph nodes' `z` attribute in-place. Returns the maximum L0 z value, which is used to set the altitude of the next layer up.

**Further reading (parameter justification):**
- `L0_BUILDING_EXCEED_THRESHOLD_JUSTIFICATION.md` — rationale for the “2% buildings may exceed the base” threshold (98th percentile baseline + local per-cell enforcement).

---

### 7. `depot_model_demand.py` — Depot Placement (Final Version)

**Role in pipeline:** Step 3. Reads step 1 output + demand data, computes optimal depot locations, writes `depot_solution_for_overlay.json`.

**What it does (two-phase approach):**

**Phase 1 — Discrete selection:**
- Loads the study area polygon and demand data from step 1's JSON
- Generates candidate depot locations in a ring around the polygon boundary
- Computes the minimum radius R needed for full coverage (eroded-R gate)
- Runs ILP (PuLP/CBC) to find the minimum number of depots needed
- Finds the smallest R that still allows full coverage with that depot count
- Optionally runs a demand-weighted MILP to minimise total weighted distance to nearest depot

**Phase 2 — Continuous refinement:**
- Takes the ILP solution (which snaps depots to discrete candidate positions)
- Runs BFGS (scipy.optimize) to continuously nudge depot positions, minimising the sum of weighted distances from demand points to their nearest depot
- Uses `refine_with_bfgs_end()` which handles the transformation from discrete to continuous optimisation

**Output JSON structure:**
```json
{
  "interior_pts": [[x1,y1], ...],      // demand sample points
  "weights": [w1, w2, ...],            // demand weight per point
  "chosen_depots": [[dx1,dy1], ...],   // final depot coordinates
  "R": 550.0                           // service radius
}
```

**Important details:**
- Uses adaptive sample refinement (`ADAPTIVE_REFINE_ITERS`) to progressively increase the number of demand sample points for more precise solutions
- Handles polygon holes (e.g. if the study area has rivers or parks excluded)
- CBC solver workdir is configured for platform compatibility (Windows path handling)

---

### 8. `dash_scene_builder.py` — Scene Construction for the Simulation

**Role in pipeline:** Bridge between the pipeline outputs and the interactive simulation. Imported by `dash_drone_sim.py`.

**What it does:**
- Loads the study area from `integrating_step1_output.json`
- Loads the depot solution from `depot_solution_for_overlay.json`
- Calls `compute_layer_plan(R_final=R, S0=ATOMIC_UNIT_M)` to get the layer hierarchy
- Downloads OSM buildings within the study area polygon
- Builds a networkx graph for each grid layer using `build_grid_graph_in_circle()`
- Computes L0 heights using `compute_l0_node_heights()` from `l0_height.py`
- Connects layers vertically using KD-tree-based nearest-neighbour matching (`build_G_3d()`)
- Builds static Plotly traces for buildings, grid edges, demand points, depot markers, and depot-to-grid link lines
- When overlay data and a `GridManager` are present, computes per-layer speeds via `gm.compute_layer_speeds_mps(get_drone_dims("H&B"), ...)` and stores the result in the scene
- Returns a `Scene` dataclass containing everything the simulation needs
- Optionally caches the entire scene to `cache/dash_scene_cache.pkl` to avoid rebuilding on reruns

**Important implementation details:**
- The 3D graph `G_3d` has nodes labelled as `(layer_index, grid_node_id)` tuples and edges weighted by grid spacing (horizontal) or `LAYER_SPACING_M` (vertical)
- Vertical edges between layers use `scipy.spatial.cKDTree` for efficient nearest-neighbour queries — this was previously O(n_lo * n_hi) and was the main performance bottleneck
- The `Scene` dataclass stores `grid_0_idx` and `base_traces` (indices into the Plotly figure's trace list), plus `grid_manager`, `pathfinder`, `layer_speeds_mps`, and `layer_speed_details` when built from overlay data — so the simulation (or future traffic-aware logic) can use per-layer speeds and grid hierarchy

---

### 9. `dash_drone_sim.py` — Interactive Drone Simulation (Final Application)

**Role in pipeline:** The final output. This is what the user runs.

**What it does:**
- Imports the pre-built scene from `dash_scene_builder.py`
- Creates a Dash web application with:
  - A 3D Plotly figure showing buildings, grid layers, depots, and demand points
  - Play/Pause button and speed slider
  - Real-time drone animation

**Simulation mechanics:**
- Each drone is assigned a mission: a delivery point (random location in the service area) and a depot (nearest or path-distance-nearest)
- A* routing on the 3D graph: `astar_path_3d()` finds the shortest path from the depot (top layer) down through the grid to the delivery point (L0)
- The drone follows the path step-by-step, with `BASE_STEP_M = 3.0` metres per tick
- Drones are drawn as 3D wireframe cuboids using `box_edges_xyz()`, scaled by `DRONE_DISPLAY_SCALE = 4.0` for visibility
- Collision avoidance (when enabled): checks separation distance and segment crossing between drones each tick

**Two operating modes:**
1. **With overlay data** (pipeline ran): drones spawn from actual depot locations, fly to random delivery points, and are recycled after completing delivery. New drones spawn every `DELIVERY_SPAWN_TICKS_MIN` to `DELIVERY_SPAWN_TICKS_MAX` ticks.
2. **Without overlay data** (fallback): 3 drones loop from a single depot on a flat 2D grid.

**Performance optimisations:**
- Uses `dash.Patch()` for partial figure updates — only the changed traces (drone positions, routes) are sent to the browser, not the entire figure
- Pre-computes route position lists (`route_x`, `route_y`, `route_z`) at mission creation time so the tick callback just increments an index
- `MAX_DRONE_SLOTS = 80` — pre-allocates trace slots so the figure structure never changes

**Important functions:**
- `new_mission_3d(delivery_xy, depot_xy)` — computes a full 3D path from depot to delivery using A*
- `tick(n, sim_state, speed)` — the main simulation loop, called every `TICK_MS = 300` ms
- `make_figure(sim_state)` — builds the initial Plotly figure with all static and drone traces

---

## Files NOT in the Main Pipeline (Standalone / Earlier Iterations)

### 10. `london_3d.py` — First 3D Building Visualisation (Iteration 1)

**Status:** Superseded by `dash_scene_builder.py`. Standalone script, not imported by anything.

**What it was:** The very first attempt at 3D building visualisation. Downloads OSM buildings near a centre point and renders them as wireframe outlines in Plotly (ground footprint, roof outline, vertical pillars). Simple but effective for small areas.

**Why it was superseded:** Could not handle large areas (slow OSM downloads, too many Plotly traces). Led to trying QGIS, then eventually back to Plotly with Dash.

---

### 11. `WholeLondon_3D.py` — QGIS 3D Export (Iteration 2)

**Status:** Superseded. Standalone script, not imported by anything.

**What it was:** Attempted to download OSM buildings within a 30 km radius of central London, simplify their geometry, estimate heights, and export to a GeoPackage (`.gpkg`) for viewing in QGIS's 3D renderer.

**Why it was superseded:** QGIS 3D rendering was laggy, buildings came out as jagged triangles at low LOD, and the GeoPackage files were enormous (hundreds of MB). Not scalable or interactive. The approach was abandoned in favour of staying in Plotly with a smaller study area.

---

### 12. `plotly_drone_sim.py` — First Drone Simulation (Iteration 3)

**Status:** Superseded by `dash_drone_sim.py`. Standalone script, not imported by anything.

**What it was:** A Plotly-only (no Dash) animation of 3 drones delivering packages across a single flat 2D grid at 300 m altitude. Uses A* routing, pre-computes all frames, and plays them back with a speed slider.

**Why it was superseded:** No multi-layer grid (only a single flat layer), no real-time interactivity (all frames pre-computed), no depot model integration, no building height awareness. Led to the Dash-based simulation which supports all of these.

**What carried forward:** The A* routing, grid construction (`build_grid_graph_in_circle`), and drone path interpolation logic were adapted into `dash_drone_sim.py`.

---

### 13. `london_3d_overlay.py` — 3D Overlay Visualisation (Standalone Tool)

**Status:** Active standalone tool, but NOT part of the simulation pipeline. Run separately to inspect the depot solution.

**What it does:** Produces a static 3D Plotly figure showing the study area with OSM buildings, overlaid with depot locations (red diamonds) and demand sample points (green-to-red colour scale by demand weight). Also shows the layer plan if available.

**When you'd use it:** After running the depot model, to visually verify that depots are placed sensibly and demand coverage looks correct, without launching the full Dash simulation.

---

### 14. `l0_preview.py` — L0 Height Preview (Standalone Tool)

**Status:** Active standalone tool, NOT part of the simulation pipeline.

**What it does:** Loads the study area and buildings, computes L0 heights using `l0_height.py`, and shows a 3D Plotly plot of the resulting blanket surface over the buildings. Accepts CLI arguments for clearance height, smoothing iterations, and other parameters.

**When you'd use it:** To quickly iterate on L0 height parameters (clearance, smoothing, plateau softening) without running the full Dash app. Buildings are cached to disk after the first run for fast reruns.

---

### 15. `fleet_specs_graphs.py` — Drone Spec Visualisation (Standalone Tool)

**Status:** Active standalone tool, NOT part of the simulation pipeline.

**What it does:** Generates 4 Matplotlib graphs for the EPQ write-up: total takeoff weight vs range, wing span vs range, battery mass vs range, and payload efficiency vs range, for all three drone tiers.

---

### 16. `depot_model_v1.py` — Depot Model (Iteration 1)

**Status:** Superseded by `depot_model_demand.py`. Standalone script.

**What it was:** The first depot placement model. Simple approach: Poisson-sample demand points inside the study polygon, compute the minimum feasible radius R using erosion, generate candidates in a ring, solve with greedy or ILP set cover, optionally refine with BFGS.

**What changed in later versions:** No demand weighting (all points treated equally), no adaptive refinement, no smallest-R search, no multi-stage approach. Led to `depot_model_optimising.py` and then `depot_model_demand.py`.

---

### 17. `depot_model_optimising.py` — Depot Model (Iteration 2)

**Status:** Superseded by `depot_model_demand.py`. Standalone script.

**What it was:** Added several improvements over v1:
- Eroded-R feasibility gate (ensures the chosen R can actually cover all points)
- Finds the baseline minimum depot count, then searches for the smallest R that still works with that count
- Space-efficient layout: minimises overlap between depot service areas
- Greedy-add upgrade: attempts to improve the solution by adding more depots up to a cap

**What it still lacked:** No demand weighting — all demand points treated equally regardless of delivery volume. This was the key limitation that led to `depot_model_demand.py`.

---

### 18. `depot_model_spaceEff.py` — Depot Model (Iteration 3 / Variant)

**Status:** Superseded by `depot_model_demand.py`. Standalone script.

**What it was:** A variant focused specifically on minimising overlap/redundancy between depot service areas. Given a range of valid R values, it sweeps across them and for each R, solves for the depot layout that minimises the overlap metric. Picks the best (R, layout) pair.

**How it differs from the final version:** Pure overlap minimisation without demand awareness — a depot in a low-demand area is treated the same as one in a high-demand area. The final `depot_model_demand.py` incorporates demand weighting into the objective function.

---

## JSON Data Files

| File | Producer | Consumer | Contents |
|------|----------|----------|----------|
| `london_boundary.json` | `london_boundary.py` | `integrating_step1.py` | Greater London boundary coords (lat/lon) |
| `london_boundary.geojson` | `london_boundary.py` | GIS tools | Same, in GeoJSON format |
| `integrating_step1_output.json` | `integrating_step1.py` + `integrating_step2.py` | `depot_model_demand.py`, `dash_scene_builder.py` | Study area polygon, centre, radius, demand data |
| `drone_fleet_specs.json` | `fleet_specs.py` | `dash_scene_builder.py`, `dash_drone_sim.py`, etc. | Drone dims, weights, atomic_unit_m (~3.1), L0_cell_m (~12.4) |
| `depot_solution_for_overlay.json` | `depot_model_demand.py` | `dash_scene_builder.py`, `london_3d_overlay.py` | Depot positions, demand points, weights, R (550.0) |

---

## Key Constants That Flow Through the System

| Constant | Value | Origin | Used By |
|----------|-------|--------|---------|
| Atomic unit | ~3.1 m | `fleet_specs.py` (max H&B L/W + 2*padding) | `layers_with_divisions.py`, `dash_scene_builder.py`, `london_3d_overlay.py` |
| L0 cell size | ~12.4 m | `fleet_specs.py` (4 * atomic) | `dash_scene_builder.py`, `dash_drone_sim.py`, `plotly_drone_sim.py` |
| Depot radius R | 550.0 m | `depot_model_demand.py` output | `dash_scene_builder.py` → `layers_with_divisions.py` |
| L0 bucket factor | 4 | `fleet_specs.py` | `layers_with_divisions.py` |
| L0 height clearance | 20 m | `dash_scene_builder.py` | `l0_height.py` |
| Building exceed threshold | 2% | `dash_scene_builder.py` | `l0_height.py` |
| Layer spacing (vertical) | 100 m | `dash_scene_builder.py` | Grid construction |
| Base altitude (L0) | 300 m | `dash_scene_builder.py` | Grid construction |
| Per-layer speeds | list (m/s) | `dash_scene_builder.py` via `GridManager.compute_layer_speeds_mps()` | Scene; see `LAYER_SPEED_MODEL.md` |

**Further reading (design / equations):**
- `CORRIDRONE_MODEL.md` — geofence sizing and why buffer scales with speed
- `LAYER_SPEED_MODEL.md` — how per-layer speeds are chosen from CORRIDRONE and capacity

---

## Iteration Timeline Summary

```
Iteration 1:  london_3d.py              (3D buildings in Plotly — small area only)
Iteration 2:  WholeLondon_3D.py         (QGIS export — too slow, bad quality)
Iteration 3:  plotly_drone_sim.py       (Plotly drone anim — flat grid, pre-computed)
                                         |
                                         v
Iteration 4:  dash_drone_sim.py         (Dash + multi-layer + real-time)
              + dash_scene_builder.py    (separated heavy scene build from UI)
              + l0_height.py            (building-aware L0 heights)

Depot model:
  v1: depot_model_v1.py                 (basic set cover, no demand)
  v2: depot_model_optimising.py         (eroded-R, smallest-R search, space-efficient)
  v3: depot_model_spaceEff.py           (overlap minimisation variant)
  v4: depot_model_demand.py             (demand-weighted, two-phase ILP+BFGS — FINAL)

Layers:
  v1: layers_with_divisions.py          (2s and 3s, top layer = R_final cell size)
  v2: layers_with_divisions.py          (power-of-2 quadtree, Morton codes, cost optimizer — CURRENT)
```
