## Research Phase (continued) — from “Bottom layer - L0” onwards

### Drone dimensions, the Atomic Unit, and CORRIDRONE geo-fencing (new research)

In my model, drone dimensions are not just visual detail — they constrain the minimum feasible spacing in the air network, and therefore the finest grid resolution that is meaningful for routing. To keep this consistent across the project, I centralise drone-size assumptions into one set of parameters and ensure the rest of the system uses the same values.

#### Why use a bounding box + padding?

When researching how robots and UAVs are represented for collision avoidance, I found that motion planning often converts a finite-size robot into a point by inflating obstacles in configuration space. In the translational case, this inflation is equivalent to a Minkowski sum between the obstacle and robot shapes (LaValle, 2006). This is useful because it separates “geometry/safety margin” from the planning algorithm: once obstacles are inflated, the planner can treat the vehicle as a point.

I use the same idea at the *grid design* level: I represent “one drone + margin” as a minimum footprint size so the routing grid is never finer than what is physically meaningful.

#### Defining the Atomic Unit \(u\)

The most restrictive drone class is the “H&B” tier, with an enclosure of \(2.5 \\times 2.0 \\times 1.2\\) metres. The minimum square footprint that encloses it is:

\[
S = \\max(L, W) = 2.5\\ \\text{m}
\]

If I set the atomic unit equal to \(S\), then two drones in adjacent atomic cells could physically touch (or overlap under tracking error). To justify a non-arbitrary margin, I looked for published bounds on how accurately drones can hold position. Manufacturers publish hovering-accuracy figures; DJI list horizontal drift on the order of ±0.3 m with vision positioning and ±1.5 m under GPS-only modes (DJI Matrice 30 specs; DJI Mini 2 specs mirror). These values provide an evidence-based scale for the “uncertainty margin” \(e\).

A simple conservative no-overlap condition for two adjacent “slots” is:

\[
u \\ge S + 2e
\]

where \(e\) is lateral position error.

My current approach defines the atomic unit \(u\) (the AirMatrix “one-drone cell” side length) as the largest footprint side plus a lateral padding margin:

\[
u = S + 2e
\]

For the largest tier in the fleet (H\&B), \(S=2.5\ \text{m}\). Using a vision-accuracy-order padding \(e\approx 0.30\ \text{m}\) gives \(u\approx 3.1\ \text{m}\) (a \(\approx 1.25\times\) factor). This is conservative enough to avoid overlap between drones in adjacent AirMatrix cells, while keeping the routing grid resolution practical.

#### Why CORRIDRONE geo-fencing exists (and why it scales with speed)

The atomic unit controls **spatial discretisation**. It does not, by itself, guarantee safe separation between moving drones. Operational safety needs to account for speed, response delay, braking distance, wind drift, and navigation uncertainty. That is why I implement a CORRIDRONE-style protective “geo-fence” around each drone.

The key principle I found in the literature is: **the safety buffer should increase with speed**, because the drone covers more distance before avoidance/braking fully takes effect. CORRIDRONE explicitly uses the idea of keeping drones inside structured corridors with protective separation buffers (Zhang et al., 2021). More generally, geofencing research emphasises that boundaries must account for vehicle dynamics and transient response rather than acting as static “lines in the air” (Thomas and Sarhadi, 2024).

I also found that geofencing safety depends strongly on navigation uncertainty: if the position estimate is uncertain, a drone can unintentionally cross a boundary even when its controller is correct. Work on UAV geofencing explicitly discusses the need to bound position error when making geofence decisions (Nam et al., 2023). For an order-of-magnitude sense of GNSS performance in favourable conditions, devices using WAAS differential correction are often considered accurate to within a few metres most of the time, but real-world performance depends on environment and multipath (Grayson et al., 2016). Urban environments are among the hardest cases, so I treat uncertainty as an explicit term rather than assuming perfect navigation.

In my implementation, I use a simple, explainable CORRIDRONE-style model that returns a protective cylinder. The horizontal buffer is built from:

- response distance \(d_{response} = v\\tau\)
- braking distance \(d_{brake} = \\frac{v^2}{2a}\)
- wind drift during response + braking \(d_{wind} = w(\\tau + v/a)\)
- navigation uncertainty \(d_{pos}\)
- plus a body-radius term derived from the drone’s dimensions

So:

\[
r_{geofence} = r_{body} + d_{pos} + d_{wind} + d_{response} + d_{brake}
\]

#### Avoiding “double counting”

It is important not to “bake” the entire CORRIDRONE geo-fence into the atomic unit. The atomic unit is a **metres-scale discretisation choice**, while operational buffers can become **tens of metres** once wind drift and uncertainty are included. The correct separation is:

- **Atomic unit**: geometric plausibility + tractable routing resolution
- **CORRIDRONE**: operational separation buffer scaling with speed and uncertainty

---

### Bottom layer — L0 (cell sizes + height)

#### Cell sizes

L0 cells are **not atomic**. They are strict \(4 \\times 4 \\times 1\) multiples of the atomic unit:

\[
L0 = 4u
\]

With \(u \approx 3.1\ \text{m}\), this gives \(L0 \approx 12.4\ \text{m}\). I do not justify “why 4×4” here — that will be explained later in the final approach section.

#### Height for L0: the “blanket” idea

The bottom layer must be above buildings so drones can position themselves before descending. My final approach is a blanket-like surface that drapes over the city: it is mostly flat, but rises locally over tall buildings with a fixed clearance margin.

Method A works in two stages:

1) Choose a *global base height* so that only a small fraction \(\epsilon\) of buildings exceed it.
2) Apply a *local raise* in any L0 cell that still intersects a building, enforcing clearance using sample points within that cell.

This is efficient because rare tall buildings do not force the entire city-wide base surface upward, while safety is still enforced locally where needed.

---

### Why the bottom-layer height threshold is 2% (new research)

In my codebase, the parameter is:

- `L0_BUILDING_EXCEED_THRESHOLD = 0.02` (2%)

This does **not** mean “2% of buildings are allowed to be unsafe”. Safety is enforced by the per-cell local raise. Instead, the exceedance threshold is a robustness/efficiency choice: it prevents outliers (rare towers or data artefacts) from forcing the entire base blanket height to be extremely high.

When choosing the baseline height I needed a method that was robust to outliers, because building-height distributions are highly heterogeneous: most buildings are low-rise, but a small number of towers can be much taller. Urban skyline analysis highlights this heterogeneity and the presence of “tall, needlelike buildings” in city cores (Schläpfer, Lee and Bettencourt, 2015). If the baseline height were set by the maximum, a tiny number of outliers would force the entire network to fly much higher than necessary.

Quantile-based trimming/winsorisation workflows are a standard way to reduce outlier sensitivity by explicitly specifying a tail proportion (e.g., treat the top \(p\) as outliers) and using the corresponding quantile as a threshold (SciPy outliers tutorial).

Setting \(\epsilon = 0.02\) corresponds to using a **98th percentile** building height as the global base reference:

\[
h_{base} = Q_{0.98}(\\{h_{building}\\})
\]

This choice has strong precedent as a “robust near-maximum” height metric:

- In national-scale 3D building modelling from LiDAR, multiple height references are often provided because the “height reference” materially changes results; percentile-based references are standard controlled options (Dukai et al., 2019).
- In remote sensing, RH98 (98th percentile height) is widely used as a robust “top height” descriptor, specifically because maxima are too sensitive to sparse outliers (ORNL DAAC GEDI/ICESat2 guide, 2024).

---

### Layers and divisions (research → final algorithm summary)

I originally designed an algorithm that approximated the ratio between “top layer cell size” and “bottom layer cell size” using a chain of multipliers from {2, 3}. After further research, I realised this is conceptually wrong for a hierarchical spatial grid: quadtree/octree structures rely on strict powers of 2 so cell boundaries align perfectly, coordinate lookup can be done with bitwise operations, and parent/child navigation is exact.

The final implementation uses only power-of-2 scaling (k=2 or k=4) and Morton codes (Z-order curves) for O(1) spatial hashing and parent-child navigation.
The implementation details (including code excerpts and outputs) are covered in the Development section.

---

## Development Phase (expanded: iterations + key algorithms)

This section is organised by **topic** (X, Y, Z, …). Under each topic, development is described as **first iteration**, **second iteration**, **third iteration**, and so on. Key equations and calculations are included where they drive the design or the code.

---

### Main pipeline (execution order + data flow)

The project has a “main pipeline” that produces the final interactive simulation. A key design choice is that **each step writes a JSON file that the next step consumes**, so heavy computation is staged and inspectable.

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

#### JSON data artefacts (what gets produced and why)

| File | Producer | Consumer | Contents |
|------|----------|----------|----------|
| `london_boundary.json` | `london_boundary.py` | `integrating_step1.py` | Greater London boundary coords (lat/lon) |
| `london_boundary.geojson` | `london_boundary.py` | GIS tools | Same boundary, in GeoJSON format |
| `integrating_step1_output.json` | `integrating_step1.py` + `integrating_step2.py` | `depot_model_demand.py`, `dash_scene_builder.py` | Study area polygon, centre, radius, demand data |
| `drone_fleet_specs.json` | `fleet_specs.py` | `dash_scene_builder.py`, `dash_drone_sim.py`, etc. | Drone dims, weights, atomic unit (~3.1 m), L0 cell (~12.4 m) |
| `depot_solution_for_overlay.json` | `depot_model_demand.py` | `dash_scene_builder.py`, `london_3d_overlay.py` | Depot positions, demand points, weights, service radius \(R\) |

#### Key constants that flow through the system

| Constant | Value | Origin | Used By |
|----------|-------|--------|---------|
| Atomic unit | ~3.1 m | `fleet_specs.py` (max H&B L/W + 2×padding) | `layers_with_divisions.py`, `dash_scene_builder.py`, `london_3d_overlay.py` |
| L0 cell size | ~12.4 m | `fleet_specs.py` (4×atomic) | `dash_scene_builder.py`, `dash_drone_sim.py`, `plotly_drone_sim.py` |
| Depot radius \(R\) | ~550 m | `depot_model_demand.py` output | `dash_scene_builder.py` → `layers_with_divisions.py` |
| L0 bucket factor | 4 | `fleet_specs.py` | `layers_with_divisions.py` |
| L0 height clearance | 20 m | `dash_scene_builder.py` | `l0_height.py` |
| Building exceed threshold | 2% | `dash_scene_builder.py` | `l0_height.py` |
| Layer spacing (vertical) | 100 m | `dash_scene_builder.py` | 3D grid construction |
| Base altitude (L0) | 300 m | `dash_scene_builder.py` | 3D grid construction |

---

### Pipeline setup scripts (boundary → study area → demand)

These scripts are not “the simulation” themselves, but they produce the inputs the simulation needs.

#### `london_boundary.py` — London boundary definition

**Role in pipeline:** run once to fetch and cache the Greater London boundary.

**What it does (technical):**
- Uses OSMnx’s place-geometry query to obtain the official administrative boundary polygon for “Greater London, England”.
- Saves the polygon as both GeoJSON (`london_boundary.geojson`) and a simplified JSON coordinate list (`london_boundary.json`) so the boundary can be reused without re-querying OSM.
- Builds a *prepared* Shapely polygon so repeated point-in-polygon queries are fast (LaValle, 2006; Shapely documentation on prepared geometries).

**Why it matters:** every later stage needs a consistent definition of “London” so the study area is well-defined and reproducible.

#### `integrating_step1.py` — Study area definition (circle ∩ London)

**Role in pipeline:** define the service region from a centre point + radius.

**What it does (technical):**
- Takes `CENTER_POINT` (lat/lon) and `RADIUS_M`.
- Projects to a metric CRS (EPSG:27700 British National Grid) so distances are in metres.
- Builds a circular polygon of radius \(R\) around the projected centre and intersects it with the London boundary polygon.
- Writes `integrating_step1_output.json` with the study polygon in projected coordinates, centre, radius, and CRS.

**Why it matters:** by forcing a metric CRS early, every later algorithm (sampling, distances, routing, optimisation) can treat geometry consistently in metres.

#### `integrating_step2.py` — Demand modelling (where deliveries “want” to be)

**Role in pipeline:** compute a demand field inside the study polygon.

**What it does (technical):**
- Downloads OSM building footprints inside the study polygon (default demand proxy: “each building is a potential delivery destination”).
- Constructs a demand density grid (e.g., 100 m cells) by counting building centroids per cell.
- Selects a small number of hotspots (top-\(N\) densest cells) and stores hotspot polygons plus density information back into the step-1 JSON.
- The code explicitly considers Voronoi-based partitioning (Aurenhammer, 1991) but rejects it for this use case because Voronoi cells do not enforce fixed service radii.

**Why it matters:** without demand, depot placement is purely geometric; demand weights let the model explain *why* depots move toward high-activity areas.

### Development timeline (high level)

- **Urban geometry prototypes:** `london_3d.py` → `WholeLondon_3D.py` → final: Plotly + Dash (`dash_scene_builder.py` + `dash_drone_sim.py`)
- **Routing prototypes:** `plotly_drone_sim.py` (flat 2D routing) → final: multi-layer 3D routing (`dash_scene_builder.py` + `dash_drone_sim.py`)
- **Bottom-layer height:** `l0_height.py` (Method A) + tuning tool `l0_preview.py`
- **Depot placement:** `depot_model_v1.py` → `depot_model_optimising.py` / `depot_model_spaceEff.py` → final: `depot_model_demand.py`
- **Hierarchical grid:** early “2/3 multipliers” idea → final: power-of-2 hierarchy + Morton codes in `layers_with_divisions.py`

---

### Iteration timeline summary (codebase snapshot)

This is the condensed “what got replaced by what” view:

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

---

### A. Foundation: drone sizing and CORRIDRONE

The atomic unit, L0 cell size, and drone dimensions are used by the grid, scene builder, CORRIDRONE, and simulation. I fixed these first in two steps: fleet specs, then CORRIDRONE.

#### First iteration (drone sizing — `fleet_specs.py`)

**Role:** One place that defines atomic unit \(u\), L0 cell size, and per-tier dimensions.

**Key equations:** Structural fraction \(f_s\), available \(f_a = 1 - f_s\); \(\text{denominator} = f_a - k \cdot \text{range}_{km}\), \(W_{total} = W_{payload}/\text{denominator}\). Atomic unit \(u = S + 2e\) with \(S = \max(L,W)\) for H&B and \(e \approx 0.30\) m → \(u \approx 3.1\) m. L0 cell size \(L_0 = 4u \approx 12.4\) m.

**Implementation:** The atomic unit, L0 cell size, and drone dimensions are used by the grid, the scene builder, CORRIDRONE, and the simulation, so I centralised them in one place before building the rest of the pipeline. There is no separate “iteration” here — it is the shared foundation that every later step depends on.

In code:

- structural fraction \(f_s\) (frame + motors) is fixed (e.g. \(f_s = 0.30\))
- available fraction for payload + battery is \(f_a = 1 - f_s\)
- battery requirement scales approximately linearly with range (a simplified technology assumption), so the usable payload fraction decreases with range:

\[
\\text{denominator} = f_a - k\\cdot \\text{range}_{km}
\]
\[
W_{total} = \\frac{W_{payload}}{\\text{denominator}}
\]

and then:

- \(W_{struct} = f_s\\,W_{total}\)
- \(W_{battery} = W_{total} - W_{struct} - W_{payload}\)

This model is deliberately simple (not a full propulsion + aerodynamics simulation), but it is *internally consistent* and produces a single JSON snapshot of “fleet geometry” that the rest of the pipeline can depend on. It also has a clear failure mode: if the denominator becomes non-positive, the mission is infeasible at that tech level. **What to look at in the code and output below.** The first snippet shows the core of `calculate_fleet_specs()`: how the denominator is computed as `available_mass_fraction - (range_km * k_factor)` and how that feeds into total weight and the mass breakdown (structural, battery, payload) per tier. The terminal output that follows is what you get when you run the script for a typical range (e.g. 1.1 km): you should see three blocks (Standard, Oversize, H&B), each with total takeoff weight, battery mass, and frame/motor mass. That output is the human-readable counterpart of what gets written to JSON.

[CODE SNIPPET: `fleet_specs.py` — `calculate_fleet_specs()` showing `denominator = available_mass_fraction - (range_km * k_factor)` and the derived mass breakdown]

[SCREENSHOT: terminal output from running `fleet_specs.py` (range_km=1.1), showing weights for Standard / Oversize / H&B]

In the terminal screenshot above, each tier lists the same quantities used in the equations: the denominator, total mass, and the split into structural and battery mass. The H&B tier is the heaviest and drives the atomic unit and L0 cell size used everywhere else in the pipeline.

**Key output that the rest of the pipeline consumes:** `drone_fleet_specs.json`, containing:
- `atomic_unit_m` (≈ 3.1 m)
- `L0_cell_m` (≈ 12.4 m, i.e. 4× atomic unit)
- per-tier `dims_LWH_m` (length, width, height in metres)

The next snippet shows exactly where these values are written to `drone_fleet_specs.json` so that other scripts can load a single source of truth for grid spacing and drone dimensions.

[CODE SNIPPET: `fleet_specs.py` — where `atomic_unit_m` and `L0_cell_m` are written to JSON]

In that code, the JSON is built from the same `calculate_fleet_specs()` results; any script that imports or reads `drone_fleet_specs.json` therefore uses the same atomic unit and L0 cell size as the scene builder and simulation.

---

#### Second iteration (CORRIDRONE — `corridrone.py`)

After the atomic unit and grid were in place, I still had a gap: the grid gave spatial resolution but not *operational* separation — two drones on the same edge at high speed need a buffer that grows with speed. I added CORRIDRONE as a separate module so per-layer speed limits could be justified without baking the full safety buffer into the grid cell size. That way grid resolution and speed-dependent buffer stay separate and I avoid double-counting.

**File:** `corridrone.py`  
**Why it exists:** The routing grid does not guarantee safety by itself. CORRIDRONE provides a speed-dependent separation buffer that accounts for dynamics and uncertainty.

**Core algorithm (what it computes):** given speed \(v\) and a drone’s dimensions (L, W, H), compute a protective cylinder. The reason for choosing a cylinder is practicality: in a grid-like routing system we need a heading-independent reserve volume, and a circular footprint gives a conservative bound in all directions.

The model first converts a box drone into a conservative “body radius”:

\[
r_{body} = \\frac{1}{2}\\sqrt{L^2 + W^2}
\]

Then it adds margins for sensing and dynamics. The horizontal radius is:

\[
r_{geofence} = r_{body} + d_{pos} + d_{wind} + d_{response} + d_{brake}
\]

where \(d_{response} = v\\tau\), \(d_{brake} = v^2/(2a)\), and \(d_{wind} = w(\\tau + v/a)\).

**Technical reason this is good for an EPQ:** every term is interpretable (you can point to the exact physical meaning), and the speed dependence is explicit. It also clarifies what should be tuned if you change assumptions (e.g., worse GNSS → increase \(d_{pos}\); higher winds → increase \(w\); better braking → increase \(a\)).

**What to look at in the code and output below.** The snippet shows `geofence_for_speed()`: how \(r_{body}\) is computed from (L, W), how each of \(d_{pos}\), \(d_{wind}\), \(d_{response}\), and \(d_{brake}\) is computed from speed and the model parameters, and how they are summed into the final horizontal radius. The screenshot shows a small test that evaluates this function at 1, 5 and 15 m/s — you should see the radius increase with speed (monotonic growth), which is exactly the behaviour we need for per-layer speed caps.

[CODE SNIPPET: `corridrone.py` — `geofence_for_speed()` showing the individual distance terms and how they sum into a radius]

[SCREENSHOT: terminal output from a simple test that prints geofence radius at v=1, 5, 15 m/s (showing monotonic growth)]

In the screenshot, the three speeds give three radii; the higher the speed, the larger the required buffer. That relationship is what `layers_with_divisions.py` uses when computing per-layer speed limits from grid spacing and CORRIDRONE (so that two drones on the same edge can safely pass with 2× geofence radius ≤ edge spacing).

---

### B. Urban geometry and 3D buildings

I wanted 3D buildings so the simulation could route above real geometry. I started with a simple Plotly script that pulled OSM footprints, projected to EPSG:27700, inferred heights from tags or storeys, and drew a wireframe. It worked for a small patch but didn’t scale — larger radii meant too many traces and sluggish interaction, and I had no way to hook the same geometry into a routing grid. So I tried exporting a much larger area to a GeoPackage and viewing it in QGIS 3D. That gave coverage, but the workflow was wrong: heavy files, a separate tool, and simplified geometry that looked jagged. I also needed everything in one place so the simulation could use the same scene. In the end I came back to Plotly, kept the study area manageable, and introduced Dash plus a scene builder so the heavy work runs once and is cached, and the UI only updates the bits that change each tick.

#### First iteration (`london_3d.py` — Plotly-only)  
**What it did (technical):** download OSM building footprints, project them into a metric CRS (EPSG:27700 so distances are in metres), infer a height per building (using `height` tags when available, otherwise `building:levels * 3m`, otherwise a default), and render a 3D wireframe (ground outline, roof outline, vertical pillars).

**Limitation:** scalable rendering and interaction. Plotly is fine for a small region, but large radii quickly produce too many buildings and too many traces for smooth interactivity.

**What the figure below shows.** The screenshot is the output of running `london_3d.py` for a small area around the chosen centre point (e.g. Soho). You should see a 3D wireframe: building footprints on the ground, roof outlines at inferred height, and vertical edges connecting them. This proves the pipeline (OSM → project → height inference → Plotly) works, but the viewer is not scalable to large areas or to an interactive simulation.

[SCREENSHOT: output of `london_3d.py` around your chosen centre point]

In that figure, the building blocks are clearly visible and the height comes from OSM tags or level-based estimates. Needing larger coverage and a link to the routing grid led to the QGIS export next.

#### Second iteration (`WholeLondon_3D.py` — export for QGIS 3D)  
**What it did (technical):** download a much larger region (tens of km), simplify polygon geometry to reduce vertex count, infer building heights, and export to a GeoPackage (`.gpkg`) for QGIS 3D rendering. This is essentially an “offline” pipeline: you pay large preprocessing costs up front for a more complete dataset.

**Limitation:** the workflow was not interactive for simulation (export step + heavy assets), and low-detail geometry produced visually jagged buildings in 3D.

**What the figure below shows.** The screenshot shows a typical QGIS 3D view of the exported GeoPackage: buildings in 3D with a zoom-in on one area. In the zoom-in you can see the jagged triangulation — simplified polygons and low LOD lead to stepped or faceted roof lines rather than smooth outlines. That, combined with the heavy file size and the need to leave QGIS to run a simulation, motivated returning to Plotly and keeping the study area smaller so that a single in-process scene (with caching) could feed an interactive Dash app.

[SCREENSHOT: QGIS 3D render, including a zoom-in showing jagged building triangulation]

So “export and view elsewhere” was not the right path; I kept all rendering in Plotly and added Dash so the same scene drives both visualisation and the simulation loop.

#### Third iteration (`dash_scene_builder.py` + `dash_drone_sim.py`): the heavy work (OSM, grids, graphs) is done once and cached; the UI updates quickly every tick.

**Key architectural decision:** I split the system into two parts:

- a *scene builder* that does expensive preprocessing once (downloads, graph building, static Plotly traces), and
- a *simulation/UI loop* that only updates small pieces of the figure every tick.

This is why `dash_scene_builder.py` exists: it keeps the Dash callbacks lightweight enough to run interactively.

**What to look at in the code and figure below.** The snippet shows how the Dash app is wired: `app.layout` defines the 3D Plotly figure plus the play/pause button and speed slider, and the `toggle_play` callback flips a flag so the interval-based tick callback either runs or stops. The screenshot shows the resulting UI: the main 3D view (buildings, grid, depots) with the controls visible so the reader can see how the user starts and speeds up the simulation.

[CODE SNIPPET: `dash_drone_sim.py` — `app.layout` and the `toggle_play` callback]
[SCREENSHOT: Dash app UI with play/pause and speed slider visible]

In the screenshot, the play button and speed slider are the only interactive controls that affect the simulation loop; the 3D figure is the same one that later shows moving drones and routes. This ties back to the architectural decision: the layout and callbacks are minimal so that each tick only patches the traces that change (drone positions, current route), not the whole figure.

**Standalone verification tool:** alongside the main simulation, I also built `london_3d_overlay.py` as a separate Plotly visualisation that overlays the depot solution (red markers) and demand sample points (colour-coded by weight) on top of the buildings and (optionally) the layer plan. This is useful for checking “does the optimisation output make sense?” without running the full Dash simulation loop.

---

### E. L0 blanket height

Once I had buildings and grid spacing, I needed a rule for L0 node heights. A uniform “fly everywhere at 300 m” would work but wastes altitude; using the maximum building height would force the whole city up for one tower. I settled on a percentile base so a few outliers don’t dominate, then raised only the cells that actually intersect a tall building. That gave a safe surface but neighbouring nodes could jump in height, so I added smoothing. Optionally I added plateau softening so the blanket curves over raised areas instead of terraces. There weren’t separate file versions — just this progression in the design, which is what “Method A” in the code implements .

**Files:** `l0_height.py` (algorithm), `l0_preview.py` (tuning/visualisation tool)  
**Problem:** Place the L0 navigation layer above buildings while avoiding a single skyscraper forcing the entire city baseline too high.

**Key equations:** Base height \(h_{base} = Q_{0.98}(\{h_{building}\})\) (98th percentile). Base layer \(z_{base} = h_{base} + \text{clearance}\). For each L0 cell, sample 16 points; cell z = max over samples of (building height at point + clearance). Laplacian smoothing: iteratively set each node height to an average of neighbours.

**Method A (technical breakdown):**

- **First iteration (global base):** compute \(h_{base}\) as the P98 quantile (2% exceedance). Set \(z_{base} = h_{base} + clearance\).
- **Second iteration (local enforcement):** for each L0 cell that intersects a building, raise that cell enough to satisfy clearance using multiple sample points.
- **Sampling strategy (technical):** each cell is checked at 16 points (4 corners + 3 points per edge on 4 edges). This is a deliberate compromise: it is much cheaper than dense rasterisation of the cell interior, but greatly reduces the “false safe” case where a tall building sits near the centre of the cell and would be missed by corner-only checks. Each sample gives a minimum required z for clearance, and the cell z is taken as the maximum over all sample points (worst-case enforcement).
- **Third iteration (smoothing):** after local raises, the node heights are iteratively smoothed using neighbour averaging (a Laplacian-style smoothing). This reduces high-frequency spikes and produces a surface with bounded gradients, which makes both horizontal routing and vertical transitions more realistic.
- **Performance detail:** point-in-polygon checks are accelerated using `buildings.sindex` (a spatial index). Without this, naïvely testing each point against every building would be \(O(N_{samples}\\cdot N_{buildings})\) and becomes too slow as the building count rises.
- **Plateau softening:** after smoothing, the implementation can “dome” flat high plateaus so the blanket has a curved top rather than step-like terraces, while still never going below the required clearance height.

**Why smoothing matters technically:** without smoothing, a single tall building can create a sudden step between neighbouring nodes. That would force a drone to either (a) climb sharply over a short horizontal distance, or (b) route around the “spike”, both of which are undesirable artefacts of discretisation rather than real constraints. Smoothing turns these discontinuities into gradual ramps, which better matches how real flight would handle obstacle clearance.

**What to look at in the code and figures below.** The snippet shows `_method_a_base_height()` (percentile-based base height) and the local-raise loop in `compute_l0_node_heights()` — including the 16 sample points per cell and the max-over-samples rule. The first figure is the output of `l0_preview.py`: a 3D blanket that sits mostly flat but rises over taller buildings; an optional second figure can compare different exceedance thresholds (0.01, 0.02, 0.05).

[CODE SNIPPET: `l0_height.py` — `_method_a_base_height()` and the local-raise logic in `compute_l0_node_heights()` (include the sampling points function)]

[SCREENSHOT: `l0_preview.py` output showing the blanket surface rising over tall buildings]
[SCREENSHOT (optional): side-by-side comparison for thresholds 0.01 vs 0.02 vs 0.05]

In the main screenshot, the "blanket" is the surface on which L0 grid nodes sit; drones route on this surface before descending to delivery. The smooth transitions come from Laplacian smoothing and (if enabled) smoothstep reshaping, so the height field is both safe (clearance enforced at every sample point) and suitable for gradient-limited flight. This is the same surface the scene builder uses when it calls `compute_l0_node_heights()` before building the 3D graph.

---

### C. Depot placement

Most of the “real optimisation” lives here. I began with a minimal set cover: choose the fewest depots so every demand point is within radius \(R\) of at least one. That gave feasible solutions, but I soon saw that the solver could still leave gaps in narrow or concave parts of the polygon, and all demand points were treated equally. So I added feasibility checks (the eroded-R gate) and a search for the smallest \(R\) that still works, and experimented with minimising overlap between depot discs. What was still missing was demand: a busy area and a quiet one counted the same. I moved to demand-weighted objectives and a two-stage pipeline (feasibility then quality), and added a BFGS polish so depot positions aren’t stuck on the candidate grid. The current code does that.

#### First iteration (`depot_model_v1.py` — basic set cover)

**Key equations:** Coverage matrix \(A_{ij} = 1\) if demand point \(i\) is within distance \(R\) of candidate \(j\). Set cover ILP: \(\min \sum_j x_j\) subject to \(\sum_j A_{ij} x_j \ge 1\) for all \(i\), \(x_j \in \{0,1\}\).

**Implementation:**

- Generate demand sample points inside the polygon (Poisson-process style sampling to approximate continuous demand with a finite set of points)
- Generate candidate depots (ring around polygon boundary)
- Build a binary coverage matrix \(A_{ij}\\) where \(A_{ij}=1\\) if depot \(j\\) covers demand point \(i\\) (distance \(\le R\))

This produces a classic set cover ILP:

\[
\\min \\sum_j x_j
\]
\[
\\text{s.t. } \\sum_j A_{ij}x_j \\ge 1 \\quad \\forall i
\]
\[
x_j \\in \\{0,1\\}
\]

where \(x_j \in \{0,1\}\) indicates whether candidate depot \(j\) is activated; the coverage matrix \(A_{ij}\) is 1 when demand point \(i\) is within distance \(R\) of candidate \(j\), so the constraint \(\sum_j A_{ij} x_j \ge 1\) means “every demand point is covered by at least one chosen depot”.

The computational bottleneck is building the coverage matrix: it is \(O(N_{points} \\cdot N_{candidates})\) distance checks, which is why sampling density and candidate density matter for runtime.

**What to look at in the code and figure below.** The snippet should show where the PuLP (or equivalent) model is built: the binary variables \(x_j\), the objective \(\min \sum_j x_j\), and the coverage constraints \(\sum_j A_{ij}x_j \ge 1\) for each demand point \(i\). The screenshot shows the v1 Plotly output: the study polygon (possibly with a hole or irregular boundary), the ring of candidate depot positions, and the subset of chosen depots (e.g. marked in a different colour or symbol). From that figure you can see that the solver picks a minimal set of candidates that cover all demand points within radius \(R\).

[CODE SNIPPET: `depot_model_v1.py` — ILP model structure: objective + coverage constraints]
[SCREENSHOT: v1 Plotly output showing polygon, candidate ring, chosen depots]

In the figure, each chosen depot covers a disc of radius \(R\); together they must cover every demand point. This formulation doesn’t yet use demand weights or BFGS — those came later.

#### Second iteration (`depot_model_optimising.py`, `depot_model_spaceEff.py` — feasibility and overlap): I added feasibility checks (“eroded-R gate”), smallest-R search, and overlap/redundancy minimisation so depots do not waste coverage by heavily overlapping.

Without the gate, solutions could look fine but leave coverage gaps in peninsulas or concavities.

#### Third iteration (`depot_model_demand.py` — demand-weighted and polished):

**Key equations (stage 2):** Minimise \(\sum_i w_i \cdot d(x_i, \text{nearest depot})\) over chosen depots. BFGS polish: same objective with depot coordinates continuous; \(\min_{\text{depot positions}} \sum_i w_i \, d(x_i, \text{nearest depot})\). demand points carry weights so high-demand areas pull depots closer. The pipeline is staged (feasibility then quality), and a BFGS step polishes depot coordinates off the discrete candidate grid.

**Key algorithm components:**
- demand weights from step 2 output (density grid / hotspots)
- stage 1: get a good feasible coverage solution (min depots / feasible R)
- stage 2: improve the solution under a depot cap by minimising weighted distance
- final polish: BFGS continuous refinement of depot coordinates

**Output structure (what the simulation consumes):**

```json
{
  "interior_pts": [[x1,y1], ...],
  "weights": [w1, w2, ...],
  "chosen_depots": [[dx1,dy1], ...],
  "R": 550.0
}
```

**What “staged” means in practice:** stage 1 primarily answers *feasibility* (“how many depots do I need to cover the region with radius R?”). Stage 2 answers *quality* (“given that many depots, place them so demand-weighted travel distances are small”). The BFGS step then moves from the discrete candidate grid to continuous coordinates to remove “grid snapping” artefacts.

**Why BFGS is not redundant:** the discrete model is forced to choose from a finite candidate set (grid/ring). Even if the ILP selection is optimal on that discrete set, the best continuous depot position may lie between candidate points. The BFGS “polish” step reduces this discretisation error by treating depot coordinates as continuous variables and optimising a smooth proxy objective. The next two snippets show the demand-weighted upgrade stage and the BFGS refinement function; the figure after them shows the final depot solution.

[CODE SNIPPET: `depot_model_demand.py` — demand weights and the distance-minimisation upgrade stage]
[CODE SNIPPET: `depot_model_demand.py` — `refine_with_bfgs_end()` showing the continuous refinement objective + minimize call]
[SCREENSHOT: final demand-weighted depot solution plot/map]

In the first snippet you see how weights are attached to demand points and how the upgrade stage minimises total weighted distance to the nearest depot. In the second, `refine_with_bfgs_end()` treats depot coordinates as continuous variables and calls `scipy.optimize.minimize` to polish positions. The figure shows the resulting depot locations (the same ones written to `depot_solution_for_overlay.json` and used by the scene builder); depots sit closer to high-demand (e.g. red) areas than in the unweighted v1 solution.

---

### D. Hierarchical grid

I needed multiple layers (coarse far away, fine near delivery) and fast lookup. At first I used a chain of division factors 2 and 3 to go from top to bottom; the ratio was whatever product of 2s and 3s fitted. I then realised that quadtrees and spatial hashing rely on strict powers of 2 — with 2 and 3 mixed, boundaries didn’t align and there was no clean encoding for O(1) lookup. So I rewrote the grid to use only power-of-2 divisions and Morton codes, and added a cost-function sweep. The current implementation is that.

#### First iteration (divisions of 2 and 3 — superseded):

**What it was:** The first design used a chain of division factors from \(\{2, 3\}\) to go from a “top layer cell size” (then interpreted as \(R_{\textit final}\) in metres) down to the bottom layer. The ratio between top and bottom was approximated by multiplying 2s and 3s.

**Why it was replaced:** Quadtrees and spatial hashing in the literature rely on *strict powers of 2* so that cell boundaries align across layers, coordinate→index mapping is exact, and parent/child links are trivial (e.g. bit shifts). Using 2 and 3 together broke that: boundaries did not align, and there was no clean integer encoding (like Morton codes) for O(1) lookup. So the implementation was rewritten.

#### Second iteration (`layers_with_divisions.py` — power-of-2 quadtree, Morton codes, cost optimizer)

**Key equations:** Grid indices \(ix = \lfloor (x - x_{origin}) / \text{cell\_size} \rfloor\), \(iy = \lfloor (y - y_{origin}) / \text{cell\_size} \rfloor\). Morton code = bit-interleave(ix, iy). Cost function \(\text{Total Cost} = \alpha \cdot \text{NodeCount} + \beta \cdot \text{HeuristicError}\); choose configuration that minimises it. Sparse mask: only create cell if \((x_c - c_x)^2 + (y_c - c_y)^2 \le R^2\).

**Implementation:** \(R_{\textit final}\) now means depot service radius; the top-layer cell size is derived from that and the power-of-2 constraint. Divisions are only \(k = 2\) or \(k = 4\). Grid indices are encoded as Morton codes (Z-order), giving O(1) cell lookup and exact parent/child navigation. A cost-function optimizer chooses the number of layers and the division factor by trading off node count vs approximation error.

**Key implementation decisions:**
- **power-of-2 structure:** ensures each layer subdivides cleanly and coordinate mapping is simple and exact
- **bounding box rounding:** the circle of radius \(R\) is embedded in a square, and the number of L0 cells per axis is rounded up to the next power of 2 so the tree terminates at a single root
- **Morton codes (technical):** interleave bits of (ix, iy) into one integer key (the Z-order curve). This gives O(1) hashing into Python dicts (fast lookup from spatial coordinate to cell record) and exact parent/child navigation using bit shifts because each subdivision level consumes a fixed number of bits.
- **sparse masking:** only create cells whose centres lie inside the service circle, saving memory and graph size. Concretely this is a simple radius test on the cell centre: \((x-c_x)^2 + (y-c_y)^2 \\le R^2\).
- **cost-function optimizer:** sweeps candidate configurations (division factor k = 2 or 4, pruning levels) and evaluates \(\text{Total Cost} = \alpha \cdot \text{NodeCount} + \beta \cdot \text{HeuristicError}\) for each; it picks the configuration that minimises this cost so the grid is neither too fine (too many nodes) nor too coarse (large approximation error).

**The key speed win:** once (x, y) is converted to a grid index (ix, iy), the Morton code acts as a constant-time spatial hash. This removes the need for searching or scanning cells. It also means parent/child navigation is exact: moving from a child to its parent corresponds to shifting off the lowest bits of the Morton code.

**Coordinate mapping detail:** at each layer, world coordinates are mapped to indices by:

\[
ix = \\left\\lfloor \\frac{x - x_{origin}}{cell\\_size} \\right\\rfloor, \\quad
iy = \\left\\lfloor \\frac{y - y_{origin}}{cell\\_size} \\right\\rfloor
\]

Using power-of-2 cell sizes ensures boundaries align exactly across layers, which keeps parent/child mappings exact and avoids “drift” from floating rounding.

The Morton encoding (bit interleaving) and the cost-function sweep that chooses divisions are shown in the next two snippets; the figure after them shows the terminal report with the chosen grid configuration.

[CODE SNIPPET: `layers_with_divisions.py` — `morton_encode_2d()` and `_spread_bits_2d()` (bit interleaving)]
[CODE SNIPPET: `layers_with_divisions.py` — `_calculate_optimal_divisions()` showing the cost-function sweep]
[SCREENSHOT: terminal output of the grid construction report showing the chosen configuration]

The first snippet shows how grid indices (ix, iy) are turned into a single integer (Morton code) by bit interleaving, and the second shows the cost-function sweep over division factors and pruning levels that picks the layer configuration. In the terminal output you should see the chosen number of layers, cells per axis at L0, and possibly node counts per layer — that configuration is what the scene builder uses when it calls `setup_grid()` or `compute_layer_plan()`.

---

### F. Scene builder and 3D routing graph

Once I had buildings, L0 height, depots, and the hierarchical grid, I needed one place to assemble them into a 3D graph and static scene for the simulation. The scene builder is that step — it runs once (or loads from cache) when I moved to Dash, so the simulation doesn’t rebuild OSM data and grids on every run and only updates the changing traces each tick.

**File:** `dash_scene_builder.py`  
**Role:** Pre-build everything that is expensive so the UI tick loop stays light.

**Key steps:**
- load study polygon + depot solution JSONs
- download + clip OSM buildings to the polygon
- build grid graphs for each layer (nodes + edges)
- apply L0 heights (blanket) and propagate layer Z values upward
- build a combined 3D graph `G_3d` with vertical edges between layers

**Caching (why it matters):** building OSM footprints, grids, and 3D graphs is slow compared to a UI tick loop. The scene builder therefore supports a disk cache (a pickle) so repeated runs of the simulation do not redo heavy preprocessing.

**Scene contents (current):** the `Scene` dataclass now includes `grid_manager`, `pathfinder`, `layer_speeds_mps`, and `layer_speed_details` when built from overlay data. Per-layer speeds are computed via `GridManager.compute_layer_speeds_mps()` using the CORRIDRONE tradeoff (capacity vs travel time); they are available for future traffic-aware routing or per-layer step sizes (see `LAYER_SPEED_MODEL.md` and `Ideas_for_Later.md`).

**Important data-structure detail:** the combined 3D routing graph `G_3d` stores nodes as `(layer_index, grid_node_id)` tuples. Horizontal edges are weighted by grid spacing and vertical edges are weighted by a fixed layer spacing (so a shortest-path algorithm is implicitly minimising physical travel distance in 3D).

**Key calculation:** Vertical edges: for each lower-layer node \((x,y)\), query KD-tree for nearest upper-layer node within \(\varepsilon\); add edge with weight = layer spacing. This replaces \(O(n_{lo} \cdot n_{hi})\) brute force with nearest-neighbour queries.

**Important optimisation:** vertical links use a KD-tree (cKDTree) to avoid \(O(n_{lo} \\cdot n_{hi})\\) brute force matching. That makes vertical edge construction tractable even when the lower layer has thousands of nodes.

**What the KD-tree is doing:** for each node in the lower layer (x, y), it queries the nearest node(s) in the upper layer within a radius epsilon and then connects to the closest match. This approximates “stacking” layers vertically without requiring identical node coordinates at different resolutions.

**What to look at in the snippet and figure below.** The code shows how `build_G_3d()` builds vertical edges: for each node in the lower layer it uses a KD-tree to find the nearest node(s) in the upper layer within a small radius, then adds an edge with weight equal to the layer spacing. This replaces an \(O(n_{lo} \cdot n_{hi})\) double loop with nearest-neighbour queries. The screenshot shows a typical run: number of buildings loaded, number of nodes per layer, and possibly cache hit/miss — so the reader can see the scale of the scene (e.g. thousands of L0 nodes) and why caching matters.

[CODE SNIPPET: `dash_scene_builder.py` — `build_G_3d()` vertical-edge construction using `cKDTree`]
[SCREENSHOT: terminal log from scene build showing building count and nodes per layer]

In the terminal log, the building count and node counts per layer are the direct result of the study area and grid setup; the 3D graph built here is the one the simulation uses for A* routing, so this step is the bridge between the pipeline outputs and the interactive app.

---

### G. Simulation and routing

#### First iteration (`plotly_drone_sim.py`): flat 2D grid, precomputed frames (superseded)

Single flat 2D grid at fixed altitude, A* on that grid, a few drones, all frames precomputed so playback was a slider. No buildings, no multi-layer grid, no depot model, no real-time control.

#### Second iteration (`dash_drone_sim.py` — current)

**Key equations:** A* with heuristic \(h(n)\) (e.g. Euclidean distance in (x,y,z)); admissible heuristic gives optimal path. Step per tick: position += step_length (e.g. `BASE_STEP_M = 3.0` m) × speed_scale; path is precomputed so tick callback only increments index.

Create missions, plan routes with A* on the 3D graph from the scene builder, and animate drones in real time. Drones spawn from depot locations, follow A* paths down through the grid to delivery points, and the UI updates every tick via partial figure updates.

**Core routing algorithm:**
- represent routing space as a graph (multi-layer, with vertical edges)
- A* finds a shortest path between depot entry (top layer) and delivery (L0)
- drones then follow the path as an interpolated sequence of points

**Mission construction detail:** a mission is not just a list of graph nodes. The code typically combines (a) a short straight-line “launch/approach” leg near the depot with (b) the discrete A* node path through the grid. The path is converted to (x, y, z) waypoints (e.g. by interpolating along edges), and the animation advances along this polyline each tick by a step length proportional to the speed slider (e.g. `BASE_STEP_M = 3.0` m per tick). The tick callback does not re-run A*; it only increments the position index along the precomputed route.

**Why A* is appropriate here (technical):** A* is Dijkstra’s algorithm guided by a heuristic \(h(n)\). When the heuristic is admissible (never overestimates remaining cost), A* returns an optimal path while expanding far fewer nodes than Dijkstra in practice. In the 3D multi-layer graph, the heuristic is typically Euclidean distance in (x, y, z), which is a lower bound on path cost when edge weights are based on physical distances. Using a hierarchical grid reduces node expansions because long-range travel happens on coarse layers (fewer nodes), then refinement happens near the goal.

**Key performance detail:** the Dash app updates the figure using partial updates (patching only the traces that change each tick) and pre-allocates drone trace slots. This avoids rebuilding the entire Plotly figure every frame and keeps the simulation responsive even with many drones.

Concretely, the implementation:
- pre-computes each mission’s interpolated route position arrays at mission creation time (so the tick loop just increments an index),
- reserves a fixed number of trace “slots” (e.g., `MAX_DRONE_SLOTS = 80`) so the figure structure does not change during playback,
- uses a fixed baseline step distance per tick (e.g., `BASE_STEP_M = 3.0`) and then scales by the user speed slider.

**Scene data available for extensions:** `scene.layer_speeds_mps` and `scene.layer_speed_details` (and `scene.grid_manager`) are populated when the pipeline has been run; they can be used to drive per-layer speeds or traffic-aware layer choice in a future iteration (see `Ideas_for_Later.md`). The mission-creation and A* routing logic are shown in the first snippet below; the second shows the tick callback that uses `Patch()` to update only the changing traces. The figure after them shows the simulation running (zoomed out and zoomed in on a drone descending through layers).

[CODE SNIPPET: `dash_drone_sim.py` — `astar_path_3d()` and `new_mission_3d()` (mission creation + A* route)]
[CODE SNIPPET: `dash_drone_sim.py` — the `tick()` callback section that uses `Patch()` to update only route/delivery/drone traces]
[SCREENSHOT: Dash simulation running — zoomed out (whole scene) + zoomed in (one drone descending through layers)]

In the first snippet, `new_mission_3d()` builds a full 3D path from depot to delivery (using A* on `G_3d`), and `astar_path_3d()` is the actual A* implementation. The second snippet shows how each tick only updates the traces for the current route and drone positions via `Patch()`, not the whole figure. In the screenshot, the zoomed-out view shows the full scene (buildings, grid layers, depots, demand points) with one or more drones and their routes; the zoomed-in view shows a single drone descending through the grid layers toward its delivery point, which illustrates the multi-layer routing and the blanket surface over the city.

---

## Reference list (new + existing)

Ahmad, T. et al. (2024) ‘Robust digital-twin airspace discretization and trajectory optimization for autonomous unmanned aerial vehicles’, *Scientific Reports*, 14(1), pp. 1–18. Available at: `https://www.nature.com/articles/s41598-024-62421-4`

Aurenhammer, F. (1991) ‘Voronoi diagrams — a survey of a fundamental geometric data structure’, *ACM Computing Surveys*, 23(3), pp. 345–405. Available at: `https://dl.acm.org/doi/10.1145/116873.116880`

Beasley, J.E. and Chu, P.C. (1996) ‘A genetic algorithm for the set covering problem’, *European Journal of Operational Research*, 94(2), pp. 392–404. Available at: `https://www.sciencedirect.com/science/article/abs/pii/037722179500159X`

Cordeau, J.F., Furini, F. and Ljubic, I. (2024) ‘Presolving and cutting planes for the generalized maximal covering location problem’, *arXiv preprint*, arXiv:2409.09834. Available at: `https://arxiv.org/abs/2409.09834`

DJI (n.d.) Matrice 30 Series Specifications. Available at: `https://enterprise.dji.com/matrice-30/specs`

DJI (n.d.) Mini 2 Specifications (mirror). Available at: `https://drdrone.com/pages/dji-mini-2-technical-specifications`

Dronova, I., Friedman, D. and Goodchild, A. (2024) ‘Sizing of multicopter air taxis — weight, endurance, and range’, *Aerospace*, 11(3). Available at: `https://www.mdpi.com/2226-4310/11/3/200`

Dukai, B. et al. (2019) ‘A multi-height LoD1 model of all buildings in the Netherlands’, *ISPRS Annals*. Available at: `https://isprs-annals.copernicus.org/articles/IV-4-W8/51/2019/`

Grayson, K. et al. (2016) GNSS accuracy discussion (WAAS-enabled devices), *Sensors*, 16, 912. Available at: `https://mdpi-res.com/d_attachment/sensors/sensors-16-00912/article_deploy/sensors-16-00912.pdf`

Hart, P.E., Nilsson, N.J. and Raphael, B. (1968) ‘A formal basis for the heuristic determination of minimum cost paths’, *IEEE Transactions on Systems Science and Cybernetics*, 4(2), pp. 100–107.

LaValle, S.M. (2006) *Planning Algorithms*. Cambridge: Cambridge University Press (online edition). Available at: `https://lavalle.pl/planning/` (see configuration-space obstacles / Minkowski-sum inflation).

Boeing, G. (2017) ‘OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks’, *Computers, Environment and Urban Systems*, 65, pp. 126–139. Available at: `https://www.sciencedirect.com/science/article/pii/S0198971516303970`

Shapely (2026) Prepared geometries (documentation). Available at: `https://shapely.readthedocs.io/`

Morton, G.M. (1966) *A computer oriented geodetic data base and a new technique in file sequencing*. Ottawa: IBM Canada Ltd.

Nam, D. et al. (2023) ‘Along-Track Position Error Bound Estimation… for UAV Geofencing’, *Journal of Positioning, Navigation, and Timing (JPNT)*. Available at: `https://www.jpnt.org/along-track-position-error-bound-estimation-using-kalman-filterbased-raim-for-uav-geofencing/`

Nocedal, J. and Wright, S.J. (2006) *Numerical Optimization*. 2nd edn. New York: Springer.

ORNL DAAC (2024) Global vegetation height metrics (RH98 definition). Available at: `https://daac.ornl.gov/VEGETATION/guides/GEDI_ICESAT2_Global_Veg_Height.html`

SciPy (2026) Trimming and winsorization transition guide. Available at: `https://docs.scipy.org/doc/scipy/tutorial/stats/outliers.html`

Schläpfer, M., Lee, J. and Bettencourt, L.M.A. (2015) ‘Urban Skylines: building heights and shapes as measures of city size’, *arXiv preprint*, arXiv:1512.00946. Available at: `https://arxiv.org/abs/1512.00946`

Stony Brook Algorithms (n.d.) Minkowski Sum. Available at: `https://www8.cs.umu.se/kurser/TDBAfl/VT06/algorithms/BOOK/BOOK5/NODE199.HTM`

Thomas, M. and Sarhadi, A. (2024) ‘Geofencing motion planning for UAVs using an anticipatory range control algorithm’, *Machines*, 12(1). Available at: `https://www.mdpi.com/2075-1702/12/1/36`

Zhang, Y. et al. (2021) ‘CORRIDRONE: Corridors for drones — An adaptive on-demand multi-lane design and testbed’. Available at: `https://arxiv.org/pdf/2012.01019`
