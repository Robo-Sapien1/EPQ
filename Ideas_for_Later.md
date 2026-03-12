### Ideas for Later (Traffic-aware Hierarchical Flight + UTM-style Operations)

This document is a parking lot for future extensions to the drone delivery simulation. The current implementation uses a simple, local heuristic for initial cruise-layer choice; many realistic systems also incorporate **traffic**, **airspace constraints**, and **learning**.

---

### Why layer choice should depend on traffic (you’re right)

- **Congestion matters**: even if a higher layer costs extra climb energy, it may reduce **delay**, **conflict probability**, and **rerouting** caused by local congestion.
- **Structured / stratified airspace** is a common concept in UTM/U-space and UAM research: separating traffic by altitude reduces interactions and improves throughput.
- **Capacity management**: when volumes get busy, you can treat some layers/corridors as “capacity constrained” and route traffic around them (strategically or tactically).

References (starting points):
- [FAA UTM Concept of Operations v2.0 (PDF)](https://www.faa.gov/sites/faa.gov/files/2022-08/UTM_ConOps_v2.pdf) (includes UAS Volume Restrictions / dynamic constraints and coordination concepts)
- [NASA UAS Traffic Management (UTM) project page](https://www.nasa.gov/utm)
- [U-space implementation handbook (SESAR / lessons learned) (PDF)](https://www.sesarju.eu/sites/default/files/documents/events/DSD2025/Uspace%20implementation%20handbook%20what%20we%20learned.pdf)

---

### 1) Traffic-aware cruise-layer selection (replace “distance only”)

**Goal**: choose the layer that minimizes *expected total mission cost*:
\[
J = w_E \cdot \text{energy} + w_T \cdot \text{time} + w_R \cdot \text{risk} + w_N \cdot \text{noise}
\]

Possible inputs:
- **Congestion index per layer**: e.g. average occupancy, conflict rate, queue delay, or “effective speed” multiplier.
- **Known hotspots**: cells/edges with frequent conflicts.
- **Dynamic restrictions**: UVRs / no-fly bubbles; weather corridors; emergency volumes.

Practical heuristic (simple and implementable later):
- For each candidate cruise layer \(L\), estimate:
  - climb cost (energy + time),
  - branch + A* path length on \(L\rightarrow L0\),
  - congestion penalty (layer and/or edge-based),
  - then choose the argmin.

---

### 2) “RAs” broadcasting traffic + incidents (your idea, extended)

Interpret “RA” as a **regional authority / local airspace service** that provides telemetry and advisories (similar to U-space services / USS feeds).

What an RA could broadcast:
- **Per-layer load**: occupancy, throughput, average delay, conflict probability.
- **Incidents**: “traffic jam” (high density), blocked corridor, emergency vehicle volume, police cordon.
- **Dynamic geofences / volume restrictions**: temporary volumes (UVRs) that close or constrain routing.

How drones use it:
- At mission start: query RA summary and adjust cruise-layer selection.
- During flight (later): replan if a corridor becomes blocked or congested.

Related UTM concept:
- **Dynamic restrictions / UVR** appear explicitly in UTM ConOps v2.0 as a mechanism for managing airspace constraints.

---

### 3) Strategic vs tactical deconfliction (U-space style)

Future architecture could separate:
- **Strategic planning** (pre-flight): choose preferred routes/layers given predicted demand and restrictions.
- **Tactical deconfliction** (in-flight): local conflict resolution / reroute around short-term issues.

This maps well to a layered grid:
- Strategic: decide cruise layer + coarse corridors.
- Tactical: resolve conflicts at lower layers or around hotspots.

Related reading:
- [Impact assessment of strategic planning performance in shared U-space volumes](https://journals.open.tudelft.nl/ejtir/article/view/7472)

---

### 4) Learning from the system (self-improving routing)

Two “learning” tracks:

#### 4.1 Predictive (supervised) learning
- Build a **traffic forecast** model per layer/edge from historical sim data.
- Inputs: time-of-day, demand intensity, depot load, weather (later).
- Output: predicted congestion penalty map used by the planner.

#### 4.2 Policy learning (reinforcement learning / MARL)
- Use MARL to learn:
  - layer choice policies,
  - decentralized conflict resolution,
  - dynamic speed control or holding patterns.

Related starting points:
- [Decentralized traffic management of autonomous drones (Springer, 2024)](https://link.springer.com/article/10.1007/s11721-024-00241-y)
- [A Reinforcement Learning Approach to Quiet and Safe UAM Traffic Management (arXiv, 2025)](https://arxiv.org/abs/2501.08941)

---

### 5) Congestion pricing / slot reservations (high-impact idea)

Borrow from road networks and ATM:
- Assign each edge/layer a **capacity** (max drones per minute).
- Use **reservation-based** planning: drones request time slots on edges.
- If a layer is crowded, its “price” increases so traffic shifts upward/downward automatically.

Benefits:
- emergent load balancing,
- fewer collisions/conflicts,
- predictable throughput.

---

### 6) Robustness features (operational realism)

- **Contingency rules**: return-to-depot, hold at safe loiter cell, divert to alternate corridor.
- **Dynamic weather volumes**: wind corridors or turbulence zones (penalties, not just blocks).
- **Noise-aware routing**: avoid sensitive polygons at low altitude; climb earlier over sensitive areas.

Related theme:
- Noise and equity constraints appear frequently in UAM/UTM optimization literature (layering is one lever).

---

### 7) Metrics to add so the above can be evaluated

To justify “traffic-aware layers,” the sim should record:
- **average delivery time**, **95th percentile**, **throughput**,
- **energy proxy** (climb + cruise),
- **conflicts** (near-misses / separation violations),
- **replans** / detours,
- **layer utilization** (occupancy histograms),
- hotspot maps (edges/cells with highest congestion).

---

### Notes for implementation later (keeping modular)

- The **scene** already exposes what you need for layer-aware logic:
  - `scene.layer_speeds_mps` — per-layer speed (m/s) from CORRIDRONE + capacity (when overlay was used)
  - `scene.layer_speed_details` — per-layer breakdown (v_geom, v_congested, v_cap_by_N, etc.)
  - `scene.grid_manager` — full grid hierarchy and layer metadata (cell sizes, z, etc.)
- Keep **layer selection** as a pluggable strategy:
  - `LayerChooser.choose(depot_xy, target_xy, context) -> layer_index`
- Define a standard `context` payload:
  - `traffic_by_layer`, `blocked_volumes`, `edge_penalties`, `time_of_day`, `layer_speeds_mps`, etc.
- Keep a clean boundary between:
  - “static geometry” (grids/layers),
  - “dynamic state” (traffic, restrictions),
  - “policy” (heuristics vs learned controller).

