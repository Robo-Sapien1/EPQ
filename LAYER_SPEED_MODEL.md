## Programmatically choosing speeds per layer (with CORRIDRONE + capacity constraints)

You want:

1) CORRIDRONE (speed-dependent safety buffer) to apply on all layers L0…Lmax.
2) A separate “AirMatrix end-mode” near delivery where drones slow to near-hover and packing rules can change.
3) A programmatic way to decide speeds per layer based on:
   - how quickly drones can deliver (time),
   - how many drones you allow on a grid edge at once (capacity / density),
   - and safety/uncertainty (CORRIDRONE).

This document proposes a simple, defensible rule set that’s implementable in code.

---

### 1) Why speed should be layer-dependent

Layered airspace exists to trade off:

- **efficiency** (higher layers, bigger cells → straighter/shorter graph paths),
- **precision / maneuvering** (lower layers, smaller cells → better local routing),
- **capacity and safety** (more traffic means more stringent separation constraints).

Therefore a layer-speed schedule is reasonable:

- higher layers: faster cruise (fewer turns, lower conflict if separated)
- lower layers: slower, more controlled motion (more interactions, closer to obstacles)

---

### 2) Capacity constraint: “how many drones can be on an edge at once?”

Let:

- edge length \(L\) (≈ layer cell size)
- maximum drones simultaneously on that edge \(N\)

A simple spacing model is:

\[
\text{spacing} \approx L/N
\]

If CORRIDRONE gives a horizontal safety radius \(r(v)\) around each drone, then
to avoid overlap of two equal safety volumes you need:

\[
\text{spacing} \ge 2r(v)
\]

So capacity implies a **max feasible speed** (because \(r(v)\) grows with speed):

\[
2r(v) \le L/N \quad \Rightarrow \quad v \le v_{\max}(L,N)
\]

This is exactly what `corridrone.max_speed_for_edge_occupancy()` computes.

---

### 2.5) The *optimal* tradeoff speed (capacity vs travel time)

If you don’t want to hard-pick “\(N\) drones per edge”, you can instead optimise the
tradeoff directly.

Assume an edge behaves like a 1D “lane” where drones must maintain a center-to-center
headway of at least \(2r(v)\) (from CORRIDRONE). Then a standard steady-state
approximation for maximum **flow rate** is:

\[
f(v) \approx \frac{v}{2r(v)} \quad [\text{drones/s per edge}]
\]

This is exactly your tradeoff:

- higher \(v\) reduces per-drone travel time,
- but higher \(v\) increases \(r(v)\), so fewer drones “fit” behind each other.

With CORRIDRONE, the horizontal radius is quadratic in \(v\):

\[
r(v)=c_0 + c_1 v + c_2 v^2
\]

so \(f(v)\) has a finite optimum at:

\[
v^* = \sqrt{c_0/c_2} = \sqrt{2ac_0}
\]

This gives a defensible “best speed under congestion” without choosing \(N\) by hand.
It’s implemented as `corridrone.speed_for_max_edge_flow()`.

---

### 3) Why \(r(v)\) grows quickly: \(v^2\) braking + wind drift

CORRIDRONE’s horizontal radius includes:

- response distance \(v\tau\)
- braking distance \(v^2/(2a)\)
- wind drift \(w(\tau + v/a)\)
- navigation uncertainty (GNSS etc.)

So \(r(v)\) is approximately quadratic in \(v\).

That general “speed costs energy / drag costs” relationship is also well known:

- In multirotor forward flight, drag force \(D \propto U^2\) and “power for overcoming drag” \(D \times U \propto U^3\).  
  Source: Hwang et al. (2018) explicitly state drag force is proportional to \(U^2\) and drag power proportional to the third power of \(U\), and show an optimal speed exists.  
  `https://www.mdpi.com/1996-1073/11/9/2221`

This supports two points:

1) it is physically reasonable that faster speeds need “more room” (CORRIDRONE grows)
2) it is physically reasonable that faster speeds use more energy (battery sizing discussion)

---

### 4) Performance caps: maximum speed is not arbitrary

When picking layer speeds, you should cap them by plausible aircraft specs.

DJI Matrice 30 specs (example of a capable multirotor) list:

- Max Horizontal Speed: **23 m/s**
- Hovering accuracy: Horizontal **±0.3 m** (vision), **±1.5 m** (GPS)  
  `https://www.dji.com/global/matrice-30/specs`

For a “future” scenario you can justify:

- higher top speeds (better motors/energy density)
- better navigation (RTK-like or local infrastructure)

but you still need a finite cap to keep the model bounded.

---

### 5) Proposed algorithm (simple and explainable)

For each layer \(i\):

Inputs:

- cell size \(s_i\) (≈ edge length \(L\))
- allowed max drones on an edge \(N_i\)
- a “preferred” speed based on maneuvering/precision (rule-of-thumb)
- aircraft capability cap \(v_\text{cap}\)
- CORRIDRONE parameters (\(\tau, a, w, \sigma_\text{pos}\))

Steps:

1) **Capacity-implied max speed**:

\[
v_{\max,i} = \text{max\_speed\_for\_edge\_occupancy}(L=s_i, N=N_i)
\]

2) **Preferred speed** (choose one rule):

- Rule A (constant traversal time per cell):
  \[
  v_{\text{pref},i} = s_i / t_\text{edge}
  \]
  with smaller \(t_\text{edge}\) for high layers, larger for low layers.

- Rule B (power-law scaling):
  \[
  v_{\text{pref},i} = v_0 (s_i/s_0)^\beta
  \]
  with \(\beta \in [0.5, 1]\) giving “bigger cells → faster”.

3) **Final speed**:

\[
v_i = \min(v_{\text{pref},i}, v_{\max,i}, v_\text{cap})
\]

This makes speeds *emerge* from:

- your desired density (\(N_i\))
- the safety model (CORRIDRONE)
- the geometry of the layer (cell size)
- and physical caps

In code, you can also blend between “raw speed preference” and “capacity-optimal”
using a single knob (e.g. `traffic_intensity in [0,1]`), as implemented by
`GridManager.compute_layer_speeds_mps()` in `layers_with_divisions.py`.

---

### 6) AirMatrix end-mode (separate)

Near delivery you can define a separate mode, e.g.:

- \(v_\text{airmatrix}\) very low (near-hover)
- tighter position uncertainty (vision / local beacons)
- possibly a different spacing model (“multiple drones per node pocket” via scheduling)

That is intentionally separate from the global L0–Lmax cruise model, because it is
the “terminal operations” regime.

---

### 7) Implementation hooks (what’s already in code)

In `corridrone.py`:

- `geofence_for_speed(v, dims, ...)` → returns the geofence radius \(r(v)\) (and vertical half-spans)
- `max_speed_for_spacing(spacing, dims, ...)` → returns the max \(v\) such that \(2r(v) \le \text{spacing}\)
- `max_speed_for_edge_occupancy(edge_length, N, dims, ...)` → returns the max \(v\) such that \(2r(v) \le (L/N)\)

In layers_with_divisions.py: GridManager.compute_layer_speeds_mps(...) returns a dict with speeds_mps and details per layer. In dash_scene_builder.py the scene stores this as Scene.layer_speeds_mps and Scene.layer_speed_details. Additional helpers in corridrone.py: speed_for_max_edge_flow, edge_flow_rate_drones_per_s, max_drones_on_edge_at_speed.

