### CORRIDRONE geofence model (speed → safety buffer)

This project’s “CORRIDRONE” idea is to give each drone a **protective geofence volume** around it, acting as a safety buffer. The buffer should **increase with speed**, because the drone covers more distance before it can react and stop/avoid.

This repo implements a deliberately simple, explainable model in `corridrone.py`:

- `geofence_for_speed(speed_mps, drone_dims_lwh_m, ...) -> GeofenceCylinder`

It outputs a **vertical cylinder** centered on the drone:

- **Horizontal radius**: how far away the nearest obstacle/no-go boundary should be.
- **Vertical up/down half-span**: how much vertical clearance to reserve above/below the drone.

### Why should the buffer scale with speed?

Geofencing research repeatedly highlights that **vehicle dynamics** (e.g., turn performance, transient response) matter: if you trigger a return/avoidance only *at* a boundary, you can still overshoot due to inertia and control dynamics.

- Thomas & Sarhadi (2024) describe “hard” geofencing that must anticipate boundary approach and explicitly include **vehicle turning dynamics** and a **transient distance term** to cover delay/rise time on initiating a turn. They show minimum-turn radius grows with \(V^2\) and add an extra transient-distance margin \(s_t\).  
  Source: “Geofencing Motion Planning for UAVs Using an Anticipatory Range Control Algorithm”, *Machines* 2024. `https://www.mdpi.com/2075-1702/12/1/36`

Even though our drones in this EPQ are modelled more like **multirotors** (able to slow/stop rather than needing a fixed-wing turning circle), the same principle holds: **response delay + braking/avoidance distance increases with speed**.

### Core geometry and equations used

Let:

- \(v\) = drone speed (m/s)
- \(\tau\) = response time (s) before effective avoidance/braking
- \(a\) = conservative max horizontal deceleration (m/s²)
- \(w\) = crosswind speed used for a simple drift margin (m/s)

We reserve the following horizontal “extra room” beyond the physical drone size:

1) **Response distance** (distance traveled before control action fully takes effect):

\[
d_\text{response} = v \tau
\]

2) **Braking distance** (constant deceleration kinematics):

\[
d_\text{brake} = \frac{v^2}{2a}
\]

3) **Wind drift margin** during response + braking time:

\[
t_\text{stop} = \frac{v}{a}
\]
\[
d_\text{wind} = w (\tau + t_\text{stop})
\]

4) **Navigation / position uncertainty margin** \(d_\text{pos}\) (GNSS + multipath + map mismatch).

Finally we include the drone’s own footprint, conservatively represented by a **heading-independent body radius**:

\[
r_\text{body} = \frac{1}{2}\sqrt{L^2 + W^2}
\]

### Final buffer (horizontal)

\[
r_\text{geofence} = r_\text{body} + d_\text{pos} + d_\text{wind} + d_\text{response} + d_\text{brake}
\]

This is exactly what `corridrone.geofence_for_speed()` computes.

### Vertical buffer

We model the vertical reserve as:

\[
u = d = \frac{H}{2} + d_\text{vert}
\]

where \(d_\text{vert}\) is a small vertical-uncertainty allowance (barometer/GNSS/terrain model mismatch).

### Why include a position-uncertainty term?

Geofence safety depends on the **navigation uncertainty** used to decide whether you are inside/outside a boundary.

- Nam et al. (2023) explicitly note that NASA’s prototype assured geofence (SAFEGUARD) depends on position estimates and that safety risk from navigation uncertainty should be considered; they develop an along-track position error bound methodology.  
  Source: “Along-Track Position Error Bound Estimation… for UAV Geofencing”, *JPNT* 2023. `https://www.jpnt.org/along-track-position-error-bound-estimation-using-kalman-filterbased-raim-for-uav-geofencing/`

For a concrete *order-of-magnitude* number, a WAAS-enabled GNSS statement commonly cited in practice is:

- Grayson et al. (2016) state: devices equipped and used where WAAS differential correction is available are “considered accurate to within **3 m** at least **95%** of the time”, while also emphasizing environment dependence.  
  Source: *Sensors* 2016, 16, 912. `https://mdpi-res.com/d_attachment/sensors/sensors-16-00912/article_deploy/sensors-16-00912.pdf?version=1466239066`

Because drones in dense urban areas can suffer multipath and occlusion, `corridrone.py` uses a configurable `pos_uncertainty_m` (default **5 m**) as a conservative, easy-to-explain margin rather than pretending GNSS is always perfect.

### Why is the deceleration set to a conservative default?

Without a detailed autopilot + propulsion model, we should not assume “maximum possible agility” all the time. Instead we pick a conservative, stable deceleration that is plausible for multirotors carrying payloads and operating safely in cities.

As a sanity check that multirotors can generate meaningful accelerations, an educational dynamics example (Skydio-like quadrotor) shows that if each motor can produce up to ~2× hover thrust, vertical accelerations on the order of \(10\ \mathrm{m/s^2}\) are physically possible for a ~1 kg vehicle (ignoring limits like battery sag and control constraints).  
Source: “Multi-rotor Aircraft”, *Introduction to Robotics and Perception*. `https://www.roboticsbook.org/S72_drone_actions.html`

Given that, a **3 m/s²** *conservative* horizontal deceleration is intentionally not “optimistic”.

### What this model is (and is not)

- **Is**: a defensible EPQ engineering model that transparently ties geofence size to speed, uncertainty, and basic physics.
- **Is not**: a certification-grade safety case, nor a regulator-defined separation standard.

### Related functions (capacity and layer speeds)

The same model is used to derive **max speed for a given spacing** and **speed that maximises edge flow**:

- `max_speed_for_spacing(required_center_spacing_m, drone_dims_lwh_m, ...)` — returns the maximum speed (m/s) such that \(2r(v) \le\) spacing.
- `max_speed_for_edge_occupancy(edge_length_m, max_drones_on_edge, drone_dims_lwh_m, ...)` — max speed such that spacing \(L/N\) satisfies the geofence.
- `speed_for_max_edge_flow(drone_dims_lwh_m, ...)` — speed \(v^*\) that maximises flow \(v/(2r(v))\) along an edge (capacity–time tradeoff).
- `edge_flow_rate_drones_per_s(v, ...)` and `max_drones_on_edge_at_speed(v, edge_length_m, ...)` — for flow/capacity calculations.

Per-layer speed schedules that use these (CORRIDRONE + capacity) are described in **`LAYER_SPEED_MODEL.md`** and implemented in `GridManager.compute_layer_speeds_mps()` in `layers_with_divisions.py`; the resulting speeds are stored in the scene as `layer_speeds_mps` and `layer_speed_details`.

### How to use it in code

Example (H&B drone at 15 m/s):

```python
from fleet_specs import get_drone_dims
from corridrone import geofence_for_speed

dims = get_drone_dims("H&B")  # (L, W, H)
fence = geofence_for_speed(15.0, dims)
print(fence.radius_m, fence.up_m)
```

