"""
CORRIDRONE — simple performance-based geofence sizing.

This module provides a small, explicit model to size a safety buffer (a "geofence")
around a drone as a function of its speed and physical dimensions.

The goal is NOT to certify a real aircraft. It's to have a defensible, explainable
engineering model for this EPQ project that:
  - scales with speed (faster aircraft need more space),
  - includes a stopping-distance term,
  - includes a navigation/position uncertainty term (GNSS),
  - includes a wind-drift term during the time it takes to respond and stop.

Inputs to the model (see geofence_for_speed()):
  - speed_mps: drone ground speed (m/s)
  - drone_dims_lwh_m: (length, width, height) in m — e.g. from fleet_specs.get_drone_dims()
  - response_time_s: delay before braking/avoidance takes effect (default 1.0 s)
  - max_decel_mps2: conservative horizontal deceleration (default 3.0 m/s²)
  - pos_uncertainty_m: GNSS/position uncertainty margin (default 5.0 m)
  - wind_speed_mps: crosswind used for drift margin (default 5.0 m/s)
  - vertical_uncertainty_m: baro/GNSS vertical margin (default 2.0 m)

In reality, drones would dynamically change their safety buffer according to the
wind they are experiencing (e.g. from onboard or weather data). In this simulation
we assume the same wind level everywhere in the system for simplicity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Tuple, Optional


@dataclass(frozen=True)
class GeofenceCylinder:
    """
    A simple 3D geofence shape: a vertical cylinder centered on the drone.

    - radius_m: horizontal radius (same in all directions)
    - up_m/down_m: vertical half-spans above/below the drone reference point
    """

    radius_m: float
    up_m: float
    down_m: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "shape": "cylinder",
            "radius_m": float(self.radius_m),
            "up_m": float(self.up_m),
            "down_m": float(self.down_m),
        }


def geofence_for_speed(
    speed_mps: float,
    drone_dims_lwh_m: Tuple[float, float, float],
    *,
    # Response + control delay before meaningful braking/avoidance takes effect.
    response_time_s: float = 1.0,
    # Conservative horizontal deceleration capability (m/s^2) used for "stop distance".
    max_decel_mps2: float = 3.0,
    # Navigation/position uncertainty margin (metres). See CORRIDRONE_MODEL.md for rationale.
    pos_uncertainty_m: float = 5.0,
    # Crosswind magnitude for drift margin (response+braking). Sim assumes uniform wind everywhere.
    wind_speed_mps: float = 5.0,
    # Extra vertical uncertainty (baro/GNSS/terrain model mismatch), metres.
    vertical_uncertainty_m: float = 2.0,
) -> GeofenceCylinder:
    """
    Compute a speed-dependent 3D geofence around the drone.

    Inputs:
      - speed_mps: drone ground speed in m/s (>= 0)
      - drone_dims_lwh_m: (length, width, height) in metres; typically from fleet_specs.get_drone_dims()

    Model (horizontal):
      radius = body_radius
             + position_uncertainty
             + wind_drift_margin
             + response_distance
             + braking_distance

      response_distance = v * response_time
      braking_distance  = v^2 / (2 * a)
      wind_drift_margin = wind_speed * (response_time + v/a)

    Vertical:
      up/down = (height/2) + vertical_uncertainty
    """

    v = float(speed_mps)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"speed_mps must be finite and >= 0, got {speed_mps!r}")

    L, W, H = (float(drone_dims_lwh_m[0]), float(drone_dims_lwh_m[1]), float(drone_dims_lwh_m[2]))
    if not all(math.isfinite(x) and x > 0 for x in (L, W, H)):
        raise ValueError(f"drone_dims_lwh_m must be finite and > 0, got {drone_dims_lwh_m!r}")

    tau = float(response_time_s)
    a = float(max_decel_mps2)
    if not (math.isfinite(tau) and tau >= 0):
        raise ValueError(f"response_time_s must be finite and >= 0, got {response_time_s!r}")
    if not (math.isfinite(a) and a > 0):
        raise ValueError(f"max_decel_mps2 must be finite and > 0, got {max_decel_mps2!r}")

    pos_m = float(pos_uncertainty_m)
    wind_mps = float(wind_speed_mps)
    vert_m = float(vertical_uncertainty_m)
    if not (math.isfinite(pos_m) and pos_m >= 0):
        raise ValueError(f"pos_uncertainty_m must be finite and >= 0, got {pos_uncertainty_m!r}")
    if not (math.isfinite(wind_mps) and wind_mps >= 0):
        raise ValueError(f"wind_speed_mps must be finite and >= 0, got {wind_speed_mps!r}")
    if not (math.isfinite(vert_m) and vert_m >= 0):
        raise ValueError(f"vertical_uncertainty_m must be finite and >= 0, got {vertical_uncertainty_m!r}")

    # Body radius as half of the footprint diagonal (conservative for any heading).
    body_radius_m = 0.5 * math.hypot(L, W)

    response_distance_m = v * tau
    braking_distance_m = (v * v) / (2.0 * a)
    wind_drift_m = wind_mps * (tau + (v / a))

    radius_m = body_radius_m + pos_m + wind_drift_m + response_distance_m + braking_distance_m

    up_down_m = 0.5 * H + vert_m
    return GeofenceCylinder(radius_m=radius_m, up_m=up_down_m, down_m=up_down_m)


def horizontal_radius_quadratic_coeffs(
    drone_dims_lwh_m: Tuple[float, float, float],
    *,
    response_time_s: float = 1.0,
    max_decel_mps2: float = 3.0,
    pos_uncertainty_m: float = 5.0,
    wind_speed_mps: float = 5.0,
) -> Tuple[float, float, float]:
    """
    Return the quadratic coefficients (c0, c1, c2) for CORRIDRONE's horizontal
    radius model:

        r(v) = c0 + c1*v + c2*v^2

    This is useful for analysis / optimisation (e.g. computing the speed that
    maximises edge throughput).
    """
    L, W, _H = (
        float(drone_dims_lwh_m[0]),
        float(drone_dims_lwh_m[1]),
        float(drone_dims_lwh_m[2]),
    )
    if not all(math.isfinite(x) and x > 0 for x in (L, W)):
        raise ValueError(f"drone_dims_lwh_m must be finite and > 0, got {drone_dims_lwh_m!r}")

    tau = float(response_time_s)
    a = float(max_decel_mps2)
    pos_m = float(pos_uncertainty_m)
    wind_mps = float(wind_speed_mps)
    if not (math.isfinite(tau) and tau >= 0):
        raise ValueError(f"response_time_s must be finite and >= 0, got {response_time_s!r}")
    if not (math.isfinite(a) and a > 0):
        raise ValueError(f"max_decel_mps2 must be finite and > 0, got {max_decel_mps2!r}")
    if not (math.isfinite(pos_m) and pos_m >= 0):
        raise ValueError(f"pos_uncertainty_m must be finite and >= 0, got {pos_uncertainty_m!r}")
    if not (math.isfinite(wind_mps) and wind_mps >= 0):
        raise ValueError(f"wind_speed_mps must be finite and >= 0, got {wind_speed_mps!r}")

    body_radius_m = 0.5 * math.hypot(L, W)

    # r(v) = body + pos + wind*tau + (tau + wind/a)*v + (1/(2a))*v^2
    c2 = 1.0 / (2.0 * a)
    c1 = tau + (wind_mps / a)
    c0 = body_radius_m + pos_m + (wind_mps * tau)
    return float(c0), float(c1), float(c2)


def edge_flow_rate_drones_per_s(
    speed_mps: float,
    drone_dims_lwh_m: Tuple[float, float, float],
    *,
    response_time_s: float = 1.0,
    max_decel_mps2: float = 3.0,
    pos_uncertainty_m: float = 5.0,
    wind_speed_mps: float = 5.0,
) -> float:
    """
    "Lane" flow rate model induced by CORRIDRONE separation.

    If drones must keep a center-to-center headway of at least 2*r(v) along an
    edge, then the maximum steady-state flow is approximately:

        f(v) ≈ v / (2*r(v))     [drones per second per edge]

    This captures the core tradeoff you asked for:
      - higher v reduces per-drone travel time,
      - but higher v increases r(v), reducing how tightly drones can pack.
    """
    v = float(speed_mps)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"speed_mps must be finite and >= 0, got {speed_mps!r}")

    gf = geofence_for_speed(
        v,
        drone_dims_lwh_m,
        response_time_s=response_time_s,
        max_decel_mps2=max_decel_mps2,
        pos_uncertainty_m=pos_uncertainty_m,
        wind_speed_mps=wind_speed_mps,
    )
    r = float(gf.radius_m)
    if r <= 0:
        return 0.0
    return float(v / (2.0 * r))


def speed_for_max_edge_flow(
    drone_dims_lwh_m: Tuple[float, float, float],
    *,
    response_time_s: float = 1.0,
    max_decel_mps2: float = 3.0,
    pos_uncertainty_m: float = 5.0,
    wind_speed_mps: float = 5.0,
    speed_cap_mps: Optional[float] = None,
) -> float:
    """
    Return the speed v* that maximises CORRIDRONE-implied edge flow
    f(v)=v/(2*r(v)).

    With r(v)=c0 + c1*v + c2*v^2, the optimum occurs at:

        v* = sqrt(c0 / c2) = sqrt(2*a*c0)

    Notably, v* does NOT depend on c1 (response + wind linear term).
    """
    c0, _c1, c2 = horizontal_radius_quadratic_coeffs(
        drone_dims_lwh_m,
        response_time_s=response_time_s,
        max_decel_mps2=max_decel_mps2,
        pos_uncertainty_m=pos_uncertainty_m,
        wind_speed_mps=wind_speed_mps,
    )
    if c0 <= 0 or c2 <= 0:
        return 0.0
    v_star = math.sqrt(c0 / c2)
    if speed_cap_mps is None:
        return float(v_star)
    cap = float(speed_cap_mps)
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError(f"speed_cap_mps must be finite and > 0, got {speed_cap_mps!r}")
    return float(min(v_star, cap))


def max_drones_on_edge_at_speed(
    edge_length_m: float,
    speed_mps: float,
    drone_dims_lwh_m: Tuple[float, float, float],
    *,
    response_time_s: float = 1.0,
    max_decel_mps2: float = 3.0,
    pos_uncertainty_m: float = 5.0,
    wind_speed_mps: float = 5.0,
) -> int:
    """
    Approximate how many drones can be simultaneously on an edge of length L,
    if each drone requires headway >= 2*r(v) along that edge.
    """
    L = float(edge_length_m)
    if not math.isfinite(L) or L <= 0:
        raise ValueError(f"edge_length_m must be finite and > 0, got {edge_length_m!r}")
    v = float(speed_mps)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"speed_mps must be finite and >= 0, got {speed_mps!r}")
    r = geofence_for_speed(
        v,
        drone_dims_lwh_m,
        response_time_s=response_time_s,
        max_decel_mps2=max_decel_mps2,
        pos_uncertainty_m=pos_uncertainty_m,
        wind_speed_mps=wind_speed_mps,
    ).radius_m
    headway = 2.0 * float(r)
    if headway <= 0:
        return 0
    return max(0, int(L // headway))


def max_speed_for_spacing(
    required_center_spacing_m: float,
    drone_dims_lwh_m: Tuple[float, float, float],
    *,
    response_time_s: float = 1.0,
    max_decel_mps2: float = 3.0,
    pos_uncertainty_m: float = 5.0,
    wind_speed_mps: float = 5.0,
) -> float:
    """
    Given a required *center-to-center* spacing between drones, return the maximum
    speed (m/s) such that CORRIDRONE's horizontal geofence does not violate it.

    We require non-overlap of two equal-radius safety cylinders:

        spacing >= 2 * r(v)

    where r(v) is the CORRIDRONE horizontal radius. With the model in
    geofence_for_speed(), r(v) is a quadratic:

        r(v) = c0 + c1*v + c2*v^2

    so we solve for the largest v such that r(v) <= spacing/2.

    Returns:
      - v_max (>= 0). If no positive v satisfies the constraint, returns 0.0.
    """
    spacing = float(required_center_spacing_m)
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError(
            f"required_center_spacing_m must be finite and > 0, got {required_center_spacing_m!r}"
        )

    L, W, _H = (
        float(drone_dims_lwh_m[0]),
        float(drone_dims_lwh_m[1]),
        float(drone_dims_lwh_m[2]),
    )
    if not all(math.isfinite(x) and x > 0 for x in (L, W)):
        raise ValueError(f"drone_dims_lwh_m must be finite and > 0, got {drone_dims_lwh_m!r}")

    tau = float(response_time_s)
    a = float(max_decel_mps2)
    pos_m = float(pos_uncertainty_m)
    wind_mps = float(wind_speed_mps)
    if not (math.isfinite(tau) and tau >= 0):
        raise ValueError(f"response_time_s must be finite and >= 0, got {response_time_s!r}")
    if not (math.isfinite(a) and a > 0):
        raise ValueError(f"max_decel_mps2 must be finite and > 0, got {max_decel_mps2!r}")
    if not (math.isfinite(pos_m) and pos_m >= 0):
        raise ValueError(f"pos_uncertainty_m must be finite and >= 0, got {pos_uncertainty_m!r}")
    if not (math.isfinite(wind_mps) and wind_mps >= 0):
        raise ValueError(f"wind_speed_mps must be finite and >= 0, got {wind_speed_mps!r}")

    # body radius (heading-independent)
    body_radius_m = 0.5 * math.hypot(L, W)

    # r(v) = c0 + c1*v + c2*v^2
    c2 = 1.0 / (2.0 * a)
    c1 = tau + (wind_mps / a)
    c0 = body_radius_m + pos_m + (wind_mps * tau)

    r_max = 0.5 * spacing
    # Solve c2*v^2 + c1*v + (c0 - r_max) <= 0
    A = c2
    B = c1
    C = c0 - r_max
    if C <= 0:
        # Even at v=0 the radius fits; return "infinite-ish" by convention.
        return float("inf")

    disc = B * B - 4.0 * A * C
    if disc <= 0:
        return 0.0
    v_max = (-B + math.sqrt(disc)) / (2.0 * A)
    return max(0.0, float(v_max))


def max_speed_for_edge_occupancy(
    edge_length_m: float,
    max_drones_on_edge: int,
    drone_dims_lwh_m: Tuple[float, float, float],
    *,
    response_time_s: float = 1.0,
    max_decel_mps2: float = 3.0,
    pos_uncertainty_m: float = 5.0,
    wind_speed_mps: float = 5.0,
) -> float:
    """
    Convenience wrapper: convert an "edge capacity" constraint into a max speed.

    If an edge of length L contains at most N drones simultaneously, a simple
    model assumes the average center-to-center spacing along the edge is:

        spacing = L / N

    We then apply max_speed_for_spacing(spacing, ...).
    """
    L = float(edge_length_m)
    if not math.isfinite(L) or L <= 0:
        raise ValueError(f"edge_length_m must be finite and > 0, got {edge_length_m!r}")
    N = int(max_drones_on_edge)
    if N <= 0:
        raise ValueError(f"max_drones_on_edge must be >= 1, got {max_drones_on_edge!r}")
    spacing = L / float(N)
    v = max_speed_for_spacing(
        spacing,
        drone_dims_lwh_m,
        response_time_s=response_time_s,
        max_decel_mps2=max_decel_mps2,
        pos_uncertainty_m=pos_uncertainty_m,
        wind_speed_mps=wind_speed_mps,
    )
    return v

