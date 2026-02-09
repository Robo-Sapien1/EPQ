from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal
import math
import numpy as np

from shapely.geometry import Polygon, Point
from shapely.prepared import prep

import plotly.graph_objects as go

# Optional: ILP + BFGS
try:
    import pulp  # ILP (CBC solver typically bundled)
    HAS_PULP = True
except Exception:
    HAS_PULP = False

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# -----------------------------
# Geometry helpers
# -----------------------------

def make_polygon(coords: List[Tuple[float, float]]) -> Polygon:
    """Create a valid Shapely polygon from user coords."""
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)  # common fix for self-intersections
    if poly.is_empty:
        raise ValueError("Polygon is empty/invalid after fix. Check coordinates.")
    return poly


def polygon_to_xy(poly: Polygon) -> Tuple[np.ndarray, np.ndarray]:
    x, y = poly.exterior.xy
    return np.array(x), np.array(y)


def circle_xy(center: np.ndarray, R: float, n: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 2 * math.pi, n, endpoint=True)
    x = center[0] + R * np.cos(t)
    y = center[1] + R * np.sin(t)
    return x, y


# -----------------------------
# Poisson-disk sampling inside polygon (Bridson)
# -----------------------------

def sample_points_poisson_in_polygon(
    poly: Polygon,
    min_dist: float,
    k: int = 30,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Poisson-disk (Bridson) sampling inside a polygon.
    Returns array shape (N,2).
    """
    rng = np.random.default_rng(seed)

    minx, miny, maxx, maxy = poly.bounds
    width, height = maxx - minx, maxy - miny
    if width <= 0 or height <= 0:
        raise ValueError("Polygon bounds degenerate. Check polygon coordinates.")

    cell_size = min_dist / math.sqrt(2)
    grid_w = int(math.ceil(width / cell_size))
    grid_h = int(math.ceil(height / cell_size))

    grid = -np.ones((grid_h, grid_w), dtype=int)
    samples: List[np.ndarray] = []
    active: List[int] = []

    poly_prep = prep(poly)

    def point_to_cell(p: np.ndarray) -> Tuple[int, int]:
        gx = int((p[0] - minx) / cell_size)
        gy = int((p[1] - miny) / cell_size)
        return gx, gy

    def in_bounds(gx: int, gy: int) -> bool:
        return 0 <= gx < grid_w and 0 <= gy < grid_h

    def fits(p: np.ndarray) -> bool:
        if not poly_prep.contains(Point(float(p[0]), float(p[1]))):
            return False
        gx, gy = point_to_cell(p)
        for yy in range(gy - 2, gy + 3):
            for xx in range(gx - 2, gx + 3):
                if not in_bounds(xx, yy):
                    continue
                idx = grid[yy, xx]
                if idx != -1:
                    q = samples[idx]
                    if np.linalg.norm(p - q) < min_dist:
                        return False
        return True

    # pick a random start point inside polygon
    for _ in range(30_000):
        p0 = np.array([rng.uniform(minx, maxx), rng.uniform(miny, maxy)], dtype=float)
        if poly_prep.contains(Point(float(p0[0]), float(p0[1]))):
            samples.append(p0)
            gx, gy = point_to_cell(p0)
            grid[gy, gx] = 0
            active.append(0)
            break
    else:
        raise RuntimeError("Couldn't find a starting point inside polygon (sampling failed).")

    while active:
        idx = active[rng.integers(0, len(active))]
        base = samples[idx]
        found = False

        for _ in range(k):
            r = rng.uniform(min_dist, 2 * min_dist)
            theta = rng.uniform(0, 2 * math.pi)
            p = base + np.array([r * math.cos(theta), r * math.sin(theta)], dtype=float)

            if p[0] < minx or p[0] > maxx or p[1] < miny or p[1] > maxy:
                continue

            if fits(p):
                samples.append(p)
                new_idx = len(samples) - 1
                gx, gy = point_to_cell(p)
                grid[gy, gx] = new_idx
                active.append(new_idx)
                found = True

        if not found:
            active.remove(idx)

    return np.vstack(samples)


# -----------------------------
# Minimum feasible radius R* via erosion
# -----------------------------

def compute_min_feasible_R(city: Polygon, tol: float = 1.0, max_R: Optional[float] = None) -> float:
    """
    Finds minimal R* such that city.buffer(-R) is empty.
    Binary search on R. tol in same units as polygon coordinates.
    """
    minx, miny, maxx, maxy = city.bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    hi = max_R if max_R is not None else max(1.0, diag)
    lo = 0.0

    def erodes(R: float) -> bool:
        return city.buffer(-R).is_empty

    # Ensure hi erodes
    while not erodes(hi):
        hi *= 2.0
        if hi > 1e9:
            raise RuntimeError("R upper bound exploded. Check polygon units/validity.")

    # Binary search
    while (hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        if erodes(mid):
            hi = mid
        else:
            lo = mid

    return hi


# -----------------------------
# Candidate depot generation in outside ring
# -----------------------------

def generate_candidate_depots_in_ring(
    city: Polygon,
    ring_inner: float,
    ring_outer: float,
    grid_step: float,
) -> tuple[np.ndarray, Polygon]:
    """
    Candidate depot centers are grid points in the ring:
      allowed_ring = city.buffer(ring_outer) - city.buffer(ring_inner)
    Returns (candidate_points, allowed_ring_polygon_or_multipolygon).
    """
    if ring_outer <= ring_inner:
        raise ValueError("ring_outer must be > ring_inner.")
    if grid_step <= 0:
        raise ValueError("grid_step must be > 0.")

    inner = city.buffer(ring_inner)
    outer = city.buffer(ring_outer)
    allowed = outer.difference(inner)

    if allowed.is_empty:
        raise ValueError("Allowed ring is empty. Check ring_inner/ring_outer.")

    minx, miny, maxx, maxy = allowed.bounds
    allowed_prep = prep(allowed)

    xs = np.arange(minx, maxx + grid_step, grid_step)
    ys = np.arange(miny, maxy + grid_step, grid_step)

    pts = []
    for y in ys:
        for x in xs:
            if allowed_prep.contains(Point(float(x), float(y))):
                pts.append((float(x), float(y)))

    if not pts:
        raise RuntimeError("No candidate depots found. Decrease grid_step or widen ring.")

    return np.array(pts, dtype=float), allowed


# -----------------------------
# Coverage matrix
# -----------------------------

def build_coverage_matrix(
    interior_pts: np.ndarray,
    candidate_depots: np.ndarray,
    R: float
) -> List[np.ndarray]:
    """covers[j] = indices of interior points covered by candidate depot j."""
    R2 = R * R
    covers: List[np.ndarray] = []
    for d in candidate_depots:
        dx = interior_pts[:, 0] - d[0]
        dy = interior_pts[:, 1] - d[1]
        mask = (dx * dx + dy * dy) <= R2
        covers.append(np.where(mask)[0])
    return covers


# -----------------------------
# Greedy set cover
# -----------------------------

def greedy_set_cover(covers: List[np.ndarray], n_points: int) -> List[int]:
    """
    Greedy set cover: iteratively pick depot covering most uncovered interior points.
    Returns indices of chosen depots.
    """
    uncovered = set(range(n_points))
    chosen: List[int] = []

    while uncovered:
        best_j, best_gain, best_cover = None, 0, None
        for j, idxs in enumerate(covers):
            if idxs.size == 0:
                continue
            gain = sum(1 for i in idxs if int(i) in uncovered)
            if gain > best_gain:
                best_j, best_gain, best_cover = j, gain, idxs

        if best_j is None or best_gain == 0:
            raise RuntimeError(
                f"Greedy failed: {len(uncovered)} points uncovered. "
                "Increase R, widen ring, or densify candidates."
            )

        chosen.append(best_j)
        for i in best_cover:
            uncovered.discard(int(i))

    return chosen


# -----------------------------
# ILP set cover (optimal for discretised candidates)
# -----------------------------

def ilp_set_cover(
    covers: List[np.ndarray],
    n_points: int,
    time_limit_s: Optional[int] = None,
    msg: bool = False
) -> List[int]:
    """
    Solve set cover via ILP:
      min sum x_j
      s.t. for each point i: sum_{j covers i} x_j >= 1
      x_j in {0,1}
    """
    if not HAS_PULP:
        raise ImportError("PuLP not installed. pip install pulp")

    m = len(covers)
    prob = pulp.LpProblem("DepotSetCover", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat=pulp.LpBinary) for j in range(m)]

    prob += pulp.lpSum(x)  # objective

    # reverse mapping: for each point i, list depots that cover it
    coverers: List[List[int]] = [[] for _ in range(n_points)]
    for j, idxs in enumerate(covers):
        for i in idxs:
            coverers[int(i)].append(j)

    for i in range(n_points):
        if not coverers[i]:
            raise RuntimeError(
                f"ILP infeasible under discretisation: interior point {i} has no covering depot at this R."
            )
        prob += pulp.lpSum(x[j] for j in coverers[i]) >= 1

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_s) if time_limit_s else pulp.PULP_CBC_CMD(msg=msg)
    status = prob.solve(solver)
    status_name = pulp.LpStatus.get(status, str(status))

    # CBC may return "Optimal" or sometimes "Not Solved"/"Infeasible"/etc.
    if status_name not in ("Optimal", "Integer Feasible"):
        raise RuntimeError(f"ILP failed: {status_name}")

    chosen = [j for j in range(m) if pulp.value(x[j]) is not None and pulp.value(x[j]) > 0.5]
    return chosen


# -----------------------------
# Auto solver: ILP then fallback to greedy
# -----------------------------

def solve_set_cover_auto(
    covers: List[np.ndarray],
    n_points: int,
    ilp_time_limit_s: int = 30,
    prefer_ilp: bool = True,
) -> tuple[List[int], str]:
    """
    Try ILP first (time-limited). If it fails or isn't available, fall back to greedy.

    Returns: (chosen_indices, method_used) where method_used is "ilp" or "greedy".
    """
    if not prefer_ilp:
        return greedy_set_cover(covers, n_points), "greedy"

    if HAS_PULP:
        try:
            chosen = ilp_set_cover(covers, n_points, time_limit_s=ilp_time_limit_s, msg=False)
            return chosen, "ilp"
        except Exception:
            pass  # fall back

    chosen = greedy_set_cover(covers, n_points)
    return chosen, "greedy"


# -----------------------------
# Optional: BFGS refinement (constrained via penalties)
# -----------------------------

def refine_with_bfgs(
    depots_xy: np.ndarray,
    interior_pts: np.ndarray,
    R: float,
    allowed_region: Polygon,
    penalty_outside: float = 1e6,
    penalty_uncovered: float = 1e6,
    edge_band_fraction: float = 0.2,
    maxiter: int = 200,
) -> np.ndarray:
    """
    Nudges depots continuously to improve robustness (slack) without changing depot count.
    Penalties enforce:
      - depots remain inside allowed_region
      - all interior points stay covered

    Note: This cannot reduce depot count; it only tweaks positions.
    """
    if not HAS_SCIPY or len(depots_xy) == 0:
        return depots_xy

    allowed_prep = prep(allowed_region)
    n = len(depots_xy)
    x0 = depots_xy.reshape(-1)

    def objective(xvec: np.ndarray) -> float:
        centers = xvec.reshape((n, 2))

        pen = 0.0

        # Penalty for leaving allowed region (outside-only constraint)
        for c in centers:
            pt = Point(float(c[0]), float(c[1]))
            if not allowed_prep.contains(pt):
                # distance to allowed region gives a useful magnitude
                pen += penalty_outside * (pt.distance(allowed_region) + 1.0) ** 2

        # Distance from each interior point to nearest depot
        dx = interior_pts[:, None, 0] - centers[None, :, 0]
        dy = interior_pts[:, None, 1] - centers[None, :, 1]
        d2 = dx * dx + dy * dy
        min_d = np.sqrt(np.min(d2, axis=1))

        margin = R - min_d
        uncovered = np.maximum(0.0, -margin)  # uncovered where min_d > R
        near_edge = np.maximum(0.0, edge_band_fraction * R - margin)

        pen += penalty_uncovered * float(np.sum(uncovered ** 2))  # huge
        pen += float(np.sum(near_edge ** 2))  # mild robustness push

        return pen

    res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": maxiter})
    refined = res.x.reshape((n, 2))

    # Safety snap-back: if any depot left allowed ring, revert that depot
    for i in range(n):
        if not allowed_prep.contains(Point(float(refined[i, 0]), float(refined[i, 1]))):
            refined[i] = depots_xy[i]

    return refined


# -----------------------------
# Plotly visualization
# -----------------------------

def plot_solution(
    city: Polygon,
    allowed_ring: Polygon,
    interior_pts: np.ndarray,
    candidate_depots: np.ndarray,
    chosen_depots: np.ndarray,
    R: float,
    show_candidates: bool = False,
    show_circles: bool = True,
) -> go.Figure:
    fig = go.Figure()

    # City boundary
    cx, cy = polygon_to_xy(city)
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode="lines", name="City boundary"
    ))

    # Allowed ring boundary (may be multi-polygon)
    def add_poly_boundary(poly, name):
        def add_one_polygon(p: Polygon, label: str):
            x, y = p.exterior.xy
            fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", name=label, line=dict(dash="dot")))
            # holes (interiors)
            for i, hole in enumerate(p.interiors):
                hx, hy = hole.xy
                fig.add_trace(go.Scatter(x=list(hx), y=list(hy), mode="lines", name=f"{label} hole {i + 1}", line=dict(dash="dot")))

        if poly.geom_type == "Polygon":
            add_one_polygon(poly, name)
        elif poly.geom_type == "MultiPolygon":
            for k, p in enumerate(poly.geoms):
                add_one_polygon(p, f"{name} {k + 1}")

    add_poly_boundary(allowed_ring, "Allowed ring")

    # Interior samples
    fig.add_trace(go.Scatter(
        x=interior_pts[:, 0], y=interior_pts[:, 1],
        mode="markers", name="Interior samples",
        marker=dict(size=5, opacity=0.7)
    ))

    # Candidate depots (optional)
    if show_candidates:
        fig.add_trace(go.Scatter(
            x=candidate_depots[:, 0], y=candidate_depots[:, 1],
            mode="markers", name="Candidate depots",
            marker=dict(size=4, opacity=0.25)
        ))

    # Chosen depots
    fig.add_trace(go.Scatter(
        x=chosen_depots[:, 0], y=chosen_depots[:, 1],
        mode="markers", name="Chosen depots",
        marker=dict(size=10, symbol="x")
    ))

    # Coverage circles (as line traces)
    if show_circles:
        for i, d in enumerate(chosen_depots):
            x, y = circle_xy(d, R)
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                name="Coverage circles" if i == 0 else None,
                showlegend=(i == 0)
            ))

    fig.update_layout(
        title="Depot placement model (outside ring) — coverage circles",
        xaxis_title="x",
        yaxis_title="y",
        yaxis_scaleanchor="x",
        legend=dict(orientation="h")
    )
    return fig


# -----------------------------
# Main wrapper
# -----------------------------

@dataclass
class DepotModelResult:
    R_star: float
    R_used: float
    method_used: str
    chosen_indices: List[int]
    depots_xy: np.ndarray
    depots_xy_refined: Optional[np.ndarray]
    interior_xy: np.ndarray
    candidate_xy: np.ndarray
    fig: go.Figure


def run_depot_model_plotly(
    city_coords: List[Tuple[float, float]],
    ring_inner: float,
    ring_outer: float,
    poisson_min_dist: float,
    depot_grid_step: float,
    R_tol: float = 1.0,
    R_margin: float = 0.0,
    solver: Literal["auto", "greedy", "ilp"] = "auto",
    ilp_time_limit_s: Optional[int] = 30,
    do_bfgs: bool = False,
    seed: Optional[int] = 42,
    show_candidates: bool = False,
) -> DepotModelResult:
    """
    Full pipeline:
      1) build polygon
      2) Poisson sample interior points
      3) find R* by erosion
      4) generate candidate depots in outside ring
      5) solve set cover using solver ("auto" tries ILP then greedy)
      6) optional BFGS refinement (continuous nudging)
      7) plot in Plotly
    """
    city = make_polygon(city_coords)

    interior_pts = sample_points_poisson_in_polygon(city, min_dist=poisson_min_dist, seed=seed)

    R_star = compute_min_feasible_R(city, tol=R_tol)
    R_used = R_star + R_margin

    candidate_depots, allowed_ring = generate_candidate_depots_in_ring(
        city, ring_inner=ring_inner, ring_outer=ring_outer, grid_step=depot_grid_step
    )

    covers = build_coverage_matrix(interior_pts, candidate_depots, R_used)

    if solver == "greedy":
        chosen_idx = greedy_set_cover(covers, n_points=len(interior_pts))
        method_used = "greedy"
    elif solver == "ilp":
        chosen_idx = ilp_set_cover(covers, n_points=len(interior_pts), time_limit_s=ilp_time_limit_s, msg=False)
        method_used = "ilp"
    else:
        chosen_idx, method_used = solve_set_cover_auto(
            covers=covers,
            n_points=len(interior_pts),
            ilp_time_limit_s=int(ilp_time_limit_s or 30),
            prefer_ilp=True
        )

    depots = candidate_depots[chosen_idx]

    depots_refined = None
    if do_bfgs and len(depots) > 0:
        # Only useful if you allow depots to move off-grid; improves robustness, not depot count.
        depots_refined = refine_with_bfgs(
            depots_xy=depots,
            interior_pts=interior_pts,
            R=R_used,
            allowed_region=allowed_ring,
            maxiter=200
        )

    fig = plot_solution(
        city=city,
        allowed_ring=allowed_ring,
        interior_pts=interior_pts,
        candidate_depots=candidate_depots,
        chosen_depots=depots_refined if depots_refined is not None else depots,
        R=R_used,
        show_candidates=show_candidates,
        show_circles=True,
    )

    return DepotModelResult(
        R_star=R_star,
        R_used=R_used,
        method_used=method_used,
        chosen_indices=chosen_idx,
        depots_xy=depots,
        depots_xy_refined=depots_refined,
        interior_xy=interior_pts,
        candidate_xy=candidate_depots,
        fig=fig
    )


# -----------------------------
# Example usage
# -----------------------------
city_coords = [
    (0, 0), (900, 0), (900, 150),
    (650, 150), (650, 350), (1100, 350),
    (1100, 650), (800, 650), (800, 900),
    (400, 900), (400, 600), (150, 600),
    (150, 250), (0, 250)
]

res = run_depot_model_plotly(
    city_coords=city_coords,
    ring_inner=5,          # depots must be at least x units outside city
    ring_outer=50,         # search up to x units outside
    poisson_min_dist=20,    # interior sample spacing (smaller = stricter)
    depot_grid_step=10,     # candidate depot density (smaller = more candidates)
    R_tol=2.0,              # erosion binary-search tolerance for R*
    R_margin=40,           # optional safety margin added to R*
    solver="auto",          # "auto" tries ILP (time-limited) then greedy fallback
    ilp_time_limit_s=60,    # ILP time limit for auto/ilp
    do_bfgs=True,           # optional (requires scipy)
    seed=42,
    show_candidates=False
)

print("R* (min feasible)  =", res.R_star)
print("R used             =", res.R_used)
print("Set cover method   =", res.method_used)
print("# depots chosen     =", len(res.depots_xy))
if res.depots_xy_refined is not None:
    print("BFGS refinement applied.")

res.fig.show()
