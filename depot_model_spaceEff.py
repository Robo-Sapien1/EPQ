"""
Depot placement model (battery-range constrained, overlap-minimising)

You said:
- You will compute a MIN and MAX battery range for drones elsewhere (from bottom-layer cell size, etc.).
- For depot placement, you no longer want to “optimise R” (range) as a tradeoff vs depot count.
- Instead, GIVEN an allowed range window [R_min, R_max], you want the *most space-efficient* depot layout:
    -> minimise overlap / redundancy between depots (don’t waste coverage where you don’t need it)
    -> money / depot count is secondary (but we still add a tiny penalty so it doesn’t choose silly solutions)

So this program does:

Pipeline
1) Build polygon (city)
2) Poisson-disk sample interior “demand points”
3) Generate candidate depot centers in a THIN outside ring
4) Sweep R over [R_min, R_max] in steps (R_step)
5) For each R:
   - Solve a coverage optimisation:
       Coverage constraint: every interior point must be covered by >=1 chosen depot
       Objective (primary): minimise overlap redundancy
           redundancy = sum_i max(0, cover_count_i - 1)
       Objective (secondary): minimise number of depots (tiny weight)
   - Optional BFGS refinement (only kept if it improves min slack materially)
6) Choose the best R+solution by (redundancy, depot_count) lexicographically
7) Plotly:
   - (A) curve of redundancy vs R (and depot count vs R as hover)
   - (B) map of city, allowed ring, interior samples, chosen depots, and coverage circles

Dependencies:
  pip install shapely numpy plotly
Optional:
  pip install pulp     # ILP (recommended for overlap objective)
  pip install scipy    # BFGS refinement
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Dict
import math
import numpy as np

from shapely.geometry import Polygon, Point
from shapely.prepared import prep

import plotly.graph_objects as go

# Optional: ILP + BFGS
try:
    import pulp
    HAS_PULP = True
except Exception:
    HAS_PULP = False

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# =============================
# Geometry helpers
# =============================

def make_polygon(coords: List[Tuple[float, float]]) -> Polygon:
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        raise ValueError("Polygon is empty/invalid after fix. Check coordinates.")
    return poly


def polygon_to_xy(poly: Polygon) -> Tuple[np.ndarray, np.ndarray]:
    x, y = poly.exterior.xy
    return np.array(x), np.array(y)


def circle_xy(center: np.ndarray, R: float, n: int = 120) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 2 * math.pi, n, endpoint=True)
    x = center[0] + R * np.cos(t)
    y = center[1] + R * np.sin(t)
    return x, y


# =============================
# Poisson-disk sampling (Bridson) inside polygon
# =============================

def sample_points_poisson_in_polygon(
    poly: Polygon,
    min_dist: float,
    k: int = 30,
    seed: Optional[int] = None,
) -> np.ndarray:
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

    # seed point
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


# =============================
# Candidate depots in THIN ring
# =============================

def generate_candidate_depots_in_ring(
    city: Polygon,
    ring_inner: float,
    ring_outer: float,
    grid_step: float,
) -> tuple[np.ndarray, Polygon]:
    """
    allowed_ring = city.buffer(ring_outer) - city.buffer(ring_inner)
    Candidates are grid points inside allowed_ring.
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


# =============================
# Distances + coverage utilities
# =============================

def build_dist2_matrix(interior_pts: np.ndarray, candidate_depots: np.ndarray) -> np.ndarray:
    dx = interior_pts[:, None, 0] - candidate_depots[None, :, 0]
    dy = interior_pts[:, None, 1] - candidate_depots[None, :, 1]
    return dx * dx + dy * dy


def mask_for_R(dist2: np.ndarray, R: float) -> np.ndarray:
    """Boolean mask a_ij: shape (n_points, n_candidates), True if depot j covers point i."""
    return dist2 <= (R * R)


def uncovered_points_report(a: np.ndarray) -> np.ndarray:
    """Return indices of points that are not coverable by any candidate."""
    coverable = np.any(a, axis=1)
    return np.where(~coverable)[0]


def compute_redundancy(a: np.ndarray, chosen_idx: List[int]) -> int:
    """
    redundancy = sum_i max(0, cover_count_i - 1)
    where cover_count_i = number of chosen depots that cover point i
    """
    if len(chosen_idx) == 0:
        return 10**18
    sub = a[:, chosen_idx]  # (n_points, k)
    cover_count = np.sum(sub, axis=1).astype(int)
    return int(np.sum(np.maximum(0, cover_count - 1)))


# =============================
# Overlap-minimising solvers
# =============================

def ilp_min_overlap(
    a: np.ndarray,
    time_limit_s: Optional[int] = 30,
    depot_weight: float = 1e-3,
    msg: bool = False,
) -> List[int]:
    """
    ILP:
      min  sum_i z_i  + depot_weight * sum_j x_j
      s.t. sum_j a_ij x_j >= 1                         for all i
           z_i >= (sum_j a_ij x_j) - 1                 for all i
           z_i >= 0
           x_j in {0,1}
    Here z_i captures overlap (extra cover beyond 1) at point i.

    depot_weight is tiny: it only tie-breaks overlap-equal solutions by choosing fewer depots.
    """
    if not HAS_PULP:
        raise ImportError("PuLP not installed. pip install pulp")

    n_points, n_cand = a.shape
    prob = pulp.LpProblem("MinOverlapDepotCover", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{j}", 0, 1, cat=pulp.LpBinary) for j in range(n_cand)]
    z = [pulp.LpVariable(f"z_{i}", 0) for i in range(n_points)]

    # Objective
    prob += pulp.lpSum(z) + depot_weight * pulp.lpSum(x)

    # Coverage + overlap constraints
    for i in range(n_points):
        cover_expr = pulp.lpSum(x[j] for j in range(n_cand) if a[i, j])
        prob += cover_expr >= 1
        prob += z[i] >= cover_expr - 1

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_s) if time_limit_s else pulp.PULP_CBC_CMD(msg=msg)
    status = prob.solve(solver)
    status_name = pulp.LpStatus.get(status, str(status))
    if status_name not in ("Optimal", "Integer Feasible"):
        raise RuntimeError(f"ILP failed: {status_name}")

    chosen = [j for j in range(n_cand) if pulp.value(x[j]) is not None and pulp.value(x[j]) > 0.5]
    return chosen


def greedy_min_overlap(
    a: np.ndarray,
    depot_weight: float = 1e-3,
) -> List[int]:
    """
    Greedy heuristic that tries to minimise overlap:
    - Always ensure we keep making progress on uncovered points.
    - Score a candidate depot by:
        gain_uncovered - depot_weight * 1 - overlap_penalty
      where overlap_penalty counts how many already-covered points it would also cover.

    This tends to "spread" depots to cover new territory instead of stacking circles.
    """
    n_points, n_cand = a.shape
    uncovered = np.ones(n_points, dtype=bool)
    chosen: List[int] = []

    # Precompute for speed: list of point indices per depot
    pts_by_depot = [np.where(a[:, j])[0] for j in range(n_cand)]

    # Track covered count (for overlap accounting)
    covered_count = np.zeros(n_points, dtype=int)

    while np.any(uncovered):
        best_j = None
        best_score = -1e18
        best_new = None

        for j in range(n_cand):
            pts = pts_by_depot[j]
            if pts.size == 0:
                continue

            new_mask = uncovered[pts]
            gain = int(np.sum(new_mask))
            if gain == 0:
                continue  # must cover something new to progress

            # overlap penalty: points that are already covered that this depot also covers
            overlap_pts = pts[~new_mask]
            overlap_pen = int(np.sum(covered_count[overlap_pts] >= 1))

            score = gain - overlap_pen - depot_weight

            if score > best_score:
                best_score = score
                best_j = j
                best_new = pts[new_mask]

        if best_j is None:
            raise RuntimeError("Greedy overlap solver failed: cannot cover remaining points (increase R or densify candidates).")

        chosen.append(best_j)

        # Update coverage state
        pts = pts_by_depot[best_j]
        covered_count[pts] += 1
        uncovered[pts] = False

    return chosen


def solve_min_overlap_auto(
    a: np.ndarray,
    ilp_time_limit_s: int = 30,
    depot_weight: float = 1e-3,
) -> tuple[List[int], str]:
    """
    Prefer ILP (best overlap optimisation) if available, else greedy.
    """
    if HAS_PULP:
        try:
            chosen = ilp_min_overlap(a, time_limit_s=ilp_time_limit_s, depot_weight=depot_weight, msg=False)
            return chosen, "ilp_min_overlap"
        except Exception:
            pass
    chosen = greedy_min_overlap(a, depot_weight=depot_weight)
    return chosen, "greedy_min_overlap"


# =============================
# Optional: BFGS refinement (kept only if improves slack)
# =============================

def _min_slack(interior_pts: np.ndarray, depots_xy: np.ndarray, R: float) -> float:
    if len(depots_xy) == 0:
        return -float("inf")
    dx = interior_pts[:, None, 0] - depots_xy[None, :, 0]
    dy = interior_pts[:, None, 1] - depots_xy[None, :, 1]
    d2 = dx * dx + dy * dy
    min_d = np.sqrt(np.min(d2, axis=1))
    return float(np.min(R - min_d))


def refine_with_bfgs_if_helpful(
    depots_xy: np.ndarray,
    interior_pts: np.ndarray,
    R: float,
    allowed_region: Polygon,
    min_improvement: float = 2.0,
    penalty_outside: float = 1e6,
    penalty_uncovered: float = 1e6,
    edge_band_fraction: float = 0.2,
    maxiter: int = 200,
) -> Optional[np.ndarray]:
    """
    BFGS here does NOT change depot count.
    It tries to improve slack/robustness (points not sitting exactly on the boundary of coverage).
    Kept only if minimum slack improves by >= min_improvement.
    """
    if not HAS_SCIPY or len(depots_xy) == 0:
        return None

    allowed_prep = prep(allowed_region)
    n = len(depots_xy)
    x0 = depots_xy.reshape(-1)
    before = _min_slack(interior_pts, depots_xy, R)

    def objective(xvec: np.ndarray) -> float:
        centers = xvec.reshape((n, 2))
        pen = 0.0

        # keep depots inside allowed region
        for c in centers:
            pt = Point(float(c[0]), float(c[1]))
            if not allowed_prep.contains(pt):
                pen += penalty_outside * (pt.distance(allowed_region) + 1.0) ** 2

        dx = interior_pts[:, None, 0] - centers[None, :, 0]
        dy = interior_pts[:, None, 1] - centers[None, :, 1]
        d2 = dx * dx + dy * dy
        min_d = np.sqrt(np.min(d2, axis=1))

        slack = R - min_d
        uncovered = np.maximum(0.0, -slack)
        near_edge = np.maximum(0.0, edge_band_fraction * R - slack)

        pen += penalty_uncovered * float(np.sum(uncovered ** 2))
        pen += float(np.sum(near_edge ** 2))
        return pen

    res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": maxiter})
    refined = res.x.reshape((n, 2))

    # snap back any depot that left allowed ring
    for i in range(n):
        if not allowed_prep.contains(Point(float(refined[i, 0]), float(refined[i, 1]))):
            refined[i] = depots_xy[i]

    after = _min_slack(interior_pts, refined, R)
    if after - before >= min_improvement:
        return refined
    return None


# =============================
# Plotly
# =============================

def plot_solution_map(
    city: Polygon,
    allowed_ring: Polygon,
    interior_pts: np.ndarray,
    chosen_depots: np.ndarray,
    R: float,
) -> go.Figure:
    fig = go.Figure()

    cx, cy = polygon_to_xy(city)
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name="City boundary"))

    def add_poly_boundary(poly, name):
        def add_one_polygon(p: Polygon, label: str):
            x, y = p.exterior.xy
            fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines",
                                     name=label, line=dict(dash="dot")))
            for i, hole in enumerate(p.interiors):
                hx, hy = hole.xy
                fig.add_trace(go.Scatter(x=list(hx), y=list(hy), mode="lines",
                                         name=f"{label} hole {i+1}", line=dict(dash="dot")))

        if poly.geom_type == "Polygon":
            add_one_polygon(poly, name)
        elif poly.geom_type == "MultiPolygon":
            for k, p in enumerate(poly.geoms):
                add_one_polygon(p, f"{name} {k+1}")

    add_poly_boundary(allowed_ring, "Allowed ring")

    fig.add_trace(go.Scatter(
        x=interior_pts[:, 0], y=interior_pts[:, 1],
        mode="markers", name="Interior samples",
        marker=dict(size=5, opacity=0.7)
    ))

    fig.add_trace(go.Scatter(
        x=chosen_depots[:, 0], y=chosen_depots[:, 1],
        mode="markers", name="Chosen depots",
        marker=dict(size=10, symbol="x")
    ))

    for i, d in enumerate(chosen_depots):
        x, y = circle_xy(d, R)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            name="Coverage circles" if i == 0 else None,
            showlegend=(i == 0)
        ))

    fig.update_layout(
        title=f"Depot placement map (R={R:.1f})",
        xaxis_title="x",
        yaxis_title="y",
        yaxis_scaleanchor="x",
        legend=dict(orientation="h")
    )
    return fig


def plot_redundancy_curve(
    Rs: np.ndarray,
    redundancy: np.ndarray,
    depot_counts: np.ndarray,
    chosen_idx: int,
) -> go.Figure:
    fig = go.Figure()

    # Use redundancy as y. Add depot count as hover.
    fig.add_trace(go.Scatter(
        x=Rs,
        y=redundancy,
        mode="lines+markers",
        name="Redundancy (overlap)",
        customdata=np.stack([depot_counts], axis=1),
        hovertemplate="R=%{x}<br>redundancy=%{y}<br>#depots=%{customdata[0]}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[Rs[chosen_idx]],
        y=[redundancy[chosen_idx]],
        mode="markers",
        name="Chosen (min overlap)",
        marker=dict(size=12, symbol="diamond"),
        customdata=np.array([[depot_counts[chosen_idx]]]),
        hovertemplate="CHOSEN<br>R=%{x}<br>redundancy=%{y}<br>#depots=%{customdata[0]}<extra></extra>"
    ))

    fig.update_layout(
        title="Overlap objective: redundancy vs radius R",
        xaxis_title="R",
        yaxis_title="Redundancy = sum(max(0, cover_count-1))"
    )
    return fig


# =============================
# Sweep runner
# =============================

@dataclass
class SweepPoint:
    R: float
    feasible: bool
    method: Optional[str]
    n_depots: Optional[int]
    redundancy: Optional[int]
    chosen_indices: Optional[List[int]]
    reason: Optional[str]


@dataclass
class OverlapSweepResult:
    chosen_R: Optional[float]
    chosen_method: Optional[str]
    chosen_depots_xy: np.ndarray
    chosen_depots_xy_refined: Optional[np.ndarray]
    sweep: List[SweepPoint]
    fig_curve: Optional[go.Figure]
    fig_map: Optional[go.Figure]


def run_overlap_sweep_depot_model(
    city_coords: List[Tuple[float, float]],
    ring_inner: float,
    ring_outer: float,
    poisson_min_dist: float,
    depot_grid_step: float,
    R_min: float,
    R_max: float,
    R_step: float = 10.0,
    ilp_time_limit_s: int = 30,
    depot_weight: float = 1e-3,   # tiny tie-breaker toward fewer depots
    do_bfgs: bool = False,
    seed: int = 42,
) -> OverlapSweepResult:
    """
    You supply [R_min, R_max] from your battery model.

    For each R:
      - require full coverage
      - choose depot set to MINIMISE OVERLAP REDUNDANCY
      - tie-break slightly toward fewer depots (depot_weight)

    Final selection across R:
      Choose the solution with:
        1) minimum redundancy
        2) then minimum depot count
        3) then smaller R
    """
    if R_max < R_min:
        raise ValueError("R_max must be >= R_min")
    if R_step <= 0:
        raise ValueError("R_step must be > 0")

    city = make_polygon(city_coords)
    interior_pts = sample_points_poisson_in_polygon(city, min_dist=poisson_min_dist, seed=seed)
    candidate_depots, allowed_ring = generate_candidate_depots_in_ring(
        city, ring_inner=ring_inner, ring_outer=ring_outer, grid_step=depot_grid_step
    )
    dist2 = build_dist2_matrix(interior_pts, candidate_depots)

    start = float(math.ceil(R_min / R_step) * R_step)
    if start < R_min:
        start = R_min
    Rs = np.arange(start, R_max + 1e-9, R_step, dtype=float)

    sweep: List[SweepPoint] = []
    feasible_Rs: List[float] = []
    feasible_red: List[int] = []
    feasible_depots: List[int] = []
    feasible_choice: Dict[float, Tuple[List[int], str, int, int]] = {}

    for R in Rs:
        try:
            a = mask_for_R(dist2, float(R))
            bad = uncovered_points_report(a)
            if len(bad) > 0:
                sweep.append(SweepPoint(
                    R=float(R), feasible=False, method=None, n_depots=None, redundancy=None,
                    chosen_indices=None,
                    reason=f"{len(bad)} interior points not coverable by any candidate at this R."
                ))
                continue

            chosen_idx, method_used = solve_min_overlap_auto(
                a, ilp_time_limit_s=ilp_time_limit_s, depot_weight=depot_weight
            )
            n_dep = len(chosen_idx)
            red = compute_redundancy(a, chosen_idx)

            sweep.append(SweepPoint(
                R=float(R), feasible=True, method=method_used, n_depots=n_dep, redundancy=red,
                chosen_indices=chosen_idx, reason=None
            ))

            feasible_Rs.append(float(R))
            feasible_red.append(int(red))
            feasible_depots.append(int(n_dep))
            feasible_choice[float(R)] = (chosen_idx, method_used, int(red), int(n_dep))

        except Exception as e:
            sweep.append(SweepPoint(
                R=float(R), feasible=False, method=None, n_depots=None, redundancy=None,
                chosen_indices=None, reason=str(e)
            ))

    if not feasible_Rs:
        return OverlapSweepResult(
            chosen_R=None, chosen_method=None,
            chosen_depots_xy=np.zeros((0, 2)),
            chosen_depots_xy_refined=None,
            sweep=sweep,
            fig_curve=None,
            fig_map=None
        )

    Rs_arr = np.array(feasible_Rs, dtype=float)
    red_arr = np.array(feasible_red, dtype=int)
    dep_arr = np.array(feasible_depots, dtype=int)

    # Choose best by (redundancy, depot_count, R)
    best_idx = int(np.lexsort((Rs_arr, dep_arr, red_arr))[0])

    chosen_R = float(Rs_arr[best_idx])
    chosen_idx, chosen_method, chosen_red, chosen_n = feasible_choice[chosen_R]

    depots_xy = candidate_depots[chosen_idx]

    depots_refined = None
    if do_bfgs:
        refined = refine_with_bfgs_if_helpful(
            depots_xy=depots_xy,
            interior_pts=interior_pts,
            R=chosen_R,
            allowed_region=allowed_ring,
            min_improvement=2.0
        )
        if refined is not None:
            depots_refined = refined

    fig_curve = plot_redundancy_curve(Rs_arr, red_arr.astype(float), dep_arr.astype(float), best_idx)
    fig_map = plot_solution_map(
        city=city,
        allowed_ring=allowed_ring,
        interior_pts=interior_pts,
        chosen_depots=depots_refined if depots_refined is not None else depots_xy,
        R=chosen_R
    )

    return OverlapSweepResult(
        chosen_R=chosen_R,
        chosen_method=chosen_method,
        chosen_depots_xy=depots_xy,
        chosen_depots_xy_refined=depots_refined,
        sweep=sweep,
        fig_curve=fig_curve,
        fig_map=fig_map
    )


# =============================
# Example usage
# =============================

if __name__ == "__main__":
    # Example polygon (replace with your city boundary coords)
    city_coords = [
        (0, 0), (900, 0), (900, 150),
        (650, 150), (650, 350), (1100, 350),
        (1100, 650), (800, 650), (800, 900),
        (400, 900), (400, 600), (150, 600),
        (150, 250), (0, 250)
    ]

    # Pretend these came from your bottom-layer / battery model:
    R_min_battery = 270
    R_max_battery = 300

    result = run_overlap_sweep_depot_model(
        city_coords=city_coords,
        ring_inner=5,
        ring_outer=50,         # THIN ring -> prevents far-out depots
        poisson_min_dist=30,
        depot_grid_step=30,
        R_min=R_min_battery,
        R_max=R_max_battery,
        R_step=10,
        ilp_time_limit_s=60,   # ILP per R (falls back to greedy if ILP unavailable/fails)
        depot_weight=1e-3,     # tiny tie-break for fewer depots
        do_bfgs=True,
        seed=42
    )

    if result.chosen_R is None:
        print(" No feasible solution found in [R_min, R_max] with current ring/candidates.")
        print("Tips: decrease depot_grid_step, widen ring_outer, reduce ring_inner, or expand R range.")
    else:
        print(f" Chosen R (min overlap) = {result.chosen_R:.2f}")
        print(f"Method used              = {result.chosen_method}")
        print(f"# depots                 = {len(result.chosen_depots_xy)}")
        if result.chosen_depots_xy_refined is not None:
            print("BFGS refinement kept (improved slack).")

        result.fig_curve.show()
        result.fig_map.show()
