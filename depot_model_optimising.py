"""
Depot placement model (battery-range constrained) with:

1) "Eroded R" feasibility gate:
   - Compute R_required = max_i min_j dist(point_i, candidateDepot_j)
   - Start_R = max(R_min_user, R_required)
   - If Start_R > R_max_user: exit (not possible with current ring/candidates)

2) Baseline (minimum cost):
   - Compute k* = MIN #depots needed at R_max_user
   - Then find the SMALLEST R in [Start_R, R_max_user] that still allows k* depots
     (monotone -> binary search if ILP available; otherwise linear scan)
   - At that smallest R, pick the most space-efficient layout among EXACTLY k* depots
     (min overlap redundancy) if ILP is available.

3) Upgrade (distance optimisation under a depot cap):
   - User enters extra depots -> K_budget = k* + extra
   - Re-run at R_max_user (best for distance, since you aren’t penalising R):
       - Start from a baseline k* solution at R_max_user
       - Greedily add depots ONLY when they reduce total assigned distance
         (does NOT have to use all K_budget)

4) Optional BFGS refinement ONCE at the end:
   - Nudges depot coordinates inside allowed ring
   - Improves robustness (slack away from coverage boundary)
   - Optional tiny distance term (off by default)

Dependencies:
  pip install shapely numpy plotly
Recommended (exact baseline / monotone search):
  pip install pulp
Optional (final polish):
  pip install scipy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import numpy as np

from shapely.geometry import Polygon, Point
from shapely.prepared import prep

import plotly.graph_objects as go

# Optional: ILP
try:
    import pulp
    HAS_PULP = True
except Exception:
    HAS_PULP = False

# Optional: BFGS
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
    return dist2 <= (R * R)


def uncovered_points_report(a: np.ndarray) -> np.ndarray:
    return np.where(~np.any(a, axis=1))[0]


def required_R_for_coverability(dist2: np.ndarray) -> float:
    """
    Discrete "erode R" equivalent for your candidate set:
      R_required = max_i min_j distance(i,j)
    """
    nearest = np.sqrt(np.min(dist2, axis=1))
    return float(np.max(nearest))


def compute_redundancy(a: np.ndarray, chosen_idx: List[int]) -> int:
    if len(chosen_idx) == 0:
        return 10**18
    sub = a[:, chosen_idx]
    cover_count = np.sum(sub, axis=1).astype(int)
    return int(np.sum(np.maximum(0, cover_count - 1)))


def compute_total_assigned_distance(dist: np.ndarray, a: np.ndarray, chosen_idx: List[int]) -> float:
    """
    Sum over points of distance to NEAREST chosen depot that covers it.
    """
    if len(chosen_idx) == 0:
        return float("inf")
    sub_d = dist[:, chosen_idx]
    sub_a = a[:, chosen_idx]
    masked = np.where(sub_a, sub_d, np.inf)
    best = np.min(masked, axis=1)
    if np.any(~np.isfinite(best)):
        return float("inf")
    return float(np.sum(best))


# =============================
# ILP baseline solvers
# =============================

def ilp_min_depots(a: np.ndarray, time_limit_s: int = 30, msg: bool = False) -> List[int]:
    """
    Set cover:
      min sum_j x_j
      s.t. sum_j a[i,j] x_j >= 1 for all i
    """
    if not HAS_PULP:
        raise ImportError("PuLP not installed.")

    n_points, n_cand = a.shape
    prob = pulp.LpProblem("MinDepotsCover", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", 0, 1, cat=pulp.LpBinary) for j in range(n_cand)]

    prob += pulp.lpSum(x)

    for i in range(n_points):
        prob += pulp.lpSum(x[j] for j in range(n_cand) if a[i, j]) >= 1

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_s)
    status = prob.solve(solver)
    status_name = pulp.LpStatus.get(status, str(status))
    if status_name not in ("Optimal", "Integer Feasible"):
        raise RuntimeError(f"Min-depots ILP failed: {status_name}")

    return [j for j in range(n_cand) if pulp.value(x[j]) is not None and pulp.value(x[j]) > 0.5]


def ilp_min_overlap_given_exact_k(a: np.ndarray, k: int, time_limit_s: int = 30, msg: bool = False) -> List[int]:
    """
    Min overlap redundancy with EXACT k depots:
      min sum_i z_i
      s.t. coverage
           z_i >= (sum_j a[i,j] x_j) - 1
           sum_j x_j == k
    """
    if not HAS_PULP:
        raise ImportError("PuLP not installed.")

    n_points, n_cand = a.shape
    prob = pulp.LpProblem("MinOverlapGivenK", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", 0, 1, cat=pulp.LpBinary) for j in range(n_cand)]
    z = [pulp.LpVariable(f"z_{i}", 0) for i in range(n_points)]

    prob += pulp.lpSum(z)
    prob += pulp.lpSum(x) == int(k)

    for i in range(n_points):
        cover_expr = pulp.lpSum(x[j] for j in range(n_cand) if a[i, j])
        prob += cover_expr >= 1
        prob += z[i] >= cover_expr - 1

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_s)
    status = prob.solve(solver)
    status_name = pulp.LpStatus.get(status, str(status))
    if status_name not in ("Optimal", "Integer Feasible"):
        raise RuntimeError(f"Min-overlap-given-k ILP failed: {status_name}")

    return [j for j in range(n_cand) if pulp.value(x[j]) is not None and pulp.value(x[j]) > 0.5]


# =============================
# Greedy fallback (if no PuLP)
# =============================

def greedy_min_depots(a: np.ndarray) -> List[int]:
    n_points, n_cand = a.shape
    uncovered = np.ones(n_points, dtype=bool)
    chosen: List[int] = []
    pts_by_depot = [np.where(a[:, j])[0] for j in range(n_cand)]

    while np.any(uncovered):
        best_j = None
        best_gain = -1
        for j in range(n_cand):
            pts = pts_by_depot[j]
            gain = int(np.sum(uncovered[pts]))
            if gain > best_gain:
                best_gain = gain
                best_j = j
        if best_j is None or best_gain <= 0:
            raise RuntimeError("Greedy min-depots failed: cannot cover remaining points.")
        chosen.append(best_j)
        uncovered[pts_by_depot[best_j]] = False

    return chosen


def min_depots_at_R(a: np.ndarray, ilp_time_limit_s: int) -> List[int]:
    if HAS_PULP:
        return ilp_min_depots(a, time_limit_s=ilp_time_limit_s, msg=False)
    return greedy_min_depots(a)


def best_space_layout_given_k(a: np.ndarray, k: int, ilp_time_limit_s: int) -> List[int]:
    """
    Among EXACTLY k depots, pick most space-efficient (min overlap).
    If no ILP, just return a greedy min-depots (approx).
    """
    if HAS_PULP:
        return ilp_min_overlap_given_exact_k(a, k=k, time_limit_s=ilp_time_limit_s, msg=False)
    # fallback (not true min-overlap)
    return greedy_min_depots(a)


# =============================
# Find smallest R that preserves depot count
# =============================

def find_smallest_R_with_k(
    Rs: np.ndarray,
    dist2: np.ndarray,
    k_star: int,
    ilp_time_limit_s: int,
) -> float:
    """
    Find smallest R in Rs such that min_depots_count(R) <= k_star.
    With exact ILP, this is equivalent to == k_star because k_star is global min at R_max.

    If PuLP available: binary search (monotone)
    Else: linear scan (greedy can violate monotonicity).
    """
    cache_k: Dict[float, int] = {}

    def k_min(R: float) -> int:
        if R in cache_k:
            return cache_k[R]
        a = mask_for_R(dist2, R)
        if len(uncovered_points_report(a)) > 0:
            cache_k[R] = 10**9
            return cache_k[R]
        chosen = min_depots_at_R(a, ilp_time_limit_s)
        cache_k[R] = len(chosen)
        return cache_k[R]

    # Ensure last index can achieve k_star
    k_last = k_min(float(Rs[-1]))
    if k_last > k_star:
        raise RuntimeError("Unexpected: R_max cannot achieve k_star (should not happen).")

    if HAS_PULP:
        lo, hi = 0, len(Rs) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if k_min(float(Rs[mid])) <= k_star:
                hi = mid
            else:
                lo = mid + 1
        return float(Rs[lo])

    # Greedy fallback
    for R in Rs:
        if k_min(float(R)) <= k_star:
            return float(R)
    return float(Rs[-1])


# =============================
# Upgrade: greedy add depots to reduce distance (cap <= K_budget)
# =============================

@dataclass
class UpgradeResult:
    chosen_idx: List[int]
    distance_history: List[float]  # total distance after 0,1,2,... adds
    added_order: List[int]


def upgrade_by_greedy_add(
    a: np.ndarray,
    dist: np.ndarray,
    start_idx: List[int],
    K_budget: int,
    min_improvement: float = 1e-6,
) -> UpgradeResult:
    """
    Add depots until:
      - you hit K_budget, OR
      - the best improvement is < min_improvement (so you don't waste depots)
    """
    n_points, n_cand = a.shape
    chosen = list(start_idx)
    chosen_set = set(chosen)

    if len(chosen) > K_budget:
        raise ValueError("Start set already exceeds K_budget.")

    pts_by_depot = [np.where(a[:, j])[0] for j in range(n_cand)]

    # Current best distance per point
    sub_d = dist[:, chosen]
    sub_a = a[:, chosen]
    cur = np.min(np.where(sub_a, sub_d, np.inf), axis=1)
    if np.any(~np.isfinite(cur)):
        raise RuntimeError("Start set does not fully cover all points.")

    history = [float(np.sum(cur))]
    added_order: List[int] = []

    remaining = [j for j in range(n_cand) if j not in chosen_set]

    while len(chosen) < K_budget:
        best_j = None
        best_improve = 0.0

        for j in remaining:
            pts = pts_by_depot[j]
            if pts.size == 0:
                continue
            new_d = dist[pts, j]
            improve = float(np.sum(np.maximum(0.0, cur[pts] - new_d)))
            if improve > best_improve:
                best_improve = improve
                best_j = j

        if best_j is None or best_improve < float(min_improvement):
            break

        chosen.append(best_j)
        chosen_set.add(best_j)
        added_order.append(best_j)

        pts = pts_by_depot[best_j]
        cur[pts] = np.minimum(cur[pts], dist[pts, best_j])
        history.append(float(np.sum(cur)))

        remaining.remove(best_j)

    return UpgradeResult(chosen_idx=chosen, distance_history=history, added_order=added_order)


# =============================
# Final BFGS refinement (once)
# =============================

def refine_with_bfgs_end(
    depots_xy: np.ndarray,
    interior_pts: np.ndarray,
    R: float,
    allowed_region: Polygon,
    edge_band_fraction: float = 0.15,
    penalty_outside: float = 1e6,
    penalty_uncovered: float = 1e6,
    dist_weight: float = 0.0,
    maxiter: int = 250,
) -> Optional[np.ndarray]:
    if not HAS_SCIPY or len(depots_xy) == 0:
        return None

    allowed_prep = prep(allowed_region)
    n = len(depots_xy)
    x0 = depots_xy.reshape(-1)

    def objective(xvec: np.ndarray) -> float:
        centers = xvec.reshape((n, 2))
        pen = 0.0

        # keep inside allowed ring
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
        pen += penalty_uncovered * float(np.sum(uncovered ** 2))

        near_edge = np.maximum(0.0, edge_band_fraction * R - slack)
        pen += float(np.sum(near_edge ** 2))

        if dist_weight > 0.0:
            pen += float(dist_weight) * float(np.sum(min_d))

        return pen

    res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": maxiter})
    refined = res.x.reshape((n, 2))

    # snap any depot that left allowed region
    for i in range(n):
        if not allowed_prep.contains(Point(float(refined[i, 0]), float(refined[i, 1]))):
            refined[i] = depots_xy[i]

    return refined


# =============================
# Plotly
# =============================

def plot_k_curve(Rs: np.ndarray, k_vals: np.ndarray, chosen_R: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Rs, y=k_vals,
        mode="lines+markers",
        name="min #depots",
        hovertemplate="R=%{x}<br>min_depots=%{y}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[chosen_R], y=[float(k_vals[np.where(Rs == chosen_R)[0][0]]) if np.any(Rs == chosen_R) else None],
        mode="markers",
        name="Chosen baseline R",
        marker=dict(size=12, symbol="diamond"),
        hovertemplate="CHOSEN<br>R=%{x}<extra></extra>"
    ))
    fig.update_layout(
        title="Baseline sweep: minimum depot count vs R",
        xaxis_title="R",
        yaxis_title="Minimum #depots"
    )
    return fig


def plot_distance_upgrade_curve(distance_history: List[float]) -> go.Figure:
    xs = list(range(len(distance_history)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=distance_history,
        mode="lines+markers",
        name="total assigned distance",
        hovertemplate="depots_added=%{x}<br>total_distance=%{y}<extra></extra>"
    ))
    fig.update_layout(
        title="Upgrade: distance improvement as depots are added (stops when no benefit)",
        xaxis_title="Extra depots actually added",
        yaxis_title="Total assigned distance"
    )
    return fig


def plot_solution_map(
    city: Polygon,
    allowed_ring: Polygon,
    interior_pts: np.ndarray,
    chosen_depots: np.ndarray,
    R: float,
    title: str,
    show_circles: bool = True,
    max_circles: int = 60,
) -> go.Figure:
    fig = go.Figure()

    cx, cy = polygon_to_xy(city)
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name="City boundary"))

    def add_poly_boundary(poly, name):
        def add_one_polygon(p: Polygon, label: str):
            x, y = p.exterior.xy
            fig.add_trace(go.Scatter(
                x=list(x), y=list(y), mode="lines",
                name=label, line=dict(dash="dot")
            ))
            for i, hole in enumerate(p.interiors):
                hx, hy = hole.xy
                fig.add_trace(go.Scatter(
                    x=list(hx), y=list(hy), mode="lines",
                    name=f"{label} hole {i+1}", line=dict(dash="dot")
                ))

        if poly.geom_type == "Polygon":
            add_one_polygon(poly, name)
        elif poly.geom_type == "MultiPolygon":
            for k, p in enumerate(poly.geoms):
                add_one_polygon(p, f"{name} {k+1}")

    add_poly_boundary(allowed_ring, "Allowed ring")

    fig.add_trace(go.Scatter(
        x=interior_pts[:, 0], y=interior_pts[:, 1],
        mode="markers", name="Interior samples",
        marker=dict(size=5, opacity=0.6)
    ))

    fig.add_trace(go.Scatter(
        x=chosen_depots[:, 0], y=chosen_depots[:, 1],
        mode="markers", name="Chosen depots",
        marker=dict(size=10, symbol="x")
    ))

    if show_circles and len(chosen_depots) <= max_circles:
        for i, d in enumerate(chosen_depots):
            x, y = circle_xy(d, R)
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                name="Coverage circles" if i == 0 else None,
                showlegend=(i == 0),
                line=dict(width=1)
            ))

    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="y",
        yaxis_scaleanchor="x",
        legend=dict(orientation="h")
    )
    return fig


# =============================
# Main
# =============================

if __name__ == "__main__":
    print("RUNNING: depot_model_budgeted_minR_then_upgrade.py")
    print(f"PuLP available: {HAS_PULP} | SciPy available: {HAS_SCIPY}")

    # --- Example polygon (replace with your city boundary coords) ---
    city_coords = [
        (0, 0), (900, 0), (900, 150),
        (650, 150), (650, 350), (1100, 350),
        (1100, 650), (800, 650), (800, 900),
        (400, 900), (400, 600), (150, 600),
        (150, 250), (0, 250)
    ]

    # --- Battery model bounds (you provide these) ---
    R_min_user = 270
    R_max_user = 300

    # --- Placement parameters ---
    ring_inner = 5
    ring_outer = 50
    poisson_min_dist = 30
    depot_grid_step = 30
    R_step = 10
    ilp_time_limit_s = 60

    # --- Build geometry + candidates ---
    city = make_polygon(city_coords)
    interior_pts = sample_points_poisson_in_polygon(city, min_dist=poisson_min_dist, seed=42)
    candidate_depots, allowed_ring = generate_candidate_depots_in_ring(
        city, ring_inner=ring_inner, ring_outer=ring_outer, grid_step=depot_grid_step
    )

    dist2 = build_dist2_matrix(interior_pts, candidate_depots)
    dist = np.sqrt(dist2)

    # --- "Erode R" gate ---
    R_required = required_R_for_coverability(dist2)
    start_R = max(float(R_min_user), float(R_required))

    if start_R > float(R_max_user) + 1e-9:
        print("\n Not possible with current ring/candidates.")
        print(f"R_required (from candidates) = {R_required:.3f}")
        print(f"User R_max                  = {R_max_user:.3f}")
        print("Try: widen ring_outer, reduce ring_inner, decrease depot_grid_step, or increase R_max.")
        raise SystemExit(1)

    start_R = float(math.ceil(start_R / R_step) * R_step)

    Rs = np.arange(start_R, float(R_max_user) + 1e-9, float(R_step), dtype=float)
    if len(Rs) == 0 or Rs[-1] < float(R_max_user) - 1e-9:
        # ensure R_max_user included if rounding caused issues
        Rs = np.append(Rs, float(R_max_user))

    # --- Baseline at R_max: find k* (minimum cost depot count) ---
    R_for_kstar = float(R_max_user)
    a_max = mask_for_R(dist2, R_for_kstar)
    if len(uncovered_points_report(a_max)) > 0:
        print("\n Even R_max cannot cover all interior points with these candidates.")
        print("Try: widen ring_outer, reduce ring_inner, decrease depot_grid_step.")
        raise SystemExit(1)

    chosen_min_at_Rmax = min_depots_at_R(a_max, ilp_time_limit_s)
    k_star = len(chosen_min_at_Rmax)

    # --- Find smallest R that still allows k* depots ---
    R_baseline = find_smallest_R_with_k(Rs, dist2, k_star, ilp_time_limit_s)

    # --- Choose best space-efficient layout with EXACTLY k* depots at that smallest R ---
    a_base = mask_for_R(dist2, float(R_baseline))
    chosen_baseline_idx = best_space_layout_given_k(a_base, k_star, ilp_time_limit_s)

    baseline_red = compute_redundancy(a_base, chosen_baseline_idx)
    baseline_dist = compute_total_assigned_distance(dist, a_base, chosen_baseline_idx)

    print("\n=== BASELINE (minimum cost, then smallest R that keeps it) ===")
    print(f"R_required (candidates) : {R_required:.3f}")
    print(f"Start_R used            : {start_R:.3f}")
    print(f"k* (min depots @ R_max) : {k_star}")
    print(f"Chosen baseline R       : {R_baseline:.3f}")
    print(f"Baseline overlap        : {baseline_red}")
    print(f"Baseline total distance : {baseline_dist:.3f}")

    # (Optional) curve of min depot count vs R (fast enough: only set-cover ILP per R)
    k_vals = []
    for R in Rs:
        aR = mask_for_R(dist2, float(R))
        if len(uncovered_points_report(aR)) > 0:
            k_vals.append(np.nan)
        else:
            k_vals.append(len(min_depots_at_R(aR, ilp_time_limit_s)))
    k_vals = np.array(k_vals, dtype=float)

    fig_k = plot_k_curve(Rs, k_vals, float(R_baseline))
    fig_base_map = plot_solution_map(
        city=city,
        allowed_ring=allowed_ring,
        interior_pts=interior_pts,
        chosen_depots=candidate_depots[chosen_baseline_idx],
        R=float(R_baseline),
        title=f"Baseline | R={R_baseline:.1f} | depots={k_star}",
        show_circles=True,
        max_circles=60
    )
    fig_k.show()
    fig_base_map.show()

    # --- User budget: extra depots allowed ---
    while True:
        try:
            extra = int(input("\nHow many EXTRA depots do you want to allow (>=0)? ").strip())
            if extra < 0:
                print("Please enter a number >= 0.")
                continue
            break
        except Exception:
            print("Please enter an integer (e.g. 0, 3, 10).")

    K_budget = k_star + extra
    print(f"\nTotal depot cap K_budget = {K_budget} (k*={k_star} + extra={extra})")

    # --- Upgrade run (distance-focused) at R_max_user ---
    # We run at R_max_user because you are NOT penalising R, and distance cannot get worse with larger R.
    R_upgrade = float(R_max_user)
    a_up = mask_for_R(dist2, R_upgrade)

    # Start from a "nice" k* layout at R_max_user (space-efficient among k*)
    start_idx_upgrade = best_space_layout_given_k(a_up, k_star, ilp_time_limit_s)

    up = upgrade_by_greedy_add(
        a=a_up,
        dist=dist,
        start_idx=start_idx_upgrade,
        K_budget=K_budget,
        min_improvement=1e-6
    )

    used_k = len(up.chosen_idx)
    up_red = compute_redundancy(a_up, up.chosen_idx)
    up_dist = compute_total_assigned_distance(dist, a_up, up.chosen_idx)

    print("\n=== UPGRADE (distance optimisation under depot cap) ===")
    print(f"Upgrade R used               : {R_upgrade:.3f}")
    print(f"Depots used (<= cap)         : {used_k} / {K_budget}")
    print(f"Extra depots actually used   : {used_k - k_star}")
    print(f"Upgraded overlap             : {up_red}")
    print(f"Upgraded total distance      : {up_dist:.3f}")
    print(f"Distance reduction vs baseline: {baseline_dist - up_dist:.3f}")

    fig_up_curve = plot_distance_upgrade_curve(up.distance_history)
    fig_up_map = plot_solution_map(
        city=city,
        allowed_ring=allowed_ring,
        interior_pts=interior_pts,
        chosen_depots=candidate_depots[up.chosen_idx],
        R=R_upgrade,
        title=f"Upgraded | R={R_upgrade:.1f} | depots_used={used_k}/{K_budget}",
        show_circles=True,
        max_circles=60
    )
    fig_up_curve.show()
    fig_up_map.show()

    # --- Final BFGS polish (optional, once) ---
    do_bfgs = True
    if do_bfgs and HAS_SCIPY:
        depots_xy = candidate_depots[up.chosen_idx]
        refined = refine_with_bfgs_end(
            depots_xy=depots_xy,
            interior_pts=interior_pts,
            R=R_upgrade,
            allowed_region=allowed_ring,
            edge_band_fraction=0.15,
            dist_weight=0.0,   # set ~0.01 if you want slight distance polishing too
            maxiter=250
        )
        if refined is not None:
            fig_ref = plot_solution_map(
                city=city,
                allowed_ring=allowed_ring,
                interior_pts=interior_pts,
                chosen_depots=refined,
                R=R_upgrade,
                title=f"Final (after BFGS polish) | R={R_upgrade:.1f} | depots_used={used_k}/{K_budget}",
                show_circles=True,
                max_circles=60
            )
            fig_ref.show()
            print("\n BFGS polish applied (final map shown).")
        else:
            print("\n BFGS polish skipped (no scipy or no depots).")
    else:
        print("\n BFGS polish not run (scipy not installed or disabled).")
