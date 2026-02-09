# -*- coding: utf-8 -*-
"""
Depot placement model (battery-range constrained) with:

1) "Eroded R" feasibility gate:
   - Compute R_required = max_i min_j dist(point_i, candidateDepot_j)
   - Start_R = max(R_min_user, R_required)
   - If Start_R > R_max_user: exit (not possible with current ring/candidates)

2) Baseline (minimum cost):
   - Compute k* = MIN #depots needed at R_max_user  (Stage 1 set cover)
   - Then find the SMALLEST R in [Start_R, R_max_user] that still allows k* depots
     (monotone -> binary search if ILP available; otherwise linear scan)
   - At that smallest R, pick a "space-efficient" layout among EXACTLY k* depots
     (min overlap redundancy) if ILP is available.

3) Upgrade (distance optimisation under a depot cap) [DEMAND-AWARE]:
   - User enters extra depots -> K_budget = k* + extra
   - Re-run at R_max_user (best for distance, since you aren’t penalising R):
       - Try a MILP assignment/p-median-style model (time-limited). If it returns a feasible solution,
         use it (or keep greedy if greedy is better).
       - If MILP fails / times out without a feasible integer solution, fall back to greedy-add:
           - Start from a baseline k* solution at R_max_user
           - Greedily add depots ONLY when they reduce DEMAND-WEIGHTED assigned distance
             (does NOT have to use all K_budget)

4) Optional BFGS refinement ONCE at the end:
   - Nudges depot coordinates inside allowed ring
   - Improves robustness (slack away from coverage boundary)
   - Optional tiny distance term (off by default) [can be demand-weighted]

Improvements added (as discussed):
A) "Polygon holes" mitigation:
   - Add boundary samples (plus polygon vertices).
   - Optional adaptive refinement: validate coverage on a grid; add uncovered points; re-solve.

B) Stage 1 robustness + speed:
   - Always build a greedy feasible solution first (upper bound).
   - ILP is warm-started from greedy when possible.
   - If ILP times out, accept a validated feasible incumbent if it beats/equals greedy;
     otherwise fall back to greedy.
   - Preprocessing for Stage 1 min-depots ILP:
       - forced depots (points with only one covering candidate)
       - dominated candidate elimination (safe for min-depots objective)

C) Windows warm-start without leaving clutter:
   - CBC warm starts require keepFiles=True on Windows; we solve inside a TemporaryDirectory,
     so files exist only during the solve and are deleted automatically.

Dependencies:
  pip install shapely numpy plotly
Recommended (exact baseline / monotone search + stage2 MILP):
  pip install pulp
Optional (final polish):
  pip install scipy
"""

from __future__ import annotations

import json
from pathlib import Path

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Sequence, Set, Iterable
import math
import os
import platform
import tempfile
import contextlib
from collections import defaultdict

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.prepared import prep

import plotly.graph_objects as go

# Optional: ILP (PuLP + CBC)
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
# User-facing toggles
# =============================

SHOW_PLOTS = True                     # Plotly figures (no files) via browser
SAVE_SOLUTION_FOR_OVERLAY = True     # Save interior_pts, weights, depots to JSON for london_3d_overlay
USE_BOUNDARY_SAMPLES = True           # Fix #1 (important)
BOUNDARY_SPACING_FACTOR = 0.6         # boundary step = factor * poisson_min_dist
ADD_POLYGON_VERTICES = True           # include exact vertices as samples

ADAPTIVE_REFINE_ITERS = 1             # Fix #1 (optional extra robustness)
REFINE_GRID_STEP_FACTOR = 0.8         # grid step = factor * poisson_min_dist
REFINE_MAX_NEW_POINTS_PER_ITER = 250  # cap per iteration to avoid blow-up

USE_WARM_START = True                 # recommended
SOLVER_VERBOSE = False                # CBC chatter (PuLP msg=)

# --- CBC file visibility / debugging ---
# For testing warm starts, set these True so you can *see* the CBC .mps/.sol files.
CBC_WORKDIR_NAME = ".cbc_cbc_files"      # folder created in your current workspace
CBC_USE_WORKDIR = True                  # run CBC solves inside that folder
CBC_KEEP_WORKDIR = True                 # do NOT delete the folder after the run
CBC_CLEAN_WORKDIR_ON_RUN = True         # delete old CBC files at program start (replaces each run)


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
# Demand weights (interior points)
# =============================

def compute_demand_weights(
    interior_pts: np.ndarray,
    hotspots: Sequence[Tuple[float, float, float]] = (),
    sigma: float = 200.0,
    floor: float = 1.0,
    normalise_mean_to_1: bool = True,
) -> np.ndarray:
    """
    Returns positive demand weights w[i] for each interior point.

    Model:
      w = floor + sum_k (peak_k * exp(-||p - h_k||^2 / (2*sigma^2)))

    - If hotspots is empty -> uniform weights (all ones if floor=1).
    - normalise_mean_to_1 keeps objectives comparable across different hotspot settings.
    """
    if interior_pts.ndim != 2 or interior_pts.shape[1] != 2:
        raise ValueError("interior_pts must be (N,2).")

    n = interior_pts.shape[0]
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if floor <= 0:
        raise ValueError("floor must be > 0 (weights must stay positive).")

    w = np.full(n, float(floor), dtype=float)
    if hotspots:
        x = interior_pts[:, 0]
        y = interior_pts[:, 1]
        two_sigma2 = 2.0 * (float(sigma) ** 2)

        for hx, hy, peak in hotspots:
            peak = float(peak)
            if peak <= 0:
                continue
            d2 = (x - float(hx)) ** 2 + (y - float(hy)) ** 2
            w += peak * np.exp(-d2 / two_sigma2)

    if normalise_mean_to_1:
        m = float(np.mean(w))
        if m > 0:
            w = w / m

    return w


def compute_weights_from_density_map(
    interior_pts: np.ndarray,
    demand_density_map: dict,
    floor: float = 1.0,
    normalise_mean_to_1: bool = True,
) -> np.ndarray:
    """
    Compute demand weight for each interior point by looking up the demand density
    at that point in the raster from integrating_step1. Weight = floor + density at cell.
    """
    if interior_pts.ndim != 2 or interior_pts.shape[1] != 2:
        raise ValueError("interior_pts must be (N,2).")
    n = interior_pts.shape[0]
    if floor <= 0:
        raise ValueError("floor must be > 0.")

    xmin = float(demand_density_map["xmin"])
    ymin = float(demand_density_map["ymin"])
    cell_size = float(demand_density_map["cell_size"])
    ncols = int(demand_density_map["ncols"])
    nrows = int(demand_density_map["nrows"])
    values = demand_density_map["values"]

    ix = np.clip(
        (interior_pts[:, 0] - xmin) / cell_size,
        0, ncols - 1e-9
    ).astype(np.int32)
    iy = np.clip(
        (interior_pts[:, 1] - ymin) / cell_size,
        0, nrows - 1e-9
    ).astype(np.int32)
    ix = np.clip(ix, 0, ncols - 1)
    iy = np.clip(iy, 0, nrows - 1)
    w = np.array([float(values[iy[i] * ncols + ix[i]]) for i in range(n)], dtype=float)
    w = np.maximum(w, 0.0) + floor

    if normalise_mean_to_1:
        m = float(np.mean(w))
        if m > 0:
            w = w / m
    return w


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    s = float(np.sum(weights))
    if s <= 0:
        return float("nan")
    return float(np.sum(weights * values) / s)


# =============================
# Sampling helpers (Fix #1)
# =============================

def sample_points_on_boundary(poly: Polygon, step: float) -> np.ndarray:
    """
    Sample points along the polygon exterior at approximately 'step' spacing.
    Also includes the start point at distance 0.
    """
    if step <= 0:
        raise ValueError("step must be > 0.")
    boundary = poly.exterior
    L = float(boundary.length)
    if L <= 0:
        return np.zeros((0, 2), dtype=float)

    dists = np.arange(0.0, L + 1e-9, float(step))
    pts = []
    for d in dists:
        p = boundary.interpolate(float(d))
        pts.append((float(p.x), float(p.y)))
    return np.array(pts, dtype=float)


def dedupe_points(pts: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Dedupe by rounding to a grid defined by tol.
    """
    if len(pts) == 0:
        return pts
    key = np.round(pts / float(tol)).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(idx)]


def grid_points_in_polygon(poly: Polygon, step: float, max_points: int = 25000, seed: int = 0) -> np.ndarray:
    """
    Uniform grid inside polygon bounds, filtered by contains().
    Returns up to max_points points (randomly subsampled if necessary).
    """
    if step <= 0:
        raise ValueError("step must be > 0.")
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx, maxx + step, step, dtype=float)
    ys = np.arange(miny, maxy + step, step, dtype=float)

    poly_prep = prep(poly)
    pts = []
    for y in ys:
        for x in xs:
            if poly_prep.contains(Point(float(x), float(y))):
                pts.append((float(x), float(y)))

    if not pts:
        return np.zeros((0, 2), dtype=float)

    pts = np.array(pts, dtype=float)
    if len(pts) > int(max_points):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=int(max_points), replace=False)
        pts = pts[idx]
    return pts


def uncovered_by_depots(points: np.ndarray, depots: np.ndarray, R: float) -> np.ndarray:
    """
    Returns indices of 'points' that are NOT within distance R of any depot.
    """
    if len(points) == 0:
        return np.zeros((0,), dtype=int)
    if len(depots) == 0:
        return np.arange(len(points), dtype=int)
    dx = points[:, None, 0] - depots[None, :, 0]
    dy = points[:, None, 1] - depots[None, :, 1]
    d2 = dx * dx + dy * dy
    min_d2 = np.min(d2, axis=1)
    return np.where(min_d2 > (R * R + 1e-12))[0]


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


def assigned_distances(dist: np.ndarray, a: np.ndarray, chosen_idx: List[int]) -> np.ndarray:
    """
    Per-point distance to nearest chosen depot that covers it.
    Returns an array of shape (n_points,).
    """
    if len(chosen_idx) == 0:
        return np.full(dist.shape[0], np.inf, dtype=float)
    sub_d = dist[:, chosen_idx]
    sub_a = a[:, chosen_idx]
    best = np.min(np.where(sub_a, sub_d, np.inf), axis=1)
    return best


def compute_total_assigned_distance(
    dist: np.ndarray,
    a: np.ndarray,
    chosen_idx: List[int],
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Sum over points of distance to NEAREST chosen depot that covers it.
    If weights provided: sum(weights * best_distance).
    """
    best = assigned_distances(dist, a, chosen_idx)
    if np.any(~np.isfinite(best)):
        return float("inf")
    if weights is None:
        return float(np.sum(best))
    return float(np.sum(weights * best))



def _clean_cbc_workdir(workdir: Path) -> None:
    """Delete old CBC artefacts in the workdir so each *program run* 'replaces' them."""
    if not workdir.exists():
        return
    exts = {".mps", ".lp", ".sol", ".mst", ".log", ".txt"}
    for p in workdir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            try:
                p.unlink()
            except Exception:
                pass

# =============================
# Solver helpers (warm-start + workspace workdir)
# =============================

@contextlib.contextmanager
def _maybe_cbc_workdir(enabled: bool):
    """Optionally run CBC inside a dedicated folder in your *workspace*.

    Why:
      - PuLP's CBC warmStart on Windows requires keepFiles=True, which writes .mps/.sol files
        in the current working directory.
      - keepFiles=True means those files are *not deleted* after solving.
      - Putting them in a dedicated folder keeps your project tidy; cleaning-on-run makes them
        effectively 'replace each run'.
    """
    if not enabled:
        yield None
        return

    workdir = Path.cwd() / CBC_WORKDIR_NAME
    workdir.mkdir(parents=True, exist_ok=True)

    old = os.getcwd()
    os.chdir(str(workdir))
    try:
        yield str(workdir)
    finally:
        os.chdir(old)

    # Note: We intentionally do NOT delete the folder here when CBC_KEEP_WORKDIR=True,
    # so you can inspect the files after the program finishes.


def _cbc_solver(time_limit_s: int, warm_start: bool) -> "pulp.LpSolver":
    """
    Create a CBC solver with settings:
    - warmStart uses current variable initial values.
    - On Windows, warmStart requires keepFiles=True (PuLP warns otherwise).
    - keepFiles=True keeps the intermediate .mps/.sol files in the *current directory* instead of deleting them.
    """
    if not HAS_PULP:
        raise ImportError("PuLP not installed.")
    is_windows = (platform.system().lower().startswith("win"))
    keep_files = bool(CBC_USE_WORKDIR or (warm_start and is_windows))
    return pulp.PULP_CBC_CMD(
        msg=bool(SOLVER_VERBOSE),
        timeLimit=float(time_limit_s),
        warmStart=bool(warm_start),
        keepFiles=bool(keep_files),
    )


def _set_warm_start_values(x_vars: List["pulp.LpVariable"], chosen: Set[int]):
    """
    Set initial values for a binary vector x based on a chosen set.
    """
    if not HAS_PULP:
        return
    for j, var in enumerate(x_vars):
        try:
            var.setInitialValue(1 if j in chosen else 0)
        except Exception:
            # Some solver backends may not support this; ignore gracefully.
            pass


def _print_knob_suggestions():
    print("  Knobs to try if you want better/proven-optimal results:")
    print("   - Increase ilp_time_limit_s (more time).")
    print("   - Set SOLVER_VERBOSE=True to see CBC progress.")
    print("   - Increase depot_grid_step (fewer candidates) OR increase poisson_min_dist (fewer sample points).")
    print("   - Widen ring_outer / reduce ring_inner (more feasible candidates).")
    print("   - Keep ADAPTIVE_REFINE_ITERS small (it adds points/constraints).")


# =============================
# Greedy fallback (set cover)
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


# =============================
# Stage 1 preprocessing (Fix #2)
# =============================

@dataclass
class PreprocessResult:
    forced: Set[int]
    kept_candidates: List[int]  # indices in original candidate list (columns)
    reduced_points_idx: np.ndarray  # indices of points used for dominance checks (uncovered by forced)
    removed_candidates: int


def preprocess_stage1_min_depots(a: np.ndarray) -> PreprocessResult:
    """
    Safe preprocessing for min-depots (set cover):
      - forced depots: if a point has only one covering candidate.
      - dominated candidate elimination: if candidate j covers a subset of candidate k (on remaining points),
        then j can be removed (equal costs).
    """
    n_points, n_cand = a.shape

    # Forced depots
    cover_counts = np.sum(a, axis=1).astype(int)
    if np.any(cover_counts == 0):
        # infeasible; return as-is
        return PreprocessResult(forced=set(), kept_candidates=list(range(n_cand)),
                                reduced_points_idx=np.arange(n_points), removed_candidates=0)

    forced: Set[int] = set()
    singletons = np.where(cover_counts == 1)[0]
    for i in singletons:
        j = int(np.where(a[i])[0][0])
        forced.add(j)

    if forced:
        forced_cols = sorted(forced)
        covered_by_forced = np.any(a[:, forced_cols], axis=1)
        remaining_points = np.where(~covered_by_forced)[0]
    else:
        remaining_points = np.arange(n_points)

    # If all points already covered by forced, keep only forced candidates.
    if len(remaining_points) == 0:
        return PreprocessResult(forced=forced, kept_candidates=sorted(forced),
                                reduced_points_idx=np.array([], dtype=int), removed_candidates=n_cand - len(forced))

    # Dominance elimination on remaining points only
    a_rem = a[remaining_points, :]

    cov_sets: List[Optional[Set[int]]] = [None] * n_cand
    cov_sizes = np.zeros(n_cand, dtype=int)
    for j in range(n_cand):
        if j in forced:
            # keep forced
            s = set(np.where(a_rem[:, j])[0].tolist())
        else:
            s = set(np.where(a_rem[:, j])[0].tolist())
        cov_sets[j] = s
        cov_sizes[j] = len(s)

    # Candidate ordering: large cover first (potential dominators first)
    order = np.argsort(-cov_sizes)

    keep_mask = np.ones(n_cand, dtype=bool)

    # Point -> list of keeper candidates indices (in keeper list space)
    point_to_keeper_ids: Dict[int, List[int]] = defaultdict(list)
    keeper_cov: List[Set[int]] = []
    keeper_orig: List[int] = []

    for j in order:
        j = int(j)
        if j in forced:
            # forced always kept
            pass
        s = cov_sets[j] or set()
        if len(s) == 0 and (j not in forced):
            keep_mask[j] = False
            continue

        # Check if j is dominated by any current keeper (subset of a keeper)
        rep_point = next(iter(s)) if s else None
        dominated = False
        if rep_point is not None and point_to_keeper_ids.get(rep_point):
            for kid in point_to_keeper_ids[rep_point]:
                if s.issubset(keeper_cov[kid]):
                    dominated = True
                    break

        if dominated and (j not in forced):
            keep_mask[j] = False
            continue

        # Keep it
        kid_new = len(keeper_cov)
        keeper_cov.append(s)
        keeper_orig.append(j)
        if rep_point is not None:
            for p in s:
                point_to_keeper_ids[p].append(kid_new)

    kept_candidates = sorted(set(keeper_orig) | set(forced))
    removed = int(n_cand - len(kept_candidates))
    return PreprocessResult(forced=forced, kept_candidates=kept_candidates,
                            reduced_points_idx=remaining_points, removed_candidates=removed)


# =============================
# Stage 1 ILP (best-effort, warm start, robust fallback)
# =============================

@dataclass
class Stage1SolveMeta:
    method: str
    status: str
    proven_optimal: bool
    used_greedy_fallback: bool
    greedy_k: int
    chosen_k: int
    forced_k: int
    removed_candidates: int


def stage1_min_depots_best_effort(
    a: np.ndarray,
    time_limit_s: int,
    silent: bool = False,
) -> Tuple[List[int], Stage1SolveMeta]:
    """
    Always compute greedy. If PuLP is available, run ILP (with preprocessing + warm start).
    If ILP isn't proven optimal, accept a feasible incumbent if it's <= greedy; else use greedy.

    Returns chosen indices and meta.
    """
    n_points, n_cand = a.shape

    greedy_sol = greedy_min_depots(a)
    greedy_set = set(greedy_sol)

    if not HAS_PULP:
        meta = Stage1SolveMeta(
            method="GREEDY",
            status="NO_PULP",
            proven_optimal=False,
            used_greedy_fallback=True,
            greedy_k=len(greedy_sol),
            chosen_k=len(greedy_sol),
            forced_k=0,
            removed_candidates=0,
        )
        return greedy_sol, meta

    prep_res = preprocess_stage1_min_depots(a)
    forced = prep_res.forced
    kept = prep_res.kept_candidates

    # Build reduced ILP on kept candidates
    n_keep = len(kept)
    prob = pulp.LpProblem("Stage1_MinDepotsCover", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{t}", 0, 1, cat=pulp.LpBinary) for t in range(n_keep)]

    prob += pulp.lpSum(x)

    # Coverage constraints
    # For each point i: sum_{cand j covering i} x_j >= 1
    for i in range(n_points):
        cols = [idx for idx, orig_j in enumerate(kept) if a[i, orig_j]]
        if not cols:
            # should not happen if a is feasible
            continue
        prob += pulp.lpSum(x[idx] for idx in cols) >= 1

    # Force forced depots open
    forced_in_kept = {orig_j for orig_j in kept if orig_j in forced}
    for idx, orig_j in enumerate(kept):
        if orig_j in forced_in_kept:
            prob += x[idx] == 1

    # Warm start from greedy: convert greedy original indices to kept-index space
    warm = bool(USE_WARM_START)
    if warm:
        chosen_keep = {kept.index(j) for j in greedy_set if j in kept}
        _set_warm_start_values(x, chosen_keep)

    solver = _cbc_solver(time_limit_s=time_limit_s, warm_start=warm)
    # Run CBC inside a dedicated workspace folder so you can inspect the .mps/.sol files after the run.
    use_workdir = bool(CBC_USE_WORKDIR)

    with _maybe_cbc_workdir(use_workdir):
        status = prob.solve(solver)

    status_name = pulp.LpStatus.get(status, str(status))
    chosen_keep = [idx for idx in range(n_keep) if pulp.value(x[idx]) is not None and pulp.value(x[idx]) > 0.5]
    chosen = [kept[idx] for idx in chosen_keep]

    # Validate feasibility (coverage)
    feasible = (len(chosen) > 0) and np.all(np.any(a[:, chosen], axis=1))

    # Prefer ILP result if feasible and no worse than greedy
    if feasible and len(chosen) <= len(greedy_sol):
        proven = (status_name == "Optimal")
        method = "ILP_OPTIMAL" if proven else f"ILP_BEST_FOUND(time_limit={time_limit_s}s)"
        if (not silent) and (not proven):
            print(f"\n[Stage1] WARNING: min-depots ILP not proven optimal (status={status_name}).")
            print("         Using best feasible solution found within time limit.")
            _print_knob_suggestions()
        meta = Stage1SolveMeta(
            method=method,
            status=status_name,
            proven_optimal=proven,
            used_greedy_fallback=False,
            greedy_k=len(greedy_sol),
            chosen_k=len(chosen),
            forced_k=len(forced),
            removed_candidates=prep_res.removed_candidates,
        )
        return chosen, meta

    # Fallback to greedy
    if not silent:
        print(f"\n[Stage1] Using GREEDY fallback for min-depots.")
        if not feasible:
            print(f"        (ILP status={status_name}, or extracted incumbent was infeasible.)")
        else:
            print(f"        (ILP found feasible but used more depots than greedy: {len(chosen)} > {len(greedy_sol)}.)")
        _print_knob_suggestions()

    meta = Stage1SolveMeta(
        method="GREEDY_FALLBACK",
        status=status_name,
        proven_optimal=False,
        used_greedy_fallback=True,
        greedy_k=len(greedy_sol),
        chosen_k=len(greedy_sol),
        forced_k=len(forced),
        removed_candidates=prep_res.removed_candidates,
    )
    return greedy_sol, meta


def stage1_min_overlap_given_exact_k_best_effort(
    a: np.ndarray,
    k: int,
    time_limit_s: int,
    warm_start_solution: Optional[List[int]] = None,
) -> Tuple[List[int], str, bool]:
    """
    Best-effort ILP for min overlap redundancy with EXACT k depots.
    - If ILP not solved optimally, accept feasible incumbent if it has exactly k depots and covers all points.
    - If ILP fails, fall back to a min-depots solution at this R (which should have size k) if possible.
    """
    n_points, n_cand = a.shape

    if not HAS_PULP:
        # crude fallback: greedy cover (may not be exactly k if k is from ILP)
        sol = greedy_min_depots(a)
        return sol, "NO_PULP_GREEDY", False

    prob = pulp.LpProblem("Stage1_MinOverlapGivenK", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", 0, 1, cat=pulp.LpBinary) for j in range(n_cand)]
    z = [pulp.LpVariable(f"z_{i}", 0) for i in range(n_points)]

    prob += pulp.lpSum(z)
    prob += pulp.lpSum(x) == int(k)

    for i in range(n_points):
        cover_expr = pulp.lpSum(x[j] for j in range(n_cand) if a[i, j])
        prob += cover_expr >= 1
        prob += z[i] >= cover_expr - 1

    # Warm start from provided solution if given
    warm = bool(USE_WARM_START and warm_start_solution is not None)
    if warm:
        chosen_set = set(warm_start_solution or [])
        for j, var in enumerate(x):
            try:
                var.setInitialValue(1 if j in chosen_set else 0)
            except Exception:
                pass

    solver = _cbc_solver(time_limit_s=time_limit_s, warm_start=warm)

    with _maybe_cbc_workdir(bool(CBC_USE_WORKDIR)):
        status = prob.solve(solver)

    status_name = pulp.LpStatus.get(status, str(status))
    chosen = [j for j in range(n_cand) if pulp.value(x[j]) is not None and pulp.value(x[j]) > 0.5]

    feasible = (len(chosen) == k) and np.all(np.any(a[:, chosen], axis=1))
    proven = (status_name == "Optimal")

    if feasible:
        if not proven:
            print(f"\n[Stage1] WARNING: overlap ILP not proven optimal (status={status_name}).")
            print("         Using best feasible solution found within time limit.")
            _print_knob_suggestions()
        return chosen, ("ILP_OPTIMAL" if proven else f"ILP_OVERLAP_BEST_FOUND(time_limit={time_limit_s}s)"), proven

    # Fallback: min depots at this R (should usually return exactly k if k is correct)
    chosen2, meta2 = stage1_min_depots_best_effort(a, time_limit_s=time_limit_s, silent=True)
    if len(chosen2) == k and np.all(np.any(a[:, chosen2], axis=1)):
        return chosen2, f"FALLBACK_MIN_DEPOTS({meta2.method})", False

    # Last resort: greedy
    sol = greedy_min_depots(a)
    return sol, "FALLBACK_GREEDY", False


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

    If PuLP available: binary search (monotone for exact min-depots).
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
        chosen, _meta = stage1_min_depots_best_effort(a, ilp_time_limit_s, silent=True)
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
# Upgrade: MILP (time-limited) OR greedy-add
# =============================

@dataclass
class UpgradeResult:
    chosen_idx: List[int]
    objective_history: List[float]  # demand-weighted total assigned distance after steps
    added_order: List[int]


def stage2_milp_min_weighted_distance(
    a: np.ndarray,
    dist: np.ndarray,
    weights: np.ndarray,
    K_budget: int,
    time_limit_s: int = 120,
    msg: bool = False,
    eps_open: float = 0.0,
) -> Optional[List[int]]:
    """
    Time-limited MILP for Stage 2 (p-median / assignment style):
      min sum_i sum_{j in N(i)} w_i * d_ij * y_ij + eps_open * sum_j x_j
      s.t. sum_{j in N(i)} y_ij = 1                   for all i
           y_ij <= x_j                                for all i,j
           sum_j x_j <= K_budget
           x_j, y_ij binary

    Only creates y_ij for pairs where a[i,j]=1 (within R).
    Returns chosen depot indices if a feasible integer solution is obtained, else None.
    """
    if not HAS_PULP:
        return None

    n_points, n_cand = a.shape
    if weights.shape[0] != n_points:
        raise ValueError("weights must have length n_points.")

    neigh = [np.where(a[i])[0] for i in range(n_points)]
    if any(len(neigh[i]) == 0 for i in range(n_points)):
        return None

    prob = pulp.LpProblem("Stage2_MinDemandWeightedDistance", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", 0, 1, cat=pulp.LpBinary) for j in range(n_cand)]
    y: Dict[Tuple[int, int], "pulp.LpVariable"] = {}

    for i in range(n_points):
        for j in neigh[i]:
            y[(i, int(j))] = pulp.LpVariable(f"y_{i}_{int(j)}", 0, 1, cat=pulp.LpBinary)

    prob += (
        pulp.lpSum(float(weights[i]) * float(dist[i, j]) * y[(i, j)] for (i, j) in y)
        + float(eps_open) * pulp.lpSum(x)
    )

    for i in range(n_points):
        prob += pulp.lpSum(y[(i, int(j))] for j in neigh[i]) == 1

    for (i, j), var in y.items():
        prob += var <= x[j]

    prob += pulp.lpSum(x) <= int(K_budget)

    # Warm start from greedy? (optional; you can add if you want)
    warm = bool(USE_WARM_START)
    solver = _cbc_solver(time_limit_s=time_limit_s, warm_start=warm)

    with _maybe_cbc_workdir(bool(CBC_USE_WORKDIR)):
        status = prob.solve(solver)

    status_name = pulp.LpStatus.get(status, str(status))
    chosen = [j for j in range(n_cand) if pulp.value(x[j]) is not None and pulp.value(x[j]) > 0.5]

    if len(chosen) == 0 or len(chosen) > K_budget:
        return None

    best = assigned_distances(dist, a, chosen)
    if np.any(~np.isfinite(best)):
        return None

    # Accept feasible incumbent even if not "Optimal"
    return chosen


def upgrade_by_greedy_add(
    a: np.ndarray,
    dist: np.ndarray,
    weights: np.ndarray,
    start_idx: List[int],
    K_budget: int,
    min_improvement: float = 1e-6,
) -> UpgradeResult:
    """
    Add depots until:
      - you hit K_budget, OR
      - the best improvement is < min_improvement (so you don't waste depots)

    Objective being improved: sum_i weights[i] * assigned_distance[i]
    """
    n_points, n_cand = a.shape
    if weights.shape[0] != n_points:
        raise ValueError("weights must have length n_points.")

    chosen = list(start_idx)
    chosen_set = set(chosen)

    if len(chosen) > K_budget:
        raise ValueError("Start set already exceeds K_budget.")

    pts_by_depot = [np.where(a[:, j])[0] for j in range(n_cand)]

    sub_d = dist[:, chosen]
    sub_a = a[:, chosen]
    cur = np.min(np.where(sub_a, sub_d, np.inf), axis=1)
    if np.any(~np.isfinite(cur)):
        raise RuntimeError("Start set does not fully cover all points.")

    history = [float(np.sum(weights * cur))]
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
            improve = float(np.sum(weights[pts] * np.maximum(0.0, cur[pts] - new_d)))
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
        history.append(float(np.sum(weights * cur)))

        remaining.remove(best_j)

    return UpgradeResult(chosen_idx=chosen, objective_history=history, added_order=added_order)


# =============================
# Final BFGS refinement (once)
# =============================

def refine_with_bfgs_end(
    depots_xy: np.ndarray,
    interior_pts: np.ndarray,
    weights: np.ndarray,
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

    if weights.shape[0] != interior_pts.shape[0]:
        raise ValueError("weights must have length = number of interior points.")

    allowed_prep = prep(allowed_region)
    n = len(depots_xy)
    x0 = depots_xy.reshape(-1)

    def objective(xvec: np.ndarray) -> float:
        centers = xvec.reshape((n, 2))
        pen = 0.0

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
            pen += float(dist_weight) * float(np.sum(weights * min_d))

        return pen

    res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": maxiter})
    refined = res.x.reshape((n, 2))

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


def plot_distance_upgrade_curve(
    objective_history: List[float],
    title: str = "Upgrade: demand-weighted distance improvement",
    xaxis_title: str = "Step",
) -> go.Figure:
    xs = list(range(len(objective_history)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=objective_history,
        mode="lines+markers",
        name="demand-weighted total distance",
        hovertemplate="step=%{x}<br>demand_weighted_total=%{y}<extra></extra>"
    ))
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Demand-weighted total assigned distance"
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
    demand_hotspots: Optional[List[Tuple[float, float, float]]] = None,
    weights: Optional[np.ndarray] = None,
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

    # Samples: colour by demand weight (green = low, red = high) if weights provided
    if weights is not None and len(weights) == len(interior_pts):
        fig.add_trace(go.Scatter(
            x=interior_pts[:, 0], y=interior_pts[:, 1],
            mode="markers", name="Samples",
            marker=dict(
                size=6,
                opacity=0.75,
                color=weights,
                colorscale=[[0, "green"], [0.5, "yellow"], [1, "red"]],
                showscale=True,
                colorbar=dict(title="Demand"),
            ),
            hovertemplate="x=%{x:.1f}<br>y=%{y:.1f}<br>demand=%{marker.color:.3f}<extra></extra>",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=interior_pts[:, 0], y=interior_pts[:, 1],
            mode="markers", name="Samples",
            marker=dict(size=5, opacity=0.6)
        ))

    if demand_hotspots:
        hx = [h[0] for h in demand_hotspots]
        hy = [h[1] for h in demand_hotspots]
        hp = [h[2] for h in demand_hotspots]
        fig.add_trace(go.Scatter(
            x=hx, y=hy,
            mode="markers+text",
            name="Demand hotspots",
            text=[f"{p:g}" for p in hp],
            textposition="top center",
            hovertemplate="Hotspot<br>x=%{x}<br>y=%{y}<br>peak=%{text}<extra></extra>",
            marker=dict(size=16, symbol="star", color="crimson", line=dict(width=2, color="black"))
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
# Adaptive refinement (Fix #1)
# =============================

def adaptive_refine_samples(
    city: Polygon,
    pts: np.ndarray,
    candidate_depots: np.ndarray,
    R: float,
    refine_iters: int,
    grid_step: float,
    max_new: int,
    ilp_time_limit_s: int,
) -> np.ndarray:
    """
    Iteratively:
      - solve Stage 1 min-depots at radius R on current pts
      - validate on a grid
      - add uncovered grid points
    """
    if refine_iters <= 0:
        return pts

    poly_prep = prep(city)
    pts = pts.copy()

    for it in range(refine_iters):
        dist2 = build_dist2_matrix(pts, candidate_depots)
        a = mask_for_R(dist2, float(R))
        if len(uncovered_points_report(a)) > 0:
            # Can't even cover sampled points; refinement won't help
            return pts

        chosen, meta = stage1_min_depots_best_effort(a, ilp_time_limit_s, silent=True)
        chosen_xy = candidate_depots[chosen]

        grid = grid_points_in_polygon(city, step=grid_step, max_points=25000, seed=123 + it)
        unc_idx = uncovered_by_depots(grid, chosen_xy, R=float(R))
        if len(unc_idx) == 0:
            if it == 0:
                print(f"\n[Refine] Grid validation @ R={R:.1f}: no uncovered grid points found.")
            else:
                print(f"[Refine] Iter {it+1}: no uncovered grid points found.")
            return pts

        # Add up to max_new uncovered points (prefer farthest)
        unc_pts = grid[unc_idx]
        dx = unc_pts[:, None, 0] - chosen_xy[None, :, 0]
        dy = unc_pts[:, None, 1] - chosen_xy[None, :, 1]
        d2 = dx * dx + dy * dy
        min_d = np.sqrt(np.min(d2, axis=1))
        slack = min_d - float(R)
        order = np.argsort(-slack)  # worst uncovered first

        add_pts = unc_pts[order[: min(max_new, len(unc_pts))]]
        pts = dedupe_points(np.vstack([pts, add_pts]), tol=1e-6)

        print(f"\n[Refine] Iter {it+1}: found {len(unc_idx)} uncovered grid points; "
              f"added {len(add_pts)} new sample points (total now {len(pts)}).")

    return pts


# =============================
# Load city polygon + demand from integrating_step1 output
# =============================

DEFAULT_STEP1_JSON = Path(__file__).resolve().parent / "integrating_step1_output.json"
DEFAULT_SOLUTION_OVERLAY_JSON = Path(__file__).resolve().parent / "depot_solution_for_overlay.json"


def save_depot_solution_for_overlay(
    interior_pts: np.ndarray,
    weights: np.ndarray,
    chosen_depots: np.ndarray,
    city_coords: List[Tuple[float, float]],
    crs: str = "EPSG:27700",
    R: Optional[float] = None,
    path: Optional[Path] = None,
) -> None:
    """Save solution data for london_3d_overlay (depots, Poisson points, demand weights, coverage radius R)."""
    path = path or DEFAULT_SOLUTION_OVERLAY_JSON
    out = {
        "interior_pts": interior_pts.tolist(),
        "weights": weights.tolist(),
        "chosen_depots": chosen_depots.tolist(),
        "city_polygon_xy": [list(p) for p in city_coords],
        "crs": crs,
    }
    if R is not None:
        out["R"] = float(R)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved solution for overlay: {path}")


def load_city_and_demand_from_step1_json(
    path: Optional[Path] = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float]], float, Optional[dict], str]:
    """
    Load city polygon, demand hotspots, radius, demand density map, and crs from integrating_step1 JSON.
    Returns (city_coords, demand_hotspots, radius_m, demand_density_map, crs). demand_density_map may be None.
    """
    path = path or DEFAULT_STEP1_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"Step1 output not found at {path}. Run integrating_step1.py first."
        )
    with open(path) as f:
        import json
        data = json.load(f)
    city_coords = [tuple(p) for p in data["city_polygon_xy"]]
    demand_hotspots = [tuple(h) for h in data.get("demand_hotspots", [])]
    radius_m = float(data["radius_m"])
    demand_density_map = data.get("demand_density_map")
    crs = data.get("crs", "EPSG:27700")
    return city_coords, demand_hotspots, radius_m, demand_density_map, crs


def get_city_and_demand_from_circle(
    center_latlon: Tuple[float, float],
    radius_m: float,
    step1_json_path: Optional[Path] = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float]], float, Optional[dict]]:
    """
    Build city polygon from centre (lat, lon) WGS84 and radius (m), and load demand
    data from integrating_step1 output JSON.
    Returns (city_coords, demand_hotspots, radius_m, demand_density_map).
    """
    try:
        from integrating_step1 import city_circle_polygon
    except ImportError as e:
        raise ImportError(
            "Circle mode requires integrating_step1. Run from the same project or install it."
        ) from e

    city_polygon, _cx, _cy, _epsg = city_circle_polygon(center_latlon, radius_m)
    city_coords = list(city_polygon.exterior.coords)

    path = step1_json_path or DEFAULT_STEP1_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"Demand data not found at {path}. Run integrating_step1.py first to generate it."
        )
    _, demand_hotspots, radius_m, demand_density_map, _ = load_city_and_demand_from_step1_json(path)
    return city_coords, demand_hotspots, radius_m, demand_density_map


# =============================
# Main
# =============================

if __name__ == "__main__":
    import sys

    print("RUNNING: depot_model_demand.py")
    print(f"PuLP available: {HAS_PULP} | SciPy available: {HAS_SCIPY}")
    print(f"Warm start enabled: {USE_WARM_START} | CBC workdir enabled: {CBC_USE_WORKDIR} | CBC workdir: {str((Path.cwd()/CBC_WORKDIR_NAME).resolve())}")

    # Prepare CBC workdir (so you can inspect solver files after the run)
    if CBC_USE_WORKDIR:
        wd = Path.cwd() / CBC_WORKDIR_NAME
        wd.mkdir(parents=True, exist_ok=True)
        if CBC_CLEAN_WORKDIR_ON_RUN:
            _clean_cbc_workdir(wd)
        # Note: folder is intentionally kept after the run when CBC_KEEP_WORKDIR=True.
        print(f"[CBC] Solver files will be written in: {wd.resolve()}")

    # --- City polygon + demand: from integrating_step1 output (default) or CLI override ---
    # Default: load from integrating_step1_output.json (run integrating_step1.py first).
    # Optional CLI: python depot_model_demand.py <lat> <lon> <radius_m> [path_to_json]
    radius_m = 300.0  # used for ring when loading from step1
    use_step1_data = False
    demand_density_map = None  # set when loading from step1
    step1_crs = "EPSG:27700"   # for saving solution overlay; updated when loading from step1 JSON
    if len(sys.argv) >= 4:
        try:
            lat = float(sys.argv[1])
            lon = float(sys.argv[2])
            radius_m = float(sys.argv[3])
            step1_path = Path(sys.argv[4]) if len(sys.argv) >= 5 else None
            city_coords, demand_hotspots, radius_m, demand_density_map = get_city_and_demand_from_circle(
                (lat, lon), radius_m, step1_json_path=step1_path
            )
            use_step1_data = True
            print(f"\n[Step1] centre=({lat}, {lon}), radius={radius_m} m, demand from step1")
        except (ValueError, FileNotFoundError, ImportError) as e:
            print(f"\nStep1 CLI failed: {e}")
            print("Trying default step1 JSON next.")
    if not use_step1_data:
        try:
            step1_path = DEFAULT_STEP1_JSON
            city_coords, demand_hotspots, radius_m, demand_density_map, step1_crs = load_city_and_demand_from_step1_json(step1_path)
            use_step1_data = True
            print(f"\n[Step1] Loaded city + demand from {step1_path.name} (radius={radius_m} m)")
        except FileNotFoundError:
            print(f"\n[Step1] No {DEFAULT_STEP1_JSON.name} found. Run integrating_step1.py first.")
            print("Using built-in example polygon and demand for this run.")
            city_coords = [
                (0, 0), (900, 0), (900, 150),
                (650, 150), (650, 350), (1100, 350),
                (1100, 650), (800, 650), (800, 900),
                (400, 900), (400, 600), (150, 600),
                (150, 250), (0, 250)
            ]
            demand_hotspots = [(500, 450, 3.0), (100, 100, 2.0)]
            radius_m = 300.0
            demand_density_map = None

    # --- Battery model bounds ---
    R_min_user = 400
    R_max_user = 550

    # --- Depot placement ring: inner at chosen radius (city edge), outer at radius + 30 m ---
    ring_inner = 0.0   # inner ring boundary = city boundary (chosen radius)
    ring_outer = 30.0  # outer ring boundary = ring_inner + 30 m
    poisson_min_dist = 50
    depot_grid_step = 30
    R_step = 10

    # Stage 1 ILP (min depots, baseline): each call can take up to this many seconds.
    ilp_time_limit_s = 60
    # Stage 2 MILP (upgrade): minimises demand-weighted distance with K_budget depots. Very slow for large K.
    stage2_milp_time_limit_s = 60

    # --- Build geometry + candidates ---
    city = make_polygon(city_coords)
    candidate_depots, allowed_ring = generate_candidate_depots_in_ring(
        city, ring_inner=ring_inner, ring_outer=ring_outer, grid_step=depot_grid_step
    )

    # --- Sampling (Fix #1) ---
    poisson_pts = sample_points_poisson_in_polygon(city, min_dist=poisson_min_dist, seed=42)
    poisson_count = len(poisson_pts)
    interior_pts = poisson_pts

    if USE_BOUNDARY_SAMPLES:
        boundary_step = max(1e-6, BOUNDARY_SPACING_FACTOR * float(poisson_min_dist))
        boundary_pts = sample_points_on_boundary(city, step=boundary_step)
        pts = np.vstack([interior_pts, boundary_pts])

        if ADD_POLYGON_VERTICES:
            verts = np.array(list(city.exterior.coords), dtype=float)
            pts = np.vstack([pts, verts])

        interior_pts = dedupe_points(pts, tol=1e-6)
        print(f"\nSampling: poisson={poisson_count}, after boundary+verts={len(interior_pts)}")
    else:
        interior_pts = dedupe_points(interior_pts, tol=1e-6)
        print(f"\nSampling: poisson only = {len(interior_pts)}")

    # --- Optional adaptive refinement (Fix #1) ---
    if ADAPTIVE_REFINE_ITERS > 0:
        refine_step = max(1e-6, REFINE_GRID_STEP_FACTOR * float(poisson_min_dist))
        interior_pts = adaptive_refine_samples(
            city=city,
            pts=interior_pts,
            candidate_depots=candidate_depots,
            R=float(R_max_user),
            refine_iters=int(ADAPTIVE_REFINE_ITERS),
            grid_step=float(refine_step),
            max_new=int(REFINE_MAX_NEW_POINTS_PER_ITER),
            ilp_time_limit_s=int(ilp_time_limit_s),
        )

    dist2 = build_dist2_matrix(interior_pts, candidate_depots)
    dist = np.sqrt(dist2)

    # =============================
    # Demand weights: from density map (step1) or from point hotspots (fallback)
    # =============================
    demand_floor = 1.0
    if demand_density_map:
        weights = compute_weights_from_density_map(
            interior_pts=interior_pts,
            demand_density_map=demand_density_map,
            floor=demand_floor,
            normalise_mean_to_1=True,
        )
    else:
        demand_sigma = 220.0
        weights = compute_demand_weights(
            interior_pts=interior_pts,
            hotspots=demand_hotspots,
            sigma=demand_sigma,
            floor=demand_floor,
            normalise_mean_to_1=True,
        )

    # --- "Erode R" gate ---
    R_required = required_R_for_coverability(dist2)
    start_R = max(float(R_min_user), float(R_required))

    if start_R > float(R_max_user) + 1e-9:
        print("\nNot possible with current ring/candidates.")
        print(f"R_required (from candidates) = {R_required:.3f}")
        print(f"User R_max                  = {R_max_user:.3f}")
        print("Try: widen ring_outer, reduce ring_inner, decrease depot_grid_step, or increase R_max.")
        raise SystemExit(1)

    start_R = float(math.ceil(start_R / R_step) * R_step)

    Rs = np.arange(start_R, float(R_max_user) + 1e-9, float(R_step), dtype=float)
    if len(Rs) == 0 or Rs[-1] < float(R_max_user) - 1e-9:
        Rs = np.append(Rs, float(R_max_user))

    # --- Baseline at R_max: find k* ---
    R_for_kstar = float(R_max_user)
    a_max = mask_for_R(dist2, R_for_kstar)
    if len(uncovered_points_report(a_max)) > 0:
        print("\nEven R_max cannot cover all sample points with these candidates.")
        print("Try: widen ring_outer, reduce ring_inner, decrease depot_grid_step.")
        raise SystemExit(1)

    chosen_min_at_Rmax, meta_k = stage1_min_depots_best_effort(a_max, time_limit_s=ilp_time_limit_s, silent=False)
    k_star = len(chosen_min_at_Rmax)

    if meta_k.removed_candidates > 0:
        print(f"\n[Stage1] Preprocessing removed {meta_k.removed_candidates} dominated/empty candidates "
              f"(kept {len(candidate_depots) - meta_k.removed_candidates}/{len(candidate_depots)}).")
    print(f"[Stage1] k* @ R_max = {k_star}  | method={meta_k.method} | status={meta_k.status}")

    # --- Find smallest R that still allows k* depots ---
    R_baseline = find_smallest_R_with_k(Rs, dist2, k_star, ilp_time_limit_s)

    # --- Choose best space-efficient layout with EXACTLY k* depots at that smallest R ---
    a_base = mask_for_R(dist2, float(R_baseline))
    chosen_baseline_idx, overlap_method, overlap_optimal = stage1_min_overlap_given_exact_k_best_effort(
        a=a_base,
        k=k_star,
        time_limit_s=ilp_time_limit_s,
        warm_start_solution=chosen_min_at_Rmax
    )

    baseline_red = compute_redundancy(a_base, chosen_baseline_idx)
    baseline_total_dist = compute_total_assigned_distance(dist, a_base, chosen_baseline_idx)  # unweighted

    base_best = assigned_distances(dist, a_base, chosen_baseline_idx)
    base_wmean = weighted_mean(base_best, weights)
    base_max = float(np.max(base_best))

    print("\n=== BASELINE (minimum depots, then smallest R that keeps it) ===")
    print(f"R_required (candidates)                : {R_required:.3f}")
    print(f"Start_R used                           : {start_R:.3f}")
    print(f"k* (min depots @ R_max)                : {k_star}")
    print(f"Chosen baseline R                      : {R_baseline:.3f}")
    print(f"Baseline selection method (overlap)    : {overlap_method}")
    print(f"Baseline overlap redundancy            : {baseline_red}")
    print(f"Baseline total distance (unweighted)   : {baseline_total_dist:.3f}")
    print(f"Baseline demand-weighted mean distance : {base_wmean:.3f}")
    print(f"Baseline max assigned distance (safety): {base_max:.3f}")

    # (Optional) curve of min depot count vs R — use shorter limit (plot only, no need for proven optimal)
    k_curve_time_limit_s = min(30, ilp_time_limit_s)
    k_vals = []
    for R in Rs:
        aR = mask_for_R(dist2, float(R))
        if len(uncovered_points_report(aR)) > 0:
            k_vals.append(np.nan)
        else:
            ch, _m = stage1_min_depots_best_effort(aR, k_curve_time_limit_s, silent=True)
            k_vals.append(len(ch))
    k_vals = np.array(k_vals, dtype=float)

    if SHOW_PLOTS:
        fig_k = plot_k_curve(Rs, k_vals, float(R_baseline))
        fig_base_map = plot_solution_map(
            city=city,
            allowed_ring=allowed_ring,
            interior_pts=interior_pts,
            chosen_depots=candidate_depots[chosen_baseline_idx],
            R=float(R_baseline),
            title=f"Baseline | R={R_baseline:.1f} | depots={k_star}",
            show_circles=True,
            max_circles=60,
            demand_hotspots=None if demand_density_map else demand_hotspots,
            weights=weights,
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

    # --- Upgrade run at R_max_user ---
    R_upgrade = float(R_max_user)
    a_up = mask_for_R(dist2, R_upgrade)

    # Start from a "nice" k* layout at R_max_user
    start_idx_upgrade, _m2 = stage1_min_depots_best_effort(a_up, ilp_time_limit_s, silent=True)

    # Always compute greedy-add
    up_greedy = upgrade_by_greedy_add(
        a=a_up,
        dist=dist,
        weights=weights,
        start_idx=start_idx_upgrade,
        K_budget=K_budget,
        min_improvement=1e-6
    )
    greedy_weighted_total = compute_total_assigned_distance(dist, a_up, up_greedy.chosen_idx, weights=weights)

    upgrade_method = "GREEDY_ADD"
    up = up_greedy

    if HAS_PULP:
        eps_open = 1e-6 * float(np.mean(dist[a_up])) if np.any(a_up) else 0.0

        chosen_milp = stage2_milp_min_weighted_distance(
            a=a_up,
            dist=dist,
            weights=weights,
            K_budget=K_budget,
            time_limit_s=stage2_milp_time_limit_s,
            msg=False,
            eps_open=eps_open
        )

        if chosen_milp is not None:
            milp_weighted_total = compute_total_assigned_distance(dist, a_up, chosen_milp, weights=weights)

            if milp_weighted_total + 1e-9 < greedy_weighted_total:
                upgrade_method = f"MILP(time_limit={stage2_milp_time_limit_s}s)"
                start_obj = compute_total_assigned_distance(dist, a_up, start_idx_upgrade, weights=weights)
                added = [j for j in chosen_milp if j not in set(start_idx_upgrade)]
                up = UpgradeResult(
                    chosen_idx=chosen_milp,
                    objective_history=[float(start_obj), float(milp_weighted_total)],
                    added_order=added
                )
            else:
                upgrade_method = f"GREEDY_ADD (MILP tried {stage2_milp_time_limit_s}s but not better)"

    used_k = len(up.chosen_idx)
    up_red = compute_redundancy(a_up, up.chosen_idx)

    up_total_dist_unweighted = compute_total_assigned_distance(dist, a_up, up.chosen_idx)
    up_total_dist_weighted = compute_total_assigned_distance(dist, a_up, up.chosen_idx, weights=weights)

    up_best = assigned_distances(dist, a_up, up.chosen_idx)
    up_wmean = weighted_mean(up_best, weights)
    up_max = float(np.max(up_best))

    print("\n=== UPGRADE (distance optimisation under depot cap, demand-aware) ===")
    print(f"Upgrade method                           : {upgrade_method}")
    print(f"Upgrade R used                           : {R_upgrade:.3f}")
    print(f"Depots used (<= cap)                     : {used_k} / {K_budget}")
    print(f"Extra depots actually used               : {used_k - k_star}")
    print(f"Upgraded overlap redundancy              : {up_red}")
    print(f"Upgraded total distance (unweighted)     : {up_total_dist_unweighted:.3f}")
    print(f"Upgraded total distance (demand-weighted): {up_total_dist_weighted:.3f}")
    print(f"Upgraded demand-weighted mean distance   : {up_wmean:.3f}")
    print(f"Upgraded max assigned distance (safety)  : {up_max:.3f}")
    print(f"Unweighted distance reduction vs baseline: {baseline_total_dist - up_total_dist_unweighted:.3f}")

    if SHOW_PLOTS:
        if upgrade_method.startswith("MILP"):
            fig_up_curve = plot_distance_upgrade_curve(
                up.objective_history,
                title="Upgrade (MILP): start vs final demand-weighted total",
                xaxis_title="0 = start(k*), 1 = final(MILP)"
            )
        else:
            fig_up_curve = plot_distance_upgrade_curve(
                up.objective_history,
                title="Upgrade (Greedy-add): demand-weighted distance improvement as depots are added",
                xaxis_title="Extra depots actually added"
            )

        fig_up_map = plot_solution_map(
            city=city,
            allowed_ring=allowed_ring,
            interior_pts=interior_pts,
            chosen_depots=candidate_depots[up.chosen_idx],
            R=R_upgrade,
            title=f"Upgraded | R={R_upgrade:.1f} | depots_used={used_k}/{K_budget}",
            show_circles=True,
            max_circles=60,
            demand_hotspots=None if demand_density_map else demand_hotspots,
            weights=weights,
        )
        fig_up_curve.show()
        fig_up_map.show()

    # --- Final BFGS polish (optional) ---
    do_bfgs = True
    if do_bfgs and HAS_SCIPY:
        depots_xy = candidate_depots[up.chosen_idx]
        refined = refine_with_bfgs_end(
            depots_xy=depots_xy,
            interior_pts=interior_pts,
            weights=weights,
            R=R_upgrade,
            allowed_region=allowed_ring,
            edge_band_fraction=0.15,
            dist_weight=0.0,
            maxiter=250
        )
        if refined is not None and SHOW_PLOTS:
            fig_ref = plot_solution_map(
                city=city,
                allowed_ring=allowed_ring,
                interior_pts=interior_pts,
                chosen_depots=refined,
                R=R_upgrade,
                title=f"Final (after BFGS polish) | R={R_upgrade:.1f} | depots_used={used_k}/{K_budget}",
                show_circles=True,
                max_circles=60,
                demand_hotspots=None if demand_density_map else demand_hotspots,
                weights=weights,
            )
            fig_ref.show()
        print("\nBFGS polish applied." if refined is not None else "\nBFGS polish skipped.")
        final_depots = refined if refined is not None else candidate_depots[up.chosen_idx]
    else:
        print("\nBFGS polish not run (scipy not installed or disabled).")
        final_depots = candidate_depots[up.chosen_idx]

    if SAVE_SOLUTION_FOR_OVERLAY:
        save_depot_solution_for_overlay(
            interior_pts=interior_pts,
            weights=weights,
            chosen_depots=final_depots,
            city_coords=city_coords,
            crs=step1_crs,
            R=R_upgrade,
        )
