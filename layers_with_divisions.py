"""
layers_with_divisions.py

Computes the optimal number of airspace layers and cell-size divisions for a
multi-layer drone-delivery network.

Two methods are provided:

1.  compute_layer_plan()          -- ORIGINAL: picks divisions to minimise
    geometric (log) error between the achieved product and F = R_final / S0.

2.  optimize_layer_plan()         -- NEW: picks divisions to minimise the
    expected total 3-D travel cost per delivery (vertical transit + horizontal
    grid-alignment corrections at every layer transition).

Both methods share the same LayerPlan dataclass so the rest of the codebase
(dash_scene_builder, dash_drone_sim, …) is unchanged.

---------------------------------------------------------------------------
Cost model used by optimize_layer_plan()
---------------------------------------------------------------------------

When a drone delivers a package it must:

  a)  START at a depot on the top layer
  b)  CRUISE horizontally at the top layer towards the delivery zone
  c)  DESCEND through layers 1..k, at each layer correcting its horizontal
      position to align with the finer grid below
  d)  DELIVER at layer 0
  e)  REVERSE the whole journey to return to the depot

For step (c), when the drone descends from layer i (cell size S_i) to layer
i-1 (cell size S_{i-1}), it is aligned to the coarser grid of layer i.  On
the finer grid of layer i-1 it is mis-aligned by up to S_i in each axis.
The expected Manhattan correction is  S_i * correction_factor  where
correction_factor ≈ 0.5 for a uniform random offset within a cell.

The total expected cost for ONE delivery (round-trip) is therefore:

    C  =  2 * k * H                              (vertical transit)
        + D_cruise                                (top-layer horizontal cruise)
        + 2 * correction_factor * Σ S_i   (i=1..k, layer corrections)

H            = LAYER_SPACING_M (default 100 m)
D_cruise     = average horizontal distance at the top layer — roughly constant
               for a given city and depot layout, so it drops out of the
               comparison between candidates.  We include it for completeness.
correction_factor = tunable; default 0.5 (half-cell expected mis-alignment)

The optimiser evaluates all valid (k, m) combinations (same search space as
compute_layer_plan) and picks the one with the lowest cost C.

---------------------------------------------------------------------------
Inputs
---------------------------------------------------------------------------
  R_final        : top-layer cell size (from depot model)
  S0             : bottom-layer cell size (layer 0, ≈ 1.5 × max H&B dimension)
  layer_spacing_m: vertical distance between layers (default 100 m)
  city_radius_m  : radius of the service area (for cruise distance estimate)
  correction_factor: expected fraction of cell size for lateral correction
                     (default 0.5; 0.5 = half-cell average offset)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Data class (unchanged — backward compatible)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerPlan:
    k: int                          # number of divisions between layer 0 and top
    divisions: List[int]            # length-k list, each 2 or 3
    product: float                  # product(divisions)
    target_F: float                 # R_final / S0
    achieved_R: float               # S0 * product
    log_error: float                # |ln(product / target_F)|
    rel_error: float                # (achieved_R - R_final) / R_final
    layer_sizes: List[float]        # [S0, S1, ..., Sk] where Sk ≈ R_final
    total_cost: Optional[float] = None   # expected round-trip cost (m) — filled by optimize_layer_plan
    vertical_cost: Optional[float] = None
    correction_cost: Optional[float] = None
    cruise_cost: Optional[float] = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _valid_k_range(F: float) -> Tuple[int, int]:
    """
    Valid k satisfy: 2^k <= F <= 3^k  (equivalent to 2 <= F^(1/k) <= 3).
    Returns (k_min, k_max). If none, returns (1, 0).
    """
    if F <= 0:
        return (1, 0)

    # k_min = ceil(log(F)/log(3)), k_max = floor(log(F)/log(2))
    k_min = int(math.ceil(math.log(F) / math.log(3.0)))
    k_max = int(math.floor(math.log(F) / math.log(2.0)))
    return (k_min, k_max)


def _best_product_for_k(F: float, k: int) -> Tuple[float, int, float]:
    """
    For fixed k, choose m (# of 3s) to make P=2^(k-m)*3^m closest to F.
    Returns (P, m_best, log_error).
    """
    best_P = None
    best_m = None
    best_err = float("inf")

    for m in range(k + 1):
        P = (2.0 ** (k - m)) * (3.0 ** m)
        err = abs(math.log(P / F))
        if err < best_err:
            best_err = err
            best_P = P
            best_m = m

    return float(best_P), int(best_m), float(best_err)


def _interleave_divisions(k: int, m_threes: int, F: float) -> List[int]:
    """
    Construct an order for the divisions (2s and 3s) that keeps intermediate
    products close to the geometric target g^t, where g = F^(1/k).

    This doesn't change the final product, but gives a "smooth" progression.
    """
    twos_left = k - m_threes
    threes_left = m_threes
    g = F ** (1.0 / k)

    divisions: List[int] = []
    cur_prod = 1.0

    for t in range(1, k + 1):
        target = g ** t

        # Try placing a 2 or 3 (if available) and see which keeps us closer to target
        candidates = []
        if twos_left > 0:
            candidates.append(2)
        if threes_left > 0:
            candidates.append(3)

        def score(factor: int) -> float:
            return abs(math.log((cur_prod * factor) / target))

        factor = min(candidates, key=score)
        divisions.append(factor)
        cur_prod *= factor

        if factor == 2:
            twos_left -= 1
        else:
            threes_left -= 1

    return divisions


def _build_layer_sizes(S0: float, divisions: List[int]) -> List[float]:
    """Return [S0, S1, ..., Sk] from the division sequence."""
    sizes = [S0]
    cur = S0
    for d in divisions:
        cur *= d
        sizes.append(cur)
    return sizes


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def _trip_cost(
    layer_sizes: List[float],
    k: int,
    layer_spacing_m: float,
    city_radius_m: float,
    correction_factor: float,
) -> Tuple[float, float, float, float]:
    """
    Estimate expected round-trip cost for a single delivery.

    Returns (total, vertical, correction, cruise).

    Vertical cost:
        The drone climbs through k layer transitions going up and k coming back
        down.  Each transition is layer_spacing_m metres.
        V = 2 * k * layer_spacing_m

    Cruise cost:
        Average horizontal distance at the top layer.  For a depot on the
        perimeter serving a circular city, the expected straight-line distance
        to a uniformly random interior point is  2/3 * R  (exact integral).
        Round trip: D_cruise = 2 * (2/3) * city_radius_m.

    Correction cost:
        At each layer transition the drone must re-align from the coarser grid
        to the finer grid.  Expected offset ≈ correction_factor * cell_size.
        This happens at each of the k descent transitions, and again on the k
        ascent transitions.
        Correction = 2 * correction_factor * sum(layer_sizes[1:])
            (layer_sizes[1:] are the upper-layer cell sizes; the correction
             needed when transitioning from layer i down to i-1 is proportional
             to S_i, the coarser cell.)
    """
    vertical = 2.0 * k * layer_spacing_m
    cruise = 2.0 * (2.0 / 3.0) * city_radius_m
    correction = 2.0 * correction_factor * sum(layer_sizes[1:])  # S1..Sk
    total = vertical + cruise + correction
    return total, vertical, correction, cruise


# ---------------------------------------------------------------------------
# Method 1: Original — minimise geometric error  (unchanged logic)
# ---------------------------------------------------------------------------

def compute_layer_plan(R_final: float, S0: float, verbose: bool = True) -> Optional[LayerPlan]:
    """Original layer-plan algorithm.  Picks the (k, m) that minimises
    |ln(product / F)|.  Preserved for backward compatibility."""

    if R_final <= 0 or S0 <= 0:
        raise ValueError("R_final and S0 must be positive numbers.")

    F = R_final / S0

    # Trivial case: already at or below S0
    if F <= 1.0 + 1e-12:
        if verbose:
            print("R_final <= S0: no scaling needed (k=0).")
        return LayerPlan(
            k=0,
            divisions=[],
            product=1.0,
            target_F=F,
            achieved_R=S0,
            log_error=abs(math.log(1.0 / F)) if F > 0 else float("inf"),
            rel_error=(S0 - R_final) / R_final,
            layer_sizes=[S0],
        )

    k_min, k_max = _valid_k_range(F)

    if k_min > k_max:
        if verbose:
            print("\nNo valid k satisfies 2.0 <= F^(1/k) <= 3.0.")
            print(f"F = R_final/S0 = {F:.6g}")
            print("Fix: increase R_final, decrease S0, or relax the per-layer factor bounds.")
        return None

    candidates: List[LayerPlan] = []

    for k in range(k_min, k_max + 1):
        P, m, log_err = _best_product_for_k(F, k)
        divisions = _interleave_divisions(k, m_threes=m, F=F)

        achieved_R = S0 * P
        rel_error = (achieved_R - R_final) / R_final

        sizes = _build_layer_sizes(S0, divisions)

        candidates.append(LayerPlan(
            k=k,
            divisions=divisions,
            product=P,
            target_F=F,
            achieved_R=achieved_R,
            log_error=log_err,
            rel_error=rel_error,
            layer_sizes=sizes
        ))

    # Choose best overall candidate by smallest log_error; tie-break by smaller k
    candidates.sort(key=lambda c: (c.log_error, c.k))
    best = candidates[0]

    if verbose:
        print("\n=== Layer scaling plan (geometric-error method) ===")
        print(f"R_final (top cell size)  = {R_final:.6g}")
        print(f"S0 (layer-0 cell size)   = {S0:.6g}")
        print(f"F = R_final/S0           = {F:.6g}")

        print("\nValid k range from 2^k <= F <= 3^k:")
        print(f"k_min = {k_min}, k_max = {k_max}")
        print("(k = number of divisions; total layers = k+1)\n")

        print("Candidates (sorted by closeness):")
        for c in candidates:
            achieved_F = c.product
            ratio = achieved_F / c.target_F
            print(
                f"  k={c.k:2d}  product={achieved_F:8.4g}  "
                f"achieved_R={c.achieved_R:8.4g}  "
                f"log_err={c.log_error:7.4f}  "
                f"rel_err={100*c.rel_error:7.3f}%  "
                f"(ratio={ratio:6.3f}x)"
            )

        print("\n--- CHOSEN ---")
        print(f"k (divisions)          : {best.k}")
        print(f"total layers           : {best.k + 1}")
        print(f"divisions (2/3 list)   : {best.divisions}")
        print(f"product(divisions)     : {best.product:.6g} (target F={best.target_F:.6g})")
        print(f"achieved top size      : {best.achieved_R:.6g} (target R_final={R_final:.6g})")
        print(f"relative error         : {100*best.rel_error:.3f}%")
        print("\nLayer cell sizes:")
        for i, s in enumerate(best.layer_sizes):
            print(f"  Layer {i}: {s:.6g}")

    return best


# ---------------------------------------------------------------------------
# Method 2: NEW — minimise expected round-trip drone travel cost
# ---------------------------------------------------------------------------

def _all_m_candidates_for_k(F: float, k: int) -> List[Tuple[int, float, float]]:
    """
    For a given k, return ALL valid m choices (not just the best product match).
    Each entry is (m, product, log_error).
    This lets optimize_layer_plan evaluate every possible division mix by cost,
    not just the one closest to F.
    """
    results = []
    for m in range(k + 1):
        P = (2.0 ** (k - m)) * (3.0 ** m)
        err = abs(math.log(P / F))
        results.append((m, P, err))
    return results


def optimize_layer_plan(
    R_final: float,
    S0: float,
    layer_spacing_m: float = 100.0,
    city_radius_m: float = 1000.0,
    correction_factor: float = 0.5,
    verbose: bool = True,
) -> Optional[LayerPlan]:
    """
    Find the layer plan that minimises expected round-trip drone travel cost.

    Parameters
    ----------
    R_final : float
        Top-layer cell size (from depot model), metres.
    S0 : float
        Layer-0 cell size (1.5 × max H&B dimension), metres.
    layer_spacing_m : float
        Vertical distance between consecutive layers, metres.
    city_radius_m : float
        Radius of the circular service area, metres.
    correction_factor : float
        Expected fraction of cell size for lateral correction when descending.
        0.5 means the drone is, on average, half a cell-width off (uniform
        random offset inside the cell).
    verbose : bool
        If True, print a detailed comparison of all candidates.

    Returns
    -------
    LayerPlan or None
        The cost-optimal plan, or None if no valid k exists.
    """
    if R_final <= 0 or S0 <= 0:
        raise ValueError("R_final and S0 must be positive numbers.")

    F = R_final / S0

    if F <= 1.0 + 1e-12:
        cost_t, cost_v, cost_c, cost_cr = _trip_cost([S0], 0, layer_spacing_m, city_radius_m, correction_factor)
        return LayerPlan(
            k=0, divisions=[], product=1.0, target_F=F, achieved_R=S0,
            log_error=abs(math.log(1.0 / F)) if F > 0 else float("inf"),
            rel_error=(S0 - R_final) / R_final, layer_sizes=[S0],
            total_cost=cost_t, vertical_cost=cost_v, correction_cost=cost_c, cruise_cost=cost_cr,
        )

    k_min, k_max = _valid_k_range(F)
    if k_min > k_max:
        if verbose:
            print("\nNo valid k satisfies 2.0 <= F^(1/k) <= 3.0.")
            print(f"F = R_final/S0 = {F:.6g}")
        return None

    # ------------------------------------------------------------------
    # Build candidates.  For each k we evaluate the TWO best m values
    # (closest product to F from above and below) so we explore division
    # sequences that overshoot and undershoot the target.  Wildly wrong
    # products (e.g. 32 when F=147) are excluded because they break the
    # depot/grid alignment — we cap |rel_error| at 50%.
    # ------------------------------------------------------------------
    MAX_REL_ERROR = 0.50  # reject candidates whose top cell is >50% off

    all_candidates: List[LayerPlan] = []

    for k in range(k_min, k_max + 1):
        for m, P, log_err in _all_m_candidates_for_k(F, k):
            achieved_R = S0 * P
            rel_error = (achieved_R - R_final) / R_final
            if abs(rel_error) > MAX_REL_ERROR:
                continue  # product too far from target — unusable
            divisions = _interleave_divisions(k, m_threes=m, F=F)
            sizes = _build_layer_sizes(S0, divisions)
            cost_t, cost_v, cost_c, cost_cr = _trip_cost(
                sizes, k, layer_spacing_m, city_radius_m, correction_factor
            )
            all_candidates.append(LayerPlan(
                k=k, divisions=divisions, product=P, target_F=F,
                achieved_R=achieved_R, log_error=log_err, rel_error=rel_error,
                layer_sizes=sizes,
                total_cost=cost_t, vertical_cost=cost_v,
                correction_cost=cost_c, cruise_cost=cost_cr,
            ))

    if not all_candidates:
        if verbose:
            print("\nNo candidate with |rel_error| <= 50%.")
        return None

    # Sort by total travel cost; tie-break by smaller geometric error, then fewer layers
    all_candidates.sort(key=lambda c: (c.total_cost, c.log_error, c.k))
    best = all_candidates[0]

    if verbose:
        _print_optimize_report(
            R_final, S0, F, k_min, k_max, layer_spacing_m,
            city_radius_m, correction_factor, all_candidates, best,
        )

    return best


def _print_optimize_report(
    R_final, S0, F, k_min, k_max, layer_spacing_m,
    city_radius_m, correction_factor, candidates, best,
):
    """Pretty-print the full optimisation report."""

    SEP = "-" * 72

    print("\n" + "=" * 72)
    print("  OPTIMAL LAYER PLAN -- minimise expected round-trip travel cost")
    print("=" * 72)

    print(f"\n  Inputs")
    print(f"  {'R_final (top cell size)':.<40s} {R_final:.4g} m")
    print(f"  {'S0 (layer-0 cell size)':.<40s} {S0:.4g} m")
    print(f"  {'F = R_final / S0':.<40s} {F:.6g}")
    print(f"  {'Layer spacing (vertical)':.<40s} {layer_spacing_m:.1f} m")
    print(f"  {'City radius':.<40s} {city_radius_m:.1f} m")
    print(f"  {'Correction factor':.<40s} {correction_factor:.2f}")
    print(f"  {'Valid k range':.<40s} {k_min} .. {k_max}")

    # --- Per-k summary (best m for each k, by cost) ---
    print(f"\n  All candidates (best m per k, sorted by travel cost):")
    print(f"  {'k':>3s}  {'layers':>6s}  {'divs':>22s}  {'product':>9s}  "
          f"{'vert(m)':>9s}  {'corr(m)':>9s}  {'cruise(m)':>9s}  {'TOTAL(m)':>10s}  "
          f"{'log_err':>8s}  {'rel_err':>8s}")
    print(f"  {'---':>3s}  {'------':>6s}  {'-'*22:>22s}  {'-'*9:>9s}  "
          f"{'-'*9:>9s}  {'-'*9:>9s}  {'-'*9:>9s}  {'-'*10:>10s}  "
          f"{'-'*8:>8s}  {'-'*8:>8s}")

    # Group by k, show best cost per k, then also show other m values
    from collections import defaultdict
    by_k = defaultdict(list)
    for c in candidates:
        by_k[c.k].append(c)

    for k in sorted(by_k):
        group = sorted(by_k[k], key=lambda x: x.total_cost)
        for idx, c in enumerate(group):
            tag = "  <-- BEST" if c is best else ""
            divs_str = str(c.divisions)
            print(
                f"  {c.k:3d}  {c.k+1:6d}  {divs_str:>22s}  {c.product:9.2f}  "
                f"{c.vertical_cost:9.1f}  {c.correction_cost:9.1f}  {c.cruise_cost:9.1f}  "
                f"{c.total_cost:10.1f}  "
                f"{c.log_error:8.4f}  {100*c.rel_error:7.2f}%{tag}"
            )

    # --- Chosen plan detail ---
    print(f"\n  {SEP}")
    print(f"  CHOSEN PLAN")
    print(f"  {SEP}")
    print(f"  k (divisions)            : {best.k}")
    print(f"  Total layers             : {best.k + 1}")
    print(f"  Divisions                : {best.divisions}")
    print(f"  Product                  : {best.product:.6g}  (target F = {best.target_F:.6g})")
    print(f"  Achieved top cell size   : {best.achieved_R:.4g} m  (target = {R_final:.4g} m)")
    print(f"  Relative error           : {100*best.rel_error:+.3f}%")
    print(f"  Geometric log error      : {best.log_error:.6f}")
    print()
    print(f"  Cost breakdown (round-trip):")
    print(f"    Vertical transit       : {best.vertical_cost:10.1f} m  (2 x {best.k} x {layer_spacing_m:.0f})")
    print(f"    Layer corrections      : {best.correction_cost:10.1f} m  (2 x {correction_factor} x sum(S_i))")
    print(f"    Top-layer cruise       : {best.cruise_cost:10.1f} m  (2 x 2/3 x {city_radius_m:.0f})")
    print(f"    {'='*42}")
    print(f"    TOTAL                  : {best.total_cost:10.1f} m")
    print()
    print(f"  Layer cell sizes:")
    for i, s in enumerate(best.layer_sizes):
        alt = (i * layer_spacing_m) if i > 0 else 0
        print(f"    Layer {i:2d}:  cell = {s:10.4g} m   (altitude ~ L0_max + {alt:.0f} m)")

    # --- Also show the geometric-error-best for comparison ---
    geom_best = min(candidates, key=lambda c: (c.log_error, c.k))
    if geom_best is not best:
        print(f"\n  For comparison, the geometric-error-best plan:")
        print(f"    k={geom_best.k}, divisions={geom_best.divisions}, "
              f"log_err={geom_best.log_error:.6f}, "
              f"total_cost={geom_best.total_cost:.1f} m "
              f"(+{geom_best.total_cost - best.total_cost:.1f} m vs optimal)")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_positive_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            x = float(raw)
            if x <= 0:
                print("Please enter a positive number.")
                continue
            return x
        except Exception:
            print("Please enter a valid number (e.g., 300, 30.0).")


if __name__ == "__main__":
    print("RUNNING: layers_with_divisions.py\n")
    R_final = _read_positive_float("Enter R_final (top-layer cell size, e.g. 550): ")
    S0 = _read_positive_float("Enter S0 (layer-0 cell size, e.g. 3.75): ")
    H = _read_positive_float("Enter layer spacing in metres (e.g. 100): ")
    R_city = _read_positive_float("Enter city radius in metres (e.g. 1000): ")

    print("\n" + "-" * 72)
    print("  METHOD 1: Geometric-error optimisation (original)")
    print("-" * 72)
    plan_geom = compute_layer_plan(R_final=R_final, S0=S0, verbose=True)

    print("\n" + "-" * 72)
    print("  METHOD 2: Travel-cost optimisation (new)")
    print("-" * 72)
    plan_cost = optimize_layer_plan(
        R_final=R_final, S0=S0,
        layer_spacing_m=H, city_radius_m=R_city,
        verbose=True,
    )

    if plan_geom is None and plan_cost is None:
        raise SystemExit(1)
