"""
layers_with_divisions.py

Computes the optimal number of airspace layers and the cell-size at each layer
for a multi-layer drone-delivery network.

===========================================================================
MATHEMATICAL DERIVATION
===========================================================================

Problem
-------
A drone must travel from a depot (top layer, cell size ~ R) down through
k intermediate layers to layer 0 (cell size S0), deliver a package, and
return.  Each layer transition costs vertical climb/descent AND a horizontal
correction to align with the finer grid below.  We want the value of k
(number of divisions) and the per-layer branching factor r that minimise the
total expected round-trip distance.

Setup
-----
Let:
    S0      = layer-0 cell size (m)                  [fixed by drone size]
    R       = top-layer cell size (m)                [fixed by depot radius]
    F       = R / S0                                 [total scaling factor]
    k       = number of divisions (layers = k + 1)
    r       = branching factor (cell-size ratio between consecutive layers)
    H       = vertical spacing between layers (m)

With a uniform branching factor r across all k divisions:
    r^k = F   =>   r = F^(1/k)   =>   k = ln(F) / ln(r)

Layer i has cell size:   S_i = S0 * r^i

Cost model (one round trip)
---------------------------
1. VERTICAL COST:  The drone climbs k layers up and k layers down.
       C_vert = 2 * k * H

2. CORRECTION COST:  At each layer transition (descending from layer i to
   layer i-1), the drone is on the coarser grid of layer i.  Its target on
   the finer grid of layer i-1 is a random point within the coarser cell.

   For a uniform random point in a square cell of side S_i, the expected
   Manhattan distance from the nearest corner of the sub-grid is:
       E[correction at layer i] = S_i / 2

   (Each axis contributes S_i/4 on average; two axes => S_i/2.)

   Summing over all k descent transitions, and doubling for the return:

       C_corr = 2 * (1/2) * SUM_{i=1}^{k} S_i
              = SUM_{i=1}^{k} S0 * r^i
              = S0 * r * (r^k - 1) / (r - 1)
              = S0 * r * (F - 1) / (r - 1)

   For large F (our case: F ~ 150), this simplifies to:

       C_corr ~ S0 * F * r / (r - 1)  =  R * r / (r - 1)

3. CRUISE COST:  Horizontal travel at the top layer to reach the delivery
   zone.  For a depot on the perimeter of a circular city of radius R_city
   serving a uniformly random interior point, the expected distance is
   (2/3) * R_city.  Round trip: D_cruise = (4/3) * R_city.
   This is CONSTANT w.r.t. k and r, so it drops out of the optimisation.

Total variable cost (what we minimise):
---------------------------------------
With Manhattan routing factor (4/pi ~ 1.27) and a mild log-penalty alpha
for large jump ratios (to account for grid routing inefficiency on fine
grids over long distances):

    C(r) = 2*H*ln(F)/ln(r)  +  (4/pi)*(1 + alpha*ln(r)) * R*r/(r-1)

where alpha = 0.1.  This is a function of the single variable r > 1.

The first term (vertical cost) DECREASES as r increases (fewer layers).
The second term (correction cost) INCREASES as r increases: larger jumps
mean more horizontal correction, and the log-penalty (1 + 0.1*ln(r))
gently discourages extreme single jumps.

The optimal r* balances these.  It is found numerically via golden-section
search (no external dependencies needed -- the function is unimodal).

KEY INSIGHT: the old approach forced r in {2, 3}, creating 5-7 layers
with tiny cells at the bottom.  The physics says we should have FEWER
layers with LARGER jumps between them.  Typical optimal r* is 5-15,
giving k* = 2-3 divisions.

Radix economy connection
------------------------
The correction cost R*r/(r-1) is analogous to the "digits * base" cost in
radix economy (Steiner, 1850).  The vertical cost 2*H*ln(F)/ln(r) is
analogous to the "number of digits" cost (= log_r(F)).  The radix economy
result proves that the optimal base is e ~ 2.718 when digit cost equals base
cost, but our problem has DIFFERENT weights on the two terms (H vs R), so
the optimum shifts.

When vertical cost dominates (H >> R/F), optimal r -> large (fewer layers).
When correction cost dominates (R >> H*ln(F)), optimal r -> small (more layers).

Discretisation
--------------
After finding the continuous optimum r*, we:
  1. Round to nearby integers r_candidates = {floor(r*), ceil(r*), ...}
  2. For each candidate r, compute k = round(ln(F)/ln(r))
  3. Fine-tune: try (k-1, k, k+1) and pick the best
  4. Build the actual layer sizes, allowing the LAST division to differ
     (to land exactly on R or close to it)

===========================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerPlan:
    k: int                          # number of divisions (layers = k + 1)
    divisions: List[int]            # length-k list of integer branching factors
    product: float                  # product of all divisions
    target_F: float                 # R_final / S0
    achieved_R: float               # S0 * product
    log_error: float                # |ln(product / target_F)|
    rel_error: float                # (achieved_R - R_final) / R_final
    layer_sizes: List[float]        # [S0, S1, ..., Sk]
    # Cost fields
    total_cost: Optional[float] = None
    vertical_cost: Optional[float] = None
    correction_cost: Optional[float] = None
    cruise_cost: Optional[float] = None


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def _continuous_cost(r: float, F: float, H: float, R: float) -> float:
    """
    Total variable cost as a function of continuous branching factor r.

    With Manhattan routing and log-penalty for large jumps:

        C(r) = 2*H*ln(F)/ln(r)
             + 2*(4/pi)*(1 + 0.1*ln(r)) * R*r / (2*(r-1))

    The correction sum for k uniform divisions of factor r is:
        sum_{i=1}^{k} S0*r^i / 2  =  S0*r*(r^k - 1) / (2*(r-1))
                                   ~  R*r / (2*(r-1))   for large F

    With Manhattan factor and penalty, and doubled for round trip:
        C_corr ~ 2 * (4/pi) * (1 + 0.1*ln(r)) * R*r / (2*(r-1))
               = (4/pi) * (1 + 0.1*ln(r)) * R*r / (r-1)
    """
    if r <= 1.0 + 1e-12:
        return float("inf")
    MANHATTAN_FACTOR = 4.0 / math.pi
    ALPHA = 0.1
    vert = 2.0 * H * math.log(F) / math.log(r)
    corr = MANHATTAN_FACTOR * (1.0 + ALPHA * math.log(r)) * R * r / (r - 1.0)
    return vert + corr


def _find_optimal_r_continuous(F: float, H: float, R: float) -> float:
    """
    Find r* that minimises C(r) = 2*H*ln(F)/ln(r) + R*r/(r-1)
    using golden-section search on r in (1, F].

    No external dependencies needed -- C(r) is unimodal on (1, inf)
    (convex sum of a decreasing and an increasing function of r).
    """
    PHI = (math.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate ~ 0.618
    a, b = 1.001, max(F, 10.0)
    tol = 1e-8

    c = b - PHI * (b - a)
    d = a + PHI * (b - a)
    fc = _continuous_cost(c, F, H, R)
    fd = _continuous_cost(d, F, H, R)

    for _ in range(200):  # converges in ~60 iterations for tol=1e-8
        if b - a < tol:
            break
        if fc < fd:
            b = d
            d, fd = c, fc
            c = b - PHI * (b - a)
            fc = _continuous_cost(c, F, H, R)
        else:
            a = c
            c, fc = d, fd
            d = a + PHI * (b - a)
            fd = _continuous_cost(d, F, H, R)

    return (a + b) / 2.0


def _discrete_cost(
    layer_sizes: List[float],
    k: int,
    H: float,
    R_city: float,
) -> Tuple[float, float, float, float]:
    """
    Exact cost for a discrete layer plan.

    Returns (total, vertical, correction, cruise).

    Correction model (refined):
        When descending from layer i (cell S_i) to layer i-1 (cell S_{i-1}),
        the expected mis-alignment is S_i / 2.  But this correction must be
        flown on layer i-1's grid.  On a grid, the Manhattan routing penalty
        is a factor of  4/pi ~ 1.27  over Euclidean distance (expected ratio
        for a random direction on a square grid).  Additionally, for LARGE
        jumps (high ratio r_i = S_i / S_{i-1}), the correction distance
        becomes a significant fraction of the city, so we add a small penalty
        proportional to the jump ratio to discourage pathological single-jump
        configurations.

        C_corr_i  =  (4/pi) * S_i / 2  *  (1 + alpha * ln(r_i))

        where alpha is a mild log-penalty that makes a 100x jump slightly
        more expensive per metre than a 3x jump (reflecting that on the fine
        grid the path is longer, less direct, and computationally heavier).
        alpha = 0.1 is calibrated so that moderate jumps (3-10x) are nearly
        unpenalised, while extreme jumps (50-150x) get a 50-60% surcharge.

    Round-trip: multiply by 2.
    """
    MANHATTAN_FACTOR = 4.0 / math.pi   # ~1.273
    ALPHA = 0.1   # log-penalty for large jump ratios

    vertical = 2.0 * k * H
    cruise = 2.0 * (2.0 / 3.0) * R_city

    correction = 0.0
    for i in range(1, len(layer_sizes)):
        S_i = layer_sizes[i]        # coarser layer cell size
        S_prev = layer_sizes[i - 1] # finer layer cell size
        r_i = S_i / S_prev if S_prev > 0 else 1.0
        # Expected correction distance at this transition
        raw_correction = S_i / 2.0
        # Penalised for routing on finer grid + jump size
        penalty = 1.0 + ALPHA * math.log(max(r_i, 1.0))
        correction += MANHATTAN_FACTOR * raw_correction * penalty

    correction *= 2.0  # round trip

    total = vertical + cruise + correction
    return total, vertical, correction, cruise


# ---------------------------------------------------------------------------
# Core optimiser
# ---------------------------------------------------------------------------

def _build_plan_for_k_and_r(
    k: int,
    r_base: int,
    S0: float,
    R_final: float,
    F: float,
    H: float,
    R_city: float,
) -> Optional[LayerPlan]:
    """
    Build a LayerPlan using k divisions with base factor r_base.

    The first (k-1) divisions use r_base.  The last division is adjusted
    (to the nearest integer) so the product lands as close to F as possible.
    This lets us hit the target R_final accurately without forcing all
    divisions to be identical.
    """
    if k <= 0:
        return None
    if r_base < 2:
        r_base = 2

    if k == 1:
        # Only one division: it must be close to F
        d_last = max(2, round(F))
        divisions = [d_last]
    else:
        partial_product = r_base ** (k - 1)
        if partial_product <= 0:
            return None
        d_last_exact = F / partial_product
        d_last = max(2, round(d_last_exact))
        divisions = [r_base] * (k - 1) + [d_last]

    product = 1.0
    for d in divisions:
        product *= d
    achieved_R = S0 * product
    rel_error = (achieved_R - R_final) / R_final
    log_error = abs(math.log(product / F)) if product > 0 and F > 0 else float("inf")

    sizes = [S0]
    cur = S0
    for d in divisions:
        cur *= d
        sizes.append(cur)

    cost_t, cost_v, cost_c, cost_cr = _discrete_cost(sizes, k, H, R_city)

    return LayerPlan(
        k=k, divisions=divisions, product=product, target_F=F,
        achieved_R=achieved_R, log_error=log_error, rel_error=rel_error,
        layer_sizes=sizes,
        total_cost=cost_t, vertical_cost=cost_v,
        correction_cost=cost_c, cruise_cost=cost_cr,
    )


def _build_plan_for_arbitrary_divisions(
    divisions: List[int],
    S0: float,
    R_final: float,
    F: float,
    H: float,
    R_city: float,
) -> LayerPlan:
    """Build a LayerPlan from an explicit list of division factors."""
    k = len(divisions)
    product = 1.0
    for d in divisions:
        product *= d
    achieved_R = S0 * product
    rel_error = (achieved_R - R_final) / R_final
    log_error = abs(math.log(product / F)) if product > 0 and F > 0 else float("inf")

    sizes = [S0]
    cur = S0
    for d in divisions:
        cur *= d
        sizes.append(cur)

    cost_t, cost_v, cost_c, cost_cr = _discrete_cost(sizes, k, H, R_city)
    return LayerPlan(
        k=k, divisions=divisions, product=product, target_F=F,
        achieved_R=achieved_R, log_error=log_error, rel_error=rel_error,
        layer_sizes=sizes,
        total_cost=cost_t, vertical_cost=cost_v,
        correction_cost=cost_c, cruise_cost=cost_cr,
    )


def optimize_layer_plan(
    R_final: float,
    S0: float,
    layer_spacing_m: float = 100.0,
    city_radius_m: float = 1000.0,
    verbose: bool = True,
) -> Optional[LayerPlan]:
    """
    Find the layer configuration that minimises total drone travel cost.

    Uses calculus to find the continuous optimum branching factor, then
    searches nearby integer configurations exhaustively.

    Parameters
    ----------
    R_final : float
        Top-layer cell size (depot radius), metres.
    S0 : float
        Layer-0 cell size, metres.
    layer_spacing_m : float
        Vertical distance between layers, metres.
    city_radius_m : float
        Radius of the circular service area, metres.
    verbose : bool
        Print detailed analysis and derivation.

    Returns
    -------
    LayerPlan or None
    """
    if R_final <= 0 or S0 <= 0:
        raise ValueError("R_final and S0 must be positive numbers.")

    F = R_final / S0
    H = layer_spacing_m
    R = R_final

    if F <= 1.0 + 1e-12:
        cost_t, cost_v, cost_c, cost_cr = _discrete_cost([S0], 0, H, city_radius_m)
        return LayerPlan(
            k=0, divisions=[], product=1.0, target_F=F, achieved_R=S0,
            log_error=0.0, rel_error=0.0, layer_sizes=[S0],
            total_cost=cost_t, vertical_cost=cost_v,
            correction_cost=cost_c, cruise_cost=cost_cr,
        )

    # --- Step 1: Find continuous optimum ---
    r_star = _find_optimal_r_continuous(F, H, R)
    k_star_cont = math.log(F) / math.log(r_star)
    C_star = _continuous_cost(r_star, F, H, R)

    # --- Step 2: Exhaustive search over nearby integer configurations ---
    # Try r values from 2 up to min(F, r_star*3) and k from 1 to reasonable max
    r_max_search = max(int(r_star * 3), int(F) + 1, 20)
    r_max_search = min(r_max_search, int(F) + 1)
    k_max_search = max(int(math.log(F) / math.log(2.0)) + 2, 10)

    candidates: List[LayerPlan] = []

    for r_int in range(2, r_max_search + 1):
        # Natural k for this r
        if math.log(r_int) <= 0:
            continue
        k_natural = math.log(F) / math.log(r_int)
        # Try k values around the natural one
        for k in range(max(1, int(k_natural) - 1), int(k_natural) + 3):
            if k < 1 or k > k_max_search:
                continue
            plan = _build_plan_for_k_and_r(k, r_int, S0, R_final, F, H, city_radius_m)
            if plan is not None and abs(plan.rel_error) <= 0.50:
                candidates.append(plan)

    # Also try some "mixed" configurations (e.g., two large jumps)
    # For k=2: try all (d1, d2) where d1*d2 ~ F
    for d1 in range(2, min(int(math.sqrt(F)) + 5, int(F))):
        d2 = max(2, round(F / d1))
        for d2_try in [d2 - 1, d2, d2 + 1]:
            if d2_try < 2:
                continue
            plan = _build_plan_for_arbitrary_divisions(
                [d1, d2_try], S0, R_final, F, H, city_radius_m
            )
            if abs(plan.rel_error) <= 0.50:
                candidates.append(plan)

    # For k=3: try some combinations
    for d1 in range(2, min(int(F ** (1/3)) + 4, 20)):
        for d2 in range(d1, min(int(F ** (1/3)) + 4, 20)):
            d3 = max(2, round(F / (d1 * d2)))
            if d3 < 2:
                continue
            for d3_try in [d3 - 1, d3, d3 + 1]:
                if d3_try < 2:
                    continue
                plan = _build_plan_for_arbitrary_divisions(
                    [d1, d2, d3_try], S0, R_final, F, H, city_radius_m
                )
                if abs(plan.rel_error) <= 0.50:
                    candidates.append(plan)

    if not candidates:
        if verbose:
            print("No valid candidates found.")
        return None

    # --- Step 3: Pick the winner ---
    candidates.sort(key=lambda c: (c.total_cost, abs(c.rel_error), c.k))
    best = candidates[0]

    if verbose:
        _print_full_report(
            R_final, S0, F, H, city_radius_m,
            r_star, k_star_cont, C_star,
            candidates, best,
        )

    return best


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_full_report(
    R_final, S0, F, H, R_city,
    r_star, k_star_cont, C_star,
    candidates, best,
):
    """Print the full mathematical derivation and results."""

    print("\n" + "=" * 78)
    print("  OPTIMAL LAYER PLAN -- Cost-minimisation with calculus derivation")
    print("=" * 78)

    # --- Inputs ---
    print(f"\n  INPUTS")
    print(f"  {'R_final (top cell / depot radius)':.<44s} {R_final:.4g} m")
    print(f"  {'S0 (layer-0 cell size)':.<44s} {S0:.4g} m")
    print(f"  {'F = R_final / S0':.<44s} {F:.6g}")
    print(f"  {'H (vertical layer spacing)':.<44s} {H:.1f} m")
    print(f"  {'R_city (service area radius)':.<44s} {R_city:.1f} m")

    # --- Cost model ---
    MF = 4.0 / math.pi
    ALPHA = 0.1
    print(f"\n  COST MODEL")
    print(f"  C(r) = 2*H*ln(F)/ln(r)  +  (4/pi)*(1+0.1*ln(r))*R*r/(r-1)")
    print(f"       = {2*H:.0f}*{math.log(F):.4f}/ln(r)  +  {MF:.4f}*(1+0.1*ln(r))*{R_final:.1f}*r/(r-1)")
    print(f"       = {2*H*math.log(F):.1f}/ln(r)  +  {MF*R_final:.1f}*(1+0.1*ln(r))*r/(r-1)")

    # --- Continuous optimum ---
    print(f"\n  CONTINUOUS OPTIMUM (calculus)")
    print(f"  {'Optimal branching factor r*':.<44s} {r_star:.4f}")
    print(f"  {'Optimal divisions k* = ln(F)/ln(r*)':.<44s} {k_star_cont:.4f}")
    print(f"  {'Minimum variable cost C(r*)':.<44s} {C_star:.1f} m")
    print(f"  {'(+ cruise {:.1f} m = {:.1f} m total)'.format(2*(2/3)*R_city, C_star + 2*(2/3)*R_city):.<44s}")

    # --- Verify with nearby integer r values ---
    print(f"\n  COST LANDSCAPE (continuous r, showing why r*={r_star:.2f} is optimal):")
    print(f"  {'r':>6s}  {'k=ln(F)/ln(r)':>14s}  {'vert cost':>10s}  {'corr cost':>10s}  {'total var':>10s}")
    print(f"  {'---':>6s}  {'-'*14:>14s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}")
    r_values = sorted(set([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, round(r_star, 2)]))
    for r_show in r_values:
        k_show = math.log(F) / math.log(r_show) if r_show > 1 else float("inf")
        c_total = _continuous_cost(r_show, F, H, R_final)
        c_vert = 2 * H * k_show
        c_corr = c_total - c_vert
        tag = "  <-- r*" if abs(r_show - r_star) < 0.5 else ""
        print(f"  {r_show:6.2f}  {k_show:14.2f}  {c_vert:10.1f}  {c_corr:10.1f}  {c_total:10.1f}{tag}")

    # --- Discrete candidates (top 15 by cost) ---
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(c.divisions)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    top_n = unique[:15]
    print(f"\n  TOP DISCRETE CANDIDATES (by total round-trip cost):")
    print(f"  {'#':>3s}  {'k':>2s}  {'divisions':>24s}  {'top cell':>9s}  "
          f"{'vert(m)':>9s}  {'corr(m)':>9s}  {'cruise(m)':>9s}  {'TOTAL(m)':>10s}  "
          f"{'rel_err':>8s}")
    print(f"  {'---':>3s}  {'--':>2s}  {'-'*24:>24s}  {'-'*9:>9s}  "
          f"{'-'*9:>9s}  {'-'*9:>9s}  {'-'*9:>9s}  {'-'*10:>10s}  "
          f"{'-'*8:>8s}")
    for idx, c in enumerate(top_n):
        tag = "  <-- BEST" if c is best else ""
        divs_str = str(c.divisions)
        print(
            f"  {idx+1:3d}  {c.k:2d}  {divs_str:>24s}  {c.achieved_R:9.1f}  "
            f"{c.vertical_cost:9.1f}  {c.correction_cost:9.1f}  {c.cruise_cost:9.1f}  "
            f"{c.total_cost:10.1f}  {100*c.rel_error:7.2f}%{tag}"
        )

    # --- Chosen plan detail ---
    SEP = "-" * 78
    print(f"\n  {SEP}")
    print(f"  CHOSEN PLAN")
    print(f"  {SEP}")
    print(f"  Divisions (k)            : {best.k}")
    print(f"  Total layers             : {best.k + 1}")
    print(f"  Division factors         : {best.divisions}")
    print(f"  Product                  : {best.product:.6g}  (target F = {best.target_F:.6g})")
    print(f"  Achieved top cell size   : {best.achieved_R:.4g} m  (target = {R_final:.4g} m)")
    print(f"  Relative error           : {100*best.rel_error:+.3f}%")
    print()
    print(f"  Cost breakdown (round-trip):")
    print(f"    Vertical transit       : {best.vertical_cost:10.1f} m  (2 x {best.k} x {H:.0f})")
    print(f"    Layer corrections      : {best.correction_cost:10.1f} m  (sum of upper cell sizes)")
    print(f"    Top-layer cruise       : {best.cruise_cost:10.1f} m  (2 x 2/3 x {R_city:.0f})")
    print(f"    {'='*46}")
    print(f"    TOTAL                  : {best.total_cost:10.1f} m")
    print()
    print(f"  Layer structure:")
    for i, s in enumerate(best.layer_sizes):
        alt = i * H if i > 0 else 0
        div_note = f"  (x{best.divisions[i-1]})" if i > 0 else "  (base)"
        print(f"    Layer {i:2d}:  cell = {s:10.2f} m   alt ~ L0_max + {alt:5.0f} m{div_note}")

    # --- Comparison with old 2-or-3 method ---
    from layers_with_divisions import compute_layer_plan as _old_method
    old_plan = _old_method(R_final=R_final, S0=S0, verbose=False)
    if old_plan is not None:
        old_sizes = old_plan.layer_sizes
        old_cost, old_v, old_c, old_cr = _discrete_cost(old_sizes, old_plan.k, H, R_city)
        print(f"\n  COMPARISON WITH OLD METHOD (2-or-3 divisions only):")
        print(f"    Old: k={old_plan.k}, layers={old_plan.k+1}, "
              f"divs={old_plan.divisions}, cost={old_cost:.1f} m")
        print(f"    New: k={best.k}, layers={best.k+1}, "
              f"divs={best.divisions}, cost={best.total_cost:.1f} m")
        saving = old_cost - best.total_cost
        pct = 100 * saving / old_cost if old_cost > 0 else 0
        print(f"    Saving: {saving:.1f} m per round trip ({pct:.1f}%)")

    print()


# ---------------------------------------------------------------------------
# LEGACY: original method (preserved for backward compatibility)
# ---------------------------------------------------------------------------

def _valid_k_range(F: float) -> Tuple[int, int]:
    """Valid k satisfy: 2^k <= F <= 3^k."""
    if F <= 0:
        return (1, 0)
    k_min = int(math.ceil(math.log(F) / math.log(3.0)))
    k_max = int(math.floor(math.log(F) / math.log(2.0)))
    return (k_min, k_max)


def _best_product_for_k(F: float, k: int) -> Tuple[float, int, float]:
    best_P, best_m, best_err = None, None, float("inf")
    for m in range(k + 1):
        P = (2.0 ** (k - m)) * (3.0 ** m)
        err = abs(math.log(P / F))
        if err < best_err:
            best_err, best_P, best_m = err, P, m
    return float(best_P), int(best_m), float(best_err)


def _interleave_divisions(k: int, m_threes: int, F: float) -> List[int]:
    twos_left, threes_left = k - m_threes, m_threes
    g = F ** (1.0 / k)
    divisions: List[int] = []
    cur_prod = 1.0
    for t in range(1, k + 1):
        target = g ** t
        cands = []
        if twos_left > 0: cands.append(2)
        if threes_left > 0: cands.append(3)
        factor = min(cands, key=lambda f: abs(math.log((cur_prod * f) / target)))
        divisions.append(factor)
        cur_prod *= factor
        if factor == 2: twos_left -= 1
        else: threes_left -= 1
    return divisions


def compute_layer_plan(R_final: float, S0: float, verbose: bool = True) -> Optional[LayerPlan]:
    """Original layer-plan: picks divisions from {2,3} minimising geometric error."""
    if R_final <= 0 or S0 <= 0:
        raise ValueError("R_final and S0 must be positive numbers.")
    F = R_final / S0
    if F <= 1.0 + 1e-12:
        return LayerPlan(k=0, divisions=[], product=1.0, target_F=F, achieved_R=S0,
                         log_error=abs(math.log(1.0/F)) if F > 0 else float("inf"),
                         rel_error=(S0-R_final)/R_final, layer_sizes=[S0])
    k_min, k_max = _valid_k_range(F)
    if k_min > k_max:
        return None
    candidates = []
    for k in range(k_min, k_max + 1):
        P, m, log_err = _best_product_for_k(F, k)
        divisions = _interleave_divisions(k, m, F)
        achieved_R = S0 * P
        sizes = [S0]
        cur = S0
        for d in divisions:
            cur *= d
            sizes.append(cur)
        candidates.append(LayerPlan(
            k=k, divisions=divisions, product=P, target_F=F,
            achieved_R=achieved_R, log_error=log_err,
            rel_error=(achieved_R-R_final)/R_final, layer_sizes=sizes))
    candidates.sort(key=lambda c: (c.log_error, c.k))
    best = candidates[0]
    if verbose:
        print(f"\n=== Layer plan (legacy 2/3 method) ===")
        print(f"k={best.k}, layers={best.k+1}, divs={best.divisions}")
        print(f"Layer sizes: {[f'{s:.2f}' for s in best.layer_sizes]}")
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_float(prompt: str) -> float:
    while True:
        try:
            x = float(input(prompt).strip())
            if x > 0: return x
            print("Must be positive.")
        except Exception:
            print("Invalid number.")


if __name__ == "__main__":
    print("OPTIMAL LAYER PLAN CALCULATOR")
    print("=" * 50)
    R_final = _read_float("R_final (top cell / depot radius, e.g. 550): ")
    S0 = _read_float("S0 (layer-0 cell size, e.g. 3.75): ")
    H = _read_float("Layer spacing in metres (e.g. 100): ")
    R_city = _read_float("City radius in metres (e.g. 1000): ")

    plan = optimize_layer_plan(
        R_final=R_final, S0=S0,
        layer_spacing_m=H, city_radius_m=R_city,
        verbose=True,
    )
    if plan is None:
        raise SystemExit(1)
