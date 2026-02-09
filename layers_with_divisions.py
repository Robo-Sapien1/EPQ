"""
depot_model_demand.py

Given:
  - R_final : top-layer cell size (from depot model)
  - S0      : bottom-layer cell size (layer 0)

We compute:
  F = R_final / S0

Choose an integer k (number of *divisions* between layer 0 and the top layer) such that:
  2.0 <= F^(1/k) <= 3.0
(i.e., the per-layer geometric scaling is between 2× and 3×)

For each valid k:
  Choose k integers from {2, 3} (repeats allowed) so that their product is as close as possible to F.
  Since order doesn't change the product, the search is equivalent to choosing how many 3s to use:
    product = 2^(k-m) * 3^m,  m = 0..k

We pick the overall best k by minimising multiplicative error (symmetrical in log space):
  error = |ln(product / F)|

Outputs:
  - chosen k (divisions), and total layers = k+1
  - the divisions (a length-k list of 2s and 3s)
  - achieved top size and relative error
  - the implied cell size at each layer boundary
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


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


def compute_layer_plan(R_final: float, S0: float, verbose: bool = True) -> Optional[LayerPlan]:
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

        # Build layer sizes
        sizes = [S0]
        cur = S0
        for d in divisions:
            cur *= d
            sizes.append(cur)

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
        print("\n=== Layer scaling plan ===")
        print(f"R_final (top cell size)  = {R_final:.6g}")
        print(f"S0 (layer-0 cell size)   = {S0:.6g}")
        print(f"F = R_final/S0           = {F:.6g}")

        print("\nValid k range from 2^k <= F <= 3^k:")
        print(f"k_min = {k_min}, k_max = {k_max}")
        print("(k = number of divisions; total layers = k+1)\n")

        # Show all candidates
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
    print("RUNNING: depot_model_demand.py")
    R_final = _read_positive_float("Enter R_final (top-layer cell size, e.g. 300): ")
    S0 = _read_positive_float("Enter S0 (layer-0 cell size, e.g. 30): ")

    plan = compute_layer_plan(R_final=R_final, S0=S0, verbose=True)

    if plan is None:
        raise SystemExit(1)
