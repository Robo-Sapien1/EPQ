"""
h_function.py  --  Demand-driven mesh sizing function for L0

Algorithm
---------
1.  Load the demand density map (12x12 grid, 80 m cells) from the pipeline
    JSON and resample it onto the fine L0 grid (cell_size ~ 3.75 m).

2.  Greedy seed selection + region growing:
        a) Find the L0 cell with the highest unoccupied demand.
        b) Flood-fill outward (BFS in 4-connected L0 cells) accumulating
           demand until cumulative demand >= demand_threshold.
        c) Record that set of L0 cells as one "h-region".
        d) Mark them occupied and repeat until the entire L0 grid is covered.

3.  Post-processing:
        a) Compute the average area (in L0-cell units) of all h-regions.
        b) Snap that average to the nearest integer multiple of the original
           L0 cell side-length, giving the "effective L1 cell size".
        c) Feed this snapped size as S0 into layers_with_divisions.py to
           derive the upper-layer hierarchy.  The *real* L0 grid keeps its
           original fine cell size.

4.  Visualization helpers: return region boundaries so the Dash overlay can
    draw them on top of the L0 grid at the same altitude.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HRegion:
    """One region produced by the greedy growing algorithm."""
    region_id: int
    seed_ix: int                     # L0 grid-index x of seed
    seed_iy: int                     # L0 grid-index y of seed
    cells: List[Tuple[int, int]]     # list of (ix, iy) L0 grid indices
    cumulative_demand: float         # total demand accumulated
    area_m2: float                   # len(cells) * l0_cell_m^2


@dataclass
class HFunctionResult:
    """Full output of the h-function."""
    regions: List[HRegion]
    demand_threshold: float
    l0_cell_m: float                 # original L0 cell size
    avg_region_area_m2: float        # mean region area
    avg_region_side_m: float         # sqrt(avg_region_area_m2)
    snapped_cell_m: float            # avg side snapped to nearest L0 multiple
    snap_n: int                      # integer multiple  (snapped = snap_n * l0_cell_m)
    demand_grid: np.ndarray          # demand resampled onto L0 (nyn x nxn)
    grid_xmin: float
    grid_ymin: float


# ---------------------------------------------------------------------------
# Demand resampling
# ---------------------------------------------------------------------------

def _resample_demand_to_l0(
    density_map: dict,
    l0_xmin: float, l0_ymin: float,
    nxn: int, nyn: int,
    l0_cell_m: float,
) -> np.ndarray:
    """
    Resample the coarse demand density raster (80 m cells) onto the fine L0
    grid via nearest-neighbour lookup.  Returns array of shape (nyn, nxn).
    """
    d_xmin = density_map["xmin"]
    d_ymin = density_map["ymin"]
    d_cs = density_map["cell_size"]
    d_ncols = density_map["ncols"]
    d_nrows = density_map["nrows"]
    d_vals = np.array(density_map["values"], dtype=float).reshape(d_nrows, d_ncols)

    out = np.zeros((nyn, nxn), dtype=float)
    for iy in range(nyn):
        y = l0_ymin + iy * l0_cell_m + 0.5 * l0_cell_m
        diy = int((y - d_ymin) / d_cs)
        diy = max(0, min(d_nrows - 1, diy))
        for ix in range(nxn):
            x = l0_xmin + ix * l0_cell_m + 0.5 * l0_cell_m
            dix = int((x - d_xmin) / d_cs)
            dix = max(0, min(d_ncols - 1, dix))
            out[iy, ix] = d_vals[diy, dix]
    return out


# ---------------------------------------------------------------------------
# Greedy region-growing
# ---------------------------------------------------------------------------

def _grow_region(
    seed: Tuple[int, int],
    demand: np.ndarray,
    occupied: np.ndarray,
    inside: set,
    threshold: float,
    l0_cell_m: float,
) -> HRegion:
    """
    BFS flood-fill from *seed* accumulating demand until >= threshold.
    Cells are consumed in descending demand order (priority BFS).
    """
    nyn, nxn = demand.shape
    cells = []
    cum = 0.0
    region_id = -1  # assigned by caller

    # Use a simple BFS queue; at each step pick the neighbor with highest demand
    # For performance, use a deque-based BFS (4-connected, demand-weighted)
    visited = set()
    visited.add(seed)
    # Priority: use a sorted frontier (highest demand first)
    frontier = [seed]

    while frontier and cum < threshold:
        # Pop highest-demand cell from frontier
        frontier.sort(key=lambda c: demand[c[1], c[0]], reverse=True)
        cell = frontier.pop(0)
        ix, iy = cell

        cells.append(cell)
        occupied[iy, ix] = True
        cum += demand[iy, ix] * (l0_cell_m ** 2)  # demand density * cell area

        # Expand neighbors
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx_, ny_ = ix + dx, iy + dy
            nb = (nx_, ny_)
            if nb in inside and nb not in visited and not occupied[ny_, nx_]:
                visited.add(nb)
                frontier.append(nb)

    # If threshold not reached (edge of map), keep what we have
    return HRegion(
        region_id=region_id,
        seed_ix=seed[0], seed_iy=seed[1],
        cells=cells,
        cumulative_demand=cum,
        area_m2=len(cells) * l0_cell_m ** 2,
    )


def compute_h_function(
    density_map: dict,
    l0_cell_m: float,
    cx: float, cy: float, radius_m: float,
    demand_threshold: float,
) -> HFunctionResult:
    """
    Run the full h-function algorithm.

    Parameters
    ----------
    density_map : dict
        The ``demand_density_map`` from integrating_step1_output.json.
    l0_cell_m : float
        Original L0 cell size in metres (e.g. 3.75).
    cx, cy : float
        Centre of the circular study area (EPSG:27700).
    radius_m : float
        Radius of the study area in metres.
    demand_threshold : float
        Cumulative demand a region must reach before it stops growing.
        Higher value → larger regions → coarser effective L1.

    Returns
    -------
    HFunctionResult
    """
    spacing = l0_cell_m
    effective_radius = radius_m + spacing
    xmin = cx - effective_radius
    ymin = cy - effective_radius
    xmax = cx + effective_radius
    ymax = cy + effective_radius
    nxn = int((xmax - xmin) // spacing) + 1
    nyn = int((ymax - ymin) // spacing) + 1

    # 1. Resample demand onto L0 grid
    demand = _resample_demand_to_l0(density_map, xmin, ymin, nxn, nyn, spacing)

    # 2. Build set of "inside" cells (within circle)
    inside = set()
    for ix in range(nxn):
        x = xmin + ix * spacing
        dx2 = (x - cx) ** 2
        for iy in range(nyn):
            y = ymin + iy * spacing
            if dx2 + (y - cy) ** 2 <= effective_radius ** 2:
                inside.add((ix, iy))

    # 3. Greedy seed + region growing
    occupied = np.zeros((nyn, nxn), dtype=bool)
    regions: List[HRegion] = []

    # Build sorted list of all inside cells by demand (descending)
    cells_by_demand = sorted(
        inside,
        key=lambda c: demand[c[1], c[0]],
        reverse=True,
    )

    region_id = 0
    for cell in cells_by_demand:
        ix, iy = cell
        if occupied[iy, ix]:
            continue
        # This is the highest-demand unoccupied cell — use as seed
        region = _grow_region(cell, demand, occupied, inside, demand_threshold, spacing)
        region.region_id = region_id
        regions.append(region)
        region_id += 1

    # 4. Post-processing
    if regions:
        areas = [r.area_m2 for r in regions]
        avg_area = sum(areas) / len(areas)
    else:
        avg_area = spacing ** 2

    avg_side = math.sqrt(avg_area)

    # Snap to nearest integer multiple of l0_cell_m
    snap_n = max(1, round(avg_side / l0_cell_m))
    snapped = snap_n * l0_cell_m

    print(f"\n  [h-function] {len(regions)} regions grown (threshold={demand_threshold:.0f})")
    print(f"  [h-function] avg region area = {avg_area:.1f} m^2  (avg side = {avg_side:.2f} m)")
    print(f"  [h-function] snapped cell size = {snapped:.4g} m  ({snap_n} x {l0_cell_m} m)")
    region_cell_counts = [len(r.cells) for r in regions]
    print(f"  [h-function] region sizes: min={min(region_cell_counts)} cells, "
          f"max={max(region_cell_counts)} cells, "
          f"median={sorted(region_cell_counts)[len(region_cell_counts)//2]} cells")

    return HFunctionResult(
        regions=regions,
        demand_threshold=demand_threshold,
        l0_cell_m=l0_cell_m,
        avg_region_area_m2=avg_area,
        avg_region_side_m=avg_side,
        snapped_cell_m=snapped,
        snap_n=snap_n,
        demand_grid=demand,
        grid_xmin=xmin,
        grid_ymin=ymin,
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def region_boundary_segments(
    h_result: HFunctionResult,
    z_fallback: float,
    G_l0=None,
    z_offset: float = 2.0,
) -> List[dict]:
    """
    Return a list of dicts, each with 'xs', 'ys', 'zs' arrays suitable
    for a Scatter3d trace, representing the boundary edges of each region.

    If *G_l0* (the L0 NetworkX graph) is provided, each boundary edge segment
    samples its z from the L0 node at that grid index (+ z_offset) so the
    overlay follows the curved L0 surface.  Otherwise falls back to a flat
    plane at *z_fallback + z_offset*.
    """
    spacing = h_result.l0_cell_m
    xmin = h_result.grid_xmin
    ymin = h_result.grid_ymin

    # Build a fast lookup: (ix, iy) -> z from L0 graph nodes
    node_z: Dict[Tuple[int, int], float] = {}
    if G_l0 is not None:
        for nid in G_l0.nodes:
            node_z[nid] = G_l0.nodes[nid]["z"]

    def _z_at(ix: int, iy: int) -> float:
        """Look up z for the grid corner (ix, iy), with fallback."""
        if node_z:
            z = node_z.get((ix, iy))
            if z is not None:
                return z + z_offset
            # Try nearby nodes (corner might be just outside the circle)
            for dix, diy in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]:
                z = node_z.get((ix + dix, iy + diy))
                if z is not None:
                    return z + z_offset
        return z_fallback + z_offset

    traces = []
    for region in h_result.regions:
        cell_set = set(region.cells)
        xs, ys, zs = [], [], []
        for (ix, iy) in region.cells:
            # Cell corner at (xmin + ix*s, ymin + iy*s)
            x0 = xmin + ix * spacing
            y0 = ymin + iy * spacing
            x1 = x0 + spacing
            y1 = y0 + spacing
            # Corner z values from L0 surface
            z00 = _z_at(ix, iy)
            z10 = _z_at(ix + 1, iy)
            z01 = _z_at(ix, iy + 1)
            z11 = _z_at(ix + 1, iy + 1)
            # Check 4 edges; draw if neighbor is NOT in same region
            # Bottom edge (y = y0, from x0 to x1)
            if (ix, iy - 1) not in cell_set:
                xs.extend([x0, x1, None]); ys.extend([y0, y0, None]); zs.extend([z00, z10, None])
            # Top edge (y = y1, from x0 to x1)
            if (ix, iy + 1) not in cell_set:
                xs.extend([x0, x1, None]); ys.extend([y1, y1, None]); zs.extend([z01, z11, None])
            # Left edge (x = x0, from y0 to y1)
            if (ix - 1, iy) not in cell_set:
                xs.extend([x0, x0, None]); ys.extend([y0, y1, None]); zs.extend([z00, z01, None])
            # Right edge (x = x1, from y0 to y1)
            if (ix + 1, iy) not in cell_set:
                xs.extend([x1, x1, None]); ys.extend([y0, y1, None]); zs.extend([z10, z11, None])
        traces.append({"xs": xs, "ys": ys, "zs": zs, "region_id": region.region_id})
    return traces


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def _auto_demand_threshold(density_map: dict, l0_cell_m: float, target_n_regions: int = 30) -> float:
    """
    Estimate a demand_threshold that produces roughly target_n_regions regions.

    Total demand = sum(density * cell_area) over the demand grid.
    threshold ≈ total_demand / target_n_regions.
    """
    vals = np.array(density_map["values"], dtype=float)
    cell_area = density_map["cell_size"] ** 2
    total = float(vals.sum()) * cell_area / (density_map["cell_size"] / l0_cell_m) ** 2
    # Adjust: the L0 cells are much smaller, so total demand on L0 is the same
    # but spread over more cells.  The density values are per-coarse-cell, so
    # when resampled each L0 cell gets the same density.
    # total_demand_on_l0 = sum(density_i * l0_cell_area) for all L0 cells in circle
    # ≈ sum(density_i * coarse_area) because density is constant within each coarse cell
    total_demand = float(vals.sum()) * (l0_cell_m ** 2)
    # But each coarse cell is replicated onto (coarse_cs / l0_cs)^2 L0 cells
    n_per_coarse = (density_map["cell_size"] / l0_cell_m) ** 2
    total_demand = float(vals.sum()) * (l0_cell_m ** 2) * n_per_coarse
    # = sum(vals) * coarse_cs^2  (same as coarse total)
    thresh = total_demand / target_n_regions
    return max(thresh, 1.0)


if __name__ == "__main__":
    step1_path = Path(__file__).resolve().parent / "integrating_step1_output.json"
    if not step1_path.exists():
        print("No integrating_step1_output.json found.")
        raise SystemExit(1)

    with open(step1_path) as f:
        data = json.load(f)

    density_map = data.get("demand_density_map")
    if density_map is None:
        print("No demand_density_map in JSON.")
        raise SystemExit(1)

    from fleet_specs import get_l0_cell_m
    l0 = get_l0_cell_m()
    center = data.get("center_projected", [0, 0])
    cx, cy = float(center[0]), float(center[1])
    radius_m = float(data.get("radius_m", 500))

    threshold = _auto_demand_threshold(density_map, l0, target_n_regions=30)
    print(f"Auto threshold = {threshold:.1f}")

    result = compute_h_function(density_map, l0, cx, cy, radius_m, threshold)

    print(f"\n  Result: {len(result.regions)} regions")
    print(f"  Snapped effective L1 cell = {result.snapped_cell_m:.4g} m "
          f"({result.snap_n} x {result.l0_cell_m} m)")
    print(f"  This feeds into layers_with_divisions as the new S0 for upper layers.")

    # Run layer plan with the snapped size
    from layers_with_divisions import optimize_layer_plan
    depot_json = Path(__file__).resolve().parent / "depot_solution_for_overlay.json"
    if depot_json.exists():
        with open(depot_json) as f:
            sol = json.load(f)
        R = float(sol.get("R", 550))
    else:
        R = 550.0

    plan = optimize_layer_plan(
        R_final=R,
        S0=result.snapped_cell_m,
        layer_spacing_m=100.0,
        city_radius_m=radius_m,
        verbose=True,
    )
