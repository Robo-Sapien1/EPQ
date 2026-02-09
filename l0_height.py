# -*- coding: utf-8 -*-
"""
L0 (bottom layer) height setting for the airspace grid.

Method A (default): Raise a base height until only a small fraction of buildings exceed it
(see building_exceed_threshold, e.g. 2%). Then for L0 cells still intersected by a building,
use corners + 3 points per edge; each point at least clearance_m above the building below.
Cells not intersected use the base height. Upper layers: second = max(L0 z) + 100 m; +100 m each.

Used by dash_scene_builder when building the multi-layer grid.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# Number of points per edge (excluding corners)
EDGE_POINTS = 3


def height_below_at_point(
    x: float, y: float, buildings: gpd.GeoDataFrame, sindex: Optional[Any] = None
) -> float:
    """
    Return the height of the surface at (x, y) in metres: 0 if no building,
    else the maximum height_m of any building whose footprint contains the point.
    If sindex is provided (buildings.sindex), use it to speed up the query.
    """
    pt = Point(x, y)
    best = 0.0
    if sindex is not None:
        try:
            possible = list(sindex.intersection(pt.bounds))
            for idx in possible:
                row = buildings.iloc[idx]
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                if hasattr(geom, "geoms"):
                    for g in geom.geoms:
                        if g.contains(pt):
                            best = max(best, float(row.get("height_m", 0.0)))
                            break
                else:
                    if geom.contains(pt):
                        best = max(best, float(row.get("height_m", 0.0)))
            return best
        except Exception:
            pass
    for _, row in buildings.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if hasattr(geom, "geoms"):
            for g in geom.geoms:
                if g.contains(pt):
                    best = max(best, float(row.get("height_m", 0.0)))
                    break
        else:
            if geom.contains(pt):
                best = max(best, float(row.get("height_m", 0.0)))
    return best


def cell_sample_points(x0: float, y0: float, cell_size: float) -> List[Tuple[float, float]]:
    """
    Return 4 corners + 3 equally spaced points on each of the 4 edges (not including corners).
    Total 4 + 4*3 = 16 points. Order: corners (bottom-left, bottom-right, top-right, top-left),
    then edge points.
    """
    s = cell_size
    corners = [
        (x0, y0),
        (x0 + s, y0),
        (x0 + s, y0 + s),
        (x0, y0 + s),
    ]
    # Edges: bottom, right, top, left (each 3 points at 0.25, 0.5, 0.75 along edge)
    t = np.linspace(0.25, 0.75, EDGE_POINTS)
    edges = []
    # bottom: (x0,y0) -> (x0+s,y0)
    edges.extend([(x0 + ti * s, y0) for ti in t])
    # right: (x0+s,y0) -> (x0+s,y0+s)
    edges.extend([(x0 + s, y0 + ti * s) for ti in t])
    # top: (x0+s,y0+s) -> (x0,y0+s)
    edges.extend([(x0 + (1 - ti) * s, y0 + s) for ti in t])
    # left: (x0,y0+s) -> (x0,y0)
    edges.extend([(x0, y0 + (1 - ti) * s) for ti in t])
    return corners + edges


def _min_cell_z_over_points(
    points: List[Tuple[float, float]],
    buildings: gpd.GeoDataFrame,
    clearance_m: float,
    sindex: Optional[Any] = None,
) -> float:
    """Minimum z such that every point is at least clearance_m above the surface below."""
    z_candidates = [
        height_below_at_point(px, py, buildings, sindex) + clearance_m
        for (px, py) in points
    ]
    return max(z_candidates) if z_candidates else clearance_m


def compute_cell_z(
    x0: float,
    y0: float,
    cell_size: float,
    buildings: gpd.GeoDataFrame,
    clearance_m: float,
    sindex: Optional[Any] = None,
) -> float:
    """
    Required flying height for a cell so that all 16 sample points are at least
    clearance_m above the building/ground below.
    """
    points = cell_sample_points(x0, y0, cell_size)
    return _min_cell_z_over_points(points, buildings, clearance_m, sindex)


def _method_a_base_height(
    buildings: gpd.GeoDataFrame,
    building_exceed_threshold: float,
) -> float:
    """
    Height h such that at most building_exceed_threshold (e.g. 0.02 = 2%) of
    buildings have height_m > h. So (1 - threshold) of buildings are at or below h.
    """
    if len(buildings) == 0:
        return 0.0
    heights = np.array([float(row.get("height_m", 0.0)) for _, row in buildings.iterrows()])
    heights.sort()
    # Percentile: (1 - threshold) of buildings below or at h
    pct = 100.0 * (1.0 - building_exceed_threshold)
    idx = max(0, min(len(heights) - 1, int(round((pct / 100.0) * (len(heights) - 1)))))
    return float(heights[idx])


def cell_intersects_building(
    x0: float,
    y0: float,
    cell_size: float,
    buildings: gpd.GeoDataFrame,
    sindex: Optional[Any] = None,
) -> bool:
    """True if any of the 16 sample points has a building below (height_below > 0)."""
    points = cell_sample_points(x0, y0, cell_size)
    for (px, py) in points:
        if height_below_at_point(px, py, buildings, sindex) > 0:
            return True
    return False


def compute_l0_node_heights(
    buildings: gpd.GeoDataFrame,
    G: Any,
    meta: Dict[str, Any],
    cx: float,
    cy: float,
    radius_m: float,
    clearance_m: float,
    building_exceed_threshold: float = 0.02,
    smooth_iterations: int = 3,
    smoothstep_reshape: bool = True,
    plateau_soften_iterations: int = 10,
    plateau_soften_alpha: float = 0.35,
    plateau_soften_radius: int = 2,
) -> float:
    """
    Method A: Base height so at most building_exceed_threshold (e.g. 2%) of buildings
    exceed it; base_z = base_height + clearance_m. For cells still intersected by a
    building, corners + 3 pts per edge each >= clearance_m above building; else use base_z.
    Then smooth, optional smoothstep reshape, then optional plateau softening so tops
    become curved (dome) instead of flat; never go below required.
    Returns max z over all L0 nodes (for upper layers).
    """
    xmin = meta["xmin"]
    ymin = meta["ymin"]
    spacing = meta["grid_spacing"]
    sindex = getattr(buildings, "sindex", None)

    # Base height: raise until only building_exceed_threshold of buildings exceed it
    base_h = _method_a_base_height(buildings, building_exceed_threshold)
    base_z = base_h + clearance_m

    cell_heights: Dict[Tuple[int, int], float] = {}
    for nid in G.nodes:
        ix, iy = nid
        x0 = xmin + ix * spacing
        y0 = ymin + iy * spacing
        if cell_intersects_building(x0, y0, spacing, buildings, sindex):
            z_cell = max(base_z, compute_cell_z(x0, y0, spacing, buildings, clearance_m, sindex))
        else:
            z_cell = base_z
        cell_heights[(ix, iy)] = z_cell

    # Node z = max of all cells that have this node as a corner
    # Node (ix, iy) is corner for cells (ix, iy), (ix-1, iy), (ix, iy-1), (ix-1, iy-1)
    for nid in G.nodes:
        ix, iy = nid
        candidates = [
            cell_heights.get((ix, iy)),
            cell_heights.get((ix - 1, iy)),
            cell_heights.get((ix, iy - 1)),
            cell_heights.get((ix - 1, iy - 1)),
        ]
        z_node = max((c for c in candidates if c is not None), default=clearance_m)
        G.nodes[nid]["z"] = z_node

    # Smooth bumps: neighbors rise toward high points so the layer has a gradual slope
    required = {n: G.nodes[n]["z"] for n in G.nodes}
    if smooth_iterations > 0:
        _smooth_l0_heights(G, smooth_iterations)
    # Optional S-curve reshape: gradient increases then decreases (smooth roll-off at top and bottom)
    if smoothstep_reshape:
        _smoothstep_reshape(G, required)
    # Optional plateau softening: allow z to move toward local average (can go down) so the
    # boundary between flat tops and slopes is less abrupt; never go below required
    if plateau_soften_iterations > 0 and plateau_soften_alpha > 0:
        _plateau_soften(G, required, plateau_soften_iterations, plateau_soften_alpha, plateau_soften_radius)

    max_z = max((G.nodes[n]["z"] for n in G.nodes), default=0.0)
    return max_z


def _smoothstep(t: float) -> float:
    """Hermite smoothstep: 0 at t=0, 1 at t=1, zero derivative at both ends (S-curve)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _smoothstep_reshape(G: Any, required: Dict[Tuple[int, int], float]) -> None:
    """
    Reshape the transition so the gradient increases then decreases (B in the sketch).
    Within each node's local neighborhood [z_lo, z_hi], remap height using smoothstep
    so the slope is gentle at the bottom and top and steeper in the middle.
    """
    if G.number_of_nodes() == 0:
        return
    n_nodes = list(G.nodes)
    eps = 1e-6
    new_z = {}
    for n in n_nodes:
        z_self = G.nodes[n]["z"]
        neighbors = list(G.neighbors(n))
        if not neighbors:
            new_z[n] = z_self
            continue
        z_lo = min(z_self, min(G.nodes[u]["z"] for u in neighbors))
        z_hi = max(z_self, max(G.nodes[u]["z"] for u in neighbors))
        if z_hi - z_lo < eps:
            new_z[n] = z_self
            continue
        t = (z_self - z_lo) / (z_hi - z_lo)
        z_new = z_lo + (z_hi - z_lo) * _smoothstep(t)
        new_z[n] = max(required[n], z_new)
    for n in n_nodes:
        G.nodes[n]["z"] = new_z[n]


def _khop_neighbors(G: Any, n: Any, radius: int) -> List[Any]:
    """Return set of nodes within `radius` hops of n (including n). radius=1 = self + neighbors."""
    if radius <= 0:
        return [n]
    neighborhood = {n}
    frontier = {n}
    for _ in range(radius):
        next_frontier = set()
        for u in frontier:
            for v in G.neighbors(u):
                if v not in neighborhood:
                    neighborhood.add(v)
                    next_frontier.add(v)
        frontier = next_frontier
    return list(neighborhood)


def _plateau_soften(
    G: Any,
    required: Dict[Tuple[int, int], float],
    num_iterations: int,
    alpha: float,
    neighborhood_radius: int = 2,
) -> None:
    """
    Let mound tops become curved (dome) instead of flat by blending each node toward
    the average over a *wider* neighborhood (radius > 1). Interior plateau nodes then
    see lower slope nodes in that neighborhood and get pulled down; never below required.
    z_new = max(required, (1-alpha)*z_self + alpha*local_avg). Larger radius = more curved tops.
    """
    if num_iterations <= 0 or alpha <= 0 or G.number_of_nodes() == 0:
        return
    radius = max(1, neighborhood_radius)
    n_nodes = list(G.nodes)
    for _ in range(num_iterations):
        new_z = {}
        for n in n_nodes:
            z_self = G.nodes[n]["z"]
            neighborhood = _khop_neighbors(G, n, radius)
            if len(neighborhood) <= 1:
                new_z[n] = z_self
                continue
            local_avg = sum(G.nodes[u]["z"] for u in neighborhood) / len(neighborhood)
            z_blend = (1.0 - alpha) * z_self + alpha * local_avg
            new_z[n] = max(required[n], z_blend)
        for n in n_nodes:
            G.nodes[n]["z"] = new_z[n]


def _smooth_l0_heights(G: Any, num_iterations: int) -> None:
    """
    Smooth L0 node heights in-place so bumps create a gradual rise. Each node's z
    is set to max(required_min, average of self and graph neighbors); we never go
    below the current z (safety). Multiple iterations let the high values propagate.
    """
    if num_iterations <= 0 or G.number_of_nodes() == 0:
        return
    # Required minimum per node (current z before smoothing)
    required = {n: G.nodes[n]["z"] for n in G.nodes}
    n_nodes = list(G.nodes)
    for _ in range(num_iterations):
        new_z = {}
        for n in n_nodes:
            z_self = G.nodes[n]["z"]
            neighbors = list(G.neighbors(n))
            if not neighbors:
                new_z[n] = z_self
                continue
            z_avg = (z_self + sum(G.nodes[u]["z"] for u in neighbors)) / (1 + len(neighbors))
            new_z[n] = max(required[n], z_avg)
        for n in n_nodes:
            G.nodes[n]["z"] = new_z[n]


def get_l0_max_z(G: Any) -> float:
    """Return the maximum z over all nodes in the L0 graph."""
    if G is None or G.number_of_nodes() == 0:
        return 0.0
    return max(G.nodes[n]["z"] for n in G.nodes)
