"""
layers_with_divisions.py  --  Hierarchical Spatial Grid for Urban Drone Delivery

==============================================================================
MATHEMATICAL FOUNDATION
==============================================================================

We construct a recursive 3-D spatial index (conceptually an Octree in 3-D,
Quadtree in the horizontal plane) that partitions a circular depot-service
area into cells at multiple resolutions.

CONCRETE VALUES (derived from fleet_specs.py + depot_solution_for_overlay.json)
--------------------------------------------------------------------------------
  H&B drone enclosure          : 2.5 x 2.0 x 1.2 m
  Atomic unit  u               : max(L,W) + 2*padding = ~3.1 m
    (AirMatrix cell side: largest drone footprint plus lateral padding)
  L0 cell size                 : 4 * u = ~12.5 m
  Depot service radius R       : 550.0 m   (from depot model JSON)
  Bounding box side            : next_pow2(ceil(2*R / L0)) * L0
    = next_pow2(ceil(1100 / 15)) * 15
    = next_pow2(74) * 15
    = 128 * 15 = 1920.0 m

Key Design Decisions & Justifications
--------------------------------------

1. ATOMIC UNIT  u  (AirMatrix cell)
   ~~~~~~~~~~~~~~~~~~~~~~~~
   The AirMatrix cell side length that fits exactly 1 H&B drone (largest tier)
   plus a lateral padding margin to prevent overlap with a drone in an adjacent
   cell due to small positioning errors.

   Why add padding at all?
     If a drone is commanded to the centre of a cell, its actual position
     will deviate around that centre (sensor noise, control error). Padding
     ensures that two drones in adjacent cells do not overlap in worst-case
     lateral error.

   Why not smaller?
     u is the physical limit for discrete occupancy: any finer grid would
     represent sub-drone precision, which is not meaningful for packing.

2. BOTTOM LAYER  L0 = 4u  (a 4x4 "bucket" of 16 atomic units)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   Why NOT atomic?
     A 550 m radius at atomic resolution would produce tens of thousands of cells
     (just for L0).  A 4x4 bucket groups 16 atoms into one graph node,
     cutting node count to ~4,200.  This is the "funnel" for terminal
     approach: a drone is routed through coarse layers to the correct
     L0 bucket, then does fine maneuvering within the 4x4 at the
     atomic level (which is NOT part of the pathfinding graph -- it is
     handled by local flight control).

   Why 4 and not 2 or 8?
     - k=2: L0 = 7.5 m.  Only 4 atoms per bucket, minimal graph
       reduction (4x vs 16x).  Node count is still ~17,000.
     - k=4: L0 is a practical sweet spot: 16 atoms per bucket, 16x
       reduction.  This puts L0 in the ~10–15 m scale, which is within
       typical GNSS uncertainty margins, so much finer graph resolution
       adds little routing benefit.
     - k=8: L0 = 30.0 m.  Too coarse for terminal approach -- a drone
       could be told to "go to this 30 m cell" but be 15 m off target.
     - 4 is a power of 2, enabling bitwise index arithmetic.

3. TOP LAYER  L_max  --  covers the entire depot service area
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   The depot radius R = 550 m defines a circle.  We embed it in a square
   bounding box of side = next_power_of_2(ceil(2R / L0)) * L0.

   Why round up to power of 2?
     If cells_per_axis at L0 = 2^p, then every 2x or 4x coarsening gives
     integer cells_per_axis at every layer, and the tree terminates cleanly
     at a single root cell.  This is the fundamental invariant of quadtree
     subdivision.

   With R = 550 m:
     min_side    = 2 * 550 = 1100 m
     n_l0_raw    = ceil(1100 / 15) = 74
     n_l0        = next_pow2(74) = 128 = 2^7
     actual_side = 128 * 15 = 1920.0 m
     overshoot   = (1920 - 1100) / 1100 = 74.5%

   The 74.5% overshoot means the bounding box is larger than the depot
   circle.  This is necessary and acceptable: the sparse flag ensures we
   only instantiate cells whose centres fall inside the 550 m circle, so
   the "wasted" corners of the square consume zero memory.

4. POWER-OF-2 DIVISIONS  k in {2, 4}
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   Each parent cell is subdivided into k x k children.
   Using k = 2 (quadtree) or k = 4 (16-tree), both powers of 2:

     Cell index from world coords:
       ix = int((x - origin_x) / cell_size)

     For power-of-2 cell sizes this is a right-shift on the integer-
     scaled coordinate:
       ix = (x_int - origin_int) >> shift_bits

     Parent-child traversal:
       parent_ix = child_ix >> log2(k)       [1 or 2 bit shift]
       parent_morton = child_morton >> (2 * log2(k))   [2 or 4 bit shift]

     Morton (Z-order) codes give O(1) encode/decode via bit interleaving,
     making get_cell(x, y) O(1) per layer.

5. COST FUNCTION
   ~~~~~~~~~~~~~
   Total_Cost = alpha * N  +  beta * epsilon

   where:
     N       = total cells across all layers
     epsilon = L0_cell / 2  (worst-case heuristic displacement)
     alpha   = cost per node (RAM + A* expansion overhead)
     beta    = cost of heuristic inaccuracy (path quality loss)

   The optimizer sweeps candidate (k, prune_level) combinations and picks
   the one that minimises Total_Cost.

==============================================================================
"""

from __future__ import annotations

import math
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ====================================================================== #
#                        MORTON CODE UTILITIES                            #
#  Z-order / Lebesgue curve -- O(1) encode/decode via bit interleaving   #
# ====================================================================== #


def _spread_bits_2d(v: int) -> int:
    """
    Spread the lower 16 bits of *v* so each bit is separated by one
    zero bit.  Used to build a 2-D Morton code by interleaving x and y.

    Algorithm: "magic bits" (Henry S. Warren, *Hacker's Delight*).
    Successively spread bits apart by doubling gaps.

    Why Morton codes?
      They turn a 2-D grid index (ix, iy) into a single integer that
      preserves spatial locality.  Nearby cells in 2-D are nearby in
      1-D Morton order, which improves cache coherence and enables O(1)
      parent/child navigation (parent = code >> 2 for k=2).
    """
    v &= 0x0000FFFF
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    v = (v | (v << 1)) & 0x55555555
    return v


def _compact_bits_2d(v: int) -> int:
    """Inverse of _spread_bits_2d -- extract every other bit back to a
    contiguous integer.  Decodes one axis from a 2-D Morton code."""
    v &= 0x55555555
    v = (v | (v >> 1)) & 0x33333333
    v = (v | (v >> 2)) & 0x0F0F0F0F
    v = (v | (v >> 4)) & 0x00FF00FF
    v = (v | (v >> 8)) & 0x0000FFFF
    return v


def morton_encode_2d(ix: int, iy: int) -> int:
    """Encode 2-D grid indices into a single Morton (Z-order) code.

    morton = ...y2 x2 y1 x1 y0 x0   (bits interleaved)

    O(1) encode.  Parent lookup = morton >> 2 (for k=2 quadtree).
    Supports ix, iy in [0, 65535] -> 32-bit Morton code.
    """
    return _spread_bits_2d(ix) | (_spread_bits_2d(iy) << 1)


def morton_decode_2d(code: int) -> Tuple[int, int]:
    """Decode a 2-D Morton code back to (ix, iy).  O(1) bitwise."""
    ix = _compact_bits_2d(code)
    iy = _compact_bits_2d(code >> 1)
    return ix, iy


# ====================================================================== #
#                            CELL CLASS                                   #
# ====================================================================== #


@dataclass(slots=True)
class Cell:
    """
    One cell in the hierarchical grid.

    Attributes
    ----------
    ix, iy : int
        Column / row index within this cell's layer.
    layer_index : int
        Which layer this cell belongs to (0 = bottom / finest).
    morton : int
        Morton (Z-order) code encoding (ix, iy).  Used as a fast spatial
        hash and for O(1) parent-child navigation.
    x_min, y_min : float
        South-west corner of this cell in world coordinates (metres).
    cell_size : float
        Side length of this cell (metres).
    z_base : float
        Base altitude for this cell (metres above datum).
    metadata : dict
        Extensible slot (obstacle flags, demand weight, height-map, etc.).
    """
    ix: int
    iy: int
    layer_index: int
    morton: int
    x_min: float
    y_min: float
    cell_size: float
    z_base: float = 0.0
    metadata: Dict = field(default_factory=dict)

    @property
    def x_max(self) -> float:
        return self.x_min + self.cell_size

    @property
    def y_max(self) -> float:
        return self.y_min + self.cell_size

    @property
    def centre(self) -> Tuple[float, float]:
        half = self.cell_size * 0.5
        return (self.x_min + half, self.y_min + half)

    def contains(self, x: float, y: float) -> bool:
        """True if world point (x, y) lies inside this cell (half-open)."""
        return (self.x_min <= x < self.x_max) and (self.y_min <= y < self.y_max)

    def parent_morton(self, division_factor: int = 2) -> int:
        """
        Morton code of the parent cell in the next coarser layer.

        For k=2 (quadtree):   parent_morton = child_morton >> 2
        For k=4 (16-tree):    parent_morton = child_morton >> 4

        General: parent_morton = child_morton >> (2 * log2(k))

        This works because Morton codes interleave 2 axes, so each tree
        level occupies 2*log2(k) bits.  A right-shift is a single CPU
        instruction -- O(1).
        """
        bits_per_level = 2 * int(math.log2(division_factor))
        return self.morton >> bits_per_level


# ====================================================================== #
#                            LAYER CLASS                                  #
# ====================================================================== #


@dataclass
class Layer:
    """
    One resolution level of the hierarchical grid.

    Attributes
    ----------
    index : int
        Layer number (0 = bottom / finest).
    cell_size : float
        Side length of each cell at this layer (metres).
    cells_per_axis : int
        Number of cells along each axis of the bounding box.
    z_base : float
        Default altitude for cells at this layer (metres).
    cells : dict[int, Cell]
        Morton code -> Cell.  Hash map gives O(1) lookup by Morton code.
    origin_x, origin_y : float
        SW corner of the bounding box in world coordinates.
    is_sparse : bool
        If True, only cells inside the depot circle are instantiated.
        Saves ~21% memory (circle = pi/4 ~ 78.5% of bounding square).
    """
    index: int
    cell_size: float
    cells_per_axis: int
    z_base: float
    origin_x: float
    origin_y: float
    is_sparse: bool = True
    cells: Dict[int, Cell] = field(default_factory=dict, repr=False)

    @property
    def total_cells(self) -> int:
        return len(self.cells)

    @property
    def dense_cell_count(self) -> int:
        """How many cells if fully dense (square grid)."""
        return self.cells_per_axis * self.cells_per_axis

    def get_cell_by_index(self, ix: int, iy: int) -> Optional[Cell]:
        """Look up a cell by grid index.  O(1) via Morton hash."""
        code = morton_encode_2d(ix, iy)
        return self.cells.get(code)

    def get_cell_at_world(self, x: float, y: float) -> Optional[Cell]:
        """
        Convert world coordinates -> cell index -> Morton lookup.

        Complexity: O(1)  (one division + one hash lookup).

        For power-of-2 cell sizes relative to the atomic unit, this
        division can be replaced by a bit-shift:
            ix = (x_int - origin_int) >> shift_bits
        We keep float division here for generality; the bit-shift
        optimisation is documented for C/Cython hot paths.
        """
        if x < self.origin_x or y < self.origin_y:
            return None
        ix = int((x - self.origin_x) / self.cell_size)
        iy = int((y - self.origin_y) / self.cell_size)
        if ix < 0 or ix >= self.cells_per_axis or iy < 0 or iy >= self.cells_per_axis:
            return None
        return self.get_cell_by_index(ix, iy)

    def build_cells(
        self,
        centre_x: float,
        centre_y: float,
        radius: float,
    ) -> None:
        """
        Populate the cell dict.  If is_sparse, only cells whose centre
        falls within the depot circle are created.

        Sparse rationale:
          Bounding box area = side^2.  Inscribed circle area = pi*R^2.
          Ratio = pi*R^2 / (2R)^2 = pi/4 ~ 0.785.
          So ~21.5% of dense cells fall entirely outside the service area.
          Skipping them saves memory proportionally and reduces graph size.
        """
        r_sq = radius * radius
        for iy in range(self.cells_per_axis):
            cy = self.origin_y + (iy + 0.5) * self.cell_size
            dy = cy - centre_y
            dy_sq = dy * dy
            for ix in range(self.cells_per_axis):
                cx = self.origin_x + (ix + 0.5) * self.cell_size
                dx = cx - centre_x
                if self.is_sparse and (dx * dx + dy_sq) > r_sq:
                    continue
                code = morton_encode_2d(ix, iy)
                self.cells[code] = Cell(
                    ix=ix,
                    iy=iy,
                    layer_index=self.index,
                    morton=code,
                    x_min=self.origin_x + ix * self.cell_size,
                    y_min=self.origin_y + iy * self.cell_size,
                    cell_size=self.cell_size,
                    z_base=self.z_base,
                )


# ====================================================================== #
#                     LAYER PLAN (backward compat)                        #
# ====================================================================== #


@dataclass(frozen=True)
class LayerPlan:
    """
    Computed plan describing the hierarchy from L0 to L_max.

    Backward-compatible with the old layers_with_divisions.py API so
    dash_scene_builder.py can consume .layer_sizes, .divisions, .k.
    """
    k: int                           # number of inter-layer divisions
    divisions: List[int]             # per-step division factor (powers of 2)
    product: float                   # product of all divisions
    target_F: float                  # bbox_side / L0_cell
    achieved_R: float                # L0_cell * product (actual bbox side)
    log_error: float                 # |ln(product / target_F)|
    rel_error: float                 # (achieved - target) / target
    layer_sizes: List[float]         # [L0, L1, ..., L_max] cell sizes


# ====================================================================== #
#                          GRID MANAGER                                   #
# ====================================================================== #


class GridManager:
    """
    Central manager for the hierarchical 3-D spatial grid.

    Owns the layer hierarchy, the cost-function optimizer, and
    O(1)/O(log N) lookup methods.

    Construction
    ------------
    >>> gm = GridManager.from_parameters(
    ...     depot_radius=550.0,
    ...     atomic_unit=3.1,
    ... )

    Or via the convenience function:
    >>> gm = setup_grid(depot_radius=550, atomic_unit_size=3.1)
    """

    def __init__(
        self,
        layers: List[Layer],
        depot_radius: float,
        centre_x: float,
        centre_y: float,
        atomic_unit: float,
        l0_cell_size: float,
        division_factors: List[int],
        layer_plan: LayerPlan,
    ) -> None:
        self.layers: List[Layer] = layers
        self.depot_radius: float = depot_radius
        self.centre_x: float = centre_x
        self.centre_y: float = centre_y
        self.atomic_unit: float = atomic_unit
        self.l0_cell_size: float = l0_cell_size
        self.division_factors: List[int] = division_factors
        self.layer_plan: LayerPlan = layer_plan

    # ------------------------------------------------------------------ #
    #  Factory                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_parameters(
        cls,
        depot_radius: float,
        centre_x: float = 0.0,
        centre_y: float = 0.0,
        atomic_unit: float = 3.1,
        l0_bucket_factor: int = 4,
        base_division: int = 2,
        alpha: float = 1.0,
        beta: float = 1000.0,
        base_z: float = 300.0,
        layer_spacing_z: float = 100.0,
        sparse: bool = True,
        full_box: bool = False,
        layer_altitudes: Optional[List[float]] = None,
        verbose: bool = True,
    ) -> "GridManager":
        """
        Build the full grid hierarchy from physical parameters.

        Parameters
        ----------
        depot_radius : float
            Service radius in metres (e.g. 550.0).
        centre_x, centre_y : float
            Centre of the service area (projected CRS, metres).
        atomic_unit : float
            AirMatrix cell side (metres).  Default ~3.1 m.
        l0_bucket_factor : int
            L0 cell = l0_bucket_factor * atomic_unit.  Default 4.
            Must be a power of 2.
        base_division : int
            Per-layer division (2 = quadtree, 4 = 16-tree).  Power of 2.
        alpha, beta : float
            Cost function weights (node count vs heuristic error).
        base_z : float
            Altitude of the L0 layer (metres).  Used only when
            *layer_altitudes* is None.
        layer_spacing_z : float
            Vertical spacing between layers (metres).  Used only when
            *layer_altitudes* is None.
        sparse : bool
            Skip cells outside the depot circle (circular clipping).
            Ignored when *full_box* is True.
        full_box : bool
            If True, **all** cells in the rectangular bounding box are
            populated (no circular clipping).  This makes the grid
            extend laterally over the entire 1920 m box so that any
            point in the simulation area maps to a valid grid node.
        layer_altitudes : list[float] | None
            Explicit altitudes (metres) for each layer, L0 first.
            Length must match the number of layers produced by the
            optimizer.  If None, heights fall back to
            ``base_z + i * layer_spacing_z``.
        verbose : bool
            Print the construction report.
        """
        # -- Validate --
        if depot_radius <= 0:
            raise ValueError("depot_radius must be positive.")
        if atomic_unit <= 0:
            raise ValueError("atomic_unit must be positive.")
        if l0_bucket_factor < 1 or (l0_bucket_factor & (l0_bucket_factor - 1)) != 0:
            raise ValueError("l0_bucket_factor must be a power of 2.")
        if base_division < 2 or (base_division & (base_division - 1)) != 0:
            raise ValueError("base_division must be a power of 2 >= 2.")

        l0_cell = float(l0_bucket_factor) * atomic_unit

        # -- Bounding box --
        # Side must be >= 2*R AND a power-of-2 multiple of l0_cell so the
        # quadtree subdivides cleanly at every level.
        min_side = 2.0 * depot_radius
        n_l0_raw = math.ceil(min_side / l0_cell)
        # Round up to next power of 2:
        #   This guarantees cells_per_axis = 2^p at L0, and every k-fold
        #   coarsening gives integer cells_per_axis = 2^p / k^i.
        n_l0 = _next_power_of_2(n_l0_raw)
        side = float(n_l0) * l0_cell

        origin_x = centre_x - side / 2.0
        origin_y = centre_y - side / 2.0

        if verbose:
            p = int(round(math.log2(n_l0)))
            overshoot = (side - min_side) / min_side * 100.0
            print(f"\n{'='*70}")
            print("  HIERARCHICAL SPATIAL GRID -- CONSTRUCTION REPORT")
            print(f"{'='*70}")
            print(f"  Depot radius R           : {depot_radius:.1f} m")
            print(f"  Atomic unit u            : {atomic_unit:.4f} m  "
                  f"(AirMatrix cell side)")
            print(f"  L0 cell = {l0_bucket_factor}u             : {l0_cell:.2f} m  "
                  f"({l0_bucket_factor}x{l0_bucket_factor} = "
                  f"{l0_bucket_factor**2} atoms per bucket)")
            print(f"  Min bbox side = 2R       : {min_side:.1f} m")
            print(f"  L0 cells/axis (raw)      : {n_l0_raw}")
            print(f"  L0 cells/axis (2^p)      : {n_l0}  (2^{p})")
            print(f"  Actual bbox side         : {side:.1f} m  "
                  f"(+{overshoot:.1f}% overshoot)")
            print(f"  Origin (SW corner)       : ({origin_x:.1f}, {origin_y:.1f})")
            if full_box:
                print(f"  Mode                     : FULL BOX (all cells populated)")

        # -- Optimise layers --
        opt = cls._calculate_optimal_divisions(
            n_l0_cells_per_axis=n_l0,
            l0_cell_size=l0_cell,
            depot_radius=depot_radius,
            base_division=base_division,
            alpha=alpha,
            beta=beta,
            verbose=verbose,
        )
        division_factors = opt["division_factors"]
        num_layers = opt["num_layers"]
        layer_sizes = opt["layer_sizes"]
        cells_per_axis_list = opt["cells_per_axis"]

        # -- LayerPlan (backward compat) --
        product = 1.0
        for d in division_factors:
            product *= d
        target_F = side / l0_cell
        achieved_side = l0_cell * product
        log_err = abs(math.log(product / target_F)) if target_F > 0 else 0.0
        rel_err = ((achieved_side - side) / side) if side > 0 else 0.0

        plan = LayerPlan(
            k=len(division_factors),
            divisions=division_factors,
            product=product,
            target_F=target_F,
            achieved_R=achieved_side,
            log_error=log_err,
            rel_error=rel_err,
            layer_sizes=layer_sizes,
        )

        # -- Resolve per-layer altitudes --
        if layer_altitudes is not None:
            if len(layer_altitudes) != num_layers:
                raise ValueError(
                    f"layer_altitudes has {len(layer_altitudes)} entries but "
                    f"the optimizer chose {num_layers} layers."
                )
            z_values = [float(z) for z in layer_altitudes]
        else:
            z_values = [base_z + i * layer_spacing_z for i in range(num_layers)]

        # -- Decide sparsity per layer --
        use_sparse = (not full_box) and sparse

        # -- Build layers, populate cells --
        layers: List[Layer] = []
        for i in range(num_layers):
            z_i = z_values[i]
            layer = Layer(
                index=i,
                cell_size=layer_sizes[i],
                cells_per_axis=cells_per_axis_list[i],
                z_base=z_i,
                origin_x=origin_x,
                origin_y=origin_y,
                is_sparse=use_sparse,
            )
            layer.build_cells(centre_x, centre_y, depot_radius)
            layers.append(layer)

        if verbose:
            print(f"\n  {'Layer':<8} {'Cell (m)':<12} {'Cells/axis':<14} "
                  f"{'Populated':<12} {'Dense':<12} {'Z (m)':<10}")
            print(f"  {'-'*68}")
            for layer in layers:
                print(f"  L{layer.index:<6} {layer.cell_size:<12.2f} "
                      f"{layer.cells_per_axis:<14} {layer.total_cells:<12} "
                      f"{layer.dense_cell_count:<12} {layer.z_base:<10.0f}")
            total_cells = sum(l.total_cells for l in layers)
            print(f"\n  Total populated cells : {total_cells:,}")
            print(f"  Layers               : {num_layers}")
            print(f"  Division factors     : {division_factors}")
            print(f"{'='*70}\n")

        return cls(
            layers=layers,
            depot_radius=depot_radius,
            centre_x=centre_x,
            centre_y=centre_y,
            atomic_unit=atomic_unit,
            l0_cell_size=l0_cell,
            division_factors=division_factors,
            layer_plan=plan,
        )

    # ------------------------------------------------------------------ #
    #  Cost-Function Optimizer                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _calculate_optimal_divisions(
        n_l0_cells_per_axis: int,
        l0_cell_size: float,
        depot_radius: float,
        base_division: int = 2,
        alpha: float = 1.0,
        beta: float = 1000.0,
        verbose: bool = True,
    ) -> dict:
        """
        Determine the optimal layer count and division factors to minimise:

            Total_Cost = alpha * N_total  +  beta * epsilon

        N_total = sum of populated cells across all layers
        epsilon = effective_L0 / 2   (worst-case heuristic displacement)

        The epsilon term is dominated by the bottom layer because that is
        where terminal routing happens.  Coarser layers handle long-range
        segments where absolute error matters less.

        However, more layers means more nodes and slower A* traversal.
        There is a diminishing-returns point: adding a finer bottom layer
        costs more in graph size than it saves in path quality.

        Candidates
        ~~~~~~~~~~
        For each k in {2, 4} and each "prune" level (0, 1, 2):
          - prune=0: full depth, L0 cell = l0_cell_size
          - prune=1: drop 1 level, effective L0 = l0_cell * k
          - prune=2: drop 2 levels, effective L0 = l0_cell * k^2

        Pruning increases heuristic error but dramatically reduces nodes.
        The cost function decides the right trade-off.

        Returns dict with: division_factors, num_layers, layer_sizes,
        cells_per_axis, total_cost, node_count, heuristic_error.
        """
        p = int(round(math.log2(n_l0_cells_per_axis)))
        assert 2 ** p == n_l0_cells_per_axis, (
            f"n_l0_cells_per_axis must be a power of 2, got {n_l0_cells_per_axis}"
        )

        candidate_k_values = [2, 4]
        best: Optional[dict] = None
        best_cost = float("inf")
        all_candidates: List[dict] = []

        for k in candidate_k_values:
            log_k = int(round(math.log2(k)))  # 1 for k=2, 2 for k=4
            if p % log_k != 0:
                continue

            full_depth = p // log_k
            for prune in range(min(3, full_depth)):
                depth = full_depth - prune
                num_layers = depth + 1

                effective_l0 = l0_cell_size * (k ** prune)
                sizes: List[float] = []
                cpa: List[int] = []
                s = effective_l0
                n_axis = n_l0_cells_per_axis // (k ** prune)
                for i in range(num_layers):
                    sizes.append(s)
                    cpa.append(n_axis)
                    s *= k
                    n_axis = max(1, n_axis // k)

                # Node count (sparse: pi/4 ratio for circular area)
                circle_ratio = math.pi / 4.0
                node_count = sum(int(c * c * circle_ratio) for c in cpa)

                # Heuristic error = half the bottom cell size
                heuristic_error = effective_l0 / 2.0

                total_cost = alpha * node_count + beta * heuristic_error

                divisions = [k] * depth
                entry = {
                    "k": k,
                    "prune": prune,
                    "division_factors": divisions,
                    "num_layers": num_layers,
                    "layer_sizes": sizes,
                    "cells_per_axis": cpa,
                    "total_cost": total_cost,
                    "node_count": node_count,
                    "heuristic_error": heuristic_error,
                    "effective_l0": effective_l0,
                }
                all_candidates.append(entry)

                if total_cost < best_cost:
                    best_cost = total_cost
                    best = entry

        if verbose and all_candidates:
            print(f"\n  --- Cost-Function Optimizer (a={alpha}, b={beta}) ---")
            print(f"  {'k':<4} {'prune':<7} {'layers':<8} {'eff.L0(m)':<11} "
                  f"{'nodes':<12} {'e (m)':<9} {'cost':<14} {'note'}")
            print(f"  {'-'*75}")
            for c in sorted(all_candidates, key=lambda x: x["total_cost"]):
                marker = " <-- CHOSEN" if c is best else ""
                print(f"  {c['k']:<4} {c['prune']:<7} {c['num_layers']:<8} "
                      f"{c['effective_l0']:<11.2f} {c['node_count']:<12,} "
                      f"{c['heuristic_error']:<9.2f} "
                      f"{c['total_cost']:<14,.0f}{marker}")

        if best is None:
            raise RuntimeError(
                f"No valid layer config found (n_l0={n_l0_cells_per_axis}, "
                f"base_division={base_division})."
            )

        return best

    # ------------------------------------------------------------------ #
    #  Spatial Lookups                                                    #
    # ------------------------------------------------------------------ #

    def get_cell(self, x: float, y: float, layer: int = 0) -> Optional[Cell]:
        """
        Look up the cell containing world point (x, y) at the given layer.

        Complexity: O(1) -- one float division, one Morton encode, one
        hash lookup.

        This is the primary spatial query for drone routing.  During A*,
        each node expansion needs to know which cell a position falls in.
        """
        if layer < 0 or layer >= len(self.layers):
            return None
        return self.layers[layer].get_cell_at_world(x, y)

    def get_cell_stack(self, x: float, y: float) -> List[Optional[Cell]]:
        """
        Return the cell at (x, y) for every layer, L0 (finest) to L_max
        (coarsest).

        Useful for hierarchical A*: start on a coarse layer, refine near
        start/goal.

        Complexity: O(num_layers) -- one O(1) lookup per layer.
        """
        return [self.layers[i].get_cell_at_world(x, y)
                for i in range(len(self.layers))]

    def parent_of(self, cell: Cell) -> Optional[Cell]:
        """
        Navigate from child to parent in the next coarser layer.

        parent_ix = child_ix // k
        parent_iy = child_iy // k

        Then look up by Morton code.  Complexity: O(1).
        """
        child_layer = cell.layer_index
        parent_layer = child_layer + 1
        if parent_layer >= len(self.layers):
            return None
        k = self.division_factors[child_layer] if child_layer < len(self.division_factors) else 2
        parent_ix = cell.ix // k
        parent_iy = cell.iy // k
        return self.layers[parent_layer].get_cell_by_index(parent_ix, parent_iy)

    def children_of(self, cell: Cell) -> List[Cell]:
        """
        Return all children in the next finer layer belonging to this cell.

        Parent (px, py) with factor k has children at:
            ix in [px*k, px*k + k)
            iy in [py*k, py*k + k)

        Complexity: O(k^2) -- typically 4 or 16.
        """
        parent_layer = cell.layer_index
        child_layer = parent_layer - 1
        if child_layer < 0:
            return []
        k = self.division_factors[child_layer] if child_layer < len(self.division_factors) else 2
        children = []
        base_ix = cell.ix * k
        base_iy = cell.iy * k
        child_layer_obj = self.layers[child_layer]
        for diy in range(k):
            for dix in range(k):
                c = child_layer_obj.get_cell_by_index(base_ix + dix, base_iy + diy)
                if c is not None:
                    children.append(c)
        return children

    def neighbours_of(self, cell: Cell) -> List[Cell]:
        """
        Return the 4-connected (cardinal) neighbours within the same layer.

        4-connectivity (not 8) because:
          - Matches Manhattan heuristic (no inadmissible diagonals).
          - Edge count = 2N vs 4N -- halves memory.
          - Diagonals reachable via two cardinal steps; no loss of
            reachability.

        Complexity: O(1) -- 4 fixed lookups.
        """
        layer_obj = self.layers[cell.layer_index]
        ix, iy = cell.ix, cell.iy
        result = []
        for dix, diy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nix, niy = ix + dix, iy + diy
            if 0 <= nix < layer_obj.cells_per_axis and 0 <= niy < layer_obj.cells_per_axis:
                n = layer_obj.get_cell_by_index(nix, niy)
                if n is not None:
                    result.append(n)
        return result

    # ------------------------------------------------------------------ #
    #  Summary                                                            #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        lines = [
            f"GridManager  depot_radius={self.depot_radius:.0f}m  "
            f"atomic={self.atomic_unit:.2f}m  L0_cell={self.l0_cell_size:.2f}m",
            f"  Layers: {len(self.layers)}  Division factors: {self.division_factors}",
        ]
        total = 0
        for layer in self.layers:
            total += layer.total_cells
            lines.append(
                f"  L{layer.index}: cell={layer.cell_size:.2f}m  "
                f"cells/axis={layer.cells_per_axis}  "
                f"populated={layer.total_cells:,}  z={layer.z_base:.0f}m"
            )
        lines.append(f"  Total cells: {total:,}")
        return "\n".join(lines)

    def compute_layer_speeds_mps(
        self,
        drone_dims_lwh_m: Tuple[float, float, float],
        *,
        # High-level behaviour knob: 0 => prioritise raw speed; 1 => prioritise capacity/flow.
        traffic_intensity: float = 0.7,
        # Geometry-based "smaller cells => slower" rule: desired edge traversal times.
        edge_traversal_time_low_s: float = 6.0,
        edge_traversal_time_high_s: float = 2.0,
        # Physical cap (e.g. DJI Matrice 30 max horizontal speed ~23 m/s).
        speed_cap_mps: float = 23.0,
        # Optional hard capacity constraint per layer (max drones simultaneously on one edge).
        max_drones_on_edge: Optional[List[int]] = None,
        # CORRIDRONE parameters (shared across layers unless you later make them layer-dependent).
        response_time_s: float = 1.0,
        max_decel_mps2: float = 3.0,
        pos_uncertainty_m: float = 5.0,
        wind_speed_mps: float = 5.0,
    ) -> dict:
        """
        Compute a defensible speed schedule for layers L0..Lmax given the
        delivery-speed vs CORRIDRONE-capacity tradeoff.

        Key idea:
          CORRIDRONE implies a minimum headway (center spacing) of ~2*r(v).
          Along an edge, that yields an approximate steady-state flow capacity:

              f(v) ≈ v / (2*r(v))   [drones/s per edge]

          This has a finite optimum speed v* that maximises capacity. Under
          congestion, choosing speeds near v* improves *both* throughput and
          realised delivery time (less queuing), compared with "always go max".

        We blend between:
          - a geometry preference (smaller cells -> slower), and
          - the CORRIDRONE throughput-optimal speed.
        """
        from corridrone import (
            geofence_for_speed,
            speed_for_max_edge_flow,
            edge_flow_rate_drones_per_s,
            max_drones_on_edge_at_speed,
            max_speed_for_edge_occupancy,
        )

        if not (0.0 <= float(traffic_intensity) <= 1.0):
            raise ValueError("traffic_intensity must be in [0, 1].")
        n_layers = len(self.layers)
        if n_layers == 0:
            return {"speeds_mps": [], "details": []}

        cap = float(speed_cap_mps)
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError("speed_cap_mps must be finite and > 0.")

        if max_drones_on_edge is not None and len(max_drones_on_edge) != n_layers:
            raise ValueError(
                f"max_drones_on_edge must have length {n_layers} (one per layer) "
                f"or be None."
            )

        v_flow_opt = speed_for_max_edge_flow(
            drone_dims_lwh_m,
            response_time_s=response_time_s,
            max_decel_mps2=max_decel_mps2,
            pos_uncertainty_m=pos_uncertainty_m,
            wind_speed_mps=wind_speed_mps,
            speed_cap_mps=cap,
        )

        t_low = float(edge_traversal_time_low_s)
        t_high = float(edge_traversal_time_high_s)
        if not (math.isfinite(t_low) and t_low > 0 and math.isfinite(t_high) and t_high > 0):
            raise ValueError("edge_traversal_time_*_s must be finite and > 0.")

        speeds: List[float] = []
        details: List[dict] = []

        for i, layer in enumerate(self.layers):
            s = float(layer.cell_size)
            # Linearly interpolate desired traversal time from low->high layers.
            if n_layers == 1:
                t_edge = t_low
            else:
                frac = i / float(n_layers - 1)
                t_edge = t_low + frac * (t_high - t_low)

            v_geom = s / t_edge
            v_time_pref = min(cap, v_geom)

            # Under congestion, bias toward the CORRIDRONE throughput optimum.
            v_congested = min(v_time_pref, v_flow_opt)
            v = (1.0 - float(traffic_intensity)) * v_time_pref + float(traffic_intensity) * v_congested

            # Optional hard capacity constraint "N drones per edge".
            v_cap_by_N: Optional[float] = None
            if max_drones_on_edge is not None:
                N = int(max_drones_on_edge[i])
                vN = max_speed_for_edge_occupancy(
                    edge_length_m=s,
                    max_drones_on_edge=N,
                    drone_dims_lwh_m=drone_dims_lwh_m,
                    response_time_s=response_time_s,
                    max_decel_mps2=max_decel_mps2,
                    pos_uncertainty_m=pos_uncertainty_m,
                    wind_speed_mps=wind_speed_mps,
                )
                if math.isfinite(vN):
                    v_cap_by_N = float(vN)
                    v = min(v, v_cap_by_N)

            v = max(0.0, float(v))
            gf = geofence_for_speed(
                v,
                drone_dims_lwh_m,
                response_time_s=response_time_s,
                max_decel_mps2=max_decel_mps2,
                pos_uncertainty_m=pos_uncertainty_m,
                wind_speed_mps=wind_speed_mps,
            )
            flow = edge_flow_rate_drones_per_s(
                v,
                drone_dims_lwh_m,
                response_time_s=response_time_s,
                max_decel_mps2=max_decel_mps2,
                pos_uncertainty_m=pos_uncertainty_m,
                wind_speed_mps=wind_speed_mps,
            )
            n_on_edge = max_drones_on_edge_at_speed(
                s,
                v,
                drone_dims_lwh_m,
                response_time_s=response_time_s,
                max_decel_mps2=max_decel_mps2,
                pos_uncertainty_m=pos_uncertainty_m,
                wind_speed_mps=wind_speed_mps,
            )

            speeds.append(v)
            details.append(
                {
                    "layer": i,
                    "cell_size_m": s,
                    "speed_mps": v,
                    "speed_geom_pref_mps": v_time_pref,
                    "speed_flow_opt_mps": v_flow_opt,
                    "speed_cap_by_edge_occupancy_mps": v_cap_by_N,
                    "geofence_radius_m": float(gf.radius_m),
                    "edge_flow_drones_per_s": float(flow),
                    "approx_max_drones_on_edge": int(n_on_edge),
                    "edge_traversal_time_s": float(t_edge),
                }
            )

        return {"speeds_mps": speeds, "details": details}

    def __repr__(self) -> str:
        return (f"GridManager(layers={len(self.layers)}, "
                f"depot_r={self.depot_radius:.0f}m, "
                f"l0={self.l0_cell_size:.1f}m)")


# ====================================================================== #
#                     CONVENIENCE ENTRY POINTS                            #
# ====================================================================== #


def setup_grid(
    depot_radius: float,
    atomic_unit_size: float = 3.1,
    centre_x: float = 0.0,
    centre_y: float = 0.0,
    l0_bucket_factor: int = 4,
    base_division: int = 2,
    alpha: float = 1.0,
    beta: float = 1000.0,
    base_z: float = 300.0,
    layer_spacing_z: float = 100.0,
    sparse: bool = True,
    full_box: bool = False,
    layer_altitudes: Optional[List[float]] = None,
    verbose: bool = True,
) -> GridManager:
    """
    Build the full hierarchical spatial grid.

    Primary entry point.

    Parameters
    ----------
    depot_radius : float
        Depot service radius (metres).
    atomic_unit_size : float
        AirMatrix cell side (metres).  Default ~3.1 m.
    l0_bucket_factor : int
        L0 cell = l0_bucket_factor * atomic_unit.  Default 4.
    base_division : int
        Quadtree division factor (2 or 4).
    alpha, beta : float
        Cost function weights.
    full_box : bool
        If True, populate every cell in the rectangular bounding box
        (no circular clipping).
    layer_altitudes : list[float] | None
        Explicit per-layer altitudes (L0 first).
    """
    return GridManager.from_parameters(
        depot_radius=depot_radius,
        centre_x=centre_x,
        centre_y=centre_y,
        atomic_unit=atomic_unit_size,
        l0_bucket_factor=l0_bucket_factor,
        base_division=base_division,
        alpha=alpha,
        beta=beta,
        base_z=base_z,
        layer_spacing_z=layer_spacing_z,
        sparse=sparse,
        full_box=full_box,
        layer_altitudes=layer_altitudes,
        verbose=verbose,
    )


def compute_layer_plan(
    R_final: float,
    S0: float,
    verbose: bool = True,
) -> Optional[LayerPlan]:
    """
    Backward-compatible entry point for dash_scene_builder.py.

    Parameters (as called by the scene builder):
        R_final = depot service radius (from depot_solution_for_overlay.json "R")
        S0      = atomic unit size (from fleet_specs.get_atomic_unit_m())

    Internally computes L0 = 4 * S0, builds the power-of-2 bounding box,
    runs the optimizer, and returns a LayerPlan.
    """
    if R_final <= 0 or S0 <= 0:
        raise ValueError("R_final and S0 must be positive.")

    L0_BUCKET = 4
    l0_cell = L0_BUCKET * S0

    n_l0_raw = math.ceil((2.0 * R_final) / l0_cell)
    if n_l0_raw <= 1:
        return LayerPlan(
            k=0, divisions=[], product=1.0,
            target_F=1.0, achieved_R=l0_cell,
            log_error=0.0, rel_error=0.0,
            layer_sizes=[l0_cell],
        )

    n_l0 = _next_power_of_2(n_l0_raw)
    side = float(n_l0) * l0_cell

    opt = GridManager._calculate_optimal_divisions(
        n_l0_cells_per_axis=n_l0,
        l0_cell_size=l0_cell,
        depot_radius=R_final,
        base_division=2,
        alpha=1.0,
        beta=1000.0,
        verbose=verbose,
    )

    divisions = opt["division_factors"]
    product = 1.0
    for d in divisions:
        product *= d
    target_F = side / l0_cell
    achieved = l0_cell * product
    log_err = abs(math.log(product / target_F)) if target_F > 1e-12 else 0.0
    rel_err = (achieved - side) / side if side > 0 else 0.0

    return LayerPlan(
        k=len(divisions),
        divisions=divisions,
        product=product,
        target_F=target_F,
        achieved_R=achieved,
        log_error=log_err,
        rel_error=rel_err,
        layer_sizes=opt["layer_sizes"],
    )


# ====================================================================== #
#                        HELPER FUNCTIONS                                 #
# ====================================================================== #


def _next_power_of_2(n: int) -> int:
    """
    Smallest power of 2 >= n.  O(1) bit-twiddling, no loops.

    Why needed:
      cells_per_axis must be 2^p so every k-fold coarsening yields
      integer subdivision at every layer.
    """
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


# ====================================================================== #
#  CRUISE-LAYER SELECTION                                                 #
# ====================================================================== #


def select_cruise_layer(
    gm: GridManager,
    distance_m: float,
) -> int:
    """
    Choose the optimal cruise layer for a trip of *distance_m*.

    Research basis  (UTM / Urban Air Mobility literature)
    =====================================================

    1. **Stratified airspace** -- UTM proposals organise urban drone
       traffic into altitude-separated layers to reduce conflicts and
       noise exposure  (Sunil et al., 2015; NASA UTM ConOps v2).

    2. **Energy cost of climb** -- Multirotors consume substantially
       more power during vertical ascent than in horizontal cruise
       (Stolaroff et al., 2018 *Nature Energy*; Rodrigues et al., 2021
       *Transportation Research Part D*).  Power in hover/climb scales
       roughly with thrust = m*g, while cruise power benefits from
       translational lift.

    3. **Traffic separation** -- Higher, coarser layers carry fewer
       corridors, reducing potential conflicts.  Short trips that stay
       low keep the upper airspace clear for long-haul traffic
       ("routes get shorter as paths become more localised").

    4. **Energy-optimal break-even** -- Climbing to a higher layer is
       only worthwhile when the horizontal distance saved by cruising
       on a coarser, straighter corridor exceeds the extra climb
       energy.  The break-even distance scales with the altitude
       difference between layers.

    Heuristic
    ---------
    We model the break-even as:

        break_even(layer_i) = alpha * delta_z(layer_i)

    where delta_z is the altitude the drone must climb (layer_i z minus
    ground), and alpha is a dimensionless ratio ≈ 5-8 representing the
    cruise-to-climb energy ratio for a typical multirotor.

    Typical values (with L0=137m, spacing=100m):
        L1 (z≈237m): break-even ≈ 237 * 5 ≈ 1185m  -- but L1 is the
            minimum cruise layer (always used as default)
        L2 (z≈337m): break-even ≈ 337 * 5 ≈ 1685m
        L3 (z≈437m): break-even ≈ 437 * 5 ≈ 2185m

    These numbers are larger than the depot radius (550m), so within a
    single depot service area most trips stay on L1 or L2.  This is
    the correct physical behaviour: the grid hierarchy exists so A*
    can step down through layers, not so drones always fly at the top.

    The drone picks the **highest** layer whose break-even distance it
    exceeds.  For very short trips where even climbing from L0 to L1
    is not energy-efficient, the drone stays on L0 and navigates the
    fine grid directly to its target cell.
    """
    ALPHA = 5.0  # cruise-to-climb energy ratio (conservative estimate)

    n = len(gm.layers)
    if n <= 1:
        return 0

    max_layer = n - 1

    # Walk from highest layer down; pick the first whose energy
    # break-even the trip distance exceeds.
    for li in range(max_layer, 1, -1):
        climb_altitude = gm.layers[li].z_base  # height above ground
        break_even_m = ALPHA * climb_altitude
        if distance_m >= break_even_m:
            return li

    # Short/medium trips: compare L1 vs L0.
    # L1 adds an extra 100 m of climb above L0.  That extra climb is
    # only worthwhile if the trip is long enough to benefit from L1's
    # coarser grid (fewer turns, straighter path).
    # Break-even for climbing from L0 to L1 ≈ ALPHA * (z_L1 - z_L0).
    if n >= 2:
        extra_climb = gm.layers[1].z_base - gm.layers[0].z_base
        l1_break_even = ALPHA * extra_climb
        if distance_m >= l1_break_even:
            return 1

    # Very short trip: cruise directly on L0.  The drone only needs
    # to climb to the blanket height and navigate the fine grid to
    # its target cell.  This avoids wasting energy on an unnecessary
    # climb to L1.
    return 0


# ====================================================================== #
#  PATHFINDER -- Hierarchical flight-profile A*                           #
#                                                                          #
#  Flight profile:                                                         #
#    Step A  select cruise_layer from distance                             #
#    Step B  vertical climb (x_start, y_start, 0) -> (x_s, y_s, z_cruise) #
#    Step C  branch to nearest grid node on cruise layer                   #
#    Step D  hierarchical A* on grid, stepping down layers toward L0       #
#    Stop    at the centre of the target L0 cell                           #
# ====================================================================== #


class Pathfinder:
    """
    Generates drone flight paths through the hierarchical grid.

    The pathfinder is **modular**: it produces a path that ends at the
    centre of a target L0 cell.  No "final approach" or landing logic
    is included -- that will be added in a future update.

    Usage
    -----
    >>> pf = Pathfinder(grid_manager)
    >>> result = pf.get_path(start_xy=(x, y), target_xy=(tx, ty))
    >>> result["positions"]   # list of (x, y, z) waypoints
    """

    def __init__(self, gm: GridManager) -> None:
        self.gm = gm

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def get_path(
        self,
        start_xy: Tuple[float, float],
        target_xy: Tuple[float, float],
        step_m: float = 3.0,
        cruise_layer_override: Optional[int] = None,
    ) -> dict:
        """
        Compute a full flight path from a ground-level depot to a
        target L0 cell centre.

        Parameters
        ----------
        start_xy : (x, y)
            Depot position in world coordinates (launch point, z=0).
        target_xy : (x, y)
            Delivery target in world coordinates (will be snapped to
            the centre of the containing L0 cell).
        step_m : float
            Interpolation step for smooth waypoints.
        cruise_layer_override : int | None
            Force a specific cruise layer index.  If None, the layer
            is selected automatically from the target distance.

        Returns
        -------
        dict with keys:
            cruise_layer : int          -- chosen cruise layer index
            target_l0_cell : Cell       -- the L0 cell the path targets
            grid_path : list[Cell]      -- sequence of cells traversed by A*
            positions : list[(x,y,z)]   -- interpolated waypoints
            segments : dict             -- named sub-paths for debugging:
                "climb", "branch", "cruise"
        """
        gm = self.gm
        sx, sy = float(start_xy[0]), float(start_xy[1])
        tx, ty = float(target_xy[0]), float(target_xy[1])

        # -- Resolve target L0 cell --
        target_cell = gm.get_cell(tx, ty, layer=0)
        if target_cell is None:
            raise ValueError(
                f"Target ({tx:.1f}, {ty:.1f}) is outside the L0 grid."
            )
        target_centre = target_cell.centre
        target_z = target_cell.z_base

        # -- Step A: select cruise layer --
        dist = math.hypot(tx - sx, ty - sy)
        if cruise_layer_override is not None:
            cruise_idx = max(0, min(cruise_layer_override, len(gm.layers) - 1))
        else:
            cruise_idx = select_cruise_layer(gm, dist)

        cruise_layer = gm.layers[cruise_idx]
        cruise_z = cruise_layer.z_base

        # -- Step B: vertical climb --
        ground_pos = (sx, sy, 0.0)
        climb_top = (sx, sy, cruise_z)
        climb_segment = _interpolate_segment(ground_pos, climb_top, step_m)

        # -- Step C: branch to nearest grid node on cruise layer --
        entry_cell = cruise_layer.get_cell_at_world(sx, sy)
        if entry_cell is None:
            entry_cell = _nearest_cell_on_layer(cruise_layer, sx, sy)
        if entry_cell is None:
            raise ValueError(
                f"Cannot find an entry cell on cruise layer L{cruise_idx} "
                f"near ({sx:.1f}, {sy:.1f})."
            )
        entry_centre = entry_cell.centre
        entry_pos = (entry_centre[0], entry_centre[1], cruise_z)
        branch_segment = _interpolate_segment(climb_top, entry_pos, step_m)

        # -- Step D: hierarchical A* (cruise layer -> L0 target) --
        goal_cell_on_target_layer = target_cell
        grid_path = _hierarchical_astar(
            gm, entry_cell, goal_cell_on_target_layer, cruise_idx,
        )

        # Convert grid_path cells into (x, y, z) waypoints
        cruise_segment: List[Tuple[float, float, float]] = []
        if len(grid_path) >= 2:
            for i in range(len(grid_path) - 1):
                c0 = grid_path[i]
                c1 = grid_path[i + 1]
                p0 = (c0.centre[0], c0.centre[1], c0.z_base)
                p1 = (c1.centre[0], c1.centre[1], c1.z_base)
                cruise_segment.extend(_interpolate_segment(p0, p1, step_m))

        # Final point: exact centre of target L0 cell
        final_pos = (target_centre[0], target_centre[1], target_z)

        # Assemble full path
        positions: List[Tuple[float, float, float]] = [ground_pos]
        positions.extend(climb_segment)
        positions.extend(branch_segment)
        positions.extend(cruise_segment)
        # Ensure the very last point is the L0 cell centre
        if not positions or positions[-1] != final_pos:
            positions.append(final_pos)

        return {
            "cruise_layer": cruise_idx,
            "target_l0_cell": target_cell,
            "grid_path": grid_path,
            "positions": positions,
            "segments": {
                "climb": [ground_pos] + climb_segment,
                "branch": [climb_top] + branch_segment,
                "cruise": cruise_segment,
            },
        }


# ====================================================================== #
#  A* ON THE HIERARCHICAL CELL GRID                                       #
# ====================================================================== #


def _hierarchical_astar(
    gm: GridManager,
    start_cell: Cell,
    goal_cell: Cell,
    start_layer_idx: int,
) -> List[Cell]:
    """
    A* search across the hierarchical grid from *start_cell* (on layer
    *start_layer_idx*) down to *goal_cell* (on L0).

    Nodes are ``(layer_index, morton)`` tuples.  Edges are:
      - Horizontal: 4-connected neighbours within the same layer.
      - Vertical (down): parent -> child mapping when stepping to a
        finer layer.  A node on layer L can descend to any of its
        children on layer L-1 that is closer to the goal.

    The heuristic is 3-D Euclidean distance (admissible).

    Returns an ordered list of Cell objects from start to goal.
    """
    goal_cx, goal_cy = goal_cell.centre
    goal_z = goal_cell.z_base
    goal_layer = goal_cell.layer_index

    def _heuristic(cell: Cell) -> float:
        cx, cy = cell.centre
        return math.sqrt(
            (cx - goal_cx) ** 2 + (cy - goal_cy) ** 2 +
            (cell.z_base - goal_z) ** 2
        )

    def _edge_cost(c0: Cell, c1: Cell) -> float:
        cx0, cy0 = c0.centre
        cx1, cy1 = c1.centre
        return math.sqrt(
            (cx1 - cx0) ** 2 + (cy1 - cy0) ** 2 +
            (c1.z_base - c0.z_base) ** 2
        )

    # Key: (layer_index, morton)
    start_key = (start_cell.layer_index, start_cell.morton)
    goal_key = (goal_cell.layer_index, goal_cell.morton)

    # Cell lookup cache: key -> Cell
    cell_cache: Dict[Tuple[int, int], Cell] = {
        start_key: start_cell,
        goal_key: goal_cell,
    }

    def _get_cell(key: Tuple[int, int]) -> Optional[Cell]:
        if key in cell_cache:
            return cell_cache[key]
        li, mort = key
        if 0 <= li < len(gm.layers):
            c = gm.layers[li].cells.get(mort)
            if c is not None:
                cell_cache[key] = c
            return c
        return None

    # open_set is a min-heap of (f_score, counter, key)
    counter = 0
    open_heap: List[Tuple[float, int, Tuple[int, int]]] = []
    g_score: Dict[Tuple[int, int], float] = {start_key: 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    in_open: set = {start_key}

    h0 = _heuristic(start_cell)
    heapq.heappush(open_heap, (h0, counter, start_key))
    counter += 1

    while open_heap:
        f_curr, _, current_key = heapq.heappop(open_heap)

        if current_key not in in_open:
            continue
        in_open.discard(current_key)

        if current_key == goal_key:
            # Reconstruct path
            path_keys: List[Tuple[int, int]] = [goal_key]
            k = goal_key
            while k in came_from:
                k = came_from[k]
                path_keys.append(k)
            path_keys.reverse()
            return [cell_cache[pk] for pk in path_keys]

        current_cell = _get_cell(current_key)
        if current_cell is None:
            continue
        current_g = g_score[current_key]
        current_layer = current_cell.layer_index

        # --- Horizontal neighbours (same layer) ---
        for nb in gm.neighbours_of(current_cell):
            nb_key = (nb.layer_index, nb.morton)
            cost = _edge_cost(current_cell, nb)
            tentative_g = current_g + cost
            if tentative_g < g_score.get(nb_key, float("inf")):
                g_score[nb_key] = tentative_g
                came_from[nb_key] = current_key
                cell_cache[nb_key] = nb
                f = tentative_g + _heuristic(nb)
                heapq.heappush(open_heap, (f, counter, nb_key))
                counter += 1
                in_open.add(nb_key)

        # --- Vertical down: step to finer layer ---
        if current_layer > goal_layer:
            children = gm.children_of(current_cell)
            for ch in children:
                ch_key = (ch.layer_index, ch.morton)
                cost = _edge_cost(current_cell, ch)
                tentative_g = current_g + cost
                if tentative_g < g_score.get(ch_key, float("inf")):
                    g_score[ch_key] = tentative_g
                    came_from[ch_key] = current_key
                    cell_cache[ch_key] = ch
                    f = tentative_g + _heuristic(ch)
                    heapq.heappush(open_heap, (f, counter, ch_key))
                    counter += 1
                    in_open.add(ch_key)

        # --- Vertical up: step to coarser layer (only if below cruise) ---
        if current_layer < start_cell.layer_index:
            parent = gm.parent_of(current_cell)
            if parent is not None:
                p_key = (parent.layer_index, parent.morton)
                cost = _edge_cost(current_cell, parent)
                tentative_g = current_g + cost
                if tentative_g < g_score.get(p_key, float("inf")):
                    g_score[p_key] = tentative_g
                    came_from[p_key] = current_key
                    cell_cache[p_key] = parent
                    f = tentative_g + _heuristic(parent)
                    heapq.heappush(open_heap, (f, counter, p_key))
                    counter += 1
                    in_open.add(p_key)

    # No path found -- return direct descent as fallback
    return [start_cell, goal_cell]


# ====================================================================== #
#  INTERPOLATION & NEAREST-CELL HELPERS                                    #
# ====================================================================== #


def _interpolate_segment(
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    step_m: float,
) -> List[Tuple[float, float, float]]:
    """Linear interpolation between two 3-D points at *step_m* intervals."""
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    seg_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
    if seg_len < 1e-9:
        return []
    n_steps = max(1, int(seg_len / step_m))
    pts: List[Tuple[float, float, float]] = []
    for s in range(1, n_steps + 1):
        t = min(1.0, (s * step_m) / seg_len)
        pts.append((
            x0 + t * (x1 - x0),
            y0 + t * (y1 - y0),
            z0 + t * (z1 - z0),
        ))
    return pts


def _nearest_cell_on_layer(layer: Layer, x: float, y: float) -> Optional[Cell]:
    """Brute-force nearest cell on *layer* to world point (x, y)."""
    best: Optional[Cell] = None
    best_d2 = float("inf")
    for cell in layer.cells.values():
        cx, cy = cell.centre
        d2 = (cx - x) ** 2 + (cy - y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = cell
    return best


# ====================================================================== #
#                          INTERACTIVE MAIN                               #
# ====================================================================== #


if __name__ == "__main__":
    print("=" * 70)
    print("  Hierarchical Spatial Grid -- Interactive Builder")
    print("=" * 70)

    def _read_float(prompt: str, default: float) -> float:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        return float(raw)

    def _read_int(prompt: str, default: int) -> int:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        return int(raw)

    depot_r = _read_float("Depot radius (m)", 550.0)
    atomic = _read_float("Atomic unit size (m)", 3.1)
    bucket = _read_int("L0 bucket factor (power of 2)", 4)
    div = _read_int("Base division factor (2 or 4)", 2)
    a_val = _read_float("Cost weight alpha (node count)", 1.0)
    b_val = _read_float("Cost weight beta (heuristic error)", 1000.0)

    gm = setup_grid(
        depot_radius=depot_r,
        atomic_unit_size=atomic,
        l0_bucket_factor=bucket,
        base_division=div,
        alpha=a_val,
        beta=b_val,
    )

    print("\n" + gm.summary())

    # Quick lookup demo
    print("\n--- Lookup Demo ---")
    x_test = gm.centre_x + depot_r * 0.3
    y_test = gm.centre_y + depot_r * 0.3
    for i in range(len(gm.layers)):
        cell = gm.get_cell(x_test, y_test, layer=i)
        if cell:
            print(f"  L{i}: ({cell.ix}, {cell.iy})  morton={cell.morton}  "
                  f"centre=({cell.centre[0]:.1f}, {cell.centre[1]:.1f})  "
                  f"size={cell.cell_size:.2f}m")
        else:
            print(f"  L{i}: outside grid")
