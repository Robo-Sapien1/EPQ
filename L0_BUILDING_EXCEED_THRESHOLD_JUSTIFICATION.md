## Justifying `L0_BUILDING_EXCEED_THRESHOLD` (the “% of buildings above the base L0 height”)

### What the parameter actually does (in this codebase)

In `dash_scene_builder.py` and `l0_preview.py` we set:

- `L0_BUILDING_EXCEED_THRESHOLD = 0.02` (2%)

`l0_height.compute_l0_node_heights()` implements **Method A**:

1) Compute a **global base building height** \(h_\text{base}\) from the building dataset such that at most a fraction \(\epsilon\) of buildings exceed it.
2) Set the default L0 blanket altitude to:

\[
z_\text{base} = h_\text{base} + \text{clearance}
\]

3) For the (usually small) set of **cells that still intersect buildings**, locally raise those cells to satisfy clearance using 16 sample points (corners + edge samples).

So \(\epsilon\) is **not a “safety override”**. Safety is enforced per-cell via the local sampling step; \(\epsilon\) is a **robustness / efficiency knob** that prevents rare tall buildings (or data artifacts) from forcing the entire city-wide base surface to be very high.

### Why use a percentile / exceedance fraction at all?

Urban building height distributions are typically **heavy-tailed**: most buildings are relatively low, while a small number of towers are much taller. If you set the base height equal to the maximum, you make *every* route pay an energy/altitude penalty because of a tiny number of outliers.

Using a high percentile is a standard way to define a **reference height** while controlling sensitivity to outliers.

- In statistics, trimming/winsorization workflows often pick thresholds by specifying a **tail proportion** (e.g., trim the top \(p\) of values), and compute the corresponding **quantile threshold**. SciPy’s outlier tutorial describes identifying thresholds via quantiles corresponding to “a certain percentage of the data in each tail”.  
  Source: SciPy “Trimming and winsorization transition guide”. `https://docs.scipy.org/doc/scipy/tutorial/stats/outliers.html`

- In large-scale 3D building modelling from point clouds, it is common to define building heights using **percentiles** of the point height distribution (e.g., roof-95, roof-99) because “height reference” choices materially change results and percentiles provide controlled, interpretable options.  
  Source: Dukai et al. (2019), “A multi-height LoD1 model of all buildings in the Netherlands”, ISPRS Annals. `https://isprs-annals.copernicus.org/articles/IV-4-W8/51/2019/isprs-annals-IV-4-W8-51-2019.pdf`

- In NASA’s spaceborne LiDAR vegetation products, a very common “robust near-maximum” height metric is the **98th percentile** (RH98): the height at which the **98th percentile** of returned energy is reached relative to the ground. This is used explicitly as a measure of canopy height/structure while avoiding the sensitivity of the absolute maximum to sparse outliers.  
  Source: ORNL DAAC user guide “Global Vegetation Height Metrics from GEDI and ICESat2” (revision 2024‑10‑02). `https://daac.ornl.gov/VEGETATION/guides/GEDI_ICESAT2_Global_Veg_Height.html`

This project applies the same idea one level up: instead of percentiles **within** a building footprint, we use a percentile **across** the city’s building set to define a global baseline.

### Why 2% specifically?

Let \(\epsilon =\) `L0_BUILDING_EXCEED_THRESHOLD`. The code selects:

\[
h_\text{base} = Q_{1-\epsilon}(\{\text{building heights}\})
\]

So:

- \(\epsilon = 0.02\) means \(h_\text{base}\) is approximately the **98th percentile** building height.
- About **98%** of buildings are at or below the baseline; only the tallest ~2% are treated as “exceptions”.

This choice is defensible as a balanced point between two competing costs, and it aligns with the common use of a 98th-percentile “robust top height” metric in remote sensing (RH98).

1) **If \(\epsilon\) is too small** (e.g., 0.1%): the baseline height approaches the maximum and becomes dominated by rare towers, increasing:
   - the L0 blanket height everywhere,
   - the altitude of every upper layer (because upper layers are set from max(L0)),
   - energy/time for most routes.

2) **If \(\epsilon\) is too large** (e.g., 10%): many more buildings exceed the baseline, increasing:
   - the number/extent of local “mounds” that must be raised,
   - the roughness/complexity of the L0 surface (even after smoothing),
   - the chance of steep gradients that are undesirable for routing/vertical transitions.

The 2% exceedance target is a compact, explainable compromise:

- **Small enough** that the baseline clears “almost all” buildings and keeps the L0 surface mostly flat.
- **Large enough** to be robust to rare very-tall structures and to height-data artifacts (e.g., missing tags, mis-entered heights), which would otherwise inflate the baseline.

In other words, \(2\%\) is a practical “outlier budget” for the base layer while preserving safety via the per-cell clearance enforcement.

### Final chosen value

**Final: keep `building_exceed_threshold = 0.02` (2%).**

Reason: it corresponds to a **P98** baseline (robust to rare tall outliers and data artifacts), and there is strong precedent for 98th-percentile height metrics being used as robust “top height” descriptors (RH98) rather than raw maxima. The per-cell raise step still enforces clearance, so \(\epsilon\) is about *efficiency + robustness*, not safety.

### How to defend it in writing (what you can literally say)

You can write something like:

> The base L0 height is chosen as the 98th percentile of building heights (i.e., allowing 2% exceedance) to make the baseline robust to rare tall outliers while still clearing the overwhelming majority of buildings. The remaining tall buildings are handled locally by raising only those L0 cells that actually intersect building footprints, so the percentile choice affects efficiency and smoothness rather than safety.

### Sensitivity (what to mention if challenged)

If you need to show you didn’t pick 2% arbitrarily, the strongest EPQ-style argument is a **sensitivity sweep**:

- Compare \(\epsilon \in \{0.01, 0.02, 0.05\}\) and report:
  - resulting \(h_\text{base}\),
  - max(L0) after local bumps,
  - fraction of cells raised above base,
  - qualitative smoothness (via `l0_preview.py`).

Even if you don’t run the sweep, you can justify that 2% is intended to be “small but not vanishing,” and the codebase keeps it configurable to support such sensitivity analysis.

