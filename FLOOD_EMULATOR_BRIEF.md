# India Flood Depth Emulator — Project Brief

## What we're building

A tool that produces flood depth grids for **any return period, any year, any SSP** across
India at 30 m, without running HEC-RAS again.

We have exactly one hydraulic run: **RP100, 2030, India-wide, 30 m**, built as basin-wise
HEC-RAS models with RMSI basin ratios used to disaggregate rainfall over 24 h. That run is
the only calibration anchor and it will not be repeated.

The method is a **HAND-based stage emulator with volume-balance calibration**. Scenarios are
resolved by computing a runoff volume, inverting a terrain-derived stage–volume curve to get
water surface elevation, and differencing that against HAND. Depth generation for any
scenario becomes a raster subtraction: seconds for a district, minutes for India.

---

## Why not just scale depth by rainfall

Read this before proposing simplifications. The obvious approach is `k = d₁₀₀/P₁₀₀` per
pixel, then `d' = k·P'`. It was considered and rejected:

1. **The wet mask is frozen.** `k = 0` wherever `d₁₀₀ = 0`, so RP500 floods exactly the same
   cells as RP100, just deeper. Extent never expands. Since extent expansion is most of the
   risk signal in a climate scenario, the method returns zero on the quantity of interest.
2. **Depth is not a local property.** Water surface elevation is approximately flat across a
   floodplain cross-section; depth varies because the ground does. Per-pixel depth scaling
   breaks that invariant — it puts extra water where water already is, deepening channels
   instead of spreading onto margins.
3. **Wrong exponent.** `dV/dh = A(h)` and wetted area grows with stage, so depth response to
   rainfall is sublinear (typically `d ∝ P^0.3–0.6` in wide floodplains). Naive scaling
   assumes exponent 1 everywhere. Worst error is in flat alluvial terrain — Gangetic plain,
   Assam, north Bihar — which is where the exposure is.
4. **Wrong driver.** `k` is local, but floodplain depth is set by integrated upstream volume.
   CMIP6 deltas are spatially non-uniform, so a scenario that wets the upper catchment and
   dries the lower one gets handled backwards.

The stage method reproduces the HEC-RAS map exactly at `P' = P₁₀₀`. The two only diverge
away from the anchor, which is the entire use case.

---

## Method

### Notation

| Symbol | Meaning |
|---|---|
| `DEM` | 30 m conditioned DEM (same one used for the HEC-RAS runs) |
| `HAND` | height above nearest drainage, m |
| `d₁₀₀` | RP100 2030 depth grid (the anchor), m |
| `h` | reach stage — water surface elevation above the drainage datum, m |
| `V(h)`, `A(h)` | stage–volume and stage–area curves per reach |
| `α` | effective runoff/storage coefficient per catchment |
| `P` | catchment-mean 24 h rainfall depth, m |

### Step 1 — HAND and reach delineation

Condition the DEM (breach depressions, not fill — preserve flow paths under road/rail
embankments where culverts exist). Compute D8 flow direction and accumulation. Extract the
drainage network at a flow-accumulation threshold; prefer snapping to the HEC-RAS
centerlines where available so reach IDs align with the original models.

Produce three static rasters, aligned and tiled identically to `d₁₀₀`:

- `hand.tif` — float32, m
- `reach_id.tif` — int32, catchment-of-nearest-drainage-segment
- `catchment_id.tif` — int32, contributing catchment per reach

### Step 2 — Stage–volume–area curves

For each reach, over ~20 stage increments spanning 0 to ~1.5 × the observed anchor stage:

```
A(h) = count(HAND < h)              × cell_area
V(h) = Σ (h − HAND) over HAND < h   × cell_area
```

This is the piece the naive method has no equivalent of. It encodes whether a reach is a
confined valley (steep `V(h)`, stage rises fast, depth response near-linear) or a flat plain
(`A(h)` explodes, volume spreads laterally at almost constant depth).

Store as parquet: `(reach_id, h, volume, area)`.

### Step 3 — Read the anchor into stage space

Per reach, over cells where `d₁₀₀ > 0`:

```
h₁₀₀ = median(d₁₀₀ + HAND)
```

This works because a flat water surface implies `d₁₀₀ + HAND` is constant across the reach.
Use the median, not the mean — it's robust to the DEM/HAND artefacts that produce outliers
at reach edges.

**Reconstruction QC (this is the gate — do not proceed if it fails):**

```
d_recon = max(0, h₁₀₀ − HAND)
```

Compare `d_recon` against `d₁₀₀`. Report per reach:
- CSI at depth thresholds 0.15 / 0.30 / 0.50 / 1.0 m
- RMSE over cells wet in either grid
- inundated area ratio
- IQR of `(d₁₀₀ + HAND)` — a wide spread means the flat-WSE assumption is failing there

Flag reaches below CSI 0.80 at the 0.3 m threshold. These are usually steep terrain, dense
urban, or bad reach delineation. Fix delineation before blaming the method.

Also cross-check `V(h₁₀₀)` from Step 2 against `Σ d₁₀₀ × cell_area`. They should agree
within a few percent; if not, the HAND conditioning or reach assignment is wrong.

### Step 4 — Calibrate α

Per catchment:

```
V_obs    = Σ d₁₀₀ × cell_area
V_rain   = P₁₀₀_mean × A_catchment
α        = V_obs / V_rain
```

`α` absorbs infiltration, routing losses, hyetograph shape, and the RMSI disaggregation —
all of which are held fixed. Expect roughly 0.05–0.4. Values outside that need
investigation, not clamping.

### Step 5 — Scenario evaluation

```
V'  = α × P'_mean × A_catchment
h'  = V⁻¹(V')                      # monotone interpolation on the stored curve
d'  = max(0, h' − HAND)
```

Clamp `h'` to the tabulated stage range and flag extrapolated cells rather than letting the
interpolation run off the end.

### Step 6 — Pluvial cells need separate treatment

Depression-storage flooding does not obey reach stage. Split the domain by HAND and flow
accumulation:

- **Fluvial** (low HAND, high flow-acc): reach stage as above, driven by catchment-integrated
  volume.
- **Pluvial** (high HAND, isolated depressions): fill the DEM to identify sinks, build a
  volume–elevation curve per depression, calibrate a separate `α` against the `d₁₀₀` inside
  it, drive with local rainfall over the depression's own contributing area.

Without this split, urban and interior ponding is systematically over-scaled.

### Step 7 — Rainfall multiplier

Two independent axes, both resolving to a multiplier applied before Step 5:

- **Return period**: GEV fits to IMD gridded daily precip, regionalised by L-moments. Gives
  `P_RP / P₁₀₀` per grid cell.
- **Year / SSP**: NEX-GDDP-CMIP6 deltas on annual maximum 1-day precip. Multi-model median
  plus spread. Sanity-bound at Clausius–Clapeyron, ~7 %/°C.

**Hold the RMSI temporal ratios fixed.** Hyetograph shape is baked into `α`; changing it
silently invalidates the calibration.

---

## Repo layout

```
data/                 # GCS paths / symlinks — never commit rasters
src/
  hand.py             # DEM conditioning, flow dir/acc, HAND, reach + catchment IDs
  curves.py           # stage-volume-area tables per reach
  calibrate.py        # h100 from anchor, alpha per catchment, reconstruction QC
  pluvial.py          # depression delineation + separate calibration
  rainfall.py         # GEV growth factors + CMIP6 deltas -> multiplier grid
  render.py           # h' -> depth tiles
  regime.py           # fluvial/pluvial/low-confidence classification
validate/
  reconstruct.py      # Step 3 gate
  holdout.py          # spatial holdout on alpha
  events.py           # Sentinel-1 observed-event comparison
tests/
```

## Stack

Use what's already in the existing GIS pipelines:

- `pysheds` or WhiteboxTools — flow routing, HAND, depression handling
- `rasterio` / `rioxarray` — raster I/O, COG output
- `exactextract` — per-reach and per-catchment zonal sums (same pattern as the PIN-code
  hazard aggregation pipeline)
- `geopandas` + parquet — curve and calibration tables
- `dask` — only if going for a single national pass rather than basin-by-basin

## Build order

**Do one basin end-to-end before touching anything national.** Mahanadi or Godavari — mixed
terrain, so both regimes show up.

1. `hand.py` on one basin, visually inspect HAND against known floodplains
2. `curves.py`, check monotonicity and that `A(h)` behaves sanely
3. `calibrate.py` + the Step 3 reconstruction QC — **this is the gate**
4. `rainfall.py`, verify against a known IDF curve at a few gauge locations
5. `render.py`, produce RP25 / RP100 / RP500 for the test basin and eyeball the extent
   progression
6. Only then scale to India

---

## Validation (no new HEC-RAS runs)

- **Reconstruction** (Step 3) — the primary gate.
- **Spatial holdout** — fit an `α`-to-catchment-attribute regression on 80 % of catchments,
  predict `α` for the held-out 20 %, reconstruct their RP100 maps. Tests whether `α` is
  transferable, which is the main structural risk.
- **Observed events** — Sentinel-1 derived inundation for Kerala 2018, Assam 2022, Bihar,
  with IMD rainfall as input. CSI at the 0.3 m threshold.
- **Published maps** — compare RP25 and RP500 against Fathom / Aqueduct / JRC on *area
  ratios*, not absolute values. A wildly off RP500/RP100 area ratio means the growth curve
  or `α` stationarity is broken.

Metrics throughout: hit rate, false alarm ratio, CSI at 0.15 / 0.30 / 0.50 / 1.0 m, RMSE over
union-wet cells, total inundated area error.

---

## Known limitations — document these in the README

The tool must not be presented as a substitute for hydraulic modelling.

- No velocities, no shear stress, no dynamic timing — max depth only.
- No backwater or confluence interaction changes. If two tributaries re-phase under a new
  scenario, the method won't see it.
- Steep terrain and dense urban areas: HAND is a poor proxy for flow paths.
- Below ~RP10 flow is channel-contained and qualitatively different from the anchor regime.
- Coastal reaches: sea level rise enters as a downstream stage offset added to `h`, **not**
  as a rainfall multiplier. Separate axis.
- Geometry frozen at 2030 — no embankments, urbanisation, or channel change.
- `α` stationarity across return periods is assumed. Weak at low RP, mildly optimistic at
  high RP.

Every output cell carries a confidence flag driven by (a) distance from the anchor,
`|log(P'/P₁₀₀)|`, and (b) regime class from `regime.py`.

---

## Deliverable

Not a stack of depth rasters. The product is:

- `hand.tif`, `reach_id.tif`, `catchment_id.tif` — static COGs on GCS
- `curves.parquet` — `(reach_id, h, volume, area)` at ~20 increments
- `calibration.parquet` — `(catchment_id, alpha, h100, qc_metrics)`
- rainfall multiplier module keyed on `(RP, year, SSP)`
- on-the-fly tile renderer computing `max(0, h' − HAND)`
