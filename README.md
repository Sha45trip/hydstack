# hydstack

A HAND-based flood depth emulator for India: produces flood depth grids for **any
return period, any year, any SSP**, at 30 m, without re-running HEC-RAS.

One hydraulic run — **RP100, 2030, India-wide, 30 m** — is the sole calibration
anchor. Scenarios are resolved by computing a runoff volume, inverting a
terrain-derived stage–volume curve to get water surface elevation, and
differencing that against HAND (height above nearest drainage). Depth
generation for any scenario is then a raster subtraction: seconds for a
district, minutes for India.


## Method, in one pass

1. **`src/hand.py`** — condition the DEM (breach, not fill), derive flow
   direction/accumulation, extract the drainage network, compute HAND, and
   assign reach/catchment IDs (optionally snapped to HEC-RAS centerlines).
2. **`src/curves.py`** — per-reach stage–volume–area curves: `A(h)` and `V(h)`
   from HAND, over ~20 stage increments.
3. **`src/calibrate.py`** — read the RP100 anchor into stage space
   (`h100 = median(d100 + HAND)`), reconstruct `d100` from it, and gate on
   CSI/RMSE/area-ratio per reach. **This QC is the primary gate** — do not
   proceed to national scale if it fails.
4. **`src/calibrate.py`** — calibrate `alpha = V_obs / V_rain` per catchment.
5. **`src/render.py`** — for a scenario: `V' = alpha * P'_mean * A_catchment`,
   invert the stored curve for `h'`, then `d' = max(0, h' - HAND)`.
6. **`src/regime.py`** / **`src/pluvial.py`** — split fluvial (reach-stage
   driven) from pluvial (depression-storage) cells; pluvial cells get their
   own local volume-elevation curve and calibration.
7. **`src/rainfall.py`** — the scenario multiplier: a GEV/L-moments return-period
   growth factor times a CMIP6 year/SSP delta, bounded at Clausius-Clapeyron.

## Repo layout

```
data/                 # GCS paths / symlinks — never commit rasters (see data/README.md)
src/
  hand.py             # DEM conditioning, flow dir/acc, HAND, reach + catchment IDs
  curves.py           # stage-volume-area tables per reach
  calibrate.py        # h100 from anchor, alpha per catchment, reconstruction QC
  pluvial.py          # depression delineation + separate calibration
  rainfall.py         # GEV growth factors + CMIP6 deltas -> multiplier grid
  render.py           # h' -> depth tiles
  regime.py           # fluvial/pluvial/low-confidence classification
validate/
  reconstruct.py      # Step 3 gate, as a standalone pass/fail report
  holdout.py          # spatial holdout on alpha
  events.py           # Sentinel-1 observed-event comparison
tests/                 # unit tests for every pure-logic path above
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

`rasterio`, `geopandas`, `numpy`/`pandas`/`scipy`/`pyarrow` are required to
import most of `src/`. `whitebox` is only needed to actually run
`hand.py`/`pluvial.py` against real rasters (DEM conditioning, flow routing,
depression delineation) — it's lazy-imported, so the rest of the package stays
usable without it.

## Running the tests

```bash
pytest
```

Tests cover the pure numpy/pandas logic (curve math, calibration, regime
classification, GEV fitting, rendering, validation metrics) with synthetic
rasters, so they run without real DEM/rainfall data or the `whitebox` binary.

## Build order

**Do one basin end-to-end before touching anything national** — Mahanadi or
Godavari, mixed terrain so both regimes show up:

1. `python -m src.hand <dem.tif> <out_dir>` — inspect `hand.tif` against known
   floodplains.
2. `python -m src.curves <hand.tif> <reach_id.tif> <d100.tif> <curves.parquet>`
   — check monotonicity.
3. `python -m src.calibrate ...` / `python -m validate.reconstruct ...` — the
   Step 3 gate. Fix delineation before blaming the method.
4. `python -m src.rainfall <annual_max.parquet> <region_growth_curves.parquet>`
   — verify against a known IDF curve at a few gauge locations.
5. `python -m src.render ...` — produce RP25 / RP100 / RP500 for the test
   basin and eyeball the extent progression.
6. Only then scale to India.

## Validation

- **Reconstruction** (`validate/reconstruct.py`) — the primary gate.
- **Spatial holdout** (`validate/holdout.py`) — is alpha transferable to
  unseen catchments?
- **Observed events** (`validate/events.py`) — Sentinel-1 vs. rendered depth
  for Kerala 2018, Assam 2022, Bihar.
- **Published maps** — RP25/RP500 area ratios against Fathom / Aqueduct / JRC
  (not absolute values).

## Known limitations

Max depth only — no velocities, no timing, no backwater/confluence
interaction. HAND is a poor flow-path proxy in steep or dense-urban terrain.
Below ~RP10, flow is channel-contained and outside the anchor regime. Coastal
sea-level rise is a stage offset, not a rainfall multiplier. Geometry is
frozen at 2030. Every output cell carries a confidence flag — see
`src/regime.py` and the brief's "Known limitations" section for the full list.
