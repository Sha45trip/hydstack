"""Step 6, pluvial branch — depression delineation and a separate local
calibration for depression-storage flooding, which does not obey reach stage.

Fill the DEM to identify sinks, build a volume-elevation curve per depression
(the same math as curves.py, just keyed on "height above the depression's own
floor" instead of HAND), calibrate a separate alpha against d100 inside it, and
drive scenarios with local rainfall over the depression's own contributing
area — not the reach-level catchment from hand.py/calibrate.py.

Without this split, urban and interior ponding is systematically over-scaled
by the fluvial volume-balance calibration (see regime.py for the domain split
that routes cells here in the first place).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.calibrate import broadcast_reach_values, calibrate_alpha
from src.curves import invert_stage, stage_volume_area_curves
from src.hand import flow_direction, read_raster, write_raster


def _wbt(working_dir: Path):
    from whitebox import WhiteboxTools

    wbt = WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.set_working_dir(str(working_dir))
    return wbt


def fill_dem(dem_path, out_path):
    out_path = Path(out_path)
    wbt = _wbt(out_path.parent)
    wbt.fill_depressions(dem=str(dem_path), output=str(out_path))
    return out_path


def delineate_depressions(dem_path, out_path):
    """Unique ID per depression (sink) in the raw DEM — these footprints are
    the pluvial domain, distinct from the reach-based fluvial one."""
    out_path = Path(out_path)
    wbt = _wbt(out_path.parent)
    wbt.sink(dem=str(dem_path), output=str(out_path), zero_background=True)
    return out_path


def depression_pour_points(dem, depression_id, profile, out_path):
    """Lowest cell in each depression, as a point raster for wbt.watershed."""
    pour = np.zeros_like(depression_id, dtype="int32")
    for did in np.unique(depression_id):
        if did <= 0:
            continue
        masked_dem = np.where(depression_id == did, dem, np.inf)
        r, c = np.unravel_index(np.argmin(masked_dem), dem.shape)
        pour[r, c] = did
    write_raster(out_path, pour, profile, dtype="int32", nodata=0)
    return out_path


def depression_contributing_area(dem_path, depression_id_path, out_dir):
    """D8 watershed draining to each depression's own pour point — the
    depression's local contributing area, not the stream-network catchment
    from hand.py (an internally-drained depression may not connect to it)."""
    out_dir = Path(out_dir)
    dem, profile = read_raster(dem_path)
    depression_id, _ = read_raster(depression_id_path)

    pointer_path = flow_direction(dem_path, out_dir / "d8_pointer_raw.tif")
    pour_path = depression_pour_points(dem, depression_id, profile, out_dir / "pour_points.tif")

    catchment_path = out_dir / "depression_catchment_id.tif"
    wbt = _wbt(out_dir)
    wbt.watershed(d8_pntr=str(pointer_path), pour_pts=str(pour_path), output=str(catchment_path))
    return catchment_path


def depression_relative_elevation(dem, depression_id):
    """Height above each depression's own floor — the pluvial analogue of HAND,
    so curves.py's stage_volume_area_curves can be reused unmodified."""
    wet = depression_id > 0
    df = pd.DataFrame({"depression_id": depression_id[wet], "elev": dem[wet]})
    floor = df.groupby("depression_id")["elev"].min()
    floor_grid = broadcast_reach_values(depression_id, floor)
    return dem - floor_grid


def depression_curves(dem, depression_id, cell_area, n_steps=20, max_stage_multiple=1.5):
    """Per-depression volume-elevation curve, using depression_relative_elevation
    as the stage datum instead of HAND."""
    hd = depression_relative_elevation(dem, depression_id)
    return stage_volume_area_curves(
        hd, depression_id, cell_area, n_steps=n_steps, max_stage_multiple=max_stage_multiple,
    )


def calibrate_depression_alpha(d100, depression_id, cell_area, p100_mean_by_depression,
                                contributing_area_by_depression):
    """Same volume-balance formula as calibrate.calibrate_alpha, driven by each
    depression's own local rainfall and contributing area rather than the
    catchment-level values used for fluvial reaches."""
    return calibrate_alpha(
        d100, depression_id, cell_area, p100_mean_by_depression, contributing_area_by_depression,
    )


def render_pluvial_depth(hd, depression_id, curves_df, alpha_by_depression,
                          p_prime_mean_by_depression, area_by_depression):
    """Step 5's formula, applied per depression:

        V' = alpha * P'_local_mean * A_depression
        h' = V^-1(V')          # invert this depression's own curve
        d' = max(0, h' - hd)

    Returns (depth, extrapolated_mask).
    """
    h_by_depression = {}
    extrapolated_by_depression = {}
    for did in np.unique(depression_id):
        if did <= 0:
            continue
        alpha = alpha_by_depression.get(did, np.nan)
        p_mean = p_prime_mean_by_depression.get(did, np.nan)
        area = area_by_depression.get(did, np.nan)
        if not (np.isfinite(alpha) and np.isfinite(p_mean) and np.isfinite(area)):
            continue
        v_prime = alpha * p_mean * area
        h_prime, extrapolated = invert_stage(curves_df, did, v_prime)
        h_by_depression[did] = h_prime
        extrapolated_by_depression[did] = float(extrapolated)

    h_grid = broadcast_reach_values(depression_id, pd.Series(h_by_depression, dtype="float64"))
    depth = np.clip(h_grid - hd, 0, None)

    extrapolated_grid = broadcast_reach_values(
        depression_id, pd.Series(extrapolated_by_depression, dtype="float64"),
    )
    extrapolated_mask = np.nan_to_num(extrapolated_grid, nan=0.0) > 0.5

    return depth, extrapolated_mask


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dem", help="raw (unbreached) DEM — depressions here must survive conditioning")
    parser.add_argument("d100_raster", help="RP100 2030 anchor depth grid")
    parser.add_argument("out_dir")
    parser.add_argument("--n-steps", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    depression_id_path = delineate_depressions(args.dem, out_dir / "depression_id.tif")
    dem, profile = read_raster(args.dem)
    depression_id, _ = read_raster(depression_id_path)
    cell_area = abs(profile["transform"].a * profile["transform"].e)

    curves_df = depression_curves(dem, depression_id, cell_area, n_steps=args.n_steps)
    curves_df.to_parquet(out_dir / "depression_curves.parquet", index=False)
    print(f"wrote curves for {curves_df['reach_id'].nunique()} depressions")

    contributing_area_path = depression_contributing_area(args.dem, depression_id_path, out_dir)
    print(f"depression contributing areas: {contributing_area_path}")


if __name__ == "__main__":
    main()
