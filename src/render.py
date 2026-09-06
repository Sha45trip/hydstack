"""Step 5 — scenario evaluation, and the on-the-fly tile renderer.

    V'  = alpha * P'_mean * A_catchment
    h'  = V^-1(V')                      # monotone interpolation on curves.py's table
    d'  = max(0, h' - HAND)

This module renders the fluvial domain (regime.py's REGIME_FLUVIAL) and merges
in the pluvial domain from pluvial.py's render_pluvial_depth, split by
regime.py's classify_regime. Every output cell carries an extrapolation flag
(h' fell outside the tabulated stage range and was clamped) and, when scenario
inputs are supplied, the confidence flag from regime.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibrate import broadcast_reach_values
from src.curves import index_curves, invert_stage_indexed
from src.pluvial import depression_relative_elevation, render_pluvial_depth
from src.regime import REGIME_FLUVIAL, REGIME_PLUVIAL, classify_regime, confidence_flag


def render_fluvial_depth(hand, catchment_id, curves_df, alpha_by_catchment,
                          p_prime_mean_by_catchment, catchment_area_by_id):
    """Step 5's formula for the fluvial domain. catchment_id doubles as the
    reach_id key into curves_df — hand.py's subbasins step assigns each
    catchment the ID of the reach link it drains to (see calibrate.py's
    write_calibration).

    Returns (depth, extrapolated_mask). Cells whose catchment is missing any
    of alpha/rainfall/area are left as NaN depth (unknown), not 0 (no flood).
    """
    indexed_curves = index_curves(curves_df)

    h_by_catchment = {}
    extrapolated_by_catchment = {}
    for cid in np.unique(catchment_id):
        if cid <= 0:
            continue
        alpha = alpha_by_catchment.get(cid, np.nan)
        p_mean = p_prime_mean_by_catchment.get(cid, np.nan)
        area = catchment_area_by_id.get(cid, np.nan)
        if not (np.isfinite(alpha) and np.isfinite(p_mean) and np.isfinite(area)):
            continue
        v_prime = alpha * p_mean * area
        h_prime, extrapolated = invert_stage_indexed(indexed_curves, cid, v_prime)
        h_by_catchment[cid] = h_prime
        extrapolated_by_catchment[cid] = float(extrapolated)

    h_grid = broadcast_reach_values(catchment_id, pd.Series(h_by_catchment, dtype="float64"))
    depth = np.clip(h_grid - hand, 0, None)

    extrapolated_grid = broadcast_reach_values(
        catchment_id, pd.Series(extrapolated_by_catchment, dtype="float64"),
    )
    extrapolated_mask = np.nan_to_num(extrapolated_grid, nan=0.0) > 0.5

    return depth, extrapolated_mask


def calibrate_channel_ratio(d100, catchment_id, rain_p100, exclude_mask=None):
    """Per-catchment depth/rainfall ratio: k = median(d100 / rain_p100) over
    wet, non-excluded cells.

    This is a pragmatic stand-in for render_fluvial_depth's curve-based method
    when that fails calibrate.py's reconstruction QC on real data — it is the
    same naive d' = (d100/P100)*P' the brief's own "Why not just scale depth
    by rainfall" section critiques, just calibrated per catchment instead of
    one global ratio so nearby reaches with different terrain/rainfall don't
    share a single number. It still inherits the naive method's structural
    limitations: a frozen wet mask (extent can't expand beyond where d100 is
    already wet) and a linear response (no sublinear d~P^0.3-0.6
    floodplain-widening). Prefer render_fluvial_depth wherever it passes
    reconstruction_qc; fall back to this only where it doesn't.

    exclude_mask, if given, drops cells from calibration — e.g. cells already
    covered by pluvial.py's depression model, so the same water isn't fit
    twice under two different methods.

    Returns a DataFrame: (catchment_id, k).
    """
    valid = (d100 > 0) & (rain_p100 > 0) & (catchment_id > 0)
    if exclude_mask is not None:
        valid = valid & ~exclude_mask

    df = pd.DataFrame({
        "catchment_id": catchment_id[valid],
        "k": d100[valid] / rain_p100[valid],
    })
    return df.groupby("catchment_id")["k"].median().reset_index()


def render_channel_ratio_depth(d100, catchment_id, rain_p_prime, k_df):
    """d' = k * P'_local, per catchment, restricted to the frozen wet mask
    (d100 > 0) — see calibrate_channel_ratio's docstring for the method's
    known limitations.

    Cells with no catchment or no calibrated k are NaN (unknown), not 0.
    Originally-dry cells (d100 <= 0) are 0 — this method never predicts new
    flooding beyond the anchor's wet footprint — not NaN.
    """
    k_by_catchment = k_df.set_index("catchment_id")["k"]
    k_grid = broadcast_reach_values(catchment_id, k_by_catchment)
    depth = k_grid * rain_p_prime
    return np.where(d100 > 0, depth, 0.0)


def combine_channel_and_depression_depth(channel_depth, depression_depth, depression_id):
    """Union render_channel_ratio_depth's output with pluvial.render_pluvial_depth's:
    depression cells take the (better-calibrated) depression model's value,
    everything else takes the channel-ratio value."""
    return np.where(depression_id > 0, depression_depth, channel_depth)


def render_depth(hand, catchment_id, flow_acc, curves_df, alpha_by_catchment,
                  p_prime_mean_by_catchment, catchment_area_by_id,
                  dem=None, depression_id=None, depression_curves_df=None,
                  alpha_by_depression=None, p_prime_mean_by_depression=None,
                  depression_area_by_id=None,
                  log_rainfall_ratio=None, slope=None, urban_mask=None,
                  return_period_years=None, coastal_mask=None):
    """Full tile render: fluvial + pluvial, merged by regime.py's domain split.

    Pluvial inputs (dem, depression_id, depression_curves_df,
    alpha_by_depression, p_prime_mean_by_depression, depression_area_by_id)
    are optional — omit them to render fluvial-only (pluvial cells come back
    as NaN, matching the "unknown" convention rather than silently 0).

    Returns a dict of grids aligned to hand: depth, extrapolated, regime, and
    (when the scenario inputs for it are supplied) low_confidence.
    """
    regime = classify_regime(hand, flow_acc)

    fluvial_depth, fluvial_extrapolated = render_fluvial_depth(
        hand, catchment_id, curves_df, alpha_by_catchment, p_prime_mean_by_catchment, catchment_area_by_id,
    )
    depth = np.where(regime == REGIME_FLUVIAL, fluvial_depth, np.nan)
    extrapolated = np.where(regime == REGIME_FLUVIAL, fluvial_extrapolated, False)

    if depression_id is not None:
        hd = depression_relative_elevation(dem, depression_id)
        pluvial_depth, pluvial_extrapolated = render_pluvial_depth(
            hd, depression_id, depression_curves_df, alpha_by_depression,
            p_prime_mean_by_depression, depression_area_by_id,
        )
        depth = np.where(regime == REGIME_PLUVIAL, pluvial_depth, depth)
        extrapolated = np.where(regime == REGIME_PLUVIAL, pluvial_extrapolated, extrapolated)

    result = {"depth": depth, "extrapolated": extrapolated, "regime": regime}

    if log_rainfall_ratio is not None:
        result["low_confidence"] = confidence_flag(
            regime, log_rainfall_ratio, slope=slope, urban_mask=urban_mask,
            return_period_years=return_period_years, coastal_mask=coastal_mask,
        )

    return result


def main():
    import argparse

    from src.hand import read_raster, write_raster

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hand_raster")
    parser.add_argument("catchment_id_raster")
    parser.add_argument("flow_accum_raster")
    parser.add_argument("curves_parquet")
    parser.add_argument("calibration_parquet", help="from calibrate.write_calibration: catchment_id, alpha, ...")
    parser.add_argument("p_prime_mean_raster", help="scenario mean rainfall depth per catchment, pre-multiplied by rainfall.py's multiplier")
    parser.add_argument("out_depth_raster")
    args = parser.parse_args()

    hand, profile = read_raster(args.hand_raster)
    catchment_id, _ = read_raster(args.catchment_id_raster)
    flow_acc, _ = read_raster(args.flow_accum_raster)
    p_prime_mean, _ = read_raster(args.p_prime_mean_raster)

    curves_df = pd.read_parquet(args.curves_parquet)
    calibration_df = pd.read_parquet(args.calibration_parquet)

    alpha_by_catchment = calibration_df.set_index("catchment_id")["alpha"].to_dict()

    cell_area = abs(profile["transform"].a * profile["transform"].e)
    catchment_area_by_id = {
        cid: int(np.count_nonzero(catchment_id == cid)) * cell_area
        for cid in np.unique(catchment_id) if cid > 0
    }
    p_prime_mean_by_catchment = {
        cid: float(np.mean(p_prime_mean[catchment_id == cid]))
        for cid in np.unique(catchment_id) if cid > 0
    }

    result = render_depth(
        hand, catchment_id, flow_acc, curves_df, alpha_by_catchment,
        p_prime_mean_by_catchment, catchment_area_by_id,
    )

    write_raster(args.out_depth_raster, result["depth"], profile, dtype="float32", nodata=np.nan)
    n_extrapolated = int(np.count_nonzero(result["extrapolated"]))
    print(f"wrote {args.out_depth_raster} ({n_extrapolated} extrapolated cells)")


if __name__ == "__main__":
    main()
