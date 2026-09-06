"""Step 7 — the scenario rainfall multiplier, on two independent axes:

- Return period: GEV fits to IMD gridded daily precip, regionalised by
  L-moments (index-flood method). Gives P_RP / P100 per grid cell.
- Year / SSP: NEX-GDDP-CMIP6 deltas on annual maximum 1-day precip.
  Multi-model median, sanity-bound at Clausius-Clapeyron (~7%/degC).

Both resolve to a multiplier applied before Step 5's V' = alpha * P' * A.
Hold the RMSI temporal ratios fixed — hyetograph shape is baked into alpha in
calibrate.py; changing it silently invalidates that calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gamma

from src.calibrate import broadcast_reach_values

DEFAULT_ANCHOR_RETURN_PERIOD = 100
DEFAULT_CC_RATE_PER_DEGC = 0.07  # Clausius-Clapeyron, ~7% per degree C
EULER_GAMMA = 0.5772156649015329
GUMBEL_SHAPE_TOLERANCE = 1e-6  # below this, treat k as exactly 0 to avoid catastrophic cancellation in (1 - 2**-k)


def station_lmoments(annual_max_values):
    """Sample L-moments (l1, l2, t3) from unbiased probability-weighted moments,
    plus record length n for weighting in the regional pool."""
    x = np.sort(np.asarray(annual_max_values, dtype=float))
    n = x.size
    if n < 3:
        return {"l1": np.nan, "l2": np.nan, "l3": np.nan, "t3": np.nan, "n": n}

    i = np.arange(1, n + 1, dtype=float)
    b0 = np.mean(x)
    b1 = np.sum((i - 1) / (n - 1) * x) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n

    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    t3 = l3 / l2 if l2 != 0 else np.nan

    return {"l1": l1, "l2": l2, "l3": l3, "t3": t3, "n": n}


def fit_gev_from_lmoments(l1, l2, t3):
    """Hosking (1990) approximation for GEV parameters from L-moment ratios.

    Valid for -0.5 < k < 0.5, which covers essentially all rainfall extremes
    regionalisation in practice. Parameterisation follows Hosking's convention:

        F(x) = exp{ -[1 - k(x - xi)/alpha]^(1/k) },  k != 0
    """
    c = 2 / (3 + t3) - np.log(2) / np.log(3)
    k = 7.8590 * c + 2.9554 * c ** 2

    if abs(k) < GUMBEL_SHAPE_TOLERANCE:
        # (1 - 2**-k) catastrophically cancels for k this close to zero; use the
        # k -> 0 (Gumbel) limit directly instead of the general GEV formula.
        alpha = l2 / np.log(2)
        xi = l1 - alpha * EULER_GAMMA
        return {"loc": xi, "scale": alpha, "shape": 0.0}

    g = gamma(1 + k)
    alpha = l2 * k / ((1 - 2 ** (-k)) * g)
    xi = l1 - alpha * (1 - g) / k
    return {"loc": xi, "scale": alpha, "shape": k}


def gev_quantile(params, non_exceedance_prob):
    """Inverse CDF for the fitted GEV, in Hosking's parameterisation."""
    xi, alpha, k = params["loc"], params["scale"], params["shape"]
    f = np.asarray(non_exceedance_prob, dtype=float)
    if abs(k) < 1e-8:
        return xi - alpha * np.log(-np.log(f))
    return xi + (alpha / k) * (1 - (-np.log(f)) ** k)


def gev_return_level(params, return_period_years):
    """Return level for annual maxima: F = 1 - 1/T."""
    t = np.asarray(return_period_years, dtype=float)
    return gev_quantile(params, 1 - 1 / t)


def regional_growth_curves(annual_max_df):
    """Index-flood regionalisation: pool L-moment ratios across each region,
    weighted by station record length, and fit one GEV per region to the
    standardized (index-flood = 1) distribution.

    annual_max_df columns: station_id, region_id, value (one row per
    station-year annual maximum).

    Returns {region_id: gev_params}. Apply via growth_factor() together with
    each site's own index flood (its mean annual max) to get absolute P(T).
    """
    lmoments = {
        station_id: station_lmoments(group["value"].to_numpy())
        for station_id, group in annual_max_df.groupby("station_id")
    }
    lmom_df = pd.DataFrame(lmoments).T
    region_by_station = annual_max_df.drop_duplicates("station_id").set_index("station_id")["region_id"]
    lmom_df["region_id"] = lmom_df.index.map(region_by_station)
    lmom_df["lcv"] = lmom_df["l2"] / lmom_df["l1"]
    lmom_df = lmom_df.dropna(subset=["lcv", "t3", "n"])

    region_params = {}
    for region_id, group in lmom_df.groupby("region_id"):
        weights = group["n"]
        lcv_r = float(np.average(group["lcv"], weights=weights))
        t3_r = float(np.average(group["t3"], weights=weights))
        region_params[region_id] = fit_gev_from_lmoments(l1=1.0, l2=lcv_r, t3=t3_r)
    return region_params


def growth_factor(region_params, return_period_years):
    """Q(T) / Q(mean) for a region's standardized growth curve."""
    return gev_return_level(region_params, return_period_years)


def return_period_multiplier_grid(region_id_grid, region_params_by_region, return_period_years,
                                   anchor_return_period=DEFAULT_ANCHOR_RETURN_PERIOD):
    """P_RP / P100 per pixel.

    The index flood cancels in this ratio — P_RP = index_flood * growth(RP) and
    P100 = index_flood * growth(100) — so only the regional growth curve, keyed
    by region_id, is needed per pixel.
    """
    ratio_by_region = {}
    for region_id, params in region_params_by_region.items():
        q_rp = gev_return_level(params, return_period_years)
        q_anchor = gev_return_level(params, anchor_return_period)
        ratio_by_region[region_id] = float(q_rp / q_anchor)

    return broadcast_reach_values(region_id_grid, pd.Series(ratio_by_region, dtype="float64"))


def cmip6_delta_multiplier(delta_pct_by_model, temp_delta_c=None, cc_rate_per_degc=DEFAULT_CC_RATE_PER_DEGC):
    """Multi-model median delta on annual-max 1-day precip, as a multiplier.

    delta_pct_by_model: per-model % change for one (year, SSP), shape
    (n_models, ...) — a scalar per model, or a grid per model.

    When temp_delta_c is given, the median delta is clipped to the
    Clausius-Clapeyron bound (~7%/degC): CMIP6 models can overshoot CC scaling
    in the tail, and this keeps the multiplier physically sane rather than
    trusting model spread blindly.

    Returns (multiplier, spread) — spread (std across models, in % points) is
    for downstream uncertainty reporting, not applied to the multiplier.
    """
    delta = np.asarray(delta_pct_by_model, dtype=float)
    median_delta = np.median(delta, axis=0)
    spread = np.std(delta, axis=0)

    if temp_delta_c is not None:
        cc_bound_pct = cc_rate_per_degc * 100 * np.asarray(temp_delta_c, dtype=float)
        median_delta = np.clip(median_delta, -cc_bound_pct, cc_bound_pct)

    multiplier = 1.0 + median_delta / 100.0
    return multiplier, spread


def scenario_multiplier(region_id_grid, region_params_by_region, return_period_years,
                         cmip6_delta_pct_by_model, temp_delta_c=None,
                         anchor_return_period=DEFAULT_ANCHOR_RETURN_PERIOD):
    """Combine both axes into the multiplier consumed by render.py's Step 5:

        P'/P100 = (P_RP/P100) * (1 + CMIP6 delta)

    Returns (multiplier_grid, cmip6_spread).
    """
    rp_multiplier = return_period_multiplier_grid(
        region_id_grid, region_params_by_region, return_period_years, anchor_return_period,
    )
    cmip6_multiplier, cmip6_spread = cmip6_delta_multiplier(cmip6_delta_pct_by_model, temp_delta_c=temp_delta_c)
    return rp_multiplier * cmip6_multiplier, cmip6_spread


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annual_max_parquet", help="columns: station_id, region_id, value")
    parser.add_argument("out_parquet", help="where to write fitted region_id -> GEV params")
    args = parser.parse_args()

    annual_max_df = pd.read_parquet(args.annual_max_parquet)
    region_params = regional_growth_curves(annual_max_df)

    out_df = pd.DataFrame.from_dict(region_params, orient="index")
    out_df.index.name = "region_id"
    out_df.reset_index().to_parquet(args.out_parquet, index=False)
    print(f"fitted {len(region_params)} regional growth curves -> {args.out_parquet}")


if __name__ == "__main__":
    main()
