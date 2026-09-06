"""Spatial holdout — the test of whether alpha is transferable, which the
brief calls out as the main structural risk (see "Validation" in
FLOOD_EMULATOR_BRIEF.md).

Fit an alpha-to-catchment-attribute regression on a training fraction of
catchments, predict alpha for the held-out ones, and reconstruct their RP100
maps with the predicted (not calibrated) alpha. A held-out spatial block, not
a random per-catchment split, is what actually tests spatial transferability —
a random split leaks spatial autocorrelation and is optimistic.

Uses plain OLS via numpy.linalg.lstsq rather than scikit-learn, so this stays
runnable without an extra heavy dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HOLDOUT_FRAC = 0.2


def spatial_holdout_split(catchment_ids, block_id_by_catchment=None, holdout_frac=DEFAULT_HOLDOUT_FRAC, seed=0):
    """Split catchments into train/test.

    With block_id_by_catchment (a coarser spatial grouping — basin, grid cell,
    region), splits at the block level so whole spatial neighbourhoods are
    held out together. Without it, falls back to a random per-catchment
    split, which is optimistic about spatial transferability.
    """
    rng = np.random.default_rng(seed)
    catchment_ids = np.asarray(catchment_ids)

    if block_id_by_catchment is not None:
        block_id_by_catchment = dict(block_id_by_catchment)
        block_ids = np.array([block_id_by_catchment[c] for c in catchment_ids])
        unique_blocks = np.unique(block_ids)
        rng.shuffle(unique_blocks)
        n_holdout_blocks = max(1, int(round(len(unique_blocks) * holdout_frac)))
        holdout_blocks = set(unique_blocks[:n_holdout_blocks].tolist())
        test_mask = np.isin(block_ids, list(holdout_blocks))
    else:
        shuffled = catchment_ids.copy()
        rng.shuffle(shuffled)
        n_holdout = max(1, int(round(len(shuffled) * holdout_frac)))
        holdout_set = set(shuffled[:n_holdout].tolist())
        test_mask = np.isin(catchment_ids, list(holdout_set))

    return catchment_ids[~test_mask], catchment_ids[test_mask]


def fit_alpha_regression(attributes_df, alpha_series, feature_cols):
    """OLS: alpha ~ intercept + features. attributes_df is indexed by
    catchment_id; alpha_series is indexed the same way. Returns fitted
    coefficients as a Series (intercept + one per feature)."""
    X = attributes_df.loc[alpha_series.index, feature_cols].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X)), X])
    y = alpha_series.to_numpy(dtype=float)
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return pd.Series(coefs, index=["intercept"] + list(feature_cols))


def predict_alpha(attributes_df, coefs, feature_cols):
    X = attributes_df[feature_cols].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X)), X])
    return pd.Series(X @ coefs.to_numpy(), index=attributes_df.index)


def run_spatial_holdout(alpha_df, attributes_df, feature_cols, block_id_by_catchment=None,
                         holdout_frac=DEFAULT_HOLDOUT_FRAC, seed=0):
    """alpha_df: catchment_id, alpha (from calibrate.calibrate_alpha).
    attributes_df: catchment_id-indexed predictors (area, slope, land cover
    fractions, etc. — whatever's hypothesised to explain infiltration/routing
    variation between catchments).

    Fits on the training catchments, predicts alpha for the held-out ones, and
    reports prediction error. Returns (coefs, results_df).
    """
    alpha_series = alpha_df.set_index("catchment_id")["alpha"].dropna()
    common_ids = alpha_series.index.intersection(attributes_df.index)
    alpha_series = alpha_series.loc[common_ids]
    attributes_df = attributes_df.loc[common_ids]

    train_ids, test_ids = spatial_holdout_split(
        common_ids.to_numpy(), block_id_by_catchment=block_id_by_catchment,
        holdout_frac=holdout_frac, seed=seed,
    )

    coefs = fit_alpha_regression(attributes_df, alpha_series.loc[train_ids], feature_cols)
    predicted = predict_alpha(attributes_df.loc[test_ids], coefs, feature_cols)
    observed = alpha_series.loc[test_ids]

    results_df = pd.DataFrame({
        "catchment_id": test_ids,
        "alpha_observed": observed.to_numpy(),
        "alpha_predicted": predicted.reindex(test_ids).to_numpy(),
    })
    results_df["abs_error"] = (results_df["alpha_predicted"] - results_df["alpha_observed"]).abs()
    results_df["rel_error"] = results_df["abs_error"] / results_df["alpha_observed"].abs()

    return coefs, results_df


def holdout_summary(results_df):
    return {
        "n_test_catchments": len(results_df),
        "median_abs_error": float(results_df["abs_error"].median()),
        "median_rel_error": float(results_df["rel_error"].median()),
        "rmse": float(np.sqrt(np.mean(results_df["abs_error"] ** 2))),
    }


def reconstruct_with_predicted_alpha(hand, catchment_id, curves_df, predicted_alpha_by_catchment,
                                      p100_mean_by_catchment, catchment_area_by_id, test_ids):
    """Use predicted (not calibrated) alpha to reconstruct RP100 depth for the
    held-out catchments. This is the deeper transferability check the brief
    asks for, beyond just alpha's numeric error — feed the result into
    calibrate.reconstruction_qc against d100 for the CSI/RMSE/area-ratio
    picture on unseen catchments.
    """
    from src.render import render_fluvial_depth

    test_ids = set(test_ids)
    alpha_subset = {cid: a for cid, a in predicted_alpha_by_catchment.items() if cid in test_ids}
    catchment_id_test = np.where(np.isin(catchment_id, list(test_ids)), catchment_id, 0)

    return render_fluvial_depth(
        hand, catchment_id_test, curves_df, alpha_subset, p100_mean_by_catchment, catchment_area_by_id,
    )


def main():
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alpha_parquet", help="catchment_id, alpha (calibrate.calibrate_alpha output)")
    parser.add_argument("attributes_parquet", help="catchment_id-indexed predictor table")
    parser.add_argument("out_report_csv")
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--block-col", default=None, help="column in attributes_parquet to hold out by, instead of a random per-catchment split")
    parser.add_argument("--holdout-frac", type=float, default=DEFAULT_HOLDOUT_FRAC)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    alpha_df = pd.read_parquet(args.alpha_parquet)
    attributes_df = pd.read_parquet(args.attributes_parquet).set_index("catchment_id")

    block_map = attributes_df[args.block_col].to_dict() if args.block_col else None

    _, results_df = run_spatial_holdout(
        alpha_df, attributes_df, args.features, block_id_by_catchment=block_map,
        holdout_frac=args.holdout_frac, seed=args.seed,
    )
    results_df.to_csv(args.out_report_csv, index=False)

    for key, value in holdout_summary(results_df).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
