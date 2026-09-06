import numpy as np
import pandas as pd
import pytest

from src.curves import stage_volume_area_curves
from validate.holdout import (
    fit_alpha_regression,
    holdout_summary,
    predict_alpha,
    reconstruct_with_predicted_alpha,
    run_spatial_holdout,
    spatial_holdout_split,
)


def test_spatial_holdout_split_random_partitions_without_overlap():
    catchment_ids = np.arange(1, 101)
    train_ids, test_ids = spatial_holdout_split(catchment_ids, holdout_frac=0.2, seed=0)

    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(train_ids) + len(test_ids) == len(catchment_ids)
    assert len(test_ids) == 20


def test_spatial_holdout_split_by_block_keeps_whole_blocks_together():
    catchment_ids = np.arange(1, 21)
    # 4 blocks of 5 catchments each
    block_map = {cid: (cid - 1) // 5 for cid in catchment_ids}

    train_ids, test_ids = spatial_holdout_split(catchment_ids, block_id_by_catchment=block_map,
                                                  holdout_frac=0.25, seed=1)

    test_blocks = {block_map[c] for c in test_ids}
    train_blocks = {block_map[c] for c in train_ids}
    assert len(test_blocks) == 1  # 25% of 4 blocks = 1 block held out whole
    assert test_blocks.isdisjoint(train_blocks)


def test_fit_and_predict_alpha_recovers_exact_linear_relationship():
    rng = np.random.default_rng(0)
    n = 50
    attributes_df = pd.DataFrame({"area": rng.uniform(1, 10, n), "slope": rng.uniform(0, 1, n)},
                                  index=pd.Index(range(n), name="catchment_id"))
    true_alpha = 0.1 + 0.02 * attributes_df["area"] - 0.05 * attributes_df["slope"]

    coefs = fit_alpha_regression(attributes_df, true_alpha, ["area", "slope"])
    predicted = predict_alpha(attributes_df, coefs, ["area", "slope"])

    np.testing.assert_allclose(predicted.to_numpy(), true_alpha.to_numpy(), atol=1e-8)


def test_run_spatial_holdout_low_error_for_a_learnable_relationship():
    rng = np.random.default_rng(2)
    n = 200
    catchment_ids = np.arange(1, n + 1)
    attributes_df = pd.DataFrame(
        {"area": rng.uniform(1, 10, n), "slope": rng.uniform(0, 1, n)}, index=catchment_ids,
    )
    attributes_df.index.name = "catchment_id"
    alpha_values = 0.1 + 0.02 * attributes_df["area"] - 0.05 * attributes_df["slope"]
    alpha_df = pd.DataFrame({"catchment_id": catchment_ids, "alpha": alpha_values.to_numpy()})

    coefs, results_df = run_spatial_holdout(alpha_df, attributes_df, ["area", "slope"], seed=3)
    stats = holdout_summary(results_df)

    assert stats["n_test_catchments"] == 40
    assert stats["median_rel_error"] < 0.01  # near-exact linear relationship should transfer cleanly


def test_run_spatial_holdout_high_error_when_alpha_is_unrelated_to_attributes():
    rng = np.random.default_rng(4)
    n = 200
    catchment_ids = np.arange(1, n + 1)
    attributes_df = pd.DataFrame(
        {"area": rng.uniform(1, 10, n), "slope": rng.uniform(0, 1, n)}, index=catchment_ids,
    )
    attributes_df.index.name = "catchment_id"
    alpha_values = rng.uniform(0.05, 0.4, n)  # no relationship to attributes at all
    alpha_df = pd.DataFrame({"catchment_id": catchment_ids, "alpha": alpha_values})

    _, results_df = run_spatial_holdout(alpha_df, attributes_df, ["area", "slope"], seed=5)
    stats = holdout_summary(results_df)

    assert stats["median_rel_error"] > 0.1  # regression shouldn't fake transferability out of noise


def test_reconstruct_with_predicted_alpha_only_touches_test_catchments():
    hand = np.linspace(0.1, 0.9, 18).reshape(3, 6)
    catchment_id = np.ones((3, 6), dtype=int)
    curves_df = stage_volume_area_curves(hand, catchment_id, cell_area=1.0, n_steps=10)

    depth, extrapolated = reconstruct_with_predicted_alpha(
        hand, catchment_id, curves_df, {1: 1.0}, {1: 0.5}, {1: 1.0}, test_ids=[1],
    )
    assert depth.shape == hand.shape
    assert extrapolated.shape == hand.shape
    assert not np.all(np.isnan(depth))  # catchment 1 is in test_ids, so it should render

    depth_excluded, _ = reconstruct_with_predicted_alpha(
        hand, catchment_id, curves_df, {1: 1.0}, {1: 0.5}, {1: 1.0}, test_ids=[999],
    )
    assert np.all(np.isnan(depth_excluded))
