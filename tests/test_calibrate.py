import time

import numpy as np
import pandas as pd
import pytest

from src.calibrate import (
    broadcast_reach_values,
    calibrate_alpha,
    compute_h100,
    cross_check_volume,
    reconstruct_depth,
    reconstruction_qc,
    write_calibration,
)
from src.curves import stage_volume_area_curves


def _flat_water_reach(h100_true=0.8, shape=(20, 20), seed=0):
    rng = np.random.default_rng(seed)
    hand = rng.uniform(0, 1.5, size=shape)
    reach_id = np.ones(shape, dtype=int)
    d100 = np.clip(h100_true - hand, 0, None)
    return hand, reach_id, d100


def test_broadcast_reach_values():
    reach_id = np.array([[1, 1, 2], [2, 0, 1]])
    values = pd.Series({1: 10.0, 2: 20.0})

    out = broadcast_reach_values(reach_id, values)

    assert out[0, 0] == 10.0
    assert out[0, 2] == 20.0
    assert np.isnan(out[1, 1])  # reach_id 0 -> no reach


def test_compute_h100_recovers_true_stage_exactly():
    hand, reach_id, d100 = _flat_water_reach(h100_true=0.8)
    h100_df = compute_h100(hand, reach_id, d100)

    assert h100_df.loc[h100_df["reach_id"] == 1, "h100"].iloc[0] == pytest.approx(0.8, abs=1e-9)


def test_reconstruction_is_near_perfect_for_synthetic_flat_surface():
    hand, reach_id, d100 = _flat_water_reach(h100_true=0.8)
    h100_df = compute_h100(hand, reach_id, d100)
    d_recon = reconstruct_depth(hand, reach_id, h100_df)

    np.testing.assert_allclose(d_recon, d100, atol=1e-9)

    qc_df = reconstruction_qc(d100, d_recon, reach_id, h100_df)
    row = qc_df.iloc[0]
    assert row["csi_0.3"] == pytest.approx(1.0)
    assert row["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert row["area_ratio"] == pytest.approx(1.0)
    assert row["flag_low_csi"] == False  # noqa: E712


def test_reconstruction_qc_flags_poor_match():
    hand, reach_id, d100 = _flat_water_reach(h100_true=0.8, seed=1)
    # deliberately wrong h100 -> reconstruction should diverge and get flagged
    bad_h100_df = pd.DataFrame({"reach_id": [1], "h100": [0.1], "stage_iqr": [0.0]})
    d_recon = reconstruct_depth(hand, reach_id, bad_h100_df)

    qc_df = reconstruction_qc(d100, d_recon, reach_id, bad_h100_df)

    assert qc_df.iloc[0]["flag_low_csi"] == True  # noqa: E712


def test_cross_check_volume_agrees_within_tolerance():
    hand, reach_id, d100 = _flat_water_reach(h100_true=0.8, shape=(60, 60), seed=2)
    cell_area = 900.0
    h100_df = compute_h100(hand, reach_id, d100)
    curves_df = stage_volume_area_curves(hand, reach_id, cell_area, anchor_stage=None, n_steps=40)

    check_df = cross_check_volume(curves_df, h100_df, d100, reach_id, cell_area, tol=0.05)

    assert check_df.iloc[0]["rel_diff"] < 0.05
    assert check_df.iloc[0]["flag_volume_mismatch"] == False  # noqa: E712


def test_calibrate_alpha_matches_manual_computation():
    catchment_id = np.array([[1, 1], [1, 1]])
    d100 = np.array([[0.5, 0.5], [0.5, 0.5]])
    cell_area = 100.0
    p100_mean = {1: 0.05}  # 50 mm
    catchment_area = {1: 4 * cell_area}

    alpha_df = calibrate_alpha(d100, catchment_id, cell_area, p100_mean, catchment_area)

    v_obs = float(np.sum(d100)) * cell_area
    v_rain = p100_mean[1] * catchment_area[1]
    expected_alpha = v_obs / v_rain

    row = alpha_df.iloc[0]
    assert row["alpha"] == pytest.approx(expected_alpha)
    assert row["flag_alpha_out_of_range"] == (not (0.05 <= expected_alpha <= 0.4))


def test_calibrate_alpha_flags_out_of_range():
    catchment_id = np.ones((2, 2), dtype=int)
    d100 = np.full((2, 2), 5.0)  # implausibly deep given tiny rainfall -> alpha way above 0.4
    cell_area = 100.0
    p100_mean = {1: 0.01}
    catchment_area = {1: 400.0}

    alpha_df = calibrate_alpha(d100, catchment_id, cell_area, p100_mean, catchment_area)

    assert alpha_df.iloc[0]["flag_alpha_out_of_range"] == True  # noqa: E712


def test_write_calibration_joins_alpha_and_qc_on_shared_id(tmp_path):
    hand, reach_id, d100 = _flat_water_reach(h100_true=0.8, seed=3)
    h100_df = compute_h100(hand, reach_id, d100)
    d_recon = reconstruct_depth(hand, reach_id, h100_df)
    qc_df = reconstruction_qc(d100, d_recon, reach_id, h100_df)

    alpha_df = calibrate_alpha(d100, reach_id, 900.0, {1: 0.05}, {1: 400 * 900.0})

    out_path = tmp_path / "calibration.parquet"
    calibration_df = write_calibration(qc_df, alpha_df, out_path)

    assert out_path.exists()
    assert {"catchment_id", "alpha", "h100"}.issubset(calibration_df.columns)
    assert calibration_df.iloc[0]["catchment_id"] == 1


def test_reconstruction_and_calibration_scale_to_many_reaches():
    """Regression test: reconstruction_qc, cross_check_volume, and
    calibrate_alpha used to mask the whole grid once per reach/catchment in a
    loop -- O(n_reaches * n_cells) -- which was unusable on a real basin's
    tens of thousands of reaches. Should stay roughly linear in grid size."""
    rng = np.random.default_rng(1)
    n_reaches = 3000
    cells_per_reach = 50
    hand = rng.uniform(0, 2, size=n_reaches * cells_per_reach)
    reach_id = np.repeat(np.arange(1, n_reaches + 1), cells_per_reach)
    h100_true = rng.uniform(0.5, 1.5, size=n_reaches)
    d100 = np.clip(np.repeat(h100_true, cells_per_reach) - hand, 0, None)
    cell_area = 900.0

    start = time.perf_counter()
    h100_df = compute_h100(hand, reach_id, d100)
    d_recon = reconstruct_depth(hand, reach_id, h100_df)
    qc_df = reconstruction_qc(d100, d_recon, reach_id, h100_df)
    curves_df = stage_volume_area_curves(hand, reach_id, cell_area, n_steps=20)
    volume_check_df = cross_check_volume(curves_df, h100_df, d100, reach_id, cell_area)
    p100_mean = {rid: 0.05 for rid in range(1, n_reaches + 1)}
    catchment_area = {rid: cells_per_reach * cell_area for rid in range(1, n_reaches + 1)}
    alpha_df = calibrate_alpha(d100, reach_id, cell_area, p100_mean, catchment_area)
    elapsed = time.perf_counter() - start

    assert len(qc_df) == n_reaches
    assert len(volume_check_df) == n_reaches
    assert len(alpha_df) == n_reaches
    assert elapsed < 20.0, f"took {elapsed:.1f}s for {n_reaches} reaches -- looks quadratic again"
