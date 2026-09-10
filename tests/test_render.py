import numpy as np
import pandas as pd
import pytest

from src.curves import stage_volume_area_curves
from src.pluvial import depression_curves
from src.regime import REGIME_FLUVIAL, REGIME_PLUVIAL
from src.render import (
    calibrate_channel_ratio,
    combine_channel_and_depression_depth,
    render_channel_ratio_depth,
    render_depth,
    render_fluvial_depth,
    render_naive_ratio_depth,
)


def test_render_fluvial_depth_reproduces_curve_at_calibration_point():
    hand = np.linspace(0.1, 0.9, 18).reshape(3, 6)
    catchment_id = np.ones((3, 6), dtype=int)
    curves_df = stage_volume_area_curves(hand, catchment_id, cell_area=1.0, n_steps=30)

    row = curves_df.iloc[len(curves_df) // 2]
    target_h, target_v = row["h"], row["volume"]

    depth, extrapolated = render_fluvial_depth(
        hand, catchment_id, curves_df, {1: 1.0}, {1: target_v}, {1: 1.0},
    )

    expected = np.clip(target_h - hand, 0, None)
    np.testing.assert_allclose(depth, expected, atol=1e-6)
    assert not extrapolated.any()


def test_render_fluvial_depth_flags_extrapolation():
    hand = np.linspace(0.1, 0.9, 18).reshape(3, 6)
    catchment_id = np.ones((3, 6), dtype=int)
    curves_df = stage_volume_area_curves(hand, catchment_id, cell_area=1.0, n_steps=10)

    huge_volume = curves_df["volume"].max() * 100
    depth, extrapolated = render_fluvial_depth(
        hand, catchment_id, curves_df, {1: 1.0}, {1: huge_volume}, {1: 1.0},
    )
    assert extrapolated.any()


def test_render_fluvial_depth_missing_inputs_is_nan_not_zero():
    hand = np.linspace(0.1, 0.9, 18).reshape(3, 6)
    catchment_id = np.ones((3, 6), dtype=int)
    curves_df = stage_volume_area_curves(hand, catchment_id, cell_area=1.0, n_steps=10)

    depth, extrapolated = render_fluvial_depth(hand, catchment_id, curves_df, {}, {}, {})

    assert np.all(np.isnan(depth))
    assert not extrapolated.any()


def _mixed_fluvial_pluvial_grid():
    """Rows 0-2: a fluvial reach (low HAND, high flow-acc). Rows 3-5: a pluvial
    bowl-shaped depression (high HAND, low flow-acc, isolated)."""
    hand = np.zeros((6, 6))
    flow_acc = np.zeros((6, 6))
    catchment_id = np.zeros((6, 6), dtype=int)
    depression_id = np.zeros((6, 6), dtype=int)
    dem = np.zeros((6, 6))

    hand[0:3, :] = np.linspace(0.1, 0.9, 18).reshape(3, 6)
    flow_acc[0:3, :] = 1000
    catchment_id[0:3, :] = 1

    hand[3:6, :] = 10.0
    flow_acc[3:6, :] = 1
    depression_id[3:6, :] = 1
    yy, xx = np.mgrid[0:3, 0:6]
    r = np.sqrt((yy - 1) ** 2 + (xx - 2.5) ** 2)
    dem[3:6, :] = 20.0 + r

    return hand, flow_acc, catchment_id, depression_id, dem


def test_render_depth_splits_regime_correctly():
    hand, flow_acc, catchment_id, depression_id, dem = _mixed_fluvial_pluvial_grid()
    fluvial_curves = stage_volume_area_curves(hand, catchment_id, cell_area=1.0, n_steps=10)

    result = render_depth(
        hand, catchment_id, flow_acc, fluvial_curves, {1: 1.0}, {1: 0.0}, {1: 1.0},
        dem=dem, depression_id=depression_id,
        depression_curves_df=depression_curves(dem, depression_id, cell_area=1.0, n_steps=10),
        alpha_by_depression={1: 1.0}, p_prime_mean_by_depression={1: 0.0},
        depression_area_by_id={1: 1.0},
    )

    assert (result["regime"][0:3, :] == REGIME_FLUVIAL).all()
    assert (result["regime"][3:6, :] == REGIME_PLUVIAL).all()


def test_render_depth_combines_fluvial_and_pluvial_values():
    hand, flow_acc, catchment_id, depression_id, dem = _mixed_fluvial_pluvial_grid()

    fluvial_curves = stage_volume_area_curves(hand, catchment_id, cell_area=1.0, n_steps=20)
    fluvial_row = fluvial_curves.iloc[len(fluvial_curves) // 2]

    pluvial_curves = depression_curves(dem, depression_id, cell_area=1.0, n_steps=20, max_stage_multiple=1.0)
    pluvial_row = pluvial_curves.iloc[len(pluvial_curves) // 2]

    result = render_depth(
        hand, catchment_id, flow_acc, fluvial_curves,
        {1: 1.0}, {1: fluvial_row["volume"]}, {1: 1.0},
        dem=dem, depression_id=depression_id, depression_curves_df=pluvial_curves,
        alpha_by_depression={1: 1.0}, p_prime_mean_by_depression={1: pluvial_row["volume"]},
        depression_area_by_id={1: 1.0},
    )

    expected_fluvial = np.clip(fluvial_row["h"] - hand[0:3, :], 0, None)
    np.testing.assert_allclose(result["depth"][0:3, :], expected_fluvial, atol=1e-6)

    from src.pluvial import depression_relative_elevation
    hd = depression_relative_elevation(dem, depression_id)
    expected_pluvial = np.clip(pluvial_row["h"] - hd[3:6, :], 0, None)
    np.testing.assert_allclose(result["depth"][3:6, :], expected_pluvial, atol=1e-6)

    assert not result["extrapolated"].any()


def test_render_depth_fluvial_only_leaves_pluvial_cells_nan():
    hand, flow_acc, catchment_id, _, _ = _mixed_fluvial_pluvial_grid()
    fluvial_curves = stage_volume_area_curves(hand, catchment_id, cell_area=1.0, n_steps=10)
    row = fluvial_curves.iloc[len(fluvial_curves) // 2]

    result = render_depth(
        hand, catchment_id, flow_acc, fluvial_curves, {1: 1.0}, {1: row["volume"]}, {1: 1.0},
    )

    assert np.all(np.isnan(result["depth"][3:6, :]))


def test_render_depth_attaches_low_confidence_flag_when_requested():
    hand, flow_acc, catchment_id, _, _ = _mixed_fluvial_pluvial_grid()
    fluvial_curves = stage_volume_area_curves(hand, catchment_id, cell_area=1.0, n_steps=10)
    row = fluvial_curves.iloc[len(fluvial_curves) // 2]

    log_ratio = np.zeros_like(hand)
    log_ratio[0, 0] = 5.0  # far from the anchor -> should trip low confidence

    result = render_depth(
        hand, catchment_id, flow_acc, fluvial_curves, {1: 1.0}, {1: row["volume"]}, {1: 1.0},
        log_rainfall_ratio=log_ratio,
    )

    assert "low_confidence" in result
    assert result["low_confidence"][0, 0]
    assert not result["low_confidence"][1, 1]


def test_calibrate_channel_ratio_recovers_exact_ratio():
    d100 = np.array([[2.0, 2.0], [0.0, 2.0]])
    catchment_id = np.array([[1, 1], [1, 1]])
    rain_p100 = np.full((2, 2), 10.0)

    k_df = calibrate_channel_ratio(d100, catchment_id, rain_p100)

    assert len(k_df) == 1
    assert k_df.iloc[0]["catchment_id"] == 1
    assert k_df.iloc[0]["k"] == pytest.approx(0.2)


def test_calibrate_channel_ratio_uses_median_not_mean():
    # ratios 0.1, 0.2, 0.3, 10.0 -- mean is dragged way up by the outlier,
    # median (0.25) is robust to it, matching curves.py's median-based h100.
    d100 = np.array([[1.0, 2.0, 3.0, 100.0]])
    catchment_id = np.ones((1, 4), dtype=int)
    rain_p100 = np.full((1, 4), 10.0)

    k_df = calibrate_channel_ratio(d100, catchment_id, rain_p100)

    assert k_df.iloc[0]["k"] == pytest.approx(0.25)


def test_calibrate_channel_ratio_separates_catchments():
    d100 = np.array([[2.0, 6.0]])
    catchment_id = np.array([[1, 2]])
    rain_p100 = np.full((1, 2), 10.0)

    k_df = calibrate_channel_ratio(d100, catchment_id, rain_p100).set_index("catchment_id")["k"]

    assert k_df.loc[1] == pytest.approx(0.2)
    assert k_df.loc[2] == pytest.approx(0.6)


def test_calibrate_channel_ratio_exclude_mask_drops_cells():
    d100 = np.array([[2.0, 20.0]])  # second cell is a depression outlier
    catchment_id = np.array([[1, 1]])
    rain_p100 = np.full((1, 2), 10.0)
    exclude_mask = np.array([[False, True]])

    k_df = calibrate_channel_ratio(d100, catchment_id, rain_p100, exclude_mask=exclude_mask)

    assert k_df.iloc[0]["k"] == pytest.approx(0.2)  # only the non-excluded cell counted


def test_render_channel_ratio_depth_scales_by_rainfall_ratio():
    d100 = np.array([[2.0, 2.0]])
    catchment_id = np.array([[1, 1]])
    k_df = pd.DataFrame({"catchment_id": [1], "k": [0.2]})
    rain_p_prime = np.full((1, 2), 50.0)  # 5x the P100 that produced k=0.2 at d100=2.0/10.0

    depth = render_channel_ratio_depth(d100, catchment_id, rain_p_prime, k_df)

    np.testing.assert_allclose(depth, [[10.0, 10.0]])


def test_render_channel_ratio_depth_frozen_mask_zeros_dry_cells():
    d100 = np.array([[2.0, 0.0]])
    catchment_id = np.array([[1, 1]])
    k_df = pd.DataFrame({"catchment_id": [1], "k": [0.2]})
    rain_p_prime = np.full((1, 2), 50.0)

    depth = render_channel_ratio_depth(d100, catchment_id, rain_p_prime, k_df)

    assert depth[0, 1] == 0.0  # originally dry -> stays dry, not NaN


def test_render_channel_ratio_depth_unknown_catchment_is_nan_not_zero():
    d100 = np.array([[2.0]])
    catchment_id = np.array([[2]])  # no k calibrated for catchment 2
    k_df = pd.DataFrame({"catchment_id": [1], "k": [0.2]})
    rain_p_prime = np.full((1, 1), 50.0)

    depth = render_channel_ratio_depth(d100, catchment_id, rain_p_prime, k_df)

    assert np.isnan(depth[0, 0])


def test_combine_channel_and_depression_depth_picks_by_domain():
    channel_depth = np.array([[1.0, 1.0], [1.0, 1.0]])
    depression_depth = np.array([[9.0, 9.0], [9.0, 9.0]])
    depression_id = np.array([[0, 1], [0, 1]])

    combined = combine_channel_and_depression_depth(channel_depth, depression_depth, depression_id)

    np.testing.assert_array_equal(combined, [[1.0, 9.0], [1.0, 9.0]])


def test_render_naive_ratio_depth_scales_by_rainfall_ratio():
    d100 = np.array([[2.0, 2.0]])
    rain_p100 = np.full((1, 2), 10.0)
    rain_p_prime = np.full((1, 2), 50.0)  # 5x P100

    depth = render_naive_ratio_depth(d100, rain_p100, rain_p_prime)

    np.testing.assert_allclose(depth, [[10.0, 10.0]])


def test_render_naive_ratio_depth_is_per_pixel_not_per_catchment():
    # two pixels with different local ratios -- unlike calibrate_channel_ratio,
    # the naive method must NOT pool them into one shared value.
    d100 = np.array([[2.0, 6.0]])
    rain_p100 = np.full((1, 2), 10.0)
    rain_p_prime = np.full((1, 2), 10.0)

    depth = render_naive_ratio_depth(d100, rain_p100, rain_p_prime)

    np.testing.assert_allclose(depth, [[2.0, 6.0]])


def test_render_naive_ratio_depth_frozen_mask_zeros_dry_cells():
    d100 = np.array([[2.0, 0.0]])
    rain_p100 = np.full((1, 2), 10.0)
    rain_p_prime = np.full((1, 2), 50.0)

    depth = render_naive_ratio_depth(d100, rain_p100, rain_p_prime)

    assert depth[0, 1] == 0.0


def test_render_naive_ratio_depth_unknown_when_no_rain_is_nan_not_zero():
    d100 = np.array([[2.0]])
    rain_p100 = np.array([[0.0]])  # no anchor rainfall -> no ratio can be formed
    rain_p_prime = np.array([[50.0]])

    depth = render_naive_ratio_depth(d100, rain_p100, rain_p_prime)

    assert np.isnan(depth[0, 0])
