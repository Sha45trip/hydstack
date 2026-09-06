import numpy as np
import pytest

from src.curves import stage_volume_area_curves
from src.pluvial import depression_curves
from src.regime import REGIME_FLUVIAL, REGIME_PLUVIAL
from src.render import render_depth, render_fluvial_depth


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
