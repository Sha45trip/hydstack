import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from src.pluvial import (
    calibrate_depression_alpha,
    depression_curves,
    depression_pour_points,
    depression_relative_elevation,
    render_pluvial_depth,
)


def _bowl_dem(depth=2.0, shape=(9, 9)):
    """A single circular depression, floor at the center, rim at the edges."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = r.max()
    dem = 10.0 + depth * (r / r_max)  # lowest (10.0) at center, rises to 10+depth at rim
    depression_id = np.ones(shape, dtype=int)
    return dem, depression_id


def test_depression_relative_elevation_floor_is_zero():
    dem, depression_id = _bowl_dem()
    hd = depression_relative_elevation(dem, depression_id)

    assert hd.min() == pytest.approx(0.0, abs=1e-9)
    # the floor cell (center) is where hd == 0
    center = (4, 4)
    assert hd[center] == pytest.approx(0.0, abs=1e-9)


def test_depression_curves_monotonic_and_capped_at_rim():
    dem, depression_id = _bowl_dem(depth=2.0)
    curves_df = depression_curves(dem, depression_id, cell_area=1.0, n_steps=15, max_stage_multiple=1.0)

    group = curves_df.sort_values("h")
    assert (group["volume"].diff().dropna() >= 0).all()
    assert (group["area"].diff().dropna() >= 0).all()
    assert group["h"].max() <= 2.0 + 1e-9


def test_depression_pour_points_finds_lowest_cell(tmp_path):
    dem, depression_id = _bowl_dem()
    profile = {
        "driver": "GTiff", "height": dem.shape[0], "width": dem.shape[1],
        "count": 1, "dtype": "float32", "crs": "EPSG:4326",
        "transform": from_origin(0, 0, 1, 1),
    }
    out_path = tmp_path / "pour_points.tif"

    depression_pour_points(dem, depression_id, profile, out_path)

    with rasterio.open(out_path) as src:
        pour = src.read(1)

    assert pour[4, 4] == 1
    assert np.count_nonzero(pour) == 1


def test_calibrate_depression_alpha_matches_manual_computation():
    depression_id = np.ones((2, 2), dtype=int)
    d100 = np.full((2, 2), 0.5)
    cell_area = 100.0
    p100_mean = {1: 0.05}
    contributing_area = {1: 4 * cell_area}

    alpha_df = calibrate_depression_alpha(d100, depression_id, cell_area, p100_mean, contributing_area)

    v_obs = float(np.sum(d100)) * cell_area
    v_rain = p100_mean[1] * contributing_area[1]
    assert alpha_df.iloc[0]["alpha"] == pytest.approx(v_obs / v_rain)


def test_render_pluvial_depth_reproduces_curve_at_calibration_point():
    dem, depression_id = _bowl_dem(depth=2.0)
    hd = depression_relative_elevation(dem, depression_id)
    cell_area = 1.0
    curves_df = depression_curves(dem, depression_id, cell_area, n_steps=30, max_stage_multiple=1.0)

    # pick a known (h, V) pair from the table and drive render_pluvial_depth
    # with alpha/rainfall/area chosen so V' reproduces exactly that volume.
    row = curves_df.iloc[len(curves_df) // 2]
    target_h, target_v = row["h"], row["volume"]

    alpha_by_depression = {1: 1.0}
    p_mean_by_depression = {1: target_v}  # alpha * p_mean * area == target_v with area=1
    area_by_depression = {1: 1.0}

    depth, extrapolated = render_pluvial_depth(
        hd, depression_id, curves_df, alpha_by_depression, p_mean_by_depression, area_by_depression,
    )

    expected_depth = np.clip(target_h - hd, 0, None)
    np.testing.assert_allclose(depth, expected_depth, atol=1e-6)
    assert not extrapolated.any()


def test_render_pluvial_depth_flags_extrapolation():
    dem, depression_id = _bowl_dem(depth=1.0)
    hd = depression_relative_elevation(dem, depression_id)
    curves_df = depression_curves(dem, depression_id, cell_area=1.0, n_steps=10, max_stage_multiple=1.0)

    huge_volume = curves_df["volume"].max() * 100
    depth, extrapolated = render_pluvial_depth(
        hd, depression_id, curves_df, {1: 1.0}, {1: huge_volume}, {1: 1.0},
    )

    assert extrapolated.any()


def test_render_pluvial_depth_skips_depression_missing_inputs():
    dem, depression_id = _bowl_dem()
    hd = depression_relative_elevation(dem, depression_id)
    curves_df = depression_curves(dem, depression_id, cell_area=1.0, n_steps=10, max_stage_multiple=1.0)

    depth, extrapolated = render_pluvial_depth(hd, depression_id, curves_df, {}, {}, {})

    # no calibration inputs -> stage is unknown, not "no flood": depth is NaN, not 0.
    assert np.all(np.isnan(depth))
    assert not extrapolated.any()
