import numpy as np
import pandas as pd
import pytest

from src.rainfall import (
    cmip6_delta_multiplier,
    fit_gev_from_lmoments,
    gev_quantile,
    gev_return_level,
    growth_factor,
    regional_growth_curves,
    return_period_multiplier_grid,
    scenario_multiplier,
    station_lmoments,
)

EULER_GAMMA = 0.5772156649015329


def test_station_lmoments_l1_is_the_mean():
    x = [10.0, 20.0, 30.0, 40.0, 50.0]
    lm = station_lmoments(x)
    assert lm["l1"] == pytest.approx(np.mean(x))
    assert lm["l2"] > 0


def test_station_lmoments_short_record_returns_nan():
    lm = station_lmoments([1.0, 2.0])
    assert np.isnan(lm["t3"])


def test_gev_from_lmoments_recovers_standard_gumbel_analytically():
    """Standard Gumbel's population L-moments are known constants: l1 = Euler's
    gamma, l2 = ln(2), t3 = ln(9/8)/ln(2). Feeding these in should recover
    shape k ~ 0 and reproduce the Gumbel quantile function."""
    l1 = EULER_GAMMA
    l2 = np.log(2)
    t3 = np.log(9 / 8) / np.log(2)

    params = fit_gev_from_lmoments(l1, l2, t3)

    assert params["shape"] == pytest.approx(0.0, abs=0.01)
    assert params["scale"] == pytest.approx(1.0, rel=0.05)
    assert params["loc"] == pytest.approx(0.0, abs=0.05)

    median = gev_quantile(params, 0.5)
    true_gumbel_median = -np.log(np.log(2))
    assert median == pytest.approx(true_gumbel_median, abs=0.05)


def test_gev_return_level_increases_with_return_period():
    params = {"loc": 0.0, "scale": 1.0, "shape": 0.0}
    q10 = gev_return_level(params, 10)
    q100 = gev_return_level(params, 100)
    q500 = gev_return_level(params, 500)
    assert q10 < q100 < q500


def test_regional_growth_curves_and_growth_factor():
    rng = np.random.default_rng(42)
    rows = []
    for station_id, region_id in [("s1", "r1"), ("s2", "r1"), ("s3", "r2")]:
        values = rng.gumbel(loc=50, scale=15, size=40)
        for v in values:
            rows.append({"station_id": station_id, "region_id": region_id, "value": v})
    annual_max_df = pd.DataFrame(rows)

    region_params = regional_growth_curves(annual_max_df)

    assert set(region_params.keys()) == {"r1", "r2"}
    for params in region_params.values():
        g10 = growth_factor(params, 10)
        g100 = growth_factor(params, 100)
        assert g100 > g10  # growth curve must increase with return period


def test_return_period_multiplier_grid_is_one_at_the_anchor():
    region_id_grid = np.array([[1, 1], [2, 2]])
    region_params = {
        1: {"loc": 0.0, "scale": 1.0, "shape": 0.0},
        2: {"loc": 5.0, "scale": 2.0, "shape": 0.1},
    }

    multiplier = return_period_multiplier_grid(region_id_grid, region_params, return_period_years=100,
                                                 anchor_return_period=100)

    np.testing.assert_allclose(multiplier, 1.0)


def test_return_period_multiplier_grid_above_one_for_larger_rp():
    region_id_grid = np.array([[1]])
    region_params = {1: {"loc": 0.0, "scale": 1.0, "shape": 0.0}}

    multiplier = return_period_multiplier_grid(region_id_grid, region_params, return_period_years=500,
                                                 anchor_return_period=100)

    assert multiplier[0, 0] > 1.0


def test_cmip6_delta_multiplier_uses_median_across_models():
    deltas = np.array([5.0, 10.0, 15.0])
    multiplier, spread = cmip6_delta_multiplier(deltas)

    assert multiplier == pytest.approx(1.10)
    assert spread == pytest.approx(np.std(deltas))


def test_cmip6_delta_multiplier_clipped_at_clausius_clapeyron_bound():
    deltas = np.array([25.0, 30.0, 35.0])  # median 30% at only 1 degree C warming
    multiplier, _ = cmip6_delta_multiplier(deltas, temp_delta_c=1.0, cc_rate_per_degc=0.07)

    assert multiplier == pytest.approx(1.07)


def test_cmip6_delta_multiplier_not_clipped_when_within_bound():
    deltas = np.array([3.0, 5.0, 7.0])
    multiplier, _ = cmip6_delta_multiplier(deltas, temp_delta_c=2.0, cc_rate_per_degc=0.07)

    assert multiplier == pytest.approx(1.05)


def test_scenario_multiplier_combines_both_axes():
    region_id_grid = np.array([[1]])
    region_params = {1: {"loc": 0.0, "scale": 1.0, "shape": 0.0}}
    deltas = np.array([10.0, 10.0, 10.0])

    multiplier_grid, spread = scenario_multiplier(
        region_id_grid, region_params, return_period_years=100,
        cmip6_delta_pct_by_model=deltas, anchor_return_period=100,
    )

    # at the anchor RP, only the CMIP6 axis should move the multiplier
    assert multiplier_grid[0, 0] == pytest.approx(1.10)
    assert spread == pytest.approx(0.0)
