import time

import numpy as np
import pytest

from src.curves import (
    anchor_stage_by_reach,
    check_monotonic,
    index_curves,
    invert_stage,
    invert_stage_indexed,
    stage_volume_area_curves,
)


def test_anchor_stage_recovers_flat_water_surface():
    hand = np.array([[0.0, 0.3, 0.6, 0.9], [0.1, 0.4, 0.7, 1.0]])
    reach_id = np.ones_like(hand, dtype=int)
    h100_true = 0.8
    d100 = np.clip(h100_true - hand, 0, None)

    anchor = anchor_stage_by_reach(hand, reach_id, d100)

    assert 1 in anchor.index
    assert anchor.loc[1] == pytest.approx(h100_true, abs=1e-9)


def test_curves_are_monotonic_and_area_grows_with_stage():
    rng = np.random.default_rng(0)
    hand = rng.uniform(0, 3, size=(50, 50))
    reach_id = np.ones_like(hand, dtype=int)
    cell_area = 900.0  # 30 m cells

    curves_df = stage_volume_area_curves(hand, reach_id, cell_area, n_steps=15)

    assert check_monotonic(curves_df) == []
    group = curves_df.sort_values("h")
    assert group["area"].iloc[-1] > group["area"].iloc[0]
    assert group["volume"].iloc[-1] > group["volume"].iloc[0]


def test_flat_floodplain_has_sublinear_depth_response():
    """A wide flat plain should spread volume laterally: area grows much faster
    than depth for the same added volume, versus a steep confined valley."""
    flat_hand = np.linspace(0, 1, 10000)
    steep_hand = np.geomspace(0.01, 10, 10000)

    flat = stage_volume_area_curves(flat_hand, np.ones_like(flat_hand, dtype=int), cell_area=1.0, n_steps=10)
    steep = stage_volume_area_curves(steep_hand, np.ones_like(steep_hand, dtype=int), cell_area=1.0, n_steps=10)

    flat_area_ratio = flat["area"].iloc[-1] / flat["area"].iloc[len(flat) // 2]
    steep_area_ratio = steep["area"].iloc[-1] / steep["area"].iloc[len(steep) // 2]

    assert flat_area_ratio > steep_area_ratio


def test_invert_stage_round_trips_a_tabulated_point():
    hand = np.linspace(0, 2, 1000)
    reach_id = np.ones_like(hand, dtype=int)
    curves_df = stage_volume_area_curves(hand, reach_id, cell_area=1.0, n_steps=20)

    mid_row = curves_df.iloc[len(curves_df) // 2]
    h_recovered, extrapolated = invert_stage(curves_df, 1, mid_row["volume"])

    assert h_recovered == pytest.approx(mid_row["h"], abs=1e-6)
    assert not extrapolated


def test_invert_stage_flags_extrapolation():
    hand = np.linspace(0, 1, 100)
    reach_id = np.ones_like(hand, dtype=int)
    curves_df = stage_volume_area_curves(hand, reach_id, cell_area=1.0, n_steps=10)

    max_volume = curves_df["volume"].max()
    _, extrapolated = invert_stage(curves_df, 1, max_volume * 10)

    assert extrapolated


def test_invert_stage_missing_reach_raises():
    hand = np.linspace(0, 1, 10)
    reach_id = np.ones_like(hand, dtype=int)
    curves_df = stage_volume_area_curves(hand, reach_id, cell_area=1.0, n_steps=5)

    with pytest.raises(KeyError):
        invert_stage(curves_df, 999, 1.0)


def test_stage_volume_area_curves_scales_to_many_reaches():
    """Regression test: an earlier implementation masked the whole grid once
    per reach (hand[reach_id == rid] in a loop), which is O(n_reaches *
    n_cells) and took an unusable amount of time on a real basin's ~59,000
    reaches. This should stay roughly linear in grid size, not quadratic in
    reach count."""
    rng = np.random.default_rng(0)
    n_reaches = 5000
    cells_per_reach = 100
    hand = rng.uniform(0, 5, size=n_reaches * cells_per_reach)
    reach_id = np.repeat(np.arange(1, n_reaches + 1), cells_per_reach)

    start = time.perf_counter()
    curves_df = stage_volume_area_curves(hand, reach_id, cell_area=900.0, n_steps=20)
    elapsed = time.perf_counter() - start

    assert curves_df["reach_id"].nunique() == n_reaches
    assert elapsed < 10.0, f"took {elapsed:.1f}s for {n_reaches} reaches -- looks quadratic again"


def test_index_curves_matches_direct_lookup():
    hand = np.linspace(0, 2, 1000)
    reach_id = np.ones_like(hand, dtype=int)
    curves_df = stage_volume_area_curves(hand, reach_id, cell_area=1.0, n_steps=20)

    mid_row = curves_df.iloc[len(curves_df) // 2]
    indexed = index_curves(curves_df)

    h_direct, extrap_direct = invert_stage(curves_df, 1, mid_row["volume"])
    h_indexed, extrap_indexed = invert_stage_indexed(indexed, 1, mid_row["volume"])

    assert h_indexed == pytest.approx(h_direct)
    assert extrap_indexed == extrap_direct


def test_invert_stage_indexed_missing_reach_raises():
    indexed = {1: (np.array([0.0, 1.0]), np.array([0.0, 10.0]))}
    with pytest.raises(KeyError):
        invert_stage_indexed(indexed, 999, 5.0)
