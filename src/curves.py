"""Step 2 — stage-volume-area curves per reach.

This is the piece the naive per-pixel rainfall-scaling method has no equivalent
of: it encodes whether a reach is a confined valley (steep V(h), near-linear
depth response) or a flat plain (A(h) explodes, volume spreads laterally at
~constant depth).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_N_STEPS = 20
DEFAULT_MAX_STAGE_MULTIPLE = 1.5


def anchor_stage_by_reach(hand, reach_id, d100):
    """Median of (d100 + HAND) over wet cells, per reach.

    Used here only to size each reach's stage range; calibrate.py reuses this
    same function as the authoritative h100 for Step 3.
    """
    wet = (reach_id > 0) & (d100 > 0)
    df = pd.DataFrame({
        "reach_id": reach_id[wet],
        "stage": (d100 + hand)[wet],
    })
    return df.groupby("reach_id")["stage"].median()


def stage_volume_area_curves(hand, reach_id, cell_area, anchor_stage=None,
                              n_steps=DEFAULT_N_STEPS,
                              max_stage_multiple=DEFAULT_MAX_STAGE_MULTIPLE):
    """Per reach, over n_steps stage increments spanning 0 to ~max_stage_multiple
    times the observed anchor stage:

        A(h) = count(HAND < h) * cell_area
        V(h) = sum(h - HAND, where HAND < h) * cell_area

    Falls back to the reach's own max HAND when no anchor stage is available
    (e.g. a reach with no wet anchor cells). Returns a long-format DataFrame:
    (reach_id, h, volume, area).
    """
    valid = (reach_id > 0) & np.isfinite(hand)
    flat_reach = reach_id[valid]
    flat_hand = hand[valid]

    # Group by reach via a single sort, rather than re-scanning the whole grid
    # once per reach (hand[reach_id == rid] in a loop) -- that's O(n_reaches *
    # n_cells) and grinds to a halt with tens of thousands of reaches.
    order = np.argsort(flat_reach, kind="stable")
    sorted_reach, sorted_hand = flat_reach[order], flat_hand[order]
    boundaries = np.flatnonzero(np.diff(sorted_reach)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(sorted_reach)]))

    records = []
    for rid, start, end in zip(sorted_reach[starts], starts, ends):
        hand_r = sorted_hand[start:end]

        if anchor_stage is not None and rid in anchor_stage.index:
            h_max = float(anchor_stage.loc[rid]) * max_stage_multiple
        else:
            h_max = float(np.nanmax(hand_r))
        h_max = max(h_max, 1e-6)

        hand_r_sorted = np.sort(hand_r)
        cumsum = np.cumsum(hand_r_sorted)
        stages = np.linspace(0.0, h_max, n_steps)
        for h in stages:
            idx = int(np.searchsorted(hand_r_sorted, h, side="left"))
            area = idx * cell_area
            below_sum = cumsum[idx - 1] if idx > 0 else 0.0
            volume = (h * idx - below_sum) * cell_area
            records.append((rid, h, volume, area))

    return pd.DataFrame.from_records(records, columns=["reach_id", "h", "volume", "area"])


def check_monotonic(curves_df):
    """V(h) and A(h) must be non-decreasing with h within each reach.
    Returns the list of reach_ids that violate this (should be empty)."""
    bad = []
    for rid, group in curves_df.sort_values("h").groupby("reach_id"):
        if (group["volume"].diff().dropna() < -1e-9).any() or (group["area"].diff().dropna() < -1e-9).any():
            bad.append(rid)
    return bad


def invert_stage(curves_df, reach_id_value, target_volume):
    """Monotone interpolation: given a target volume, solve for stage h on the
    stored curve for one reach.

    Returns (h, extrapolated) — extrapolated is True if target_volume fell
    outside the tabulated range and h was clamped to the nearest end.
    """
    group = curves_df[curves_df["reach_id"] == reach_id_value].sort_values("h")
    if group.empty:
        raise KeyError(f"no curve stored for reach_id={reach_id_value}")

    v = group["volume"].to_numpy()
    h = group["h"].to_numpy()
    extrapolated = bool(target_volume < v[0] or target_volume > v[-1])
    h_interp = float(np.interp(target_volume, v, h))
    return h_interp, extrapolated


def index_curves(curves_df):
    """Group the curve table by reach once, for repeated O(log k) lookups
    (k = points per reach) instead of re-filtering the whole table on every
    call — curves_df[curves_df["reach_id"] == rid] inside a per-reach loop is
    O(n_reaches * n_rows) and is the dominant cost once a basin has tens of
    thousands of reaches. Use this ahead of a loop calling invert_stage or
    volume_at_stage many times; a single lookup can still use those directly.

    Returns {reach_id: (h_sorted, v_sorted)}.
    """
    return {
        rid: (group["h"].to_numpy(), group["volume"].to_numpy())
        for rid, group in curves_df.sort_values("h").groupby("reach_id")
    }


def invert_stage_indexed(indexed_curves, reach_id_value, target_volume):
    """Same as invert_stage, against an index built once by index_curves."""
    entry = indexed_curves.get(reach_id_value)
    if entry is None:
        raise KeyError(f"no curve stored for reach_id={reach_id_value}")
    h, v = entry
    extrapolated = bool(target_volume < v[0] or target_volume > v[-1])
    return float(np.interp(target_volume, v, h)), extrapolated


def volume_at_stage_indexed(indexed_curves, reach_id_value, h_value):
    """Forward lookup — V(h) for one reach — against an index built once by
    index_curves."""
    entry = indexed_curves.get(reach_id_value)
    if entry is None:
        return np.nan
    h, v = entry
    return float(np.interp(h_value, h, v))


def write_curves(curves_df, out_path):
    curves_df.to_parquet(out_path, index=False)


def main():
    import argparse

    from src.hand import read_raster

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hand_raster")
    parser.add_argument("reach_id_raster")
    parser.add_argument("d100_raster", help="RP100 2030 anchor depth grid, used to size each reach's stage range")
    parser.add_argument("out_parquet")
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--max-stage-multiple", type=float, default=DEFAULT_MAX_STAGE_MULTIPLE)
    args = parser.parse_args()

    hand, profile = read_raster(args.hand_raster)
    reach_id, _ = read_raster(args.reach_id_raster)
    d100, _ = read_raster(args.d100_raster)

    cell_area = abs(profile["transform"].a * profile["transform"].e)
    anchor_stage = anchor_stage_by_reach(hand, reach_id, d100)

    curves_df = stage_volume_area_curves(
        hand, reach_id, cell_area, anchor_stage=anchor_stage,
        n_steps=args.n_steps, max_stage_multiple=args.max_stage_multiple,
    )

    bad = check_monotonic(curves_df)
    if bad:
        print(f"WARNING: non-monotonic V(h)/A(h) in {len(bad)} reaches, e.g. {bad[:10]}")

    write_curves(curves_df, args.out_parquet)
    print(f"wrote {len(curves_df)} rows across {curves_df['reach_id'].nunique()} reaches to {args.out_parquet}")


if __name__ == "__main__":
    main()
