"""Step 3 — read the RP100 anchor into stage space, with the reconstruction QC
gate. Step 4 — calibrate the volume/rainfall coefficient alpha per catchment.

The Step 3 QC is the gate: do not proceed to national-scale calibration if it
fails. Reaches below CSI 0.80 at the 0.3 m threshold are usually steep terrain,
dense urban, or bad reach delineation — fix delineation before blaming the
method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.curves import anchor_stage_by_reach

CSI_THRESHOLDS = (0.15, 0.30, 0.50, 1.0)
CSI_GATE_THRESHOLD_M = 0.30
CSI_GATE_MIN = 0.80
ALPHA_EXPECTED_RANGE = (0.05, 0.4)
VOLUME_CROSS_CHECK_TOL = 0.05


def broadcast_reach_values(reach_id, values_by_reach):
    """Spread a per-reach Series (indexed by reach_id) across the full grid."""
    values_by_reach = values_by_reach.dropna()
    if values_by_reach.empty:
        return np.full(reach_id.shape, np.nan)

    max_id = int(max(reach_id.max(), values_by_reach.index.max()))
    table = np.full(max_id + 1, np.nan)
    idx = values_by_reach.index.to_numpy().astype(int)
    table[idx] = values_by_reach.to_numpy()

    flat_ids = np.clip(reach_id, 0, max_id)
    out = table[flat_ids]
    out[reach_id <= 0] = np.nan
    return out


def compute_h100(hand, reach_id, d100):
    """Per reach: h100 = median(d100 + HAND) over wet cells, and the IQR of that
    same quantity — a wide spread means the flat-water-surface assumption is
    failing there."""
    h100 = anchor_stage_by_reach(hand, reach_id, d100)  # median, from curves.py

    wet = (reach_id > 0) & (d100 > 0)
    df = pd.DataFrame({"reach_id": reach_id[wet], "stage": (d100 + hand)[wet]})
    iqr = df.groupby("reach_id")["stage"].apply(lambda s: s.quantile(0.75) - s.quantile(0.25))

    out = pd.DataFrame({"h100": h100, "stage_iqr": iqr}).reset_index()
    out = out.rename(columns={"index": "reach_id"})
    return out


def reconstruct_depth(hand, reach_id, h100_df):
    """d_recon = max(0, h100 - HAND), broadcast per reach across the grid."""
    h100_by_reach = h100_df.set_index("reach_id")["h100"]
    h100_grid = broadcast_reach_values(reach_id, h100_by_reach)
    return np.clip(h100_grid - hand, 0, None)


def _csi(obs_wet, recon_wet):
    hits = np.count_nonzero(obs_wet & recon_wet)
    misses = np.count_nonzero(obs_wet & ~recon_wet)
    false_alarms = np.count_nonzero(~obs_wet & recon_wet)
    denom = hits + misses + false_alarms
    return hits / denom if denom > 0 else np.nan


def reconstruction_qc(d100, d_recon, reach_id, h100_df, thresholds=CSI_THRESHOLDS):
    """Per-reach CSI at each threshold, RMSE over union-wet cells, and
    inundated area ratio. Merges in h100/stage_iqr from compute_h100 and flags
    reaches failing the CSI gate."""
    records = []
    for rid in h100_df["reach_id"]:
        mask = reach_id == rid
        obs = d100[mask]
        recon = d_recon[mask]

        row = {"reach_id": rid}
        for t in thresholds:
            row[f"csi_{t}"] = _csi(obs > t, recon > t)

        union_wet = (obs > 0) | (recon > 0)
        row["rmse"] = float(np.sqrt(np.mean((obs[union_wet] - recon[union_wet]) ** 2))) if union_wet.any() else np.nan

        area_obs = int(np.count_nonzero(obs > 0))
        area_recon = int(np.count_nonzero(recon > 0))
        row["area_ratio"] = area_recon / area_obs if area_obs > 0 else np.nan

        records.append(row)

    qc_df = pd.DataFrame.from_records(records).merge(h100_df, on="reach_id", how="left")
    gate_col = f"csi_{CSI_GATE_THRESHOLD_M}"
    qc_df["flag_low_csi"] = qc_df[gate_col] < CSI_GATE_MIN
    return qc_df


def volume_at_stage(curves_df, reach_id_value, h):
    """Forward lookup on the stored curve: V(h) for one reach, by interpolation."""
    group = curves_df[curves_df["reach_id"] == reach_id_value].sort_values("h")
    if group.empty:
        return np.nan
    return float(np.interp(h, group["h"].to_numpy(), group["volume"].to_numpy()))


def cross_check_volume(curves_df, h100_df, d100, reach_id, cell_area, tol=VOLUME_CROSS_CHECK_TOL):
    """V(h100) from the Step 2 curve should agree with sum(d100)*cell_area within
    a few percent. Large disagreement means HAND conditioning or reach
    assignment is wrong, not a calibration problem."""
    records = []
    for _, row in h100_df.iterrows():
        rid, h100 = row["reach_id"], row["h100"]
        mask = reach_id == rid
        v_direct = float(np.sum(d100[mask])) * cell_area
        v_curve = volume_at_stage(curves_df, rid, h100)
        rel_diff = abs(v_curve - v_direct) / v_direct if v_direct > 0 else np.nan
        records.append({
            "reach_id": rid, "v_direct": v_direct, "v_curve": v_curve,
            "rel_diff": rel_diff, "flag_volume_mismatch": bool(rel_diff is not np.nan and rel_diff > tol),
        })
    return pd.DataFrame.from_records(records)


def calibrate_alpha(d100, catchment_id, cell_area, p100_mean_by_catchment, catchment_area_by_id,
                     expected_range=ALPHA_EXPECTED_RANGE):
    """Per catchment:

        V_obs  = sum(d100) * cell_area
        V_rain = P100_mean * A_catchment
        alpha  = V_obs / V_rain

    alpha absorbs infiltration, routing losses, hyetograph shape, and the RMSI
    disaggregation — all held fixed. Values outside [0.05, 0.4] need
    investigation, not clamping, so they're flagged rather than corrected.
    """
    catchments = np.unique(catchment_id[catchment_id > 0])
    records = []
    for cid in catchments:
        mask = catchment_id == cid
        v_obs = float(np.sum(d100[mask])) * cell_area
        p100_mean = p100_mean_by_catchment.get(cid, np.nan)
        area = catchment_area_by_id.get(cid, np.nan)
        v_rain = p100_mean * area if np.isfinite(p100_mean) and np.isfinite(area) else np.nan
        alpha = v_obs / v_rain if v_rain and v_rain > 0 else np.nan
        in_range = expected_range[0] <= alpha <= expected_range[1] if np.isfinite(alpha) else False
        records.append({
            "catchment_id": cid, "v_obs": v_obs, "v_rain": v_rain, "alpha": alpha,
            "flag_alpha_out_of_range": not in_range,
        })
    return pd.DataFrame.from_records(records)


def write_calibration(qc_df, alpha_df, out_path):
    """Assemble the calibration.parquet deliverable: (catchment_id, alpha, h100,
    qc_metrics). catchment_id and reach_id share one ID space here — hand.py's
    subbasins step assigns each catchment the ID of the reach link it drains to.
    """
    calibration_df = alpha_df.rename(columns={"catchment_id": "reach_id"}).merge(
        qc_df, on="reach_id", how="outer",
    )
    calibration_df = calibration_df.rename(columns={"reach_id": "catchment_id"})
    calibration_df.to_parquet(out_path, index=False)
    return calibration_df


def main():
    import argparse

    from src.hand import read_raster

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hand_raster")
    parser.add_argument("reach_id_raster")
    parser.add_argument("d100_raster")
    parser.add_argument("curves_parquet")
    parser.add_argument("out_qc_csv")
    args = parser.parse_args()

    hand, profile = read_raster(args.hand_raster)
    reach_id, _ = read_raster(args.reach_id_raster)
    d100, _ = read_raster(args.d100_raster)
    cell_area = abs(profile["transform"].a * profile["transform"].e)
    curves_df = pd.read_parquet(args.curves_parquet)

    h100_df = compute_h100(hand, reach_id, d100)
    d_recon = reconstruct_depth(hand, reach_id, h100_df)
    qc_df = reconstruction_qc(d100, d_recon, reach_id, h100_df)
    volume_check_df = cross_check_volume(curves_df, h100_df, d100, reach_id, cell_area)
    qc_df = qc_df.merge(volume_check_df, on="reach_id", how="left")

    n_failed = int(qc_df["flag_low_csi"].sum())
    n_total = len(qc_df)
    print(f"reconstruction QC: {n_total - n_failed}/{n_total} reaches pass CSI>={CSI_GATE_MIN} at {CSI_GATE_THRESHOLD_M} m")
    if n_failed:
        print(f"GATE: {n_failed} reaches flagged — fix delineation before proceeding to national scale")

    qc_df.to_csv(args.out_qc_csv, index=False)
    print(f"wrote QC report to {args.out_qc_csv}")


if __name__ == "__main__":
    main()
