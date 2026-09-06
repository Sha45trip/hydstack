"""Reconstruction validation — the primary gate (Step 3 of the method).

Runs calibrate.py's h100 / reconstruction / QC / volume cross-check across a
whole basin and reports a basin-wide pass/fail against the CSI gate, on top of
the per-reach flags calibrate.py already produces. Do not proceed to
national-scale calibration if this fails.
"""
from __future__ import annotations

from src.calibrate import (
    CSI_GATE_THRESHOLD_M,
    compute_h100,
    cross_check_volume,
    reconstruct_depth,
    reconstruction_qc,
)

DEFAULT_MIN_PASS_RATE = 0.95


def run_reconstruction_gate(hand, reach_id, d100, cell_area, curves_df=None,
                             min_pass_rate=DEFAULT_MIN_PASS_RATE):
    """Full Step 3 pipeline: h100 -> reconstruction -> QC -> (optional) volume
    cross-check. Returns (qc_df, passed) where passed is whether the
    basin-wide pass rate clears the gate."""
    h100_df = compute_h100(hand, reach_id, d100)
    d_recon = reconstruct_depth(hand, reach_id, h100_df)
    qc_df = reconstruction_qc(d100, d_recon, reach_id, h100_df)

    if curves_df is not None:
        volume_check_df = cross_check_volume(curves_df, h100_df, d100, reach_id, cell_area)
        qc_df = qc_df.merge(volume_check_df, on="reach_id", how="left")

    passed = gate_passed(qc_df, min_pass_rate=min_pass_rate)
    return qc_df, passed


def gate_passed(qc_df, min_pass_rate=DEFAULT_MIN_PASS_RATE):
    """The gate passes when at least min_pass_rate of reaches clear the CSI
    threshold. Individual failing reaches stay flagged in qc_df for targeted
    delineation fixes — this is a basin-wide go/no-go on top of that."""
    if qc_df.empty:
        return False
    pass_rate = 1 - qc_df["flag_low_csi"].mean()
    return bool(pass_rate >= min_pass_rate)


def summarize(qc_df):
    """Aggregate stats for a human-readable report."""
    if qc_df.empty:
        return {"n_reaches": 0, "n_failed": 0, "pass_rate": float("nan"),
                "median_csi": float("nan"), "median_rmse": float("nan"), "median_area_ratio": float("nan")}

    csi_col = f"csi_{CSI_GATE_THRESHOLD_M}"
    return {
        "n_reaches": len(qc_df),
        "n_failed": int(qc_df["flag_low_csi"].sum()),
        "pass_rate": float(1 - qc_df["flag_low_csi"].mean()),
        "median_csi": float(qc_df[csi_col].median()),
        "median_rmse": float(qc_df["rmse"].median()),
        "median_area_ratio": float(qc_df["area_ratio"].median()),
    }


def main():
    import sys
    import argparse

    import pandas as pd

    from src.hand import read_raster

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hand_raster")
    parser.add_argument("reach_id_raster")
    parser.add_argument("d100_raster")
    parser.add_argument("out_report_csv")
    parser.add_argument("--curves-parquet", default=None, help="enables the V(h100) cross-check")
    parser.add_argument("--min-pass-rate", type=float, default=DEFAULT_MIN_PASS_RATE)
    args = parser.parse_args()

    hand, profile = read_raster(args.hand_raster)
    reach_id, _ = read_raster(args.reach_id_raster)
    d100, _ = read_raster(args.d100_raster)
    cell_area = abs(profile["transform"].a * profile["transform"].e)

    curves_df = pd.read_parquet(args.curves_parquet) if args.curves_parquet else None

    qc_df, passed = run_reconstruction_gate(
        hand, reach_id, d100, cell_area, curves_df=curves_df, min_pass_rate=args.min_pass_rate,
    )
    qc_df.to_csv(args.out_report_csv, index=False)

    for key, value in summarize(qc_df).items():
        print(f"{key}: {value}")

    if passed:
        print(f"GATE PASSED (>= {args.min_pass_rate:.0%} of reaches clear CSI @ {CSI_GATE_THRESHOLD_M} m)")
        sys.exit(0)

    print(f"GATE FAILED (< {args.min_pass_rate:.0%} of reaches clear CSI @ {CSI_GATE_THRESHOLD_M} m) "
          "— fix delineation before scaling")
    sys.exit(1)


if __name__ == "__main__":
    main()
