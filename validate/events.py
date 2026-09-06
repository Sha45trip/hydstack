"""Observed-event validation: compare rendered depth against Sentinel-1
derived inundation extent for real events (Kerala 2018, Assam 2022, Bihar),
the scenario driven with IMD rainfall as input rather than a synthetic
multiplier.

Sentinel-1 flood maps are binary extent, not depth, so the primary metric is
CSI at the 0.3 m threshold on the rendered depth grid — plus hit rate (POD),
false alarm ratio, and total inundated area error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CSI_THRESHOLDS = (0.15, 0.30, 0.50, 1.0)
EVENT_GATE_THRESHOLD_M = 0.30
EVENT_GATE_MIN_CSI = 0.5  # looser than the Step 3 reconstruction gate — see event_gate_passed


def binary_event_metrics(rendered_depth, observed_wet_mask, threshold):
    """Contingency-table metrics at one depth threshold, against a binary
    observed wet/dry mask. Cells where rendered_depth is NaN (unknown / not
    rendered) are excluded from every count."""
    known = np.isfinite(rendered_depth)
    predicted_wet = known & (rendered_depth > threshold)
    observed_wet = known & observed_wet_mask.astype(bool)

    hits = int(np.count_nonzero(predicted_wet & observed_wet))
    misses = int(np.count_nonzero(observed_wet & ~predicted_wet))
    false_alarms = int(np.count_nonzero(predicted_wet & ~observed_wet))

    denom_csi = hits + misses + false_alarms
    csi = hits / denom_csi if denom_csi > 0 else np.nan

    denom_pod = hits + misses
    hit_rate = hits / denom_pod if denom_pod > 0 else np.nan

    denom_far = hits + false_alarms
    far = false_alarms / denom_far if denom_far > 0 else np.nan

    area_observed = int(np.count_nonzero(observed_wet))
    area_predicted = int(np.count_nonzero(predicted_wet))
    area_error = (area_predicted - area_observed) / area_observed if area_observed > 0 else np.nan

    return {
        "threshold": threshold, "hits": hits, "misses": misses, "false_alarms": false_alarms,
        "csi": csi, "hit_rate": hit_rate, "far": far,
        "area_observed_cells": area_observed, "area_predicted_cells": area_predicted,
        "area_error": area_error,
    }


def evaluate_event(rendered_depth, observed_wet_mask, thresholds=DEFAULT_CSI_THRESHOLDS):
    """Metrics at each threshold, for one event."""
    return pd.DataFrame([binary_event_metrics(rendered_depth, observed_wet_mask, t) for t in thresholds])


def evaluate_events(events, thresholds=DEFAULT_CSI_THRESHOLDS):
    """events: {event_name: (rendered_depth, observed_wet_mask)}. Returns one
    concatenated DataFrame with an event column — a single Kerala 2018 /
    Assam 2022 / Bihar report."""
    frames = []
    for name, (rendered_depth, observed_wet_mask) in events.items():
        df = evaluate_event(rendered_depth, observed_wet_mask, thresholds=thresholds)
        df.insert(0, "event", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def event_gate_passed(events_df, threshold=EVENT_GATE_THRESHOLD_M, min_csi=EVENT_GATE_MIN_CSI):
    """A much looser bar than calibrate.py's reconstruction gate: observed
    events bring real rainfall-estimation and SAR-detection error the
    synthetic reconstruction QC never sees, so a lower CSI here doesn't
    indict the method the way it would in Step 3."""
    subset = events_df[events_df["threshold"] == threshold]
    if subset.empty:
        return False
    return bool((subset["csi"] >= min_csi).all())


def main():
    import argparse

    from src.hand import read_raster

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_name")
    parser.add_argument("rendered_depth_raster")
    parser.add_argument("observed_wet_mask_raster",
                         help="binary Sentinel-1 derived inundation extent, aligned to the depth raster")
    parser.add_argument("out_report_csv")
    args = parser.parse_args()

    depth, _ = read_raster(args.rendered_depth_raster)
    observed_mask, _ = read_raster(args.observed_wet_mask_raster)

    events_df = evaluate_event(depth, observed_mask.astype(bool))
    events_df.insert(0, "event", args.event_name)
    events_df.to_csv(args.out_report_csv, index=False)

    gate_row = events_df[events_df["threshold"] == EVENT_GATE_THRESHOLD_M].iloc[0]
    print(f"{args.event_name}: CSI@{EVENT_GATE_THRESHOLD_M}m={gate_row['csi']:.3f} "
          f"hit_rate={gate_row['hit_rate']:.3f} far={gate_row['far']:.3f}")


if __name__ == "__main__":
    main()
