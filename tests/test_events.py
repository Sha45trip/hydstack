import numpy as np
import pytest

from validate.events import (
    binary_event_metrics,
    event_gate_passed,
    evaluate_event,
    evaluate_events,
)


def test_binary_event_metrics_perfect_match():
    depth = np.array([[0.5, 0.0], [0.6, 0.0]])
    observed = np.array([[True, False], [True, False]])

    m = binary_event_metrics(depth, observed, threshold=0.3)

    assert m["hits"] == 2
    assert m["misses"] == 0
    assert m["false_alarms"] == 0
    assert m["csi"] == pytest.approx(1.0)
    assert m["hit_rate"] == pytest.approx(1.0)
    assert m["far"] == pytest.approx(0.0)
    assert m["area_error"] == pytest.approx(0.0)


def test_binary_event_metrics_counts_misses_and_false_alarms():
    # observed wet at (0,0) and (1,0); predicted wet at (0,0) and (0,1)
    depth = np.array([[0.5, 0.5], [0.0, 0.0]])
    observed = np.array([[True, False], [True, False]])

    m = binary_event_metrics(depth, observed, threshold=0.3)

    assert m["hits"] == 1       # (0,0)
    assert m["misses"] == 1     # (1,0): observed wet, predicted dry
    assert m["false_alarms"] == 1  # (0,1): predicted wet, observed dry
    assert m["csi"] == pytest.approx(1 / 3)
    assert m["hit_rate"] == pytest.approx(0.5)
    assert m["far"] == pytest.approx(0.5)


def test_binary_event_metrics_excludes_nan_depth_cells():
    depth = np.array([[np.nan, 0.5]])
    observed = np.array([[True, True]])

    m = binary_event_metrics(depth, observed, threshold=0.3)

    assert m["hits"] == 1
    assert m["area_observed_cells"] == 1  # the NaN cell is excluded, not counted as a miss


def test_evaluate_event_covers_all_default_thresholds():
    depth = np.full((5, 5), 0.4)
    observed = np.full((5, 5), True)

    df = evaluate_event(depth, observed)

    assert list(df["threshold"]) == [0.15, 0.30, 0.50, 1.0]
    # depth 0.4 clears 0.15 and 0.30 but not 0.50 or 1.0
    assert df.set_index("threshold").loc[0.15, "csi"] == pytest.approx(1.0)
    assert df.set_index("threshold").loc[0.30, "csi"] == pytest.approx(1.0)
    assert df.set_index("threshold").loc[0.50, "csi"] == pytest.approx(0.0)


def test_evaluate_events_concatenates_with_event_column():
    depth = np.full((3, 3), 0.5)
    observed = np.full((3, 3), True)

    df = evaluate_events({"kerala_2018": (depth, observed), "assam_2022": (depth, observed)})

    assert set(df["event"]) == {"kerala_2018", "assam_2022"}
    assert len(df) == 2 * 4  # 2 events x 4 default thresholds


def test_event_gate_passed_true_when_all_events_clear_bar():
    depth = np.full((4, 4), 0.5)
    observed = np.full((4, 4), True)
    events_df = evaluate_events({"e1": (depth, observed), "e2": (depth, observed)})

    assert event_gate_passed(events_df, min_csi=0.5)


def test_event_gate_passed_false_when_one_event_misses():
    good_depth = np.full((4, 4), 0.5)
    good_observed = np.full((4, 4), True)

    bad_depth = np.zeros((4, 4))
    bad_observed = np.full((4, 4), True)  # everything missed -> csi 0 at 0.3m

    events_df = evaluate_events({"good": (good_depth, good_observed), "bad": (bad_depth, bad_observed)})

    assert not event_gate_passed(events_df, min_csi=0.5)


def test_event_gate_passed_false_for_empty_report():
    import pandas as pd

    assert not event_gate_passed(pd.DataFrame(columns=["threshold", "csi"]))
