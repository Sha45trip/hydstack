import numpy as np
import pytest

from validate.reconstruct import gate_passed, run_reconstruction_gate, summarize


def _flat_water_reach(reach_id_value, h100_true, shape, seed):
    rng = np.random.default_rng(seed)
    hand = rng.uniform(0, 1.5, size=shape)
    reach_id = np.full(shape, reach_id_value, dtype=int)
    d100 = np.clip(h100_true - hand, 0, None)
    return hand, reach_id, d100


def _two_reach_basin(good_shape=(20, 20), bad_shape=(20, 20)):
    """Reach 1 reconstructs perfectly (synthetic flat water surface). Reach 2's
    wetting pattern is random and uncorrelated with HAND — violating the
    flat-water-surface assumption the way real steep terrain or bad
    delineation would — so its reconstruction should fail the CSI gate."""
    hand1, reach1, d100_1 = _flat_water_reach(1, 0.8, good_shape, seed=0)

    rng = np.random.default_rng(1)
    hand2 = rng.uniform(0, 1.5, size=bad_shape)
    reach2 = np.full(bad_shape, 2, dtype=int)
    d100_2 = rng.uniform(0, 1.0, size=bad_shape) * rng.integers(0, 2, size=bad_shape)

    hand = np.concatenate([hand1, hand2], axis=0)
    reach_id = np.concatenate([reach1, reach2], axis=0)
    d100 = np.concatenate([d100_1, d100_2], axis=0)
    return hand, reach_id, d100


def test_run_reconstruction_gate_passes_when_all_reaches_reconstruct_well():
    hand, reach_id, d100 = _flat_water_reach(1, 0.8, (30, 30), seed=5)
    qc_df, passed = run_reconstruction_gate(hand, reach_id, d100, cell_area=900.0)

    assert passed
    assert not qc_df["flag_low_csi"].any()


def test_run_reconstruction_gate_fails_with_a_bad_reach_at_full_pass_rate():
    hand, reach_id, d100 = _two_reach_basin()
    qc_df, passed = run_reconstruction_gate(hand, reach_id, d100, cell_area=900.0, min_pass_rate=1.0)

    assert not passed
    assert qc_df.set_index("reach_id").loc[1, "flag_low_csi"] == False  # noqa: E712
    assert qc_df.set_index("reach_id").loc[2, "flag_low_csi"] == True  # noqa: E712


def test_gate_passed_respects_min_pass_rate_threshold():
    hand, reach_id, d100 = _two_reach_basin()
    qc_df, _ = run_reconstruction_gate(hand, reach_id, d100, cell_area=900.0)

    assert gate_passed(qc_df, min_pass_rate=0.4)  # 1 of 2 reaches passing clears a 40% bar
    assert not gate_passed(qc_df, min_pass_rate=0.9)  # but not a 90% bar


def test_gate_passed_empty_qc_df_is_false():
    import pandas as pd

    assert not gate_passed(pd.DataFrame())


def test_summarize_reports_expected_keys():
    hand, reach_id, d100 = _flat_water_reach(1, 0.8, (20, 20), seed=2)
    qc_df, _ = run_reconstruction_gate(hand, reach_id, d100, cell_area=900.0)

    stats = summarize(qc_df)

    assert stats["n_reaches"] == 1
    assert stats["n_failed"] == 0
    assert stats["pass_rate"] == pytest.approx(1.0)
    assert stats["median_csi"] == pytest.approx(1.0)
