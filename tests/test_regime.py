import numpy as np

from src.regime import (
    REGIME_FLUVIAL,
    REGIME_PLUVIAL,
    classify_regime,
    confidence_flag,
    regime_summary,
)


def test_low_hand_is_fluvial_even_with_low_flow_acc():
    hand = np.array([[0.5]])
    flow_acc = np.array([[1]])
    regime = classify_regime(hand, flow_acc, hand_threshold=3.0, flowacc_threshold=500)
    assert regime[0, 0] == REGIME_FLUVIAL


def test_high_flow_acc_is_fluvial_even_with_high_hand():
    hand = np.array([[10.0]])
    flow_acc = np.array([[1000]])
    regime = classify_regime(hand, flow_acc, hand_threshold=3.0, flowacc_threshold=500)
    assert regime[0, 0] == REGIME_FLUVIAL


def test_high_hand_and_low_flow_acc_is_pluvial():
    hand = np.array([[10.0]])
    flow_acc = np.array([[1]])
    regime = classify_regime(hand, flow_acc, hand_threshold=3.0, flowacc_threshold=500)
    assert regime[0, 0] == REGIME_PLUVIAL


def test_regime_summary_fractions_sum_to_one():
    hand = np.array([[0.5, 10.0], [10.0, 0.5]])
    flow_acc = np.array([[1, 1], [1, 1]])
    regime = classify_regime(hand, flow_acc, hand_threshold=3.0, flowacc_threshold=500)

    summary = regime_summary(regime)

    assert summary["fluvial_frac"] == 0.5
    assert summary["pluvial_frac"] == 0.5
    assert summary["fluvial_frac"] + summary["pluvial_frac"] == 1.0


def test_confidence_flag_flags_large_rainfall_departure():
    regime = np.zeros((1, 3), dtype="int8")
    log_ratio = np.array([[0.0, 1.5, -1.5]])  # ~1x, ~4.5x, ~0.22x the anchor

    low_conf = confidence_flag(regime, log_ratio, log_ratio_threshold=np.log(2.0))

    assert not low_conf[0, 0]
    assert low_conf[0, 1]
    assert low_conf[0, 2]


def test_confidence_flag_combines_all_signals():
    regime = np.zeros((1, 1), dtype="int8")
    log_ratio = np.array([[0.0]])  # at the anchor, wouldn't trip on its own
    slope = np.array([[0.2]])  # steep -> should trip
    urban_mask = np.array([[False]])
    rp = np.array([[100]])
    coastal_mask = np.array([[False]])

    low_conf = confidence_flag(regime, log_ratio, slope=slope, urban_mask=urban_mask,
                                return_period_years=rp, coastal_mask=coastal_mask)

    assert low_conf[0, 0]


def test_confidence_flag_low_return_period_trips_flag():
    regime = np.zeros((1, 1), dtype="int8")
    log_ratio = np.array([[0.0]])
    rp = np.array([[5]])  # below RP10

    low_conf = confidence_flag(regime, log_ratio, return_period_years=rp)

    assert low_conf[0, 0]
