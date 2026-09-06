"""Step 6 — fluvial/pluvial domain split, plus the per-cell confidence flag
from the brief's Known Limitations section: driven by (a) distance from the
anchor, |log(P'/P100)|, and (b) regime class.
"""
from __future__ import annotations

import numpy as np

REGIME_FLUVIAL = 0
REGIME_PLUVIAL = 1

DEFAULT_HAND_PLUVIAL_THRESHOLD_M = 3.0
DEFAULT_FLOWACC_FLUVIAL_THRESHOLD_CELLS = 500  # matches hand.py's stream extraction threshold

DEFAULT_LOG_RATIO_LOW_CONFIDENCE = np.log(2.0)  # scenario rainfall > 2x or < 0.5x the anchor
DEFAULT_SLOPE_STEEP_THRESHOLD = 0.15  # ~8.5 degrees; HAND is a poor flow-path proxy above this
DEFAULT_LOW_RP_THRESHOLD_YEARS = 10  # below RP10, flow is channel-contained — a different regime


def classify_regime(hand, flow_acc, hand_threshold=DEFAULT_HAND_PLUVIAL_THRESHOLD_M,
                     flowacc_threshold=DEFAULT_FLOWACC_FLUVIAL_THRESHOLD_CELLS):
    """Two-way split per Step 6: fluvial cells are stage-driven by catchment
    volume (calibrate.py + render.py); pluvial cells are depression storage and
    need pluvial.py's separate local calibration.

        fluvial: low HAND, or well-connected to the flow network (high flow-acc)
        pluvial: everything else — high HAND, isolated depressions

    Without this split, urban and interior ponding is systematically
    over-scaled by the fluvial volume-balance calibration.
    """
    fluvial = (hand < hand_threshold) | (flow_acc >= flowacc_threshold)
    return np.where(fluvial, REGIME_FLUVIAL, REGIME_PLUVIAL).astype("int8")


def confidence_flag(regime, log_rainfall_ratio, slope=None, urban_mask=None,
                     return_period_years=None, coastal_mask=None,
                     log_ratio_threshold=DEFAULT_LOG_RATIO_LOW_CONFIDENCE,
                     slope_threshold=DEFAULT_SLOPE_STEEP_THRESHOLD,
                     low_rp_threshold=DEFAULT_LOW_RP_THRESHOLD_YEARS):
    """Low-confidence cells per the brief's Known Limitations:

    - far from the anchor scenario: |log(P'/P100)| beyond log_ratio_threshold
    - steep terrain or dense urban areas: HAND is a poor flow-path proxy there
    - below ~RP10: flow is channel-contained, unlike the anchor regime
    - coastal reaches: sea-level rise needs a stage offset, not this pipeline

    regime is accepted (and reserved as an extension point) even though the
    current rule doesn't discriminate by it — some limitations, e.g. steep
    terrain, apply within a single regime rather than across the split.

    Returns a boolean array, True where confidence is low.
    """
    low_confidence = np.abs(log_rainfall_ratio) > log_ratio_threshold

    if slope is not None:
        low_confidence = low_confidence | (slope > slope_threshold)
    if urban_mask is not None:
        low_confidence = low_confidence | urban_mask.astype(bool)
    if return_period_years is not None:
        low_confidence = low_confidence | (np.asarray(return_period_years) < low_rp_threshold)
    if coastal_mask is not None:
        low_confidence = low_confidence | coastal_mask.astype(bool)

    return low_confidence


def regime_summary(regime):
    """Fraction of cells in each regime class — a quick sanity check after
    classification, before wiring pluvial.py and render.py to the split."""
    total = regime.size
    return {
        "fluvial_frac": float(np.count_nonzero(regime == REGIME_FLUVIAL)) / total,
        "pluvial_frac": float(np.count_nonzero(regime == REGIME_PLUVIAL)) / total,
    }
