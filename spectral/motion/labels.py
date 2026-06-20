"""Motion category label definitions."""

from __future__ import annotations

from enum import IntEnum


class MotionLabel(IntEnum):
    UNKNOWN = 0
    TRAVELING_POLARIZED = 1
    MILLING = 2
    SWARMING = 3
    FOUNTAIN_EVASION = 4
    EXPANSION_BURST = 5
    CONTRACTION_COMPACTION = 6


MOTION_LABELS = {
    MotionLabel.UNKNOWN: "unknown",
    MotionLabel.TRAVELING_POLARIZED: "traveling_polarized",
    MotionLabel.MILLING: "milling",
    MotionLabel.SWARMING: "swarming",
    MotionLabel.FOUNTAIN_EVASION: "fountain_evasion",
    MotionLabel.EXPANSION_BURST: "expansion_burst",
    MotionLabel.CONTRACTION_COMPACTION: "contraction_compaction",
}

FEATURE_NAMES = [
    "directional_polarization",
    "rotational_polarization",
    "norm_angular_momentum",
    "tangential_order",
    "mean_radial_velocity",
    "spread",
]
