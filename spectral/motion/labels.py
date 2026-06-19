"""Motion category label definitions."""

from __future__ import annotations

from enum import IntEnum


class MotionLabel(IntEnum):
    UNKNOWN = 0
    TRAVELING_POLARIZED = 1
    MILLING = 2
    SWARMING = 3
    FOUNTAIN_EVASION = 4
    HYDRODYNAMIC = 5


class HydroSubLabel(IntEnum):
    NONE = -1
    INLINE = 0
    PHALANX = 1
    STAGGERED = 2


MOTION_LABELS = {
    MotionLabel.UNKNOWN: "unknown",
    MotionLabel.TRAVELING_POLARIZED: "traveling_polarized",
    MotionLabel.MILLING: "milling",
    MotionLabel.SWARMING: "swarming",
    MotionLabel.FOUNTAIN_EVASION: "fountain_evasion",
    MotionLabel.HYDRODYNAMIC: "hydrodynamic",
}

HYDRO_SUB_LABELS = {
    HydroSubLabel.NONE: "none",
    HydroSubLabel.INLINE: "inline",
    HydroSubLabel.PHALANX: "phalanx",
    HydroSubLabel.STAGGERED: "staggered",
}

FEATURE_NAMES = [
    "polarization",
    "angular_momentum",
    "tangential_order",
    "mean_radial_velocity",
    "spread",
    "heading_variance",
    "inline_score",
    "phalanx_score",
    "staggered_score",
]
