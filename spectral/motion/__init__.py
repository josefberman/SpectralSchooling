"""Collective motion classification from raw trajectories."""

from __future__ import annotations

from spectral.motion.classifier import MotionClassifier, classify_motion
from spectral.motion.labels import (
    HYDRO_SUB_LABELS,
    MOTION_LABELS,
    MotionLabel,
    HydroSubLabel,
)

__all__ = [
    "HYDRO_SUB_LABELS",
    "MOTION_LABELS",
    "HydroSubLabel",
    "MotionClassifier",
    "MotionLabel",
    "classify_motion",
]
