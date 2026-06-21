"""Collective motion classification from raw trajectories."""

from __future__ import annotations

from spectral.motion.classifier import MotionClassifier, classify_motion
from spectral.motion.features import compute_motion_features
from spectral.motion.labels import MOTION_LABELS, MotionLabel

__all__ = [
    "MOTION_LABELS",
    "MotionClassifier",
    "MotionLabel",
    "classify_motion",
    "compute_motion_features",
]
