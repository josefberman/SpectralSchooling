"""Tests for motion classification."""

from __future__ import annotations

import numpy as np

from spectral.motion.classifier import classify_motion
from spectral.motion.labels import MotionLabel
from spectral.types import FishSchoolTrajectory


def _traj(positions: np.ndarray, velocities: np.ndarray) -> FishSchoolTrajectory:
    valid = np.isfinite(positions).all(axis=2)
    return FishSchoolTrajectory(positions=positions, velocities=velocities, valid_mask=valid, fps=30.0)


def test_polarized_traveling():
    T, N = 20, 8
    pos = np.zeros((T, N, 2))
    vel = np.tile([10.0, 0.0], (T, N, 1))
    for t in range(T):
        pos[t, :, 0] = np.arange(N) * 20 + t * 10
        pos[t, :, 1] = np.zeros(N)
    pred = classify_motion(_traj(pos, vel))
    ordered = (pred.labels == MotionLabel.TRAVELING_POLARIZED).mean()
    assert ordered > 0.5


def test_swarming_disordered():
    T, N = 30, 10
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(T, N, 2)) * 30
    angles = rng.uniform(0, 2 * np.pi, (T, N))
    speed = 5.0
    vel = np.stack([speed * np.cos(angles), speed * np.sin(angles)], axis=2)
    pred = classify_motion(_traj(pos, vel))
    swarming = (pred.labels == MotionLabel.SWARMING).sum()
    assert swarming > T * 0.3
