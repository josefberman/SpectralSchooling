"""Per-frame collective motion features from position/velocity data."""

from __future__ import annotations

import numpy as np

from spectral.motion.labels import FEATURE_NAMES
from spectral.types import FishSchoolTrajectory


def compute_motion_features(trajectory: FishSchoolTrajectory) -> np.ndarray:
    positions = trajectory.positions
    velocities = trajectory.velocities
    valid_mask = trajectory.valid_mask
    num_frames = trajectory.num_frames

    features = np.full((num_frames, len(FEATURE_NAMES)), np.nan)
    for t in range(num_frames):
        feat = compute_frame_features(positions[t], velocities[t], valid_mask[t])
        if feat is not None:
            features[t] = feat
    return features


def compute_frame_features(
    positions: np.ndarray,
    velocities: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray | None:
    valid = valid_mask & np.isfinite(positions).all(axis=1) & np.isfinite(velocities).all(axis=1)
    if valid.sum() < 3:
        return None

    pos = positions[valid]
    vel = velocities[valid]
    speed = np.linalg.norm(vel, axis=1)
    moving = speed > 1e-6
    if moving.sum() < 3:
        return None

    pos = pos[moving]
    vel = vel[moving]
    speed = speed[moving]

    v_hat = vel / speed[:, None]
    mean_dir = v_hat.mean(axis=0)
    directional_polarization = float(np.linalg.norm(mean_dir))

    centroid = pos.mean(axis=0)
    rel = pos - centroid
    dist = np.linalg.norm(rel, axis=1)
    mean_dist = float(dist.mean()) + 1e-9
    spread = float(dist.std())

    raw_ang_mom = rel[:, 0] * vel[:, 1] - rel[:, 1] * vel[:, 0]
    norm_angular_momentum = float(np.mean(raw_ang_mom) / mean_dist)

    abs_ang_mom = np.abs(raw_ang_mom)
    rotational_polarization = float(np.abs(raw_ang_mom.sum()) / (abs_ang_mom.sum() + 1e-9))

    rel_hat = rel / (dist[:, None] + 1e-9)
    tangent = np.column_stack([-rel_hat[:, 1], rel_hat[:, 0]])
    tangential_alignment = np.sum(v_hat * tangent, axis=1)
    tangential_order = float(np.abs(tangential_alignment).mean())

    mean_radial_velocity = float(np.mean(np.sum(vel * rel_hat, axis=1)))

    return np.array(
        [
            directional_polarization,
            rotational_polarization,
            norm_angular_momentum,
            tangential_order,
            mean_radial_velocity,
            spread,
        ],
        dtype=float,
    )
