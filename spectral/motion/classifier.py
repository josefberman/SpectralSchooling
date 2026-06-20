"""Heuristic collective-motion classifier from position/velocity data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectral.motion.labels import (
    FEATURE_NAMES,
    MOTION_LABELS,
    MotionLabel,
)
from spectral.types import FishSchoolTrajectory, MotionClassificationConfig, MotionPrediction


@dataclass
class MotionClassifier:
    """Rule-based motion categorization for Year 1 exploratory analysis."""

    config: MotionClassificationConfig | None = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = MotionClassificationConfig()

    def predict(self, trajectory: FishSchoolTrajectory) -> MotionPrediction:
        features = compute_motion_features(trajectory)
        return self.predict_from_features(features)

    def predict_from_features(self, features: np.ndarray) -> MotionPrediction:
        num_frames = features.shape[0]
        labels, confidence = _classify_features_batch(features, self.config)
        valid = ~np.all(np.isnan(features), axis=1)
        labels[~valid] = MotionLabel.UNKNOWN

        if self.config.smooth_window > 1:
            labels = _smooth_labels(labels, self.config.smooth_window)

        return MotionPrediction(
            labels=labels,
            confidence=confidence,
            features=features,
            feature_names=list(FEATURE_NAMES),
            label_names=[MOTION_LABELS[MotionLabel(i)] for i in range(len(MOTION_LABELS))],
        )


def classify_motion(
    trajectory: FishSchoolTrajectory,
    config: MotionClassificationConfig | None = None,
) -> MotionPrediction:
    return MotionClassifier(config).predict(trajectory)


def compute_motion_features(trajectory: FishSchoolTrajectory) -> np.ndarray:
    positions = trajectory.positions
    velocities = trajectory.velocities
    valid_mask = trajectory.valid_mask
    num_frames = trajectory.num_frames

    features = np.full((num_frames, len(FEATURE_NAMES)), np.nan)
    for t in range(num_frames):
        feat = _compute_frame_features(positions[t], velocities[t], valid_mask[t])
        if feat is not None:
            features[t] = feat
    return features


def classify_motion_from_features(
    trajectory: FishSchoolTrajectory,
    config: MotionClassificationConfig | None = None,
    *,
    features: np.ndarray | None = None,
) -> MotionPrediction:
    """Classify from precomputed features (for fast threshold tuning)."""
    if features is None:
        features = compute_motion_features(trajectory)
    return MotionClassifier(config).predict_from_features(features)


def _compute_frame_features(
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

    # --- Directional polarization (0=disordered headings, 1=aligned) ---
    v_hat = vel / speed[:, None]
    mean_dir = v_hat.mean(axis=0)
    directional_polarization = float(np.linalg.norm(mean_dir))

    centroid = pos.mean(axis=0)
    rel = pos - centroid
    dist = np.linalg.norm(rel, axis=1)
    mean_dist = float(dist.mean()) + 1e-9
    spread = float(dist.std())

    # --- Normalized angular momentum (per-fish, per-unit-radius) ---
    raw_ang_mom = rel[:, 0] * vel[:, 1] - rel[:, 1] * vel[:, 0]
    norm_angular_momentum = float(np.mean(raw_ang_mom) / mean_dist)

    # --- Rotational polarization (0=balanced CW/CCW, 1=unidirectional rotation) ---
    abs_ang_mom = np.abs(raw_ang_mom)
    rotational_polarization = float(np.abs(raw_ang_mom.sum()) / (abs_ang_mom.sum() + 1e-9))

    # --- Tangential order (how much of velocity is tangent to centroid) ---
    rel_hat = rel / (dist[:, None] + 1e-9)
    tangent = np.column_stack([-rel_hat[:, 1], rel_hat[:, 0]])
    tangential_alignment = np.sum(v_hat * tangent, axis=1)
    tangential_order = float(np.abs(tangential_alignment).mean())

    # --- Mean radial velocity (expansion/contraction) ---
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


def _classify_features_batch(
    features: np.ndarray,
    config: MotionClassificationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    num_frames = features.shape[0]
    labels = np.zeros(num_frames, dtype=np.int32)
    confidence = np.zeros(num_frames, dtype=float)
    valid = ~np.all(np.isnan(features), axis=1)
    if not valid.any():
        return labels, confidence

    directional_polarization = features[:, 0]
    norm_ang_mom = np.abs(features[:, 2])
    tangential_order = features[:, 3]
    mean_radial = features[:, 4]
    abs_radial = np.abs(mean_radial)
    radial_dominant = tangential_order < config.expansion_contraction_tangential_ceiling
    assigned = np.zeros(num_frames, dtype=bool)

    fountain = valid & (abs_radial > config.fountain_radial_threshold)
    labels[fountain] = MotionLabel.FOUNTAIN_EVASION
    confidence[fountain] = np.clip(
        abs_radial[fountain] / (2.0 * config.fountain_radial_threshold),
        0,
        1,
    )
    assigned |= fountain

    expansion = (
        valid
        & ~assigned
        & radial_dominant
        & (mean_radial > config.expansion_contraction_radial_threshold)
    )
    labels[expansion] = MotionLabel.EXPANSION_BURST
    confidence[expansion] = np.clip(
        mean_radial[expansion] / config.fountain_radial_threshold
        * (
            (config.expansion_contraction_tangential_ceiling - tangential_order[expansion])
            / max(config.expansion_contraction_tangential_ceiling, 1e-9)
        ),
        0,
        1,
    )
    assigned |= expansion

    contraction = (
        valid
        & ~assigned
        & radial_dominant
        & (mean_radial < -config.expansion_contraction_radial_threshold)
    )
    labels[contraction] = MotionLabel.CONTRACTION_COMPACTION
    confidence[contraction] = np.clip(
        abs_radial[contraction] / config.fountain_radial_threshold
        * (
            (config.expansion_contraction_tangential_ceiling - tangential_order[contraction])
            / max(config.expansion_contraction_tangential_ceiling, 1e-9)
        ),
        0,
        1,
    )
    assigned |= contraction

    traveling = valid & ~assigned & (
        directional_polarization >= config.traveling_polarization_threshold
    )
    labels[traveling] = MotionLabel.TRAVELING_POLARIZED
    confidence[traveling] = directional_polarization[traveling]
    assigned |= traveling

    milling = (
        valid
        & ~assigned
        & (directional_polarization < config.milling_polarization_ceiling)
        & (norm_ang_mom > config.milling_angular_threshold)
        & (tangential_order > config.milling_tangential_threshold)
    )
    labels[milling] = MotionLabel.MILLING
    confidence[milling] = np.clip(
        tangential_order[milling] * (1 - directional_polarization[milling]),
        0,
        1,
    )
    assigned |= milling

    swarming = valid & ~assigned & (
        directional_polarization < config.swarming_polarization_threshold
    )
    labels[swarming] = MotionLabel.SWARMING
    confidence[swarming] = 1.0 - directional_polarization[swarming]
    assigned |= swarming

    fallback = valid & ~assigned
    labels[fallback] = MotionLabel.TRAVELING_POLARIZED
    confidence[fallback] = directional_polarization[fallback]
    return labels, confidence


def _classify_from_features(
    feat: np.ndarray,
    *,
    radial_velocity: float,
    config: MotionClassificationConfig,
) -> tuple[int, float]:
    directional_polarization = feat[0]
    norm_ang_mom = abs(feat[2])
    tangential_order = feat[3]
    mean_radial = radial_velocity

    # 1. Fountain/evasion: sudden extreme radial flow (inward or outward)
    abs_radial = abs(mean_radial)
    if abs_radial > config.fountain_radial_threshold:
        conf = float(np.clip(abs_radial / (2.0 * config.fountain_radial_threshold), 0, 1))
        return MotionLabel.FOUNTAIN_EVASION, conf

    # 2. Expansion/contraction: moderate radial flow with low tangential order
    radial_dominant = tangential_order < config.expansion_contraction_tangential_ceiling
    if radial_dominant and mean_radial > config.expansion_contraction_radial_threshold:
        radial_conf = mean_radial / config.fountain_radial_threshold
        tang_conf = (config.expansion_contraction_tangential_ceiling - tangential_order) / max(
            config.expansion_contraction_tangential_ceiling, 1e-9
        )
        conf = float(np.clip(radial_conf * tang_conf, 0, 1))
        return MotionLabel.EXPANSION_BURST, conf

    if radial_dominant and mean_radial < -config.expansion_contraction_radial_threshold:
        radial_conf = abs(mean_radial) / config.fountain_radial_threshold
        tang_conf = (config.expansion_contraction_tangential_ceiling - tangential_order) / max(
            config.expansion_contraction_tangential_ceiling, 1e-9
        )
        conf = float(np.clip(radial_conf * tang_conf, 0, 1))
        return MotionLabel.CONTRACTION_COMPACTION, conf

    # 3. Traveling/polarized: high directional alignment
    if directional_polarization >= config.traveling_polarization_threshold:
        return MotionLabel.TRAVELING_POLARIZED, float(directional_polarization)

    # 4. Milling: low directional alignment + consistent rotation around centroid
    if (
        directional_polarization < config.milling_polarization_ceiling
        and norm_ang_mom > config.milling_angular_threshold
        and tangential_order > config.milling_tangential_threshold
    ):
        conf = float(np.clip(tangential_order * (1 - directional_polarization), 0, 1))
        return MotionLabel.MILLING, conf

    # 5. Swarming: low directional alignment, no organized rotation
    if directional_polarization < config.swarming_polarization_threshold:
        conf = float(1.0 - directional_polarization)
        return MotionLabel.SWARMING, conf

    # 6. Intermediate — moderate directional alignment
    return MotionLabel.TRAVELING_POLARIZED, float(directional_polarization)


def _rolling_zscore(series: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(series, np.nan)
    half = window // 2
    for t in range(len(series)):
        sl = series[max(0, t - half) : min(len(series), t + half + 1)]
        sl = sl[np.isfinite(sl)]
        if len(sl) < 5:
            continue
        mu, sigma = sl.mean(), sl.std()
        if sigma > 1e-9:
            out[t] = (series[t] - mu) / sigma
    return out


def _smooth_labels(labels: np.ndarray, window: int) -> np.ndarray:
    """Majority-vote smoothing over a temporal window."""
    out = labels.copy()
    half = window // 2
    for t in range(len(labels)):
        sl = labels[max(0, t - half) : min(len(labels), t + half + 1)]
        if len(sl) == 0:
            continue
        counts = np.bincount(sl, minlength=int(labels.max()) + 1)
        out[t] = int(np.argmax(counts))
    return out
