"""Heuristic collective-motion classifier from position/velocity data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectral.motion.labels import (
    FEATURE_NAMES,
    HYDRO_SUB_LABELS,
    MOTION_LABELS,
    HydroSubLabel,
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
        positions = trajectory.positions
        velocities = trajectory.velocities
        valid_mask = trajectory.valid_mask
        num_frames = trajectory.num_frames

        features = np.full((num_frames, len(FEATURE_NAMES)), np.nan)
        labels = np.zeros(num_frames, dtype=np.int32)
        hydro_sub = np.full(num_frames, HydroSubLabel.NONE, dtype=np.int32)
        confidence = np.zeros(num_frames, dtype=float)

        spread_series = np.full(num_frames, np.nan)
        for t in range(num_frames):
            feat = _compute_frame_features(positions[t], velocities[t], valid_mask[t])
            if feat is None:
                labels[t] = MotionLabel.UNKNOWN
                continue
            features[t] = feat
            spread_series[t] = feat[4]

        spread_z = _rolling_zscore(spread_series, self.config.fountain_window)
        radial = features[:, 3]

        for t in range(num_frames):
            if labels[t] == MotionLabel.UNKNOWN and np.all(np.isnan(features[t])):
                continue
            feat = features[t]
            label, sub, conf = _classify_from_features(
                feat,
                spread_zscore=spread_z[t],
                radial_velocity=radial[t],
                config=self.config,
            )
            labels[t] = label
            hydro_sub[t] = sub
            confidence[t] = conf

        if self.config.smooth_window > 1:
            labels = _smooth_labels(labels, self.config.smooth_window)

        return MotionPrediction(
            labels=labels,
            hydro_sub_labels=hydro_sub,
            confidence=confidence,
            features=features,
            feature_names=list(FEATURE_NAMES),
            label_names=[MOTION_LABELS[MotionLabel(i)] for i in range(len(MOTION_LABELS))],
            hydro_sub_label_names=[HYDRO_SUB_LABELS[s] for s in HydroSubLabel],
        )


def classify_motion(
    trajectory: FishSchoolTrajectory,
    config: MotionClassificationConfig | None = None,
) -> MotionPrediction:
    return MotionClassifier(config).predict(trajectory)


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

    v_hat = vel / speed[:, None]
    mean_dir = v_hat.mean(axis=0)
    polarization = float(np.linalg.norm(mean_dir))

    centroid = pos.mean(axis=0)
    rel = pos - centroid
    dist = np.linalg.norm(rel, axis=1)
    spread = float(dist.std())

    angular_momentum = float(np.sum(rel[:, 0] * vel[:, 1] - rel[:, 1] * vel[:, 0]))

    rel_hat = rel / (dist[:, None] + 1e-9)
    tangent = np.column_stack([-rel_hat[:, 1], rel_hat[:, 0]])
    tangential_order = float(np.abs(np.sum(v_hat * tangent, axis=1)).mean())

    mean_radial_velocity = float(np.mean(np.sum(vel * rel_hat, axis=1)))

    headings = np.arctan2(vel[:, 1], vel[:, 0])
    heading_variance = float(np.var(headings))

    inline_score, phalanx_score, staggered_score = _hydrodynamic_scores(rel, v_hat, mean_dir)

    return np.array(
        [
            polarization,
            angular_momentum,
            tangential_order,
            mean_radial_velocity,
            spread,
            heading_variance,
            inline_score,
            phalanx_score,
            staggered_score,
        ],
        dtype=float,
    )


def _hydrodynamic_scores(
    rel: np.ndarray,
    v_hat: np.ndarray,
    mean_dir: np.ndarray,
) -> tuple[float, float, float]:
    """Scores for in-line, phalanx, and staggered formations in the velocity frame."""
    norm = np.linalg.norm(mean_dir)
    if norm < 0.5:
        return 0.0, 0.0, 0.0

    e_par = mean_dir / norm
    e_perp = np.array([-e_par[1], e_par[0]])

    u = rel @ e_par
    v = rel @ e_perp

    # In-line: positions spread mainly along travel axis, low lateral variance
    inline_score = float(1.0 - np.clip(v.std() / (np.abs(u).std() + 1e-6), 0, 1))

    # Phalanx: tight along-axis bands, spread perpendicular (side-by-side rows)
    u_bins = np.round(u / (np.std(u) + 1e-6))
    lateral_spread = []
    for band in np.unique(u_bins):
        mask = u_bins == band
        if mask.sum() >= 2:
            lateral_spread.append(v[mask].std())
    phalanx_score = float(np.mean(lateral_spread) / (v.std() + 1e-6)) if lateral_spread else 0.0
    phalanx_score = float(np.clip(phalanx_score, 0, 1))

    # Staggered: alternating lateral offsets along travel direction
    order = np.argsort(u)
    v_sorted = v[order]
    if len(v_sorted) >= 4:
        diffs = v_sorted[::2].mean() - v_sorted[1::2].mean()
        staggered_score = float(np.clip(np.abs(diffs) / (v.std() + 1e-6), 0, 1))
    else:
        staggered_score = 0.0

    return inline_score, phalanx_score, staggered_score


def _classify_from_features(
    feat: np.ndarray,
    *,
    spread_zscore: float,
    radial_velocity: float,
    config: MotionClassificationConfig,
) -> tuple[int, int, float]:
    polarization = feat[0]
    angular_momentum = abs(feat[1])
    tangential_order = feat[2]
    mean_radial = radial_velocity
    heading_var = feat[5]
    inline_score = feat[6]
    phalanx_score = feat[7]
    staggered_score = feat[8]

    # Fountain: sudden outward radial flow + expanding spread
    if (
        mean_radial > config.fountain_radial_threshold
        and np.isfinite(spread_zscore)
        and spread_zscore > config.fountain_spread_zscore
    ):
        conf = float(np.clip(mean_radial / (config.fountain_radial_threshold + 1e-6), 0, 1))
        return MotionLabel.FOUNTAIN_EVASION, HydroSubLabel.NONE, conf

    # Milling: strong tangential motion around centroid
    if tangential_order > config.milling_tangential_threshold and angular_momentum > config.milling_angular_threshold:
        conf = float(np.clip(tangential_order, 0, 1))
        return MotionLabel.MILLING, HydroSubLabel.NONE, conf

    # Swarming: low polarization, disordered headings
    if polarization < config.swarming_polarization_threshold or heading_var > config.swarming_heading_variance:
        conf = float(1.0 - polarization)
        return MotionLabel.SWARMING, HydroSubLabel.NONE, conf

    # High polarization: traveling or hydrodynamic sub-formation
    if polarization >= config.traveling_polarization_threshold:
        hydro_scores = {
            HydroSubLabel.INLINE: inline_score,
            HydroSubLabel.PHALANX: phalanx_score,
            HydroSubLabel.STAGGERED: staggered_score,
        }
        best_sub = max(hydro_scores, key=hydro_scores.get)
        best_score = hydro_scores[best_sub]

        if best_score >= config.hydro_sub_threshold:
            return MotionLabel.HYDRODYNAMIC, best_sub, float(best_score)

        return MotionLabel.TRAVELING_POLARIZED, HydroSubLabel.NONE, float(polarization)

    # Intermediate polarization
    conf = float(polarization)
    return MotionLabel.TRAVELING_POLARIZED, HydroSubLabel.NONE, conf


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
