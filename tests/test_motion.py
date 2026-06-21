"""Tests for motion classification."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from spectral.motion.classifier import classify_motion, classify_motion_from_features
from spectral.motion.features import compute_motion_features
from spectral.motion.labels import FEATURE_NAMES, MotionLabel
from spectral.motion.tree import MotionTreeModel, MotionTreeTrainConfig
from spectral.types import FishSchoolTrajectory, MotionClassifierConfig


def _traj(positions: np.ndarray, velocities: np.ndarray) -> FishSchoolTrajectory:
    valid = np.isfinite(positions).all(axis=2)
    return FishSchoolTrajectory(positions=positions, velocities=velocities, valid_mask=valid, fps=30.0)


def _synthetic_model() -> MotionTreeModel:
    x = np.array(
        [
            [0.95, 0.2, 0.1, 0.2, 0.0, 10.0],
            [0.15, 0.1, 0.8, 0.9, 0.0, 10.0],
            [0.15, 0.1, 0.1, 0.1, 0.0, 10.0],
        ]
    )
    y = np.array(
        [MotionLabel.TRAVELING_POLARIZED, MotionLabel.MILLING, MotionLabel.SWARMING],
        dtype=np.int32,
    )
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(x, y)
    return MotionTreeModel(
        tree=tree,
        feature_names=list(FEATURE_NAMES),
        smooth_window=1,
        train_config=MotionTreeTrainConfig(max_depth=3, min_samples_leaf=1, smooth_window=1),
    )


def test_tree_predicts_from_feature_rows():
    model = _synthetic_model()
    features = np.tile([0.95, 0.2, 0.1, 0.2, 0.0, 10.0], (10, 1))
    pred = model.predict_from_features(features)
    assert np.all(pred.labels == MotionLabel.TRAVELING_POLARIZED)


def test_compute_motion_features_shape():
    T, N = 10, 6
    pos = np.random.default_rng(0).normal(size=(T, N, 2))
    vel = np.random.default_rng(1).normal(size=(T, N, 2))
    features = compute_motion_features(_traj(pos, vel))
    assert features.shape == (T, len(FEATURE_NAMES))
    assert np.isfinite(features).any()


def test_classify_motion_requires_trained_model():
    with pytest.raises(FileNotFoundError):
        classify_motion(
            _traj(np.zeros((5, 3, 2)), np.zeros((5, 3, 2))),
            MotionClassifierConfig(model_path="missing.joblib"),
        )


def test_classify_motion_from_features_with_injected_model():
    model = _synthetic_model()
    features = np.tile([0.15, 0.1, 0.1, 0.1, 0.0, 10.0], (8, 1))
    pred = classify_motion_from_features(
        _traj(np.zeros((8, 3, 2)), np.zeros((8, 3, 2))),
        model=model,
        features=features,
    )
    assert np.all(pred.labels == MotionLabel.SWARMING)
