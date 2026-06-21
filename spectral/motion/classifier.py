"""Decision-tree collective motion classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spectral.motion.datasets import ANNOTATIONS_DIR
from spectral.motion.features import compute_motion_features
from spectral.motion.tree import MotionTreeModel, load_motion_tree_model
from spectral.types import FishSchoolTrajectory, MotionClassifierConfig, MotionPrediction


def _resolve_model_path(path: str) -> Path:
    model_path = Path(path)
    if model_path.is_absolute():
        return model_path
    return ANNOTATIONS_DIR.parent / model_path


@dataclass
class MotionClassifier:
    """Explainable motion categorization via a trained decision tree."""

    config: MotionClassifierConfig | None = None
    model: MotionTreeModel | None = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = MotionClassifierConfig()
        if self.model is None:
            self.model = load_motion_tree_model(_resolve_model_path(self.config.model_path))

    def predict(self, trajectory: FishSchoolTrajectory) -> MotionPrediction:
        features = compute_motion_features(trajectory)
        return self.predict_from_features(features)

    def predict_from_features(self, features: np.ndarray) -> MotionPrediction:
        assert self.model is not None
        return self.model.predict_from_features(features)


def classify_motion(
    trajectory: FishSchoolTrajectory,
    config: MotionClassifierConfig | None = None,
) -> MotionPrediction:
    return MotionClassifier(config).predict(trajectory)


def classify_motion_from_features(
    trajectory: FishSchoolTrajectory,
    config: MotionClassifierConfig | None = None,
    *,
    features: np.ndarray | None = None,
    model: MotionTreeModel | None = None,
) -> MotionPrediction:
    if features is None:
        features = compute_motion_features(trajectory)
    clf = MotionClassifier(config=config, model=model)
    return clf.predict_from_features(features)
