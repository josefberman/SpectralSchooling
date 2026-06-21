"""Lightweight decision-tree motion classifier trained on manual annotations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

from spectral.motion.annotations import MotionAnnotationSet, load_motion_annotations
from spectral.motion.features import compute_motion_features
from spectral.motion.datasets import (
    ANNOTATIONS_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_TREE_TEXT_PATH,
    list_annotation_files,
    trajectory_path,
)
from spectral.motion.evaluation import MotionEvalMetrics, evaluate_motion_prediction
from spectral.motion.labels import FEATURE_NAMES, MOTION_LABELS, MotionLabel
from spectral.state import load_trajectory
from spectral.types import FishSchoolTrajectory, MotionPrediction


@dataclass(frozen=True)
class MotionTreeTrainConfig:
    max_depth: int = 12
    min_samples_leaf: int = 25
    smooth_window: int = 7
    random_state: int = 0


@dataclass
class MotionTreeModel:
    tree: DecisionTreeClassifier
    feature_names: list[str]
    smooth_window: int
    train_config: MotionTreeTrainConfig

    def predict_from_features(self, features: np.ndarray) -> MotionPrediction:
        num_frames = features.shape[0]
        valid = ~np.all(np.isnan(features), axis=1)
        labels = np.full(num_frames, MotionLabel.UNKNOWN, dtype=np.int32)
        confidence = np.zeros(num_frames, dtype=float)

        if valid.any():
            probs = self.tree.predict_proba(features[valid])
            preds = self.tree.predict(features[valid])
            labels[valid] = preds.astype(np.int32)
            confidence[valid] = probs.max(axis=1)

        if self.smooth_window > 1:
            labels = _smooth_labels(labels, self.smooth_window)

        return MotionPrediction(
            labels=labels,
            confidence=confidence,
            features=features,
            feature_names=list(self.feature_names),
            label_names=[MOTION_LABELS[MotionLabel(i)] for i in range(len(MOTION_LABELS))],
        )

    def export_rules(self) -> str:
        return export_text(
            self.tree,
            feature_names=self.feature_names,
            class_names=[MOTION_LABELS[MotionLabel(int(c))] for c in self.tree.classes_],
        )


def save_motion_tree_model(model: MotionTreeModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_motion_tree_model(path: str | Path | None = None) -> MotionTreeModel:
    model_path = Path(path) if path else DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Motion tree model not found at {model_path}. "
            "Train one with: python main.py motion-train"
        )
    return joblib.load(model_path)


def _smooth_labels(labels: np.ndarray, window: int) -> np.ndarray:
    out = labels.copy()
    half = window // 2
    max_label = int(labels.max()) + 1
    for t in range(len(labels)):
        if labels[t] == MotionLabel.UNKNOWN:
            continue
        sl = labels[max(0, t - half) : min(len(labels), t + half + 1)]
        sl = sl[sl != MotionLabel.UNKNOWN]
        if len(sl) == 0:
            continue
        counts = np.bincount(sl, minlength=max_label)
        out[t] = int(np.argmax(counts))
    return out


def _labeled_rows(
    features: np.ndarray,
    annotations: MotionAnnotationSet,
) -> tuple[np.ndarray, np.ndarray]:
    gt = annotations.frame_labels(features.shape[0])
    mask = (gt >= 0) & ~np.all(np.isnan(features), axis=1)
    return features[mask], gt[mask]


def collect_labeled_features(
    annotations_dir: Path | None = None,
    data_root: Path | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return per-dataset (X, y) arrays from manual annotations."""
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ann_path in list_annotation_files(annotations_dir):
        ann = load_motion_annotations(ann_path)
        traj_path = trajectory_path(ann.dataset, root=data_root)
        trajectory = load_trajectory(traj_path, fps=ann.fps)
        features = compute_motion_features(trajectory)
        x, y = _labeled_rows(features, ann)
        if len(y):
            datasets[ann.dataset] = (x, y)
    return datasets


def _metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> MotionEvalMetrics:
    accuracy = float((y_true == y_pred).mean())
    class_ids = sorted(set(y_true.tolist()))
    per_class: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    for label_id in class_ids:
        name = MOTION_LABELS[MotionLabel(int(label_id))]
        tp = int(np.sum((y_pred == label_id) & (y_true == label_id)))
        fp = int(np.sum((y_pred == label_id) & (y_true != label_id)))
        fn = int(np.sum((y_pred != label_id) & (y_true == label_id)))
        support = int(np.sum(y_true == label_id))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"support": float(support), "precision": precision, "recall": recall, "f1": f1}
        f1_scores.append(f1)
    return MotionEvalMetrics(
        accuracy=accuracy,
        macro_f1=float(np.mean(f1_scores)) if f1_scores else 0.0,
        labeled_frames=int(len(y_true)),
        per_class=per_class,
    )


def leave_one_dataset_out_scores(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    train_config: MotionTreeTrainConfig,
) -> dict[str, MotionEvalMetrics]:
    results: dict[str, MotionEvalMetrics] = {}
    for held_out in datasets:
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for name, (x, y) in datasets.items():
            if name != held_out:
                x_parts.append(x)
                y_parts.append(y)
        model = _fit_tree(np.vstack(x_parts), np.concatenate(y_parts), train_config)
        x_test, y_test = datasets[held_out]
        y_pred = model.tree.predict(x_test)
        results[held_out] = _metrics_from_arrays(y_test, y_pred)
    return results


def _fit_tree(
    x: np.ndarray,
    y: np.ndarray,
    train_config: MotionTreeTrainConfig,
) -> MotionTreeModel:
    tree = DecisionTreeClassifier(
        max_depth=train_config.max_depth,
        min_samples_leaf=train_config.min_samples_leaf,
        class_weight="balanced",
        random_state=train_config.random_state,
    )
    tree.fit(x, y)
    return MotionTreeModel(
        tree=tree,
        feature_names=list(FEATURE_NAMES),
        smooth_window=train_config.smooth_window,
        train_config=train_config,
    )


def select_tree_hyperparameters(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    max_depths: tuple[int, ...] = (8, 10, 12, 14, 16, 18),
    min_samples_leaf_options: tuple[int, ...] = (10, 25, 50, 100),
    smooth_window: int = 7,
    random_state: int = 0,
) -> MotionTreeTrainConfig:
    best: MotionTreeTrainConfig | None = None
    best_score = -1.0
    best_nodes = 10**9
    for max_depth in max_depths:
        for min_samples_leaf in min_samples_leaf_options:
            cfg = MotionTreeTrainConfig(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                smooth_window=smooth_window,
                random_state=random_state,
            )
            loo = leave_one_dataset_out_scores(datasets, cfg)
            macro_f1 = float(np.mean([m.macro_f1 for m in loo.values()]))
            x_all = np.vstack([x for x, _ in datasets.values()])
            y_all = np.concatenate([y for _, y in datasets.values()])
            nodes = _fit_tree(x_all, y_all, cfg).tree.tree_.node_count
            if macro_f1 > best_score + 0.005 or (
                abs(macro_f1 - best_score) <= 0.005 and nodes < best_nodes
            ):
                best_score = macro_f1
                best_nodes = nodes
                best = cfg
    assert best is not None
    return best


def train_motion_tree(
    *,
    annotations_dir: Path | None = None,
    data_root: Path | None = None,
    train_config: MotionTreeTrainConfig | None = None,
    tune: bool = True,
) -> tuple[MotionTreeModel, dict[str, MotionEvalMetrics], MotionEvalMetrics]:
    datasets = collect_labeled_features(annotations_dir, data_root=data_root)
    if not datasets:
        raise ValueError("No labeled training data found in annotations/")

    config = train_config or MotionTreeTrainConfig()
    if tune:
        config = select_tree_hyperparameters(datasets, smooth_window=config.smooth_window)

    loo_metrics = leave_one_dataset_out_scores(datasets, config)
    x_all = np.vstack([x for x, _ in datasets.values()])
    y_all = np.concatenate([y for _, y in datasets.values()])
    model = _fit_tree(x_all, y_all, config)

    y_pred = model.tree.predict(x_all)
    train_metrics = _metrics_from_arrays(y_all, y_pred)
    return model, loo_metrics, train_metrics


def evaluate_model_on_dataset(
    model: MotionTreeModel,
    dataset: str,
    annotations_dir: Path | None = None,
    data_root: Path | None = None,
) -> MotionEvalMetrics:
    ann_path = (annotations_dir or ANNOTATIONS_DIR) / f"{dataset}_motion.json"
    ann = load_motion_annotations(ann_path)
    trajectory = load_trajectory(trajectory_path(dataset, root=data_root), fps=ann.fps)
    features = compute_motion_features(trajectory)
    pred = model.predict_from_features(features)
    return evaluate_motion_prediction(pred, ann)


def _metrics_to_dict(metrics: MotionEvalMetrics) -> dict:
    return {
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "labeled_frames": metrics.labeled_frames,
        "per_class": metrics.per_class,
    }


def save_training_report(
    model: MotionTreeModel,
    loo_metrics: dict[str, MotionEvalMetrics],
    train_metrics: MotionEvalMetrics,
    path: Path,
) -> None:
    report = {
        "train_config": asdict(model.train_config),
        "train_metrics": _metrics_to_dict(train_metrics),
        "leave_one_out": {name: _metrics_to_dict(metrics) for name, metrics in loo_metrics.items()},
        "tree_nodes": int(model.tree.tree_.node_count),
        "tree_depth": int(model.tree.get_depth()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
