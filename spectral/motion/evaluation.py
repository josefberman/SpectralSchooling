"""Evaluate and tune motion classifier against manual annotations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np

from spectral.motion.annotations import MotionAnnotationSet
from spectral.motion.classifier import classify_motion_from_features, compute_motion_features
from spectral.motion.labels import MOTION_LABELS, MotionLabel
from spectral.types import FishSchoolTrajectory, MotionClassificationConfig, MotionPrediction


@dataclass(frozen=True)
class MotionEvalMetrics:
    accuracy: float
    macro_f1: float
    labeled_frames: int
    per_class: dict[str, dict[str, float]]

    def summary(self) -> str:
        lines = [
            f"Labeled frames: {self.labeled_frames}",
            f"Accuracy: {100 * self.accuracy:.1f}%",
            f"Macro F1: {self.macro_f1:.3f}",
            "",
            "Per class (support = labeled frames with that ground truth):",
        ]
        for name, stats in sorted(self.per_class.items()):
            lines.append(
                f"  {name:24s}  n={int(stats['support']):4d}  "
                f"P={stats['precision']:.2f}  R={stats['recall']:.2f}  F1={stats['f1']:.2f}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class MotionTuneResult:
    config: MotionClassificationConfig
    metrics: MotionEvalMetrics
    trials: int


def evaluate_motion_prediction(
    prediction: MotionPrediction,
    annotations: MotionAnnotationSet,
) -> MotionEvalMetrics:
    num_frames = len(prediction.labels)
    gt = annotations.frame_labels(num_frames)
    mask = gt >= 0
    if not mask.any():
        raise ValueError("No labeled frames overlap the prediction length")

    pred = prediction.labels[mask]
    truth = gt[mask]
    accuracy = float((pred == truth).mean())

    class_names = sorted({MOTION_LABELS[MotionLabel(i)] for i in np.unique(truth)})
    per_class: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    for name in class_names:
        label_id = next(k for k, v in MOTION_LABELS.items() if v == name)
        tp = int(np.sum((pred == label_id) & (truth == label_id)))
        fp = int(np.sum((pred == label_id) & (truth != label_id)))
        fn = int(np.sum((pred != label_id) & (truth == label_id)))
        support = int(np.sum(truth == label_id))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "support": float(support),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_scores.append(f1)

    return MotionEvalMetrics(
        accuracy=accuracy,
        macro_f1=float(np.mean(f1_scores)) if f1_scores else 0.0,
        labeled_frames=int(mask.sum()),
        per_class=per_class,
    )


def confusion_matrix(
    prediction: MotionPrediction,
    annotations: MotionAnnotationSet,
) -> tuple[np.ndarray, list[str]]:
    num_frames = len(prediction.labels)
    gt = annotations.frame_labels(num_frames)
    mask = gt >= 0
    truth = gt[mask]
    pred = prediction.labels[mask]

    names = sorted(
        {MOTION_LABELS[MotionLabel(int(i))] for i in np.unique(truth)}
        | {MOTION_LABELS[MotionLabel(int(i))] for i in np.unique(pred)}
    )
    name_to_idx = {name: i for i, name in enumerate(names)}
    mat = np.zeros((len(names), len(names)), dtype=int)
    for t, p in zip(truth, pred, strict=True):
        gt_name = MOTION_LABELS[MotionLabel(int(t))]
        pred_name = MOTION_LABELS[MotionLabel(int(p))]
        mat[name_to_idx[gt_name], name_to_idx[pred_name]] += 1
    return mat, names


def tune_motion_thresholds(
    trajectory: FishSchoolTrajectory,
    annotations: MotionAnnotationSet,
    *,
    base_config: MotionClassificationConfig | None = None,
    trials: int = 4000,
    seed: int = 0,
) -> MotionTuneResult:
    """Random search over threshold hyperparameters on labeled frames."""
    base = base_config or MotionClassificationConfig()
    features = compute_motion_features(trajectory)
    rng = np.random.default_rng(seed)
    best: MotionTuneResult | None = None

    for _ in range(trials):
        cfg = _sample_config(base, rng)
        pred = classify_motion_from_features(trajectory, cfg, features=features)
        metrics = evaluate_motion_prediction(pred, annotations)
        if best is None or metrics.macro_f1 > best.metrics.macro_f1 or (
            metrics.macro_f1 == best.metrics.macro_f1 and metrics.accuracy > best.metrics.accuracy
        ):
            best = MotionTuneResult(config=cfg, metrics=metrics, trials=trials)

    assert best is not None
    tuned = replace(best.config, smooth_window=base.smooth_window)
    final_pred = classify_motion_from_features(trajectory, tuned, features=features)
    final_metrics = evaluate_motion_prediction(final_pred, annotations)
    return MotionTuneResult(config=tuned, metrics=final_metrics, trials=trials)


def save_motion_config(config: MotionClassificationConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


def load_motion_config(path: str | Path) -> MotionClassificationConfig:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    valid = {f.name for f in fields(MotionClassificationConfig)}
    return MotionClassificationConfig(**{k: v for k, v in data.items() if k in valid})


def _sample_config(
    base: MotionClassificationConfig,
    rng: np.random.Generator,
) -> MotionClassificationConfig:
    return MotionClassificationConfig(
        traveling_polarization_threshold=float(rng.uniform(0.55, 0.85)),
        swarming_polarization_threshold=float(rng.uniform(0.25, 0.55)),
        milling_polarization_ceiling=float(rng.uniform(0.45, 0.75)),
        milling_tangential_threshold=float(rng.uniform(0.45, 0.85)),
        milling_angular_threshold=float(rng.uniform(0.15, 0.65)),
        fountain_radial_threshold=float(rng.uniform(0.5, 2.5)),
        expansion_contraction_radial_threshold=float(rng.uniform(0.15, 0.65)),
        expansion_contraction_tangential_ceiling=float(rng.uniform(0.35, 0.75)),
        smooth_window=1,
    )
