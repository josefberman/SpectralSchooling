"""Evaluate motion classifier against manual annotations."""

from __future__ import annotations

import numpy as np

from spectral.motion.annotations import MotionAnnotationSet
from spectral.motion.labels import MOTION_LABELS, MotionLabel
from spectral.types import MotionPrediction


class MotionEvalMetrics:
    __slots__ = ("accuracy", "macro_f1", "labeled_frames", "per_class")

    def __init__(
        self,
        accuracy: float,
        macro_f1: float,
        labeled_frames: int,
        per_class: dict[str, dict[str, float]],
    ) -> None:
        self.accuracy = accuracy
        self.macro_f1 = macro_f1
        self.labeled_frames = labeled_frames
        self.per_class = per_class

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

    class_names = sorted({MOTION_LABELS[MotionLabel(int(i))] for i in np.unique(truth)})
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
