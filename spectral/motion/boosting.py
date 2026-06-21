"""XGBoost motion classifier for comparison with the decision tree."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from spectral.motion.labels import FEATURE_NAMES, MOTION_LABELS, MotionLabel
from spectral.motion.tree import MotionTreeTrainConfig, _metrics_from_arrays, _smooth_labels
from spectral.types import MotionPrediction


@dataclass(frozen=True)
class MotionXGBTrainConfig:
    n_estimators: int = 150
    max_depth: int = 6
    learning_rate: float = 0.08
    min_child_weight: int = 5
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    smooth_window: int = 7
    random_state: int = 0


@dataclass
class MotionXGBModel:
    booster: XGBClassifier
    feature_names: list[str]
    smooth_window: int
    train_config: MotionXGBTrainConfig
    label_encoder: LabelEncoder

    def predict_from_features(self, features: np.ndarray) -> MotionPrediction:
        num_frames = features.shape[0]
        valid = ~np.all(np.isnan(features), axis=1)
        labels = np.full(num_frames, MotionLabel.UNKNOWN, dtype=np.int32)
        confidence = np.zeros(num_frames, dtype=float)

        if valid.any():
            probs = self.booster.predict_proba(features[valid])
            preds_enc = self.booster.predict(features[valid])
            preds = self.label_encoder.inverse_transform(preds_enc.astype(int))
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

    def feature_importances(self) -> dict[str, float]:
        scores = self.booster.feature_importances_
        return {name: float(score) for name, score in zip(self.feature_names, scores, strict=True)}


def fit_xgb(
    x: np.ndarray,
    y: np.ndarray,
    train_config: MotionXGBTrainConfig,
) -> MotionXGBModel:
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    weights = compute_sample_weight(class_weight="balanced", y=y_enc)
    booster = XGBClassifier(
        n_estimators=train_config.n_estimators,
        max_depth=train_config.max_depth,
        learning_rate=train_config.learning_rate,
        min_child_weight=train_config.min_child_weight,
        subsample=train_config.subsample,
        colsample_bytree=train_config.colsample_bytree,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=train_config.random_state,
        n_jobs=-1,
    )
    booster.fit(x, y_enc, sample_weight=weights)
    return MotionXGBModel(
        booster=booster,
        feature_names=list(FEATURE_NAMES),
        smooth_window=train_config.smooth_window,
        train_config=train_config,
        label_encoder=label_encoder,
    )


def predict_xgb(model: MotionXGBModel, x: np.ndarray) -> np.ndarray:
    preds_enc = model.booster.predict(x)
    return model.label_encoder.inverse_transform(preds_enc.astype(int))


def leave_one_dataset_out_xgb(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    train_config: MotionXGBTrainConfig,
) -> dict[str, "MotionEvalMetrics"]:
    from spectral.motion.evaluation import MotionEvalMetrics

    results: dict[str, MotionEvalMetrics] = {}
    for held_out in datasets:
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for name, (x, y) in datasets.items():
            if name != held_out:
                x_parts.append(x)
                y_parts.append(y)
        model = fit_xgb(np.vstack(x_parts), np.concatenate(y_parts), train_config)
        x_test, y_test = datasets[held_out]
        y_pred = predict_xgb(model, x_test)
        results[held_out] = _metrics_from_arrays(y_test, y_pred)
    return results


def summarize_loo(loo: dict[str, "MotionEvalMetrics"]) -> dict[str, float]:
    return {
        "mean_accuracy": float(np.mean([m.accuracy for m in loo.values()])),
        "mean_macro_f1": float(np.mean([m.macro_f1 for m in loo.values()])),
    }
