"""Compare decision tree vs XGBoost on manual motion annotations (LOO)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spectral.motion.boosting import (
    MotionXGBTrainConfig,
    fit_xgb,
    leave_one_dataset_out_xgb,
    predict_xgb,
    summarize_loo,
)
from spectral.motion.datasets import ANNOTATIONS_DIR
from spectral.motion.tree import (
    MotionTreeTrainConfig,
    _fit_tree,
    _metrics_from_arrays,
    collect_labeled_features,
    leave_one_dataset_out_scores,
)


def _per_dataset_table(
    tree_loo: dict,
    xgb_loo: dict,
) -> list[dict]:
    rows = []
    for dataset in sorted(tree_loo):
        t = tree_loo[dataset]
        x = xgb_loo[dataset]
        rows.append(
            {
                "dataset": dataset,
                "tree_accuracy": t.accuracy,
                "tree_macro_f1": t.macro_f1,
                "xgb_accuracy": x.accuracy,
                "xgb_macro_f1": x.macro_f1,
                "delta_macro_f1": x.macro_f1 - t.macro_f1,
            }
        )
    return rows


def main() -> None:
    datasets = collect_labeled_features()
    tree_cfg = MotionTreeTrainConfig(max_depth=8, min_samples_leaf=5, smooth_window=7)
    xgb_cfg = MotionXGBTrainConfig(smooth_window=7)

    print("=== Leave-one-dataset-out comparison ===")
    print(f"Decision tree: max_depth={tree_cfg.max_depth}, min_samples_leaf={tree_cfg.min_samples_leaf}")
    print(
        f"XGBoost: n_estimators={xgb_cfg.n_estimators}, max_depth={xgb_cfg.max_depth}, "
        f"lr={xgb_cfg.learning_rate}\n"
    )

    tree_loo = leave_one_dataset_out_scores(datasets, tree_cfg)
    xgb_loo = leave_one_dataset_out_xgb(datasets, xgb_cfg)
    tree_summary = summarize_loo(tree_loo)
    xgb_summary = summarize_loo(xgb_loo)

    print(f"{'Dataset':<8} {'Tree acc':>9} {'Tree F1':>8} {'XGB acc':>9} {'XGB F1':>8} {'dF1':>7}")
    print("-" * 54)
    for row in _per_dataset_table(tree_loo, xgb_loo):
        print(
            f"{row['dataset']:<8} "
            f"{100 * row['tree_accuracy']:8.1f}% {row['tree_macro_f1']:7.3f} "
            f"{100 * row['xgb_accuracy']:8.1f}% {row['xgb_macro_f1']:7.3f} "
            f"{row['delta_macro_f1']:+6.3f}"
        )

    print("-" * 54)
    print(
        f"{'MEAN':<8} "
        f"{100 * tree_summary['mean_accuracy']:8.1f}% {tree_summary['mean_macro_f1']:7.3f} "
        f"{100 * xgb_summary['mean_accuracy']:8.1f}% {xgb_summary['mean_macro_f1']:7.3f} "
        f"{xgb_summary['mean_macro_f1'] - tree_summary['mean_macro_f1']:+6.3f}"
    )

    x_all = np.vstack([x for x, _ in datasets.values()])
    y_all = np.concatenate([y for _, y in datasets.values()])
    tree_model = _fit_tree(x_all, y_all, tree_cfg)
    xgb_model = fit_xgb(x_all, y_all, xgb_cfg)

    tree_train = _metrics_from_arrays(y_all, tree_model.tree.predict(x_all))
    xgb_train = _metrics_from_arrays(y_all, predict_xgb(xgb_model, x_all))

    print("\n=== Full training-set fit (all labeled frames) ===")
    print(f"Decision tree: acc={100 * tree_train.accuracy:.1f}%  macro-F1={tree_train.macro_f1:.3f}")
    print(f"XGBoost:       acc={100 * xgb_train.accuracy:.1f}%  macro-F1={xgb_train.macro_f1:.3f}")

    print("\nXGBoost feature importances:")
    for name, score in sorted(xgb_model.feature_importances().items(), key=lambda kv: -kv[1]):
        print(f"  {name:28s} {score:.3f}")

    report = {
        "decision_tree": {
            "config": {
                "max_depth": tree_cfg.max_depth,
                "min_samples_leaf": tree_cfg.min_samples_leaf,
                "smooth_window": tree_cfg.smooth_window,
            },
            "loo": tree_summary,
            "train": {"accuracy": tree_train.accuracy, "macro_f1": tree_train.macro_f1},
            "per_dataset": {
                ds: {"accuracy": m.accuracy, "macro_f1": m.macro_f1}
                for ds, m in sorted(tree_loo.items())
            },
        },
        "xgboost": {
            "config": xgb_cfg.__dict__,
            "loo": xgb_summary,
            "train": {"accuracy": xgb_train.accuracy, "macro_f1": xgb_train.macro_f1},
            "feature_importances": xgb_model.feature_importances(),
        },
        "comparison": {
            "per_dataset": _per_dataset_table(tree_loo, xgb_loo),
            "loo_delta_macro_f1": xgb_summary["mean_macro_f1"] - tree_summary["mean_macro_f1"],
            "train_delta_macro_f1": xgb_train.macro_f1 - tree_train.macro_f1,
        },
    }
    out_path = ANNOTATIONS_DIR / "motion_classifier_comparison.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report -> {out_path}")


if __name__ == "__main__":
    main()
