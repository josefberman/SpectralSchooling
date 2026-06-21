"""Compare LOO performance across decision-tree depths and save the best model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spectral.motion.datasets import ANNOTATIONS_DIR, DEFAULT_MODEL_PATH, DEFAULT_TREE_TEXT_PATH
from spectral.motion.tree import (
    MotionTreeTrainConfig,
    collect_labeled_features,
    leave_one_dataset_out_scores,
    save_motion_tree_model,
    save_training_report,
    train_motion_tree,
)


def main() -> None:
    datasets = collect_labeled_features()
    print("Depth x min_samples_leaf -> mean LOO macro-F1 (depth >= 12):\n")
    best_f1, best_depth, best_leaf = -1.0, 8, 10
    for depth in (8, 10, 12, 14, 16):
        for leaf in (5, 10, 25, 50):
            cfg = MotionTreeTrainConfig(max_depth=depth, min_samples_leaf=leaf, smooth_window=7)
            loo = leave_one_dataset_out_scores(datasets, cfg)
            mean_f1 = float(np.mean([m.macro_f1 for m in loo.values()]))
            if mean_f1 > best_f1:
                best_f1, best_depth, best_leaf = mean_f1, depth, leaf
            if depth >= 12:
                print(f"  depth={depth:2d}  leaf={leaf:3d}  macro-F1={mean_f1:.3f}")
    print(f"\nOverall best: depth={best_depth} leaf={best_leaf} macro-F1={best_f1:.3f}")

    print("\nFixed-depth comparison (leaf=10):\n")
    for depth in (8, 10, 12, 14, 16, 18, 20):
        cfg = MotionTreeTrainConfig(max_depth=depth, min_samples_leaf=10, smooth_window=7)
        loo = leave_one_dataset_out_scores(datasets, cfg)
        mean_f1 = float(np.mean([m.macro_f1 for m in loo.values()]))
        mean_acc = float(np.mean([m.accuracy for m in loo.values()]))
        print(f"  depth={depth:2d}  mean LOO acc={100 * mean_acc:5.1f}%  mean LOO macro-F1={mean_f1:.3f}")

    best_depth, best_leaf, best_f1 = best_depth, best_leaf, best_f1
    model, loo, train = train_motion_tree(
        train_config=MotionTreeTrainConfig(
            max_depth=best_depth,
            min_samples_leaf=best_leaf,
            smooth_window=7,
        ),
        tune=False,
    )
    save_motion_tree_model(model, DEFAULT_MODEL_PATH)
    DEFAULT_TREE_TEXT_PATH.write_text(model.export_rules(), encoding="utf-8")
    save_training_report(model, loo, train, ANNOTATIONS_DIR / "motion_classifier_report.json")
    print(
        f"Saved depth={best_depth} model: "
        f"nodes={model.tree.tree_.node_count}, tree_depth={model.tree.get_depth()}"
    )
    print("\nLeave-one-dataset-out:")
    for ds, m in sorted(loo.items()):
        print(f"  {ds}: acc={100 * m.accuracy:.1f}%  macro-F1={m.macro_f1:.3f}")
    print("\nTraining fit:")
    print(train.summary())


if __name__ == "__main__":
    main()
