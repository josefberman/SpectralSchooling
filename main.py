"""CLI for spectral observable pipeline (PDF sections 5.1–5.4)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spectral.dmd.analyzer import DMDAnalyzer
from spectral.graph.laplacian import GraphSpectralAnalyzer
from spectral.io import (
    load_dmd,
    load_graph_spectral,
    save_dmd,
    save_graph_spectral,
    save_koopman,
    save_motion_prediction,
)
from spectral.koopman.operator import fit_koopman_operator
from spectral.motion.annotations import load_motion_annotations
from spectral.motion.classifier import classify_motion
from spectral.motion.datasets import (
    ANNOTATIONS_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_TREE_TEXT_PATH,
    list_annotation_files,
    trajectory_path,
)
from spectral.motion.evaluation import confusion_matrix, evaluate_motion_prediction
from spectral.motion.tree import (
    MotionTreeTrainConfig,
    save_motion_tree_model,
    save_training_report,
    train_motion_tree,
)
from spectral.pipeline import run_observable_pipeline
from spectral.state import load_trajectory
from spectral.types import DMDConfig, InteractionGraphConfig, KoopmanLiftConfig, MotionClassifierConfig


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", "-i", required=True, help="loc_vel_data.h5 or .csv")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--fps", type=float, default=None, help="Frame rate (for DMD dt)")


def cmd_pipeline(args: argparse.Namespace) -> None:
    run_observable_pipeline(
        args.input,
        args.output_dir,
        graph_config=InteractionGraphConfig(k_neighbors=args.k_neighbors),
        dmd_config=DMDConfig(rank=args.dmd_rank, dt=args.fps and (1.0 / args.fps)),
        koopman_config=KoopmanLiftConfig(),
        fps=args.fps,
    )
    print(f"Pipeline complete -> {args.output_dir}")


def cmd_graph(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = load_trajectory(args.input, fps=args.fps)
    graph_spectral = GraphSpectralAnalyzer(
        InteractionGraphConfig(k_neighbors=args.k_neighbors)
    ).compute(trajectory)
    save_graph_spectral(graph_spectral, output_dir / "graph_spectral.npz")
    print(f"Saved graph spectral observables -> {output_dir / 'graph_spectral.npz'}")


def cmd_dmd(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.graph_input:
        graph_spectral = load_graph_spectral(args.graph_input)
    else:
        trajectory = load_trajectory(args.input, fps=args.fps)
        graph_spectral = GraphSpectralAnalyzer(
            InteractionGraphConfig(k_neighbors=args.k_neighbors)
        ).compute(trajectory)

    spectral_input = graph_spectral.spectral_vector.copy()
    spectral_input[~graph_spectral.valid_mask] = np.nan
    dt = (1.0 / args.fps) if args.fps else 1.0
    dmd = DMDAnalyzer(DMDConfig(rank=args.dmd_rank, dt=dt)).fit(spectral_input, dt=dt)
    save_dmd(dmd, output_dir / "dmd_observables.npz")
    print(f"Saved DMD observables -> {output_dir / 'dmd_observables.npz'}")


def cmd_koopman(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = args.graph_input or str(Path(args.output_dir) / "graph_spectral.npz")
    dmd_path = args.dmd_input or str(Path(args.output_dir) / "dmd_observables.npz")

    graph_spectral = load_graph_spectral(graph_path)
    dmd = load_dmd(dmd_path)
    koopman = fit_koopman_operator(graph_spectral, dmd, KoopmanLiftConfig())
    save_koopman(koopman, output_dir / "koopman_observables.npz")
    print(f"Saved Koopman observables -> {output_dir / 'koopman_observables.npz'}")


def cmd_motion(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = load_trajectory(args.input, fps=args.fps)
    config = MotionClassifierConfig(model_path=args.model) if args.model else MotionClassifierConfig()
    motion = classify_motion(trajectory, config)
    save_motion_prediction(motion, output_dir / "motion_prediction.npz")
    print(f"Saved motion prediction -> {output_dir / 'motion_prediction.npz'}")


def cmd_motion_eval(args: argparse.Namespace) -> None:
    annotations = load_motion_annotations(args.annotations)
    trajectory = load_trajectory(args.input, fps=args.fps or annotations.fps)
    config = MotionClassifierConfig(model_path=args.model) if args.model else MotionClassifierConfig()
    motion = classify_motion(trajectory, config)
    metrics = evaluate_motion_prediction(motion, annotations)
    print(metrics.summary())
    if args.show_confusion:
        mat, names = confusion_matrix(motion, annotations)
        print("\nConfusion matrix (rows=ground truth, cols=predicted):")
        header = " " * 22 + "  ".join(f"{n[:12]:>12s}" for n in names)
        print(header)
        for i, name in enumerate(names):
            row = "  ".join(f"{mat[i, j]:12d}" for j in range(len(names)))
            print(f"{name[:20]:20s}  {row}")


def cmd_motion_eval_all(args: argparse.Namespace) -> None:
    config = MotionClassifierConfig(model_path=args.model) if args.model else MotionClassifierConfig()
    macro_f1s: list[float] = []
    for ann_path in list_annotation_files(Path(args.annotations_dir)):
        ann = load_motion_annotations(ann_path)
        traj_path = trajectory_path(ann.dataset)
        trajectory = load_trajectory(traj_path, fps=ann.fps)
        motion = classify_motion(trajectory, config)
        metrics = evaluate_motion_prediction(motion, ann)
        macro_f1s.append(metrics.macro_f1)
        print(f"\n=== {ann.dataset} ===")
        print(metrics.summary())
    if macro_f1s:
        print(f"\nMean macro F1 across datasets: {float(np.mean(macro_f1s)):.3f}")


def cmd_motion_train(args: argparse.Namespace) -> None:
    train_config = MotionTreeTrainConfig(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        smooth_window=args.smooth_window,
        random_state=args.seed,
    )
    model, loo_metrics, train_metrics = train_motion_tree(
        annotations_dir=Path(args.annotations_dir),
        train_config=train_config,
        tune=not args.no_tune,
    )

    model_path = Path(args.model_output)
    tree_path = Path(args.tree_output)
    report_path = Path(args.report_output)

    save_motion_tree_model(model, model_path)
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    tree_path.write_text(model.export_rules(), encoding="utf-8")
    save_training_report(model, loo_metrics, train_metrics, report_path)

    print("Selected hyperparameters:")
    print(f"  max_depth={model.train_config.max_depth}")
    print(f"  min_samples_leaf={model.train_config.min_samples_leaf}")
    print(f"  smooth_window={model.train_config.smooth_window}")
    print(f"  nodes={model.tree.tree_.node_count}, depth={model.tree.get_depth()}")
    print()
    print("Training fit (all labeled frames):")
    print(train_metrics.summary())
    print("\nLeave-one-dataset-out:")
    for dataset, metrics in sorted(loo_metrics.items()):
        print(f"  {dataset}: acc={100 * metrics.accuracy:.1f}%  macro-F1={metrics.macro_f1:.3f}")
    print(f"\nSaved model -> {model_path}")
    print(f"Saved tree rules -> {tree_path}")
    print(f"Saved report -> {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spectral fish-school observable pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    pipeline = sub.add_parser("pipeline", help="Run full 5.1–5.4 pipeline")
    _add_common_args(pipeline)
    pipeline.add_argument("--k-neighbors", type=int, default=5)
    pipeline.add_argument("--dmd-rank", type=int, default=None)
    pipeline.set_defaults(func=cmd_pipeline)

    graph = sub.add_parser("graph", help="Graph Laplacian spectra only (5.2)")
    _add_common_args(graph)
    graph.add_argument("--k-neighbors", type=int, default=5)
    graph.set_defaults(func=cmd_graph)

    dmd = sub.add_parser("dmd", help="DMD on graph spectra (5.3)")
    _add_common_args(dmd)
    dmd.add_argument("--k-neighbors", type=int, default=5)
    dmd.add_argument("--dmd-rank", type=int, default=None)
    dmd.add_argument("--graph-input", default=None, help="Precomputed graph_spectral.npz")
    dmd.set_defaults(func=cmd_dmd)

    koopman = sub.add_parser("koopman", help="Koopman lift and operator fit (5.4)")
    koopman.add_argument("--input", "-i", default=None, help="loc_vel_data (if graph not saved)")
    koopman.add_argument("--output-dir", "-o", required=True)
    koopman.add_argument("--fps", type=float, default=None)
    koopman.add_argument("--graph-input", default=None)
    koopman.add_argument("--dmd-input", default=None)
    koopman.set_defaults(func=cmd_koopman)

    motion = sub.add_parser("motion", help="Collective motion classification from loc_vel data")
    _add_common_args(motion)
    motion.add_argument("--model", default=None, help="Path to trained motion_classifier.joblib")
    motion.set_defaults(func=cmd_motion)

    motion_eval = sub.add_parser("motion-eval", help="Evaluate motion classifier on manual labels")
    motion_eval.add_argument("--input", "-i", required=True, help="loc_vel_data.h5 or .csv")
    motion_eval.add_argument("--annotations", "-a", required=True, help="Annotation JSON file")
    motion_eval.add_argument("--fps", type=float, default=None)
    motion_eval.add_argument("--model", default=None, help="Path to trained motion_classifier.joblib")
    motion_eval.add_argument("--show-confusion", action="store_true")
    motion_eval.set_defaults(func=cmd_motion_eval)

    motion_eval_all = sub.add_parser(
        "motion-eval-all",
        help="Evaluate motion classifier on all annotation files",
    )
    motion_eval_all.add_argument(
        "--annotations-dir",
        default=str(ANNOTATIONS_DIR),
        help="Directory containing *_motion.json files",
    )
    motion_eval_all.add_argument("--model", default=None, help="Path to trained motion_classifier.joblib")
    motion_eval_all.set_defaults(func=cmd_motion_eval_all)

    motion_train = sub.add_parser(
        "motion-train",
        help="Train decision-tree motion classifier on manual annotations",
    )
    motion_train.add_argument(
        "--annotations-dir",
        default=str(ANNOTATIONS_DIR),
        help="Directory containing *_motion.json files",
    )
    motion_train.add_argument("--max-depth", type=int, default=12)
    motion_train.add_argument("--min-samples-leaf", type=int, default=25)
    motion_train.add_argument("--smooth-window", type=int, default=7)
    motion_train.add_argument("--seed", type=int, default=0)
    motion_train.add_argument("--no-tune", action="store_true", help="Skip LOO hyperparameter search")
    motion_train.add_argument("--model-output", default=str(DEFAULT_MODEL_PATH))
    motion_train.add_argument("--tree-output", default=str(DEFAULT_TREE_TEXT_PATH))
    motion_train.add_argument(
        "--report-output",
        default=str(ANNOTATIONS_DIR / "motion_classifier_report.json"),
    )
    motion_train.set_defaults(func=cmd_motion_train)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
