"""CLI for spectral observable pipeline (PDF sections 5.1–5.4)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
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
from spectral.motion.evaluation import (
    confusion_matrix,
    evaluate_motion_prediction,
    save_motion_config,
    tune_motion_thresholds,
)
from spectral.pipeline import run_observable_pipeline
from spectral.state import load_trajectory
from spectral.types import DMDConfig, InteractionGraphConfig, KoopmanLiftConfig, MotionClassificationConfig


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

    if args.graph-input:
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
    config = MotionClassificationConfig()
    if args.config:
        from spectral.motion.evaluation import load_motion_config

        config = load_motion_config(args.config)
    motion = classify_motion(trajectory, config)
    save_motion_prediction(motion, output_dir / "motion_prediction.npz")
    print(f"Saved motion prediction -> {output_dir / 'motion_prediction.npz'}")


def cmd_motion_eval(args: argparse.Namespace) -> None:
    annotations = load_motion_annotations(args.annotations)
    trajectory = load_trajectory(args.input, fps=args.fps or annotations.fps)
    config = MotionClassificationConfig()
    if args.config:
        from spectral.motion.evaluation import load_motion_config

        config = load_motion_config(args.config)
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


def cmd_motion_tune(args: argparse.Namespace) -> None:
    annotations = load_motion_annotations(args.annotations)
    trajectory = load_trajectory(args.input, fps=args.fps or annotations.fps)
    base = MotionClassificationConfig()
    if args.config:
        from spectral.motion.evaluation import load_motion_config

        base = load_motion_config(args.config)
    result = tune_motion_thresholds(
        trajectory,
        annotations,
        base_config=base,
        trials=args.trials,
        seed=args.seed,
    )
    print("Best thresholds:")
    print(json.dumps(asdict(result.config), indent=2))
    print()
    print(result.metrics.summary())
    if args.save_config:
        save_motion_config(result.config, args.save_config)
        print(f"\nSaved tuned config -> {args.save_config}")


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
    motion.add_argument("--config", default=None, help="JSON motion threshold config")
    motion.set_defaults(func=cmd_motion)

    motion_eval = sub.add_parser("motion-eval", help="Evaluate motion classifier on manual labels")
    motion_eval.add_argument("--input", "-i", required=True, help="loc_vel_data.h5 or .csv")
    motion_eval.add_argument("--annotations", "-a", required=True, help="Annotation JSON file")
    motion_eval.add_argument("--fps", type=float, default=None)
    motion_eval.add_argument("--config", default=None, help="JSON motion threshold config")
    motion_eval.add_argument("--show-confusion", action="store_true")
    motion_eval.set_defaults(func=cmd_motion_eval)

    motion_tune = sub.add_parser("motion-tune", help="Tune motion thresholds on manual labels")
    motion_tune.add_argument("--input", "-i", required=True, help="loc_vel_data.h5 or .csv")
    motion_tune.add_argument("--annotations", "-a", required=True, help="Annotation JSON file")
    motion_tune.add_argument("--fps", type=float, default=None)
    motion_tune.add_argument("--config", default=None, help="Starting threshold config JSON")
    motion_tune.add_argument("--trials", type=int, default=4000)
    motion_tune.add_argument("--seed", type=int, default=0)
    motion_tune.add_argument(
        "--save-config",
        default=None,
        help="Write best thresholds to this JSON path",
    )
    motion_tune.set_defaults(func=cmd_motion_tune)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
