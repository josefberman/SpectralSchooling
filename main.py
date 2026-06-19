"""CLI for spectral observable pipeline (PDF sections 5.1–5.4)."""

from __future__ import annotations

import argparse
from pathlib import Path

from spectral.dmd.analyzer import DMDAnalyzer
from spectral.graph.laplacian import GraphSpectralAnalyzer
from spectral.io import load_dmd, load_graph_spectral, save_dmd, save_graph_spectral, save_koopman
from spectral.koopman.operator import fit_koopman_operator
from spectral.pipeline import run_observable_pipeline
from spectral.state import load_trajectory
from spectral.types import DMDConfig, InteractionGraphConfig, KoopmanLiftConfig


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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
