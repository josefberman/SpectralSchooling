"""End-to-end observable pipeline (PDF sections 5.1–5.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spectral.dmd.analyzer import DMDAnalyzer
from spectral.graph.laplacian import GraphSpectralAnalyzer
from spectral.io import save_dmd, save_graph_spectral, save_koopman
from spectral.koopman.operator import fit_koopman_operator
from spectral.state import load_trajectory
from spectral.types import (
    DMDConfig,
    InteractionGraphConfig,
    KoopmanLiftConfig,
    ObservablePipelineResult,
)


def run_observable_pipeline(
    trajectory_path: str | Path,
    output_dir: str | Path,
    *,
    graph_config: InteractionGraphConfig | None = None,
    dmd_config: DMDConfig | None = None,
    koopman_config: KoopmanLiftConfig | None = None,
    fps: float | None = None,
    save_outputs: bool = True,
) -> ObservablePipelineResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_config = graph_config or InteractionGraphConfig()
    dmd_config = dmd_config or DMDConfig()
    koopman_config = koopman_config or KoopmanLiftConfig()

    trajectory = load_trajectory(trajectory_path, fps=fps)
    dt = dmd_config.dt or trajectory.dt

    graph_spectral = GraphSpectralAnalyzer(graph_config).compute(trajectory)

    # Mask invalid spectral rows before DMD
    spectral_input = graph_spectral.spectral_vector.copy()
    spectral_input[~graph_spectral.valid_mask] = np.nan

    dmd = DMDAnalyzer(dmd_config).fit(spectral_input, dt=dt)
    koopman = fit_koopman_operator(graph_spectral, dmd, koopman_config)

    if save_outputs:
        save_graph_spectral(graph_spectral, output_dir / "graph_spectral.npz")
        save_dmd(dmd, output_dir / "dmd_observables.npz")
        save_koopman(koopman, output_dir / "koopman_observables.npz")

    return ObservablePipelineResult(
        trajectory=trajectory,
        graph_spectral=graph_spectral,
        dmd=dmd,
        koopman=koopman,
    )
