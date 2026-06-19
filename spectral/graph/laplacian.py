"""Graph Laplacian spectral analysis (PDF section 5.2, eq. 4–6)."""

from __future__ import annotations

import numpy as np

from spectral.graph.interaction_graph import build_adjacency
from spectral.types import (
    FishSchoolTrajectory,
    GraphSpectralObservables,
    InteractionGraphConfig,
)


class GraphSpectralAnalyzer:
    """Compute graph Laplacian eigenvalues s_L(t) for a fish-school trajectory."""

    def __init__(self, config: InteractionGraphConfig | None = None):
        self.config = config or InteractionGraphConfig()

    def compute(self, trajectory: FishSchoolTrajectory) -> GraphSpectralObservables:
        num_frames = trajectory.num_frames
        num_fish = trajectory.num_fish
        num_modes = max(num_fish - 1, 0)

        eigenvalues = np.full((num_frames, num_modes), np.nan, dtype=float)
        eigenvectors = np.full((num_frames, num_fish, num_modes), np.nan, dtype=float)
        frame_valid = np.zeros(num_frames, dtype=bool)

        for t in range(num_frames):
            positions_t = trajectory.positions[t]
            valid_t = trajectory.valid_mask[t]
            n_valid = int(valid_t.sum())
            if n_valid < 2:
                continue

            adjacency = build_adjacency(positions_t, valid_t, self.config)
            laplacian = _combinatorial_laplacian(adjacency)
            vals, vecs = np.linalg.eigh(laplacian)

            # Skip trivial zero eigenvalue; keep lambda_2 .. lambda_N
            spectral_vals = vals[1 : num_modes + 1]
            spectral_vecs = vecs[:, 1 : num_modes + 1]

            eigenvalues[t, : len(spectral_vals)] = spectral_vals
            eigenvectors[t, :, : len(spectral_vecs)] = spectral_vecs
            frame_valid[t] = True

        algebraic_connectivity = eigenvalues[:, 0] if num_modes > 0 else np.full(num_frames, np.nan)
        spectral_vector = eigenvalues.copy()

        return GraphSpectralObservables(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            algebraic_connectivity=algebraic_connectivity,
            spectral_vector=spectral_vector,
            valid_mask=frame_valid,
        )


def _combinatorial_laplacian(adjacency: np.ndarray) -> np.ndarray:
    degree = np.diag(adjacency.sum(axis=1))
    return degree - adjacency
