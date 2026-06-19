"""Dynamic interaction graph construction (PDF section 5.2, eq. 2–3)."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from spectral.types import InteractionGraphConfig


def build_adjacency(
    positions: np.ndarray,
    valid_mask: np.ndarray,
    config: InteractionGraphConfig,
) -> np.ndarray:
    """
    Build k-nearest-neighbor adjacency for one frame.

    Args:
        positions: (num_fish, 2) positions at time t
        valid_mask: (num_fish,) bool — valid fish at time t
        config: graph construction parameters

    Returns:
        (num_fish, num_fish) adjacency matrix
    """
    num_fish = positions.shape[0]
    adjacency = np.zeros((num_fish, num_fish), dtype=float)
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) < 2:
        return adjacency

    valid_positions = positions[valid_indices]
    k = min(config.k_neighbors, len(valid_indices) - 1)
    if k < 1:
        return adjacency

    tree = cKDTree(valid_positions)
    for local_i, global_i in enumerate(valid_indices):
        distances, neighbors = tree.query(
            valid_positions[local_i],
            k=k + 1,
        )
        if np.isscalar(distances):
            distances = [distances]
            neighbors = [neighbors]
        for dist, local_j in zip(distances[1:], neighbors[1:]):
            if config.distance_cutoff is not None and dist > config.distance_cutoff:
                continue
            global_j = valid_indices[int(local_j)]
            adjacency[global_i, global_j] = 1.0

    if config.symmetrize == "mutual":
        adjacency = np.minimum(adjacency, adjacency.T)
    elif config.symmetrize == "union":
        adjacency = np.maximum(adjacency, adjacency.T)
    # "directed" leaves adjacency as built

    np.fill_diagonal(adjacency, 0.0)
    return adjacency
