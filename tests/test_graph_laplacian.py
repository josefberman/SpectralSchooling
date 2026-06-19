"""Tests for graph Laplacian spectral analysis."""

from __future__ import annotations

import numpy as np

from spectral.graph.interaction_graph import build_adjacency
from spectral.graph.laplacian import GraphSpectralAnalyzer
from spectral.types import FishSchoolTrajectory, InteractionGraphConfig


def _make_trajectory(positions: np.ndarray, fps: float = 30.0) -> FishSchoolTrajectory:
    num_frames, num_fish, _ = positions.shape
    velocities = np.empty_like(positions)
    if num_frames > 1:
        velocities[:, :-1] = np.diff(positions, axis=0)
        velocities[:, -1] = velocities[:, -2]
    else:
        velocities[:] = 0.0
    valid_mask = np.isfinite(positions).all(axis=2)
    return FishSchoolTrajectory(
        positions=positions,
        velocities=velocities,
        valid_mask=valid_mask,
        fps=fps,
    )


def test_mutual_knn_adjacency_is_symmetric():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    valid = np.ones(4, dtype=bool)
    config = InteractionGraphConfig(k_neighbors=2, symmetrize="mutual")
    adj = build_adjacency(positions, valid, config)
    assert np.allclose(adj, adj.T)
    assert adj.sum() > 0


def test_connected_school_has_positive_lambda2():
    num_fish = 6
    angles = np.linspace(0, 2 * np.pi, num_fish, endpoint=False)
    radius = 50.0
    positions = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    positions = positions[None, :, :]  # single frame
    traj = _make_trajectory(positions)
    obs = GraphSpectralAnalyzer().compute(traj)
    assert obs.valid_mask[0]
    assert obs.algebraic_connectivity[0] > 0.01


def test_split_clusters_lower_connectivity():
    num_fish = 8
    left = np.column_stack([np.linspace(0, 10, num_fish // 2), np.zeros(num_fish // 2)])
    right = np.column_stack([np.linspace(100, 110, num_fish // 2), np.zeros(num_fish // 2)])
    positions = np.vstack([left, right])[None, :, :]
    traj = _make_trajectory(positions)
    obs = GraphSpectralAnalyzer(InteractionGraphConfig(k_neighbors=3)).compute(traj)
    assert obs.algebraic_connectivity[0] < 0.5
