"""Finite-dimensional Koopman operator fit (PDF section 5.4, eq. 14)."""

from __future__ import annotations

import numpy as np

from spectral.koopman.lift import build_koopman_features
from spectral.types import (
    DMDObservables,
    GraphSpectralObservables,
    KoopmanLiftConfig,
    KoopmanObservables,
)


def fit_koopman_operator(
    graph_spectral: GraphSpectralObservables,
    dmd: DMDObservables,
    config: KoopmanLiftConfig | None = None,
) -> KoopmanObservables:
    """Fit K such that z(t+1) ≈ K @ z(t)."""
    config = config or KoopmanLiftConfig()
    lifted, feature_names = build_koopman_features(graph_spectral, dmd, config)

    valid = np.isfinite(lifted).all(axis=1)
    z = lifted[valid]

    if z.shape[0] < 2:
        dim = lifted.shape[1]
        return KoopmanObservables(
            lifted_state=lifted,
            operator=np.full((dim, dim), np.nan),
            feature_names=feature_names,
        )

    z_curr = z[:-1]
    z_next = z[1:]
    operator = z_next.T @ np.linalg.pinv(z_curr.T)

    one_step = np.full((lifted.shape[0] - 1, lifted.shape[1]), np.nan)
    valid_pairs = valid[:-1] & valid[1:]
    pair_indices = np.flatnonzero(valid_pairs)
    if len(pair_indices):
        one_step[pair_indices] = (operator @ lifted[pair_indices].T).T

    return KoopmanObservables(
        lifted_state=lifted,
        operator=operator,
        feature_names=feature_names,
        one_step_prediction=one_step,
    )
