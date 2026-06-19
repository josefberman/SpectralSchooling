"""Koopman observable lifting (PDF section 5.4, eq. 13)."""

from __future__ import annotations

import numpy as np

from spectral.types import DMDObservables, GraphSpectralObservables, KoopmanLiftConfig


def build_koopman_features(
    graph_spectral: GraphSpectralObservables,
    dmd: DMDObservables,
    config: KoopmanLiftConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Construct lifted observable z(t) from graph and DMD observables.

    z(t) = [ s_L(t), E_DMD(t), s_L(t)⊗E_DMD(t), s_L(t)², E_DMD(t)² ]
    """
    config = config or KoopmanLiftConfig()
    s_l = graph_spectral.spectral_vector
    e_dmd = dmd.modal_energies
    num_frames = s_l.shape[0]

    blocks: list[np.ndarray] = []
    names: list[str] = []

    def add_block(arr: np.ndarray, prefix: str) -> None:
        blocks.append(arr)
        for i in range(arr.shape[1]):
            names.append(f"{prefix}_{i}")

    add_block(s_l, "sL")

    add_block(e_dmd, "E_DMD")

    if config.include_cross_product:
        cross = s_l[:, :, None] * e_dmd[:, None, :]
        cross_flat = cross.reshape(num_frames, -1)
        add_block(cross_flat, "sL_x_E_DMD")

    if config.include_polynomial:
        add_block(s_l**2, "sL_sq")
        add_block(e_dmd**2, "E_DMD_sq")

    lifted = np.hstack(blocks)
    return lifted, names
