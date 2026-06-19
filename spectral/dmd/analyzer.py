"""Dynamic Mode Decomposition observables (PDF section 5.3)."""

from __future__ import annotations

import numpy as np
from pydmd import DMD

from spectral.types import DMDConfig, DMDObservables


class DMDAnalyzer:
    """Fit DMD to a time series of observable vectors y(t)."""

    def __init__(self, config: DMDConfig | None = None):
        self.config = config or DMDConfig()

    def fit(self, observable_matrix: np.ndarray, dt: float | None = None) -> DMDObservables:
        """
        Args:
            observable_matrix: (T, d) time-major observable sequence s_L(t) or combined
            dt: time step; defaults to config.dt or 1.0
        """
        y = np.asarray(observable_matrix, dtype=float)
        if y.ndim != 2:
            raise ValueError("observable_matrix must be 2D (num_frames, dim)")

        # Replace NaN rows with column means for fitting; track valid frames
        valid_rows = np.isfinite(y).all(axis=1)
        y_filled = y.copy()
        col_mean = np.nanmean(y, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        for t in range(y.shape[0]):
            if not valid_rows[t]:
                y_filled[t] = col_mean

        step = dt if dt is not None else (self.config.dt or 1.0)

        svd_rank = 0 if self.config.rank is None else self.config.rank
        dmd = DMD(svd_rank=svd_rank)
        # pydmd expects snapshots as columns: (d, T)
        dmd.fit(y_filled.T)

        modes = np.asarray(dmd.modes)  # (d, r)
        eigenvalues = np.asarray(dmd.eigs)  # (r,)
        frequencies = np.imag(np.log(eigenvalues) / step)

        # Time-varying amplitudes via projection onto modes
        r = modes.shape[1]
        amplitudes = np.full((y.shape[0], r), np.nan, dtype=float)
        for t in range(y.shape[0]):
            if valid_rows[t]:
                amplitudes[t] = np.linalg.lstsq(modes, y_filled[t], rcond=None)[0]

        modal_energies = np.abs(amplitudes) ** 2
        modal_energies[~valid_rows] = np.nan

        # Reconstruction for evaluation (eq. 25)
        reconstruction = (modes @ amplitudes.T).T
        reconstruction[~valid_rows] = np.nan

        return DMDObservables(
            modes=modes,
            eigenvalues=eigenvalues,
            frequencies=frequencies,
            amplitudes=amplitudes,
            modal_energies=modal_energies,
            reconstruction=reconstruction,
            dt=step,
        )
