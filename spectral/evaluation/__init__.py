"""Evaluation metrics stubs (PDF section 5.9, eq. 25–28)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from spectral.types import DMDObservables, GraphSpectralObservables, KoopmanObservables


class SpectralEvaluator(ABC):
    """Compare biological and simulated observable trajectories."""

    @abstractmethod
    def reconstruction_error(self, dmd: DMDObservables) -> float:
        """DMD reconstruction error (eq. 25)."""
        ...

    @abstractmethod
    def fidelity(
        self,
        bio_graph: GraphSpectralObservables,
        sim_graph: GraphSpectralObservables,
        bio_dmd: DMDObservables,
        sim_dmd: DMDObservables,
        bio_koopman: KoopmanObservables,
        sim_koopman: KoopmanObservables,
    ) -> dict[str, float]:
        """Return F_L, F_D, F_K fidelity scores (eq. 26–28)."""
        ...
