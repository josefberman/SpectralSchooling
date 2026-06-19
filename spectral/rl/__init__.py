"""Spectral imitation learning stubs (PDF section 5.7)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from spectral.types import DMDObservables, GraphSpectralObservables, KoopmanObservables


class SpectralReward(ABC):
    """MARL reward from observable-family comparison (eq. 23)."""

    @abstractmethod
    def compute(
        self,
        bio_graph: GraphSpectralObservables,
        sim_graph: GraphSpectralObservables,
        bio_dmd: DMDObservables,
        sim_dmd: DMDObservables,
        bio_koopman: KoopmanObservables,
        sim_koopman: KoopmanObservables,
    ) -> float:
        ...
