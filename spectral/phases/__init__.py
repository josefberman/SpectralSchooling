"""Phase identification stubs (PDF section 5.5)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from spectral.types import DMDObservables, GraphSpectralObservables, KoopmanObservables


class PhaseIdentifier(ABC):
    """Identify stationary and multistable phases from observables."""

    @abstractmethod
    def identify(
        self,
        graph_spectral: GraphSpectralObservables,
        dmd: DMDObservables,
        koopman: KoopmanObservables,
    ) -> np.ndarray:
        """Return per-frame phase labels."""
        ...
