"""Transition detection stubs (PDF section 5.6)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from spectral.types import DMDObservables, GraphSpectralObservables, KoopmanObservables


class TransitionDetector(ABC):
    """Detect transitions between spectral phase regions."""

    @abstractmethod
    def detect(
        self,
        graph_spectral: GraphSpectralObservables,
        dmd: DMDObservables,
        koopman: KoopmanObservables,
    ) -> np.ndarray:
        """Return boolean mask of transition frames."""
        ...
