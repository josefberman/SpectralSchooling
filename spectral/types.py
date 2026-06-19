"""Shared dataclasses and configuration for the spectral analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class FishSchoolTrajectory:
    """Per-frame fish positions and velocities (PDF eq. 1)."""

    positions: np.ndarray  # (num_frames, num_fish, 2)
    velocities: np.ndarray  # (num_frames, num_fish, 2)
    valid_mask: np.ndarray  # (num_frames, num_fish) bool
    fps: float | None = None

    @property
    def num_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def num_fish(self) -> int:
        return self.positions.shape[1]

    @property
    def dt(self) -> float:
        return 1.0 / self.fps if self.fps else 1.0


@dataclass(frozen=True)
class InteractionGraphConfig:
    k_neighbors: int = 5
    symmetrize: Literal["mutual", "union", "directed"] = "mutual"
    distance_cutoff: float | None = None


@dataclass(frozen=True)
class GraphSpectralObservables:
    """Graph Laplacian spectral observables s_L(t) (PDF eq. 6)."""

    eigenvalues: np.ndarray  # (T, num_modes) — lambda_2 .. lambda_N
    eigenvectors: np.ndarray  # (T, num_fish, num_modes)
    algebraic_connectivity: np.ndarray  # (T,) — lambda_2
    spectral_vector: np.ndarray  # (T, d_L) — same as eigenvalues layout
    valid_mask: np.ndarray  # (T,) — False when frame has <2 valid fish


@dataclass(frozen=True)
class DMDConfig:
    rank: int | None = None
    dt: float | None = None  # if None, use trajectory.dt


@dataclass
class DMDObservables:
    """Dynamic Mode Decomposition observables (PDF eq. 7–11)."""

    modes: np.ndarray  # (d, r)
    eigenvalues: np.ndarray  # (r,) complex
    frequencies: np.ndarray  # (r,) real
    amplitudes: np.ndarray  # (T, r)
    modal_energies: np.ndarray  # (T, r)
    reconstruction: np.ndarray  # (T, d)
    dt: float = 1.0


@dataclass(frozen=True)
class KoopmanLiftConfig:
    include_cross_product: bool = True
    include_polynomial: bool = True


@dataclass
class KoopmanObservables:
    """Koopman lifted observables z(t) and operator K (PDF eq. 13–14)."""

    lifted_state: np.ndarray  # (T, d_z)
    operator: np.ndarray  # (d_z, d_z)
    feature_names: list[str] = field(default_factory=list)
    one_step_prediction: np.ndarray | None = None  # (T-1, d_z)


@dataclass
class ObservablePipelineResult:
    trajectory: FishSchoolTrajectory
    graph_spectral: GraphSpectralObservables
    dmd: DMDObservables
    koopman: KoopmanObservables
    motion: MotionPrediction | None = None


@dataclass(frozen=True)
class MotionClassificationConfig:
    """Thresholds for heuristic motion classification (tunable)."""

    traveling_polarization_threshold: float = 0.55
    swarming_polarization_threshold: float = 0.35
    swarming_heading_variance: float = 1.5
    milling_tangential_threshold: float = 0.55
    milling_angular_threshold: float = 500.0
    fountain_radial_threshold: float = 15.0
    fountain_spread_zscore: float = 2.0
    fountain_window: int = 15
    hydro_sub_threshold: float = 0.45
    smooth_window: int = 5


@dataclass
class MotionPrediction:
    """Per-frame collective motion labels."""

    labels: np.ndarray  # (T,) int — MotionLabel values
    hydro_sub_labels: np.ndarray  # (T,) int — HydroSubLabel values (-1 = none)
    confidence: np.ndarray  # (T,)
    features: np.ndarray  # (T, F)
    feature_names: list[str]
    label_names: list[str]
    hydro_sub_label_names: list[str]
