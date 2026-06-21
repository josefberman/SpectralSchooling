"""Spectral transition dynamics analysis for fish schools."""

from spectral.motion.classifier import MotionClassifier, classify_motion
from spectral.pipeline import run_observable_pipeline
from spectral.state import load_trajectory, load_trajectory_from_csv, load_trajectory_from_h5
from spectral.types import (
    DMDConfig,
    DMDObservables,
    FishSchoolTrajectory,
    GraphSpectralObservables,
    InteractionGraphConfig,
    KoopmanLiftConfig,
    KoopmanObservables,
    MotionClassifierConfig,
    MotionPrediction,
    ObservablePipelineResult,
)

__all__ = [
    "DMDConfig",
    "DMDObservables",
    "FishSchoolTrajectory",
    "GraphSpectralObservables",
    "InteractionGraphConfig",
    "KoopmanLiftConfig",
    "KoopmanObservables",
    "MotionClassifierConfig",
    "MotionClassifier",
    "MotionPrediction",
    "ObservablePipelineResult",
    "classify_motion",
    "load_trajectory",
    "load_trajectory_from_csv",
    "load_trajectory_from_h5",
    "run_observable_pipeline",
]
