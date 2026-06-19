"""Save and load intermediate spectral analysis results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spectral.types import (
    DMDObservables,
    GraphSpectralObservables,
    KoopmanObservables,
    MotionPrediction,
)


def save_graph_spectral(obs: GraphSpectralObservables, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        eigenvalues=obs.eigenvalues,
        eigenvectors=obs.eigenvectors,
        algebraic_connectivity=obs.algebraic_connectivity,
        spectral_vector=obs.spectral_vector,
        valid_mask=obs.valid_mask,
    )
    return path


def load_graph_spectral(path: str | Path) -> GraphSpectralObservables:
    data = np.load(path)
    return GraphSpectralObservables(
        eigenvalues=data["eigenvalues"],
        eigenvectors=data["eigenvectors"],
        algebraic_connectivity=data["algebraic_connectivity"],
        spectral_vector=data["spectral_vector"],
        valid_mask=data["valid_mask"],
    )


def save_dmd(obs: DMDObservables, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        modes=obs.modes,
        eigenvalues=obs.eigenvalues,
        frequencies=obs.frequencies,
        amplitudes=obs.amplitudes,
        modal_energies=obs.modal_energies,
        reconstruction=obs.reconstruction,
        dt=obs.dt,
    )
    return path


def load_dmd(path: str | Path) -> DMDObservables:
    data = np.load(path)
    return DMDObservables(
        modes=data["modes"],
        eigenvalues=data["eigenvalues"],
        frequencies=data["frequencies"],
        amplitudes=data["amplitudes"],
        modal_energies=data["modal_energies"],
        reconstruction=data["reconstruction"],
        dt=float(data["dt"]),
    )


def save_koopman(obs: KoopmanObservables, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        lifted_state=obs.lifted_state,
        operator=obs.operator,
        feature_names=np.array(obs.feature_names, dtype=object),
        one_step_prediction=obs.one_step_prediction
        if obs.one_step_prediction is not None
        else np.array([]),
    )
    return path


def load_koopman(path: str | Path) -> KoopmanObservables:
    data = np.load(path, allow_pickle=True)
    one_step = data["one_step_prediction"]
    return KoopmanObservables(
        lifted_state=data["lifted_state"],
        operator=data["operator"],
        feature_names=[str(x) for x in data["feature_names"].tolist()],
        one_step_prediction=one_step if one_step.size else None,
    )


def save_motion_prediction(pred: MotionPrediction, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        labels=pred.labels,
        hydro_sub_labels=pred.hydro_sub_labels,
        confidence=pred.confidence,
        features=pred.features,
        feature_names=np.array(pred.feature_names, dtype=object),
        label_names=np.array(pred.label_names, dtype=object),
        hydro_sub_label_names=np.array(pred.hydro_sub_label_names, dtype=object),
    )
    return path


def load_motion_prediction(path: str | Path) -> MotionPrediction:
    data = np.load(path, allow_pickle=True)
    return MotionPrediction(
        labels=data["labels"],
        hydro_sub_labels=data["hydro_sub_labels"],
        confidence=data["confidence"],
        features=data["features"],
        feature_names=[str(x) for x in data["feature_names"].tolist()],
        label_names=[str(x) for x in data["label_names"].tolist()],
        hydro_sub_label_names=[str(x) for x in data["hydro_sub_label_names"].tolist()],
    )
