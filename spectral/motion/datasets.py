"""Resolve dataset paths for motion annotation training."""

from __future__ import annotations

import json
from pathlib import Path

ANNOTATIONS_DIR = Path(__file__).resolve().parents[2] / "annotations"
DATA_ROOT = ANNOTATIONS_DIR.parent / "schooling-datasets"
DEFAULT_MODEL_PATH = ANNOTATIONS_DIR / "motion_classifier.joblib"
DEFAULT_TREE_TEXT_PATH = ANNOTATIONS_DIR / "motion_classifier_tree.txt"


def load_dataset_registry(path: Path | None = None) -> dict[str, dict]:
    registry_path = path or (ANNOTATIONS_DIR / "datasets.json")
    with registry_path.open(encoding="utf-8") as f:
        return json.load(f)


def trajectory_path(dataset: str, root: Path | None = None) -> Path:
    registry = load_dataset_registry()
    if dataset not in registry:
        raise KeyError(f"Unknown dataset {dataset!r}; add it to annotations/datasets.json")
    meta = registry[dataset]
    base = root or DATA_ROOT
    return base / meta["fish_group"] / dataset / f"{dataset}_loc_vel_data.h5"


def list_annotation_files(annotations_dir: Path | None = None) -> list[Path]:
    directory = annotations_dir or ANNOTATIONS_DIR
    return sorted(directory.glob("*_motion.json"))
