"""Tests for motion annotation loading and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spectral.motion.annotations import (
    load_motion_annotations,
    parse_timestamp,
    timestamp_to_frame,
)
from spectral.motion.evaluation import evaluate_motion_prediction
from spectral.motion.labels import MotionLabel
from spectral.types import MotionPrediction


def test_parse_timestamp():
    assert parse_timestamp("00:17") == 17
    assert parse_timestamp("01:11") == 71
    assert parse_timestamp("13:40") == 820


def test_timestamp_to_frame():
    assert timestamp_to_frame("00:17", 29.97) == 509
    assert timestamp_to_frame("00:24", 29.97) == 719


def test_load_0103_annotations():
    path = Path("annotations/0103_motion.json")
    if not path.exists():
        pytest.skip("annotation file not present")
    ann = load_motion_annotations(path)
    assert ann.dataset == "0103"
    assert len(ann.segments) == 19
    assert ann.segments[2].label == MotionLabel.FOUNTAIN_EVASION
    assert ann.segments[3].label == MotionLabel.TRAVELING_POLARIZED


def test_all_annotation_files_load():
    directory = Path("annotations")
    files = sorted(directory.glob("*_motion.json"))
    if not files:
        pytest.skip("no annotation files")
    assert len(files) == 10
    for path in files:
        ann = load_motion_annotations(path)
        assert ann.segments


def test_frame_labels_overlap(tmp_path: Path):
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(
        json.dumps(
            {
                "dataset": "test",
                "fps": 10.0,
                "segments": [
                    {"start": "00:00", "end": "00:01", "label": "swarming"},
                    {"start": "00:01", "end": "00:02", "label": "milling"},
                ],
            }
        ),
        encoding="utf-8",
    )
    ann = load_motion_annotations(ann_path)
    labels = ann.frame_labels(30)
    assert labels[0] == MotionLabel.SWARMING
    assert labels[15] == MotionLabel.MILLING
    assert labels[25] == -1


def test_evaluate_perfect_prediction():
    ann_path = Path("annotations/0103_motion.json")
    if not ann_path.exists():
        pytest.skip("annotation file not present")
    ann = load_motion_annotations(ann_path)
    labels = ann.frame_labels(25000)
    mask = labels >= 0
    pred = MotionPrediction(
        labels=labels.copy(),
        confidence=np.ones_like(labels, dtype=float),
        features=np.zeros((len(labels), 6)),
        feature_names=[],
        label_names=[],
    )
    metrics = evaluate_motion_prediction(pred, ann)
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.labeled_frames == int(mask.sum())
