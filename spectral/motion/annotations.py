"""Load manual motion-type annotations for classifier training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spectral.motion.labels import MOTION_LABELS, MotionLabel

ANNOTATIONS_DIR = Path(__file__).resolve().parents[2] / "annotations"
DEFAULT_LABEL_ALIASES_PATH = ANNOTATIONS_DIR / "_label_aliases.json"


@dataclass(frozen=True)
class MotionSegmentAnnotation:
    start_frame: int
    end_frame: int
    label: MotionLabel
    label_name: str
    start_time: str
    end_time: str


@dataclass(frozen=True)
class MotionAnnotationSet:
    dataset: str
    fps: float
    segments: tuple[MotionSegmentAnnotation, ...]
    source: str = ""

    @property
    def num_labeled_frames(self) -> int:
        return int(np.sum(self.frame_mask(max_frame=10**9)))

    def frame_labels(self, num_frames: int) -> np.ndarray:
        """Per-frame ground truth; -1 where unlabeled."""
        labels = np.full(num_frames, -1, dtype=np.int32)
        for seg in self.segments:
            start = max(0, seg.start_frame)
            end = min(num_frames - 1, seg.end_frame)
            if start > end:
                continue
            labels[start : end + 1] = int(seg.label)
        return labels

    def frame_mask(self, max_frame: int) -> np.ndarray:
        return self.frame_labels(max_frame + 1) >= 0


def parse_timestamp(value: str) -> float:
    """Parse MM:SS or HH:MM:SS to seconds."""
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    raise ValueError(f"Invalid timestamp {value!r}; expected MM:SS or HH:MM:SS")


def timestamp_to_frame(value: str, fps: float) -> int:
    return int(round(parse_timestamp(value) * fps))


def _resolve_label(name: str, aliases: dict[str, str]) -> tuple[MotionLabel, str]:
    canonical = aliases.get(name, name)
    label_by_name = {v: k for k, v in MOTION_LABELS.items()}
    if canonical not in label_by_name:
        known = sorted(set(MOTION_LABELS.values()) | set(aliases.keys()))
        raise ValueError(f"Unknown label {name!r}; known names/aliases: {known}")
    label = label_by_name[canonical]
    return label, canonical


def load_default_label_aliases() -> dict[str, str]:
    if not DEFAULT_LABEL_ALIASES_PATH.exists():
        return {}
    with DEFAULT_LABEL_ALIASES_PATH.open(encoding="utf-8") as f:
        return {str(k): str(v) for k, v in json.load(f).items()}


def load_motion_annotations(path: str | Path) -> MotionAnnotationSet:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    fps = float(data["fps"])
    aliases = load_default_label_aliases()
    aliases.update({str(k): str(v) for k, v in data.get("label_aliases", {}).items()})
    segments: list[MotionSegmentAnnotation] = []
    for item in data["segments"]:
        label, label_name = _resolve_label(str(item["label"]), aliases)
        start_time = str(item["start"])
        end_time = str(item["end"])
        segments.append(
            MotionSegmentAnnotation(
                start_frame=timestamp_to_frame(start_time, fps),
                end_frame=timestamp_to_frame(end_time, fps),
                label=label,
                label_name=label_name,
                start_time=start_time,
                end_time=end_time,
            )
        )

    return MotionAnnotationSet(
        dataset=str(data.get("dataset", path.stem)),
        fps=fps,
        segments=tuple(segments),
        source=str(data.get("source", "")),
    )
