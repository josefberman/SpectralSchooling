"""Load fish-school trajectories from loc_vel data (PDF section 5.1)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from spectral.types import FishSchoolTrajectory


def _wide_table_to_trajectory(
    data: np.ndarray,
    columns: list[str],
    fps: float | None = None,
) -> FishSchoolTrajectory:
    """Convert wide (T, 4*N) table to FishSchoolTrajectory."""
    fish_ids: set[int] = set()
    for col in columns:
        if col.startswith("fish") and col.endswith(("_x", "_y", "_vx", "_vy")):
            fish_ids.add(int(col.split("_")[0].replace("fish", "")))

    if not fish_ids:
        raise ValueError("No fish columns found (expected fish{i}_x, fish{i}_y, ...)")

    num_fish = max(fish_ids) + 1
    num_frames = data.shape[0]
    positions = np.full((num_frames, num_fish, 2), np.nan, dtype=float)
    velocities = np.full((num_frames, num_fish, 2), np.nan, dtype=float)

    col_index = {name: i for i, name in enumerate(columns)}
    for fish in range(num_fish):
        for axis, suffix in enumerate(("_x", "_y")):
            pos_key = f"fish{fish}{suffix}"
            vel_key = f"fish{fish}_v{suffix[1:]}"
            if pos_key in col_index:
                positions[:, fish, axis] = data[:, col_index[pos_key]]
            if vel_key in col_index:
                velocities[:, fish, axis] = data[:, col_index[vel_key]]

    valid_mask = np.isfinite(positions).all(axis=2)
    return FishSchoolTrajectory(
        positions=positions,
        velocities=velocities,
        valid_mask=valid_mask,
        fps=fps,
    )


def load_trajectory_from_h5(path: str | Path, fps: float | None = None) -> FishSchoolTrajectory:
    path = Path(path)
    with h5py.File(path, "r") as f:
        data = np.asarray(f["data"][:], dtype=float)
        columns = [c.decode() if isinstance(c, bytes) else str(c) for c in f["columns"][:]]
        file_fps = f.attrs.get("fps")
    return _wide_table_to_trajectory(data, columns, fps=fps or file_fps)


def load_trajectory_from_csv(path: str | Path, fps: float | None = None) -> FishSchoolTrajectory:
    path = Path(path)
    df = pd.read_csv(path, index_col=0)
    return _wide_table_to_trajectory(
        df.to_numpy(dtype=float),
        list(df.columns),
        fps=fps,
    )


def load_trajectory(path: str | Path, fps: float | None = None) -> FishSchoolTrajectory:
    path = Path(path)
    if path.suffix.lower() == ".h5":
        return load_trajectory_from_h5(path, fps=fps)
    if path.suffix.lower() == ".csv":
        return load_trajectory_from_csv(path, fps=fps)
    raise ValueError(f"Unsupported trajectory format: {path.suffix}")
