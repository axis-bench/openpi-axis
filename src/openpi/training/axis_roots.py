"""The AXIS pretraining / finetuning dataset: per-task LeRobot datasets concatenated by task id.

A roots index names one LeRobot dataset per task (`{"version": 2, "base": "<dir>", "roots": {"<task
id>": "<dir>"}}`, or a bare `{"<task id>": "<dir>"}` mapping). The datasets are built and
concatenated in ascending task-id order, so a row of the concatenation is
`sum(lengths of earlier tasks) + dataset_from_index + frame` -- the rows a corpus row space's plan
indexes (`corpus_layout`). A flat row space indexes the rows a ranges file keeps instead, in the
ranges walk's order (`plan_rows_from_roots`). Each dataset's `len()` is asserted against the frame
count its episode metadata reports, because the plan's row numbers are only right if the two agree.

Relative entries resolve against `corpus_base` when the config gives one, otherwise against the
index's own `base` (relative to the index file). Nothing is read from the environment.

Online tier (openpi venv): lerobot and torch are imported inside the functions that need them.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import pathlib

import numpy as np


def ordered_roots(
    roots_index: str | pathlib.Path, corpus_base: str | pathlib.Path | None = None
) -> list[tuple[int, pathlib.Path]]:
    """`(task_id, dataset dir)` pairs, sorted by task id."""
    path = pathlib.Path(roots_index)
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "roots" in data:
        mapping = data["roots"]
        base = pathlib.Path(corpus_base) if corpus_base else path.parent / str(data.get("base", "."))
    else:
        mapping = data
        base = pathlib.Path(corpus_base) if corpus_base else None
    out = []
    for key, value in mapping.items():
        root = pathlib.Path(str(value))
        if not root.is_absolute():
            if base is None:
                raise ValueError(
                    f"{path}: entry {key!r} -> {value!r} is relative, and neither the index "
                    f"nor the config gives a base to resolve it against"
                )
            root = base / root
        out.append((int(key), root))
    return sorted(out, key=lambda kv: kv[0])


def episode_offsets(root: pathlib.Path) -> tuple[list[tuple[int, int, int]], int]:
    """`(episode_index, dataset_from_index, length)` of each episode of one dataset, in its
    meta/episodes order, and the dataset's total frames (no video decoding)."""
    import pandas as pd

    files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no meta/episodes parquet under {root}")
    ep = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "dataset_from_index", "length"]) for f in files],
        ignore_index=True,
    )
    rows = [
        (int(e), int(s), int(n))
        for e, s, n in zip(ep["episode_index"], ep["dataset_from_index"], ep["length"], strict=True)
    ]
    return rows, int(ep["length"].sum())


def episode_frames(root: pathlib.Path) -> int:
    """Total frames of one dataset, from meta/episodes/**/*.parquet (no video decoding)."""
    return episode_offsets(root)[1]


@dataclasses.dataclass(frozen=True)
class TaskLayout:
    """One dataset's place in the concatenation: its first global row and its episodes."""

    task_id: int
    base: int
    episodes: tuple[tuple[int, int, int], ...]  # (episode_index, dataset_from_index, length)
    total: int


def corpus_layout(roots: list[tuple[int, pathlib.Path]]) -> list[TaskLayout]:
    """Every dataset of `ordered_roots` placed in the concatenation `build_concat_dataset` makes:
    task t starts at the sum of the earlier tasks' frame counts."""
    out, base = [], 0
    for task_id, root in roots:
        episodes, total = episode_offsets(root)
        out.append(TaskLayout(task_id=int(task_id), base=base, episodes=tuple(episodes), total=total))
        base += total
    return out


def ranges_key(task_id: int, episode_index: int) -> str:
    """The episode key of a ranges file."""
    return f"task_{int(task_id)}--{int(episode_index)}"


def plan_rows_from_roots(layout: list[TaskLayout], ranges: dict) -> np.ndarray:
    """The concatenation row of every row of a flat row space cut by a ranges file, in order.

    A finetune's rows are this walk over the roots index and the ranges file, and the flat row
    space a finetune bundle indexes is that list: flat row r is the r-th row of it. The walk: tasks
    in ascending id, each task's episodes in its meta/episodes order, each episode's half-open spans
    in file order, clamped to [0, length); a row is `task base + dataset_from_index + frame`. The
    walk refuses, since either would put a flat row on a frame the plan was not keyed on, a ranges
    key that names no episode of these datasets, and an episode with planned frames whose
    `dataset_from_index` is not where its rows sit (the lengths of the episodes before it) or that
    is listed twice.
    """
    out, used = [], set()
    for task in layout:
        offset, seen = 0, set()
        for episode, start, length in task.episodes:
            key = ranges_key(task.task_id, episode)
            spans = ranges.get(key)
            if spans is not None:
                used.add(key)
                kept = [(lo, hi) for lo, hi in ((max(0, int(s)), min(int(e), length)) for s, e in spans) if hi > lo]
                if kept and (episode in seen or start != offset):
                    raise ValueError(
                        f"task {task.task_id}: episode {episode} has frames in the ranges file but its "
                        f"dataset_from_index {start} is not where its rows sit ({offset}) or it is "
                        f"listed twice"
                    )
                out += [np.arange(task.base + start + lo, task.base + start + hi, dtype=np.int64) for lo, hi in kept]
            seen.add(episode)
            offset += length
    unknown = sorted(set(ranges) - used)
    if unknown:
        raise ValueError(
            f"{len(unknown)} of the ranges file's {len(ranges)} episodes name no episode of the "
            f"datasets (first: {unknown[0]})"
        )
    return np.concatenate([np.empty(0, dtype=np.int64), *out])


def task_prompts(roots: list[tuple[int, pathlib.Path]], repo_id: str) -> list[str]:
    """Every task string of every dataset (the prompts the token budget is checked over)."""
    from lerobot.datasets import lerobot_dataset

    out = set()
    for _task_id, root in roots:
        meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
        out.update(str(t) for t in meta.tasks.index)
    return sorted(out)


class DataRows:
    """A dataset whose length is its data rows.

    The concatenation places a dataset's rows by its length, and every plan row is placed by the
    episode metadata walk, so the two must agree row for row. Each dataset's length is its data
    rows (`n_rows`), whatever count info.json's `total_frames` gives; indexing goes to the dataset
    unchanged, and LeRobot reads row i of its data.
    """

    def __init__(self, dataset, n_rows: int):
        self._dataset, self._n = dataset, int(n_rows)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i):
        if not 0 <= int(i) < self._n:
            raise IndexError(f"row {i} of a dataset of {self._n} rows")
        return self._dataset[i]


def data_rows(ds) -> int:
    """The data rows of a LeRobotDataset (what its rows are indexed by)."""
    if ds.hf_dataset is None:
        raise RuntimeError(f"{ds.root}: the dataset's data is not loaded, so its rows cannot be counted")
    return len(ds.hf_dataset)


def build_concat_dataset(data_config, action_horizon: int):
    """The concatenated dataset for an AxisDataConfig: each part is as long as its data rows, which
    must be the frames its episode metadata reports (`DataRows`)."""
    from lerobot.datasets import lerobot_dataset
    import torch

    from openpi import transforms as _transforms
    from openpi.training.data_loader import TransformedDataset

    parts = []
    for _task_id, root in ordered_roots(data_config.axis_roots_index, data_config.axis_corpus_base):
        total = episode_frames(root)
        meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=root)
        ds = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            root=root,
            delta_timestamps={
                k: [t / meta.fps for t in range(action_horizon)] for k in data_config.action_sequence_keys
            },
            tolerance_s=0.25 / meta.fps,
        )
        rows = data_rows(ds)
        if rows != total:
            raise RuntimeError(
                f"{root}: the dataset has {rows} data rows but its metadata reports {total} frames; "
                f"the plan's row numbers would land on the wrong frames"
            )
        part = ds if len(ds) == rows else DataRows(ds, rows)
        if data_config.prompt_from_task:
            tasks = {int(ti): str(task) for task, ti in meta.tasks["task_index"].items()}
            part = TransformedDataset(part, [_transforms.PromptFromLeRobotTask(tasks)])
        parts.append(part)
    if not parts:
        raise ValueError(f"{data_config.axis_roots_index} names no datasets")
    return torch.utils.data.ConcatDataset(parts)
