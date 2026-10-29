"""The roots index orders tasks by id and resolves entries against an explicit base only."""

import json
import types

import pandas as pd
import pytest

from openpi.training import axis_roots


def test_v2_index_resolves_against_its_own_base(tmp_path):
    idx = tmp_path / "index" / "roots.json"
    idx.parent.mkdir()
    idx.write_text(
        json.dumps({"version": 2, "base": "../corpus_d8", "roots": {"1000": "task_1000", "806": "task_806"}})
    )
    assert axis_roots.ordered_roots(idx) == [
        (806, idx.parent / "../corpus_d8/task_806"),
        (1000, idx.parent / "../corpus_d8/task_1000"),
    ]
    assert axis_roots.ordered_roots(idx, tmp_path / "elsewhere")[0] == (806, tmp_path / "elsewhere/task_806")


def test_a_bare_mapping_needs_a_base_for_relative_entries(tmp_path):
    idx = tmp_path / "roots.json"
    idx.write_text(json.dumps({"809": "ft_demos/out_eef/task_809"}))
    with pytest.raises(ValueError, match="relative"):
        axis_roots.ordered_roots(idx)
    assert axis_roots.ordered_roots(idx, tmp_path) == [(809, tmp_path / "ft_demos/out_eef/task_809")]


def test_episode_frames_reads_the_metadata(tmp_path):
    d = tmp_path / "meta" / "episodes" / "chunk-000"
    d.mkdir(parents=True)
    pd.DataFrame({"episode_index": [0, 1], "length": [10, 15], "dataset_from_index": [0, 10]}).to_parquet(
        d / "file-000.parquet"
    )
    assert axis_roots.episode_frames(tmp_path) == 25


class _Meta:
    """LeRobotDatasetMetadata for the stand-in datasets below."""

    def __init__(self, repo_id, root):
        self.fps = 15
        self.tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index([f"pick {root.name}"], name="task"))


class _LeRobot:
    """A LeRobotDataset as LeRobot 0.4.4 sizes it: `len()` is info.json's total_frames, while its
    data (hf_dataset) holds every row; row i is row i of the data."""

    def __init__(self, repo_id, root, delta_timestamps, tolerance_s):
        self.root = root
        self.hf_dataset = [None] * DATA_ROWS[root.name]

    def __len__(self):
        return INFO_FRAMES[self.root.name]

    def __getitem__(self, i):
        if not 0 <= i < len(self.hf_dataset):
            raise IndexError(i)
        return {"row": (self.root.name, int(i)), "task_index": 0}


DATA_ROWS = {"task_800": 12, "task_1000": 6}
INFO_FRAMES = {"task_800": 9, "task_1000": 6}  # task_800's info.json lists 9 frames


def test_each_part_of_the_concatenation_is_as_long_as_its_data(tmp_path, monkeypatch):
    """Rows are placed by the episode metadata (5 + 3 + 4 frames of task 800, then 6 of task 1000), so
    task 1000 starts at row 12 even though task 800's info.json says 9 frames."""
    from lerobot.datasets import lerobot_dataset

    from openpi.training.axis_bundle_test import write_corpus

    roots, base = write_corpus(tmp_path, {800: [5, 3, 4], 1000: [6]})
    monkeypatch.setattr(lerobot_dataset, "LeRobotDatasetMetadata", _Meta)
    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", _LeRobot)
    config = types.SimpleNamespace(
        axis_roots_index=str(roots),
        axis_corpus_base=str(base),
        repo_id="r",
        action_sequence_keys=("action",),
        prompt_from_task=True,
    )
    concat = axis_roots.build_concat_dataset(config, 10)
    assert len(concat) == 18 == sum(t.total for t in axis_roots.corpus_layout(axis_roots.ordered_roots(roots, base)))
    assert concat[11]["row"] == ("task_800", 11)
    assert concat[12]["row"] == ("task_1000", 0)
    assert concat[12]["prompt"] == "pick task_1000"
    monkeypatch.setitem(DATA_ROWS, "task_1000", 7)  # data rows the episode metadata does not account for
    with pytest.raises(RuntimeError, match="has 7 data rows but its metadata reports 6 frames"):
        axis_roots.build_concat_dataset(config, 10)
