"""The scripts' AXIS hooks: the record a finished run leaves for a later `run:<arm>` init, the one
launch option a generated config takes (`--resume`), and the directory `compute_norm_stats.py
--config-json` writes, which is the one training reads and which it never replaces unasked."""

import copy
import dataclasses
import importlib.util
import json
import os
import pathlib
import re
import sys
import types

import numpy as np
import pytest

from openpi.training import axis_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIGS = json.loads((pathlib.Path(__file__).parent / "axis_config_test_configs.json").read_text())["configs"]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"axis_{name}_script", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = _script("train")
NORM = _script("compute_norm_stats")


class _Manager:
    """The two calls the end of a run makes on orbax's CheckpointManager."""

    def __init__(self, latest: int | None):
        self.latest, self.waited = latest, False

    def wait_until_finished(self) -> None:
        self.waited = True

    def latest_step(self) -> int | None:
        return self.latest


def _run(tmp_path, *, smoke: bool = False):
    d = copy.deepcopy(CONFIGS["pretrain_no_bind"])
    d["checkpoint_base_dir"] = str(tmp_path)
    d["data"]["smoke"] = smoke
    config = axis_config.train_config_from_dict(d)
    last = config.num_train_steps - 1
    for step in (last - 1, last):
        (config.checkpoint_dir / str(step) / "params").mkdir(parents=True)
    return config, last


def _summary(config) -> dict:
    return json.loads((config.checkpoint_dir / "train_summary.json").read_text())


def test_the_end_of_a_run_links_the_latest_checkpoint_and_writes_the_summary_last(tmp_path):
    """A later run's `run:<arm>` init reads <runs>/<arm>/train/final/params and the summary beside it."""
    config, last = _run(tmp_path)
    (config.checkpoint_dir / "final").symlink_to(str(last - 1))  # an earlier attempt's link is replaced
    manager = _Manager(last)
    TRAIN._finish_run(config, manager, types.SimpleNamespace(step=np.int32(config.num_train_steps)))  # noqa: SLF001
    final = config.checkpoint_dir / "final"
    assert manager.waited
    assert final.is_symlink()
    assert os.readlink(final) == str(last)
    assert (final / "params").resolve() == (config.checkpoint_dir / str(last) / "params")
    assert _summary(config) == {
        "schema": "qvla.train_summary/1",
        "name": config.name,
        "declared_steps": 20605,
        "trained_steps": 20605,
        "final_step": 20604,
        "smoke": False,
    }
    # Nothing but the step directories, the link and the summary: no temporary file is left behind.
    assert sorted(p.name for p in config.checkpoint_dir.iterdir()) == ["20603", "20604", "final", "train_summary.json"]


def test_the_summary_records_the_train_states_step_and_the_managers_latest_checkpoint(tmp_path):
    """Neither is taken from the config: a run that stopped early says so, and bind refuses it as an init."""
    config, last = _run(tmp_path, smoke=True)
    TRAIN._finish_run(config, _Manager(last - 1), types.SimpleNamespace(step=np.int32(1234)))  # noqa: SLF001
    s = _summary(config)
    assert (s["declared_steps"], s["trained_steps"], s["final_step"], s["smoke"]) == (20605, 1234, last - 1, True)
    assert os.readlink(config.checkpoint_dir / "final") == str(last - 1)


def test_only_process_zero_writes_the_record(tmp_path, monkeypatch):
    config, last = _run(tmp_path)
    monkeypatch.setattr(TRAIN.jax, "process_index", lambda: 1)
    manager = _Manager(last)
    TRAIN._finish_run(config, manager, types.SimpleNamespace(step=np.int32(20605)))  # noqa: SLF001
    assert manager.waited
    assert not (config.checkpoint_dir / "final").exists()
    assert not (config.checkpoint_dir / "train_summary.json").exists()


def test_no_summary_without_its_link(tmp_path):
    """The summary is written after the link: when the link cannot be made, no summary claims a final step."""
    config, last = _run(tmp_path)
    (config.checkpoint_dir / "final").mkdir()  # a directory, not a link: os.replace refuses it
    with pytest.raises(IsADirectoryError):
        TRAIN._finish_run(config, _Manager(last), types.SimpleNamespace(step=np.int32(20605)))  # noqa: SLF001
    assert not (config.checkpoint_dir / "train_summary.json").exists()
    config2, _ = _run(tmp_path / "b")
    with pytest.raises(RuntimeError, match="wrote no checkpoint"):
        TRAIN._finish_run(config2, _Manager(None), types.SimpleNamespace(step=np.int32(20605)))  # noqa: SLF001
    assert not (config2.checkpoint_dir / "train_summary.json").exists()


class _Franka:
    """Eight raw AXIS Franka rows; row i has state i in every dimension."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, i):
        return {
            "observation.images.third_person": np.zeros((3, 36, 64), np.float32),
            "observation.images.wrist": np.zeros((3, 36, 64), np.float32),
            "observation.state": np.full(8, float(i), np.float32),
            "action": np.full((10, 8), 2.0 * i, np.float32),
            "prompt": "put the tomato beside the fork",
        }


def test_norm_stats_of_a_generated_config_land_where_training_reads_them(tmp_path, monkeypatch):
    d = copy.deepcopy(CONFIGS["finetune_bound"])
    d.update(batch_size=4, checkpoint_base_dir=str(tmp_path / "runs"), assets_base_dir=str(tmp_path / "assets"))
    d["data"]["norm_stats_dir"] = str(tmp_path / "norm")
    path = tmp_path / "openpi_config.json"
    path.write_text(json.dumps(d))
    monkeypatch.setattr(_data_loader, "create_torch_dataset", lambda data_config, horizon, model: _Franka())
    NORM.main(config_json=str(path), num_workers=0)
    asset = d["data"]["asset_id"]
    assert (tmp_path / "norm" / asset / "norm_stats.json").is_file()
    assert not (tmp_path / "assets").exists()  # upstream's <assets_dirs>/<repo id> is not written
    config = axis_config.load_train_config(path)
    assert NORM.stats_output_path(config, config.data.create(config.assets_dirs, config.model)) == (
        tmp_path / "norm" / asset
    )
    # What training loads (DataConfigFactory.create_base_config) is the file just written.
    stats = config.data.create(config.assets_dirs, config.model).norm_stats
    np.testing.assert_allclose(stats["state"].mean, np.full(8, 3.5), rtol=1e-6)
    np.testing.assert_allclose(stats["actions"].mean, np.full(8, 7.0), rtol=1e-6)


def _norm_config(tmp_path) -> pathlib.Path:
    d = copy.deepcopy(CONFIGS["finetune_bound"])
    d.update(batch_size=4, checkpoint_base_dir=str(tmp_path / "runs"), assets_base_dir=str(tmp_path / "assets"))
    d["data"]["norm_stats_dir"] = str(tmp_path / "norm")
    path = tmp_path / "openpi_config.json"
    path.write_text(json.dumps(d))
    return path


class _Scaled(_Franka):
    def __getitem__(self, i):
        row = super().__getitem__(i)
        return {**row, "observation.state": row["observation.state"] * 10}


def test_existing_norm_stats_of_a_generated_config_are_replaced_only_with_overwrite(tmp_path, monkeypatch):
    path = _norm_config(tmp_path)
    monkeypatch.setattr(_data_loader, "create_torch_dataset", lambda data_config, horizon, model: _Franka())
    NORM.main(config_json=str(path), num_workers=0)
    written = tmp_path / "norm" / CONFIGS["finetune_bound"]["data"]["asset_id"] / "norm_stats.json"
    before = written.read_bytes()

    def no_frames(*args):
        raise AssertionError("refused before any frame is read")

    monkeypatch.setattr(_data_loader, "create_torch_dataset", no_frames)
    with pytest.raises(FileExistsError, match=rf"^{re.escape(str(written))} exists and is not replaced"):
        NORM.main(config_json=str(path), num_workers=0)
    assert written.read_bytes() == before
    monkeypatch.setattr(_data_loader, "create_torch_dataset", lambda data_config, horizon, model: _Scaled())
    NORM.main(config_json=str(path), num_workers=0, overwrite=True)
    assert written.read_bytes() != before


def test_the_one_flag_a_generated_config_takes_is_resume(tmp_path, monkeypatch):
    def cli(*args: str):
        monkeypatch.setattr(sys, "argv", ["scripts/train.py", "--config-json", *args])
        return axis_config.cli()

    path = tmp_path / "openpi_config.json"
    path.write_text(json.dumps(CONFIGS["pretrain_no_bind"]))
    first, resumed = cli(str(path)), cli(str(path), "--resume")
    assert (first.resume, resumed.resume) == (False, True)

    def fields(c) -> dict:
        return {f.name: getattr(c, f.name) for f in dataclasses.fields(c) if f.name not in ("resume", "freeze_filter")}

    assert fields(resumed) == fields(first)  # a resumed run trains the config its first launch trained
    for extra in (("--overwrite",), ("--resume", "--resume"), ("--resume", "--fsdp-devices=1"), ()):
        with pytest.raises(SystemExit, match="only a launch option follows it"):
            cli(*((str(path), *extra) if extra else ()))
    smoke = copy.deepcopy(CONFIGS["pretrain_no_bind"])
    smoke.update(overwrite=True, exp_name="smoke")
    smoke["data"]["smoke"] = True
    path.write_text(json.dumps(smoke))
    with pytest.raises(SystemExit, match="a smoke config starts over at every launch"):
        cli(str(path), "--resume")


def test_an_openpi_config_keeps_upstreams_norm_stats_path():
    config = _config.get_config("pi05_libero")
    data = types.SimpleNamespace(repo_id="physical-intelligence/libero", asset_id="physical-intelligence/libero")
    assert NORM.stats_output_path(config, data) == config.assets_dirs / "physical-intelligence/libero"
