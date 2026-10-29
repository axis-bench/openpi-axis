"""The bundle loader end to end on CPU: which frame each planned row reads, where the stream ends,
that a bundle that does not fit is refused before scripts/train.py builds the model, and how a
refused, a resumed and a mis-sharded launch start.

The datasets are stand-ins (row i of the concatenation has state i / 100 in every dimension, so the
row a sample came from can be read back after quantile normalisation, (x - 0) / (1 - 0) * 2 - 1);
their episode metadata is real parquet, which is what places every row.
"""

import copy
import dataclasses
import functools
import hashlib
import json
import logging

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import tokenizer as _tokenizer
from openpi.shared import normalize as _normalize
from openpi.training import axis_bundle
from openpi.training import axis_config
from openpi.training import axis_roots
from openpi.training import data_loader as _data_loader
from openpi.training.axis_bundle_test import FORK_RANGE_ROWS
from openpi.training.axis_bundle_test import TOY_EPISODES
from openpi.training.axis_bundle_test import TOY_RANGES
from openpi.training.axis_bundle_test import save_norm_stats
from openpi.training.axis_bundle_test import write_bundle
from openpi.training.axis_bundle_test import write_corpus
from openpi.training.axis_config_test import FT
from openpi.training.axis_scripts_test import TRAIN

PROMPT = "put the tomato beside the fork"


@functools.cache
def _tokenizer_200() -> _tokenizer.PaligemmaTokenizer:
    return _tokenizer.PaligemmaTokenizer(200, strict=True)


FLAT_PLAN = np.asarray([[6, 0], [3, 5], [1, 2]], dtype=np.int64)  # rows of the 7-row flat row space


class _Concat:
    """The concatenated datasets: row i has state i / 100."""

    def __init__(self, n: int):
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i):
        if not 0 <= i < self._n:
            raise IndexError(i)
        return {
            "observation.images.third_person": np.zeros((3, 36, 64), np.float32),
            "observation.images.wrist": np.zeros((3, 36, 64), np.float32),
            "observation.state": np.full(8, i / 100, np.float32),
            "action": np.zeros((10, 8), np.float32),
            "prompt": PROMPT,
        }


def _rows_of(observation) -> list[int]:
    return np.rint((np.asarray(observation.state)[:, 0] + 1) / 2 * 100).astype(int).tolist()


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path, bdir, *, roots, base, ranges=None, steps=3, batch=2, bundle_id=None, smoke=False):
    norm = tmp_path / "norm"
    norm_id = save_norm_stats(norm)
    d = copy.deepcopy(FT)
    d.update(
        num_train_steps=steps,
        batch_size=batch,
        num_workers=0,
        fsdp_devices=1,
        checkpoint_base_dir=str(tmp_path / "runs"),
        assets_base_dir=str(tmp_path / "assets"),
    )
    d["data"].update(
        bundle=str(bdir),
        bundle_id=bundle_id or f"sha256:{bdir.name}",
        norm_stats_dir=str(norm),
        roots_index=str(roots),
        corpus_base=str(base),
        ranges_path=str(ranges or tmp_path / "ranges-not-read.json"),
        smoke=smoke,
    )
    d["identity"].update(bundle=d["data"]["bundle_id"], norm=norm_id)
    return axis_config.train_config_from_dict(d)


def _flat(tmp_path, *, cond=None, vocab=None, decode=axis_bundle.FLAT_ROW_DECODE, plan=FLAT_PLAN, **kw):
    """A weighted bundle over the toy pair's 7 ranges-cut rows, cut by the files this run reads."""
    roots, base = write_corpus(tmp_path / "corpus", TOY_EPISODES)
    ranges = tmp_path / "ranges.json"
    ranges.write_text(json.dumps(TOY_RANGES))
    bdir = write_bundle(
        tmp_path / "bundles",
        plan=plan,
        weights=np.arange(7, dtype=np.float32) + 1,
        reducer="global_ratio",
        n_rows=7,
        kind="flat",
        cond=cond,
        vocab=vocab,
        requires=("plan", "loss_weights", *(() if cond is None else ("cond_prompt",))),
        params={"row_decode": decode},
        inputs={"roots_index": _sha256(roots), "ranges": _sha256(ranges)},
    )
    return _config(tmp_path, bdir, roots=roots, base=base, ranges=ranges, **kw), ranges


def _corpus(tmp_path, *, datasets=None, plan=None, **kw):
    """An unweighted bundle over one task of 100 frames (episodes of 60 and 40)."""
    roots, base = write_corpus(tmp_path / "corpus", datasets or {800: [60, 40]})
    episodes = {
        "episode_task": [800, 800],
        "episode_index": [0, 1],
        "episode_from": [0, 60],
        "episode_length": [60, 40],
    }
    plan = np.asarray([[7, 70], [99, 0], [42, 61]], dtype=np.int64) if plan is None else plan
    bdir = write_bundle(tmp_path / "bundles", plan=plan, n_rows=100, episodes=episodes)
    return _config(tmp_path, bdir, roots=roots, base=base, **kw)


@pytest.fixture
def concat(monkeypatch):
    built = []

    def build(data_config, horizon):
        n = sum(
            t.total
            for t in axis_roots.corpus_layout(
                axis_roots.ordered_roots(data_config.axis_roots_index, data_config.axis_corpus_base)
            )
        )
        built.append(n)
        return _Concat(n)

    monkeypatch.setattr(axis_roots, "build_concat_dataset", build)
    monkeypatch.setattr(axis_roots, "task_prompts", lambda roots, repo_id: [PROMPT])
    return built


def test_a_flat_bundle_reads_row_r_from_the_rth_ranges_cut_row(tmp_path, concat):
    """Plan row r is FORK_RANGE_ROWS[r] of the 18-row concatenation, never concat row r."""
    cond = np.asarray([[[5], [0]], [[0], [5]], [[5], [5]]], dtype=np.uint8)
    config, _ = _flat(tmp_path, cond=cond, vocab={5: "\nQuality: 5"})
    batches = list(_data_loader.create_data_loader(config))
    assert concat == [18]
    assert len(batches) == 3
    want = np.asarray(FORK_RANGE_ROWS)[FLAT_PLAN]
    assert want.tolist() != FLAT_PLAN.tolist()
    for s, (observation, actions, weights) in enumerate(batches):
        assert _rows_of(observation) == want[s].tolist()
        np.testing.assert_array_equal(np.asarray(weights), FLAT_PLAN[s] + 1)  # weights index the bundle's rows
        assert np.asarray(actions).shape == (2, 10, 32)
        # Code 5 appends "\nQuality: 5" to the task prompt and code 0 leaves it bare: each sample's
        # tokens are exactly the tokenizer's for that prompt and the sample's own normalised state.
        for i in range(2):
            prompt = PROMPT + ("\nQuality: 5" if cond[s, i, 0] == 5 else "")
            tokens, mask = _tokenizer_200().tokenize(prompt, np.asarray(observation.state)[i, :8])
            np.testing.assert_array_equal(np.asarray(observation.tokenized_prompt)[i], tokens)
            np.testing.assert_array_equal(np.asarray(observation.tokenized_prompt_mask)[i], mask)


def test_the_stream_ends_with_the_plan_and_a_resume_continues_it(tmp_path, concat):
    config = _corpus(tmp_path)
    plan = [[7, 70], [99, 0], [42, 61]]
    it = iter(_data_loader.create_data_loader(config))
    assert [_rows_of(next(it)[0]) for _ in range(3)] == plan
    with pytest.raises(StopIteration):  # nothing past the plan's last step, no wrap to its start
        next(it)
    assert [_rows_of(b[0]) for b in _data_loader.create_data_loader(config, plan_start_step=1)] == plan[1:]
    with pytest.raises(axis_bundle.BundleError, match="there is no step 3 to resume the plan at"):
        _data_loader.create_data_loader(config, plan_start_step=3)


def test_a_smoke_run_trains_the_plans_first_steps(tmp_path, concat):
    plan = np.asarray([[7, 70], [99, 0], [42, 61], [1, 2]], dtype=np.int64)
    config = _corpus(tmp_path, plan=plan, steps=2, smoke=True)
    assert [_rows_of(b[0]) for b in _data_loader.create_data_loader(config)] == plan[:2].tolist()


def test_a_flat_bundle_cut_by_other_ranges_is_refused_before_any_dataset_is_built(tmp_path, concat):
    config, ranges = _flat(tmp_path)
    ranges.write_text(json.dumps({**TOY_RANGES, "task_1000--0": [[0, 2]]}))  # still 7 rows, other frames
    with pytest.raises(axis_bundle.BundleError, match="the ranges file this run reads has sha256"):
        _data_loader.create_data_loader(config)
    assert concat == []


def test_a_corpus_bundle_over_other_datasets_with_as_many_rows_is_refused(tmp_path, concat):
    config = _corpus(tmp_path, datasets={800: [50, 50]})
    with pytest.raises(
        axis_bundle.BundleError, match=r"order is .* = \(800, 0, 0, 50\) in the datasets and \(800, 0, 0, 60\)"
    ):
        _data_loader.create_data_loader(config)
    assert concat == []


@flax.struct.dataclass
class _State:
    step: int
    params: dict


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """scripts/train.py's main() with the model replaced: init_train_state returns a step counter,
    train_step records the rows and weights of each batch it is given, and nothing is saved."""
    calls = {"init": 0, "steps": [], "saved": [], "finished": None}

    def init_train_state(config, init_rng, mesh, *, resume):
        calls["init"] += 1
        return _State(step=0, params={}), None

    def train_step(config, rng, state, batch):
        observation, _actions, *weights = batch
        calls["steps"].append((_rows_of(observation), np.asarray(weights[0]).tolist() if weights else None))
        return state.replace(step=state.step + 1), {"loss": jnp.zeros(())}

    monkeypatch.setattr(TRAIN, "init_logging", lambda: None)
    monkeypatch.setattr(TRAIN, "init_train_state", init_train_state)
    monkeypatch.setattr(TRAIN, "train_step", train_step)
    monkeypatch.setattr(TRAIN.jax, "jit", lambda f, **kw: f)
    monkeypatch.setattr(TRAIN._checkpoints, "save_state", lambda m, s, d, step: calls["saved"].append(step))  # noqa: SLF001
    monkeypatch.setattr(TRAIN, "_finish_run", lambda c, m, s: calls.update(finished=int(s.step)))
    # main() points JAX's persistent compilation cache at this directory; afterwards the cache is
    # put back as it was, so later tests in the session do not write into it.
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "jax_cache"))
    old = jax.config.jax_compilation_cache_dir
    yield calls
    jax.config.update("jax_compilation_cache_dir", old)
    from jax._src import compilation_cache

    compilation_cache.reset_cache()


def test_the_training_loop_trains_every_planned_batch_and_draws_no_more(tmp_path, concat, harness, caplog):
    config, _ = _flat(tmp_path)
    with caplog.at_level(logging.INFO):
        TRAIN.main(config)
    want = np.asarray(FORK_RANGE_ROWS)[FLAT_PLAN].tolist()
    assert harness["steps"] == [(want[s], (FLAT_PLAN[s] + 1).astype(float).tolist()) for s in range(3)]
    assert harness["saved"] == [2]
    assert harness["finished"] == 3
    line = f"axis bundle {config.data.bundle_id}: steps=3 batch=2 rows=7 start_step=0 weights=True cond=False"
    # The decoded row map's digest: the dataset row of every flat row, little-endian int64.
    rows_sha = hashlib.sha256(np.asarray(FORK_RANGE_ROWS, dtype="<i8").tobytes()).hexdigest()
    assert any(r.getMessage() == f"{line} reducer=global_ratio rows_sha256={rows_sha}" for r in caplog.records)


@pytest.mark.parametrize("problem", ["id", "ranges", "decode", "batch", "layout", "rows"])
def test_a_bundle_that_does_not_fit_never_reaches_model_init(tmp_path, concat, harness, problem):
    if problem == "id":
        config, _ = _flat(tmp_path, bundle_id="sha256:" + "0" * 64)
    elif problem == "ranges":
        config, ranges = _flat(tmp_path)
        ranges.write_text(json.dumps({**TOY_RANGES, "task_800--1": [[0, 1]]}))
    elif problem == "decode":
        config, _ = _flat(tmp_path, decode="flat row r = concat row r")
    elif problem == "batch":
        config, _ = _flat(tmp_path, batch=4)
    elif problem == "layout":
        config = _corpus(tmp_path, datasets={800: [40, 60]})
    else:
        config = _corpus(tmp_path, datasets={800: [60, 40], 900: [5]})
    with pytest.raises(axis_bundle.BundleError, match="refused"):
        TRAIN.main(config)
    assert harness["init"] == 0
    assert harness["steps"] == []
    assert concat == []
    assert not config.checkpoint_dir.exists()


def test_a_refused_launch_leaves_no_checkpoint_directory_and_runs_unchanged_once_its_input_is_fixed(
    tmp_path, concat, harness
):
    config, ranges = _flat(tmp_path)
    cut_by = ranges.read_text()
    ranges.write_text(json.dumps({**TOY_RANGES, "task_800--1": [[0, 1]]}))
    with pytest.raises(axis_bundle.BundleError, match="the ranges file this run reads has sha256"):
        TRAIN.main(config)
    assert not config.checkpoint_dir.exists()
    ranges.write_text(cut_by)
    TRAIN.main(config)  # the same config: no FileExistsError from the refused launch's directory
    assert harness["init"] == 1
    assert [rows for rows, _ in harness["steps"]] == np.asarray(FORK_RANGE_ROWS)[FLAT_PLAN].tolist()
    assert config.checkpoint_dir.is_dir()


def test_norm_statistics_other_than_the_arms_are_refused_before_any_dataset_is_built(tmp_path, concat):
    config, _ = _flat(tmp_path)
    stats = _normalize.NormStats(mean=np.zeros(8), std=np.full(8, 2.0), q01=np.zeros(8), q99=np.full(8, 2.0))
    _normalize.save(tmp_path / "norm" / "Devon018" / "Franka-Datasets-v2", {"state": stats, "actions": stats})
    with pytest.raises(
        axis_bundle.BundleError,
        match=r"norm_stats.json have sha256 [0-9a-f]{12}, the arm names [0-9a-f]{12} \(identity.norm\)",
    ):
        _data_loader.create_data_loader(config)
    assert concat == []


def test_the_statistics_that_train_are_the_bytes_whose_digest_was_checked(tmp_path, concat):
    config, _ = _flat(tmp_path)
    loader = _data_loader.create_data_loader(config)
    path = tmp_path / "norm" / "Devon018" / "Franka-Datasets-v2" / "norm_stats.json"
    want = _normalize.deserialize_json(path.read_text())
    got = loader.data_config().norm_stats
    assert sorted(got) == sorted(want)
    for key in want:
        np.testing.assert_array_equal(got[key].q99, want[key].q99)


def test_a_server_config_is_refused_by_the_training_loader(tmp_path, concat):
    config, _ = _flat(tmp_path)
    serve = dataclasses.replace(config, data=dataclasses.replace(config.data, bundle=None, bundle_id=None))
    with pytest.raises(axis_bundle.BundleError, match=r"binds no bundle \(data.bundle is null\)"):
        _data_loader.create_data_loader(serve)
    assert concat == []


def test_an_untagged_bundle_is_refused_when_a_task_string_is_over_the_token_budget(tmp_path, concat, monkeypatch):
    config = _corpus(tmp_path)
    monkeypatch.setattr(axis_roots, "task_prompts", lambda roots, repo_id: ["word " * 300])
    with pytest.raises(axis_bundle.BundleError, match="over the budget of 200"):
        _data_loader.create_data_loader(config)
    assert concat == []


class _Stopped:
    """The checkpoint manager of a run that stopped after saving step `latest`."""

    def __init__(self, latest: int):
        self.latest = latest

    def latest_step(self) -> int:
        return self.latest

    def all_steps(self) -> tuple[int, ...]:
        return (self.latest,)

    def close(self) -> None:
        raise AssertionError("a resumed run's checkpoint directory is never removed")


def test_a_resumed_launch_replays_the_plan_from_the_step_after_its_last_checkpoint(
    tmp_path, concat, harness, monkeypatch
):
    """A checkpoint saved at step s holds the state after s + 1 steps, so the plan resumes at s + 1."""
    config, _ = _flat(tmp_path)
    config = dataclasses.replace(config, resume=True)
    config.checkpoint_dir.mkdir(parents=True)  # the stopped run's directory, with step 0 saved
    seen = {}

    def initialize(checkpoint_dir, *, keep_period, overwrite, resume):
        seen.update(overwrite=overwrite, resume=resume)
        return _Stopped(latest=0), True

    monkeypatch.setattr(TRAIN._checkpoints, "initialize_checkpoint_dir", initialize)  # noqa: SLF001
    monkeypatch.setattr(TRAIN._checkpoints, "restore_state", lambda m, state, loader: state.replace(step=1))  # noqa: SLF001
    TRAIN.main(config)
    assert seen == {"overwrite": False, "resume": True}
    want = np.asarray(FORK_RANGE_ROWS)[FLAT_PLAN].tolist()
    assert [rows for rows, _ in harness["steps"]] == want[1:]
    assert harness["finished"] == 3
    # A restored state that is not at the plan's resume step is refused, and the directory is kept.
    monkeypatch.setattr(TRAIN._checkpoints, "restore_state", lambda m, state, loader: state.replace(step=2))  # noqa: SLF001
    with pytest.raises(ValueError, match="restored step 2 but the plan resumes at 1"):
        TRAIN.main(config)
    assert config.checkpoint_dir.is_dir()


def test_fsdp_devices_that_do_not_divide_the_visible_devices_are_refused_before_anything_is_written(
    tmp_path, concat, harness
):
    config, _ = _flat(tmp_path)
    config = dataclasses.replace(config, fsdp_devices=jax.device_count() + 1)
    with pytest.raises(ValueError, match=r"fsdp_devices \d+ does not divide the \d+ visible devices"):
        TRAIN.main(config)
    assert not (tmp_path / "runs").exists()
    assert harness["init"] == 0
