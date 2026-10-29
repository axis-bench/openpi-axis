"""The bundle consumer: verify, refuse, then replay the plan exactly (CPU, no real dataset)."""

import copy
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from openpi.shared import normalize as _normalize
from openpi.training import axis_bundle
from openpi.training import axis_config
from openpi.training import axis_roots
from openpi.training.axis_config_test import FT
from openpi.training.weighted_loss import LOSS_WEIGHT_KEY

TESTDATA = pathlib.Path(__file__).with_name("testdata") / "axis_bundle"

# The ranges walk of the paper's training code over task 800 -- episodes 0, 1, 2 of 5, 3 and 4
# frames -- and task 1000 -- one episode of 6 -- with TOY_RANGES (out-of-order, clamped and missing
# spans): the concat rows it keeps, in order, and the (episode, frame) at each. Measured by running
# that code; the same vector pins the finetune compiler's episode table (AXIS-QVLA
# tests/unit/train/test_ft_bundle.py, FORK_RANGE_ROWS).
TOY_EPISODES = {800: [5, 3, 4], 1000: [6]}
TOY_RANGES = {"task_1000--0": [[4, 99]], "task_800--2": [[2, 4], [-3, 1]], "task_800--0": [[1, 3]]}
FORK_RANGE_ROWS = [1, 2, 10, 11, 8, 16, 17]
FORK_EPISODE_FRAME = [(0, 1), (0, 2), (2, 2), (2, 3), (2, 0), (0, 4), (0, 5)]


def write_bundle(
    root: pathlib.Path,
    *,
    plan,
    weights=None,
    cond=None,
    n_rows=100,
    reducer="mean",
    vocab=None,
    serve=({"fields": {}, "uncond_fields": None, "beta": 1.0},),
    requires=("plan",),
    kind="corpus",
    episodes=None,
    params=None,
    inputs=None,
) -> pathlib.Path:
    """A bundle in the signal compiler's on-disk form (axis.signals.bundle.write_bundle): canonical
    manifest, `<name>.npy` arrays, a digest over dtype, shape and bytes per array. `vocab` is
    {code: text} of one field, `quality`. A corpus row space carries its episode table (by default
    one episode of task 800 holding every row); a flat one carries none."""
    tmp = root / "incoming"
    tmp.mkdir(parents=True, exist_ok=True)
    named = {"plan": plan, "weights": weights, "cond": cond}
    rowspace_arrays = {}
    if kind == "corpus":
        episodes = episodes or {
            "episode_task": [800],
            "episode_index": [0],
            "episode_from": [0],
            "episode_length": [n_rows],
        }
        for name in axis_bundle.EPISODE_TABLE:
            named[f"rowspace.{name}"] = np.asarray(episodes[name], dtype=np.int64)
    arrays = {}
    for name, a in named.items():
        if a is None:
            continue
        np.save(tmp / f"{name}.npy", np.ascontiguousarray(a), allow_pickle=False)
        arrays[name] = {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": axis_bundle.array_digest(a)}
        if name.startswith("rowspace."):
            rowspace_arrays[name[len("rowspace.") :]] = arrays[name]
    text = (
        None
        if vocab is None
        else {
            "fields": ["quality"],
            "prompts": [],
            "text": {"quality": [[int(c), v] for c, v in sorted(vocab.items())]},
        }
    )
    manifest = {
        "schema": axis_bundle.SCHEMA,
        "program": {"method": "test", "version": 1, "params": params or {}},
        "program_sha": "0" * 64,
        "seed": 0,
        "controls": [],
        "diagnostics": {},
        "inputs": inputs or {},
        "legacy": [],
        "arrays": arrays,
        "budget": {"steps": int(plan.shape[0]), "global_batch": int(plan.shape[1])},
        "reducer": reducer,
        "requires": sorted(requires),
        "serve": list(serve),
        "vocab": text,
        "rowspace": {
            "id": "0" * 64,
            "name": "fixture",
            "kind": kind,
            "n_rows": n_rows,
            "n_trainable": n_rows,
            "arrays": rowspace_arrays,
        },
    }
    raw = axis_bundle.canonical_bytes(manifest)
    (tmp / "manifest.json").write_bytes(raw)
    final = root / hashlib.sha256(raw).hexdigest()
    tmp.rename(final)
    return final


def write_corpus(root: pathlib.Path, episodes: dict[int, list[int]], *, starts: dict | None = None):
    """Per-task datasets' episode metadata (meta/episodes/*.parquet) and a bare roots index relative
    to `root`: what axis_roots reads to place every row. Returns (roots index, corpus base)."""
    for tid, lengths in episodes.items():
        meta = root / f"task_{tid}" / "meta" / "episodes" / "chunk-000"
        meta.mkdir(parents=True)
        first = (starts or {}).get(tid) or [sum(lengths[:i]) for i in range(len(lengths))]
        pd.DataFrame({"episode_index": range(len(lengths)), "dataset_from_index": first, "length": lengths}).to_parquet(
            meta / "file-000.parquet"
        )
    index = root / "roots.json"
    index.write_text(json.dumps({str(tid): f"task_{tid}" for tid in episodes}))
    return index, root


def save_norm_stats(norm_dir: pathlib.Path, *, name: str = "heldout24_qual_v2") -> str:
    """Identity statistics (mean 0, std 1, q01 0, q99 1) for the Franka asset under `norm_dir`, and
    the norm id `<name>#sha256:<digest of norm_stats.json>` a config names them by."""
    stats = _normalize.NormStats(mean=np.zeros(8), std=np.ones(8), q01=np.zeros(8), q99=np.ones(8))
    asset = norm_dir / "Devon018" / "Franka-Datasets-v2"
    _normalize.save(asset, {"state": stats, "actions": stats})
    return f"{name}#sha256:{hashlib.sha256((asset / 'norm_stats.json').read_bytes()).hexdigest()}"


def _plan(steps=4, batch=2, n_rows=100):
    return (np.arange(steps * batch, dtype=np.int64) * 7 % n_rows).reshape(steps, batch)


def test_a_bundle_is_verified_by_content(tmp_path):
    d = write_bundle(tmp_path, plan=_plan())
    b = axis_bundle.load_bundle(d)
    assert b.bundle_id == f"sha256:{d.name}"
    assert b.plan.shape == (4, 2)
    assert b.weights is None
    assert b.episodes["episode_length"].tolist() == [100]
    other = tmp_path / ("f" * 64)
    d.rename(other)
    with pytest.raises(axis_bundle.BundleError, match="is not its manifest's digest"):
        axis_bundle.load_bundle(other)


@pytest.mark.parametrize("name", ["plan", "rowspace.episode_from"])
def test_a_tampered_array_is_refused(tmp_path, name):
    d = write_bundle(tmp_path, plan=_plan())
    np.save(d / f"{name}.npy", np.load(d / f"{name}.npy") + 1)
    with pytest.raises(axis_bundle.BundleError, match="not the one the manifest names"):
        axis_bundle.load_bundle(d)


def _check(tmp_path, *, steps=4, smoke=False, n_dataset_rows=100, **kw):
    b = axis_bundle.load_bundle(write_bundle(tmp_path, **kw))
    return axis_bundle.check(b, batch_size=2, num_train_steps=steps, smoke=smoke, n_dataset_rows=n_dataset_rows)


def test_a_fitting_bundle_passes_and_a_smoke_run_trains_a_prefix(tmp_path):
    assert _check(tmp_path / "a", plan=_plan()) == []
    assert _check(tmp_path / "b", plan=_plan(steps=40), steps=20, smoke=True) == []
    assert "only a smoke run" in _check(tmp_path / "c", plan=_plan(steps=40), steps=20)[0]


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"plan": _plan(), "requires": ("plan", "flow_time")}, "does not implement"),
        ({"plan": _plan(), "reducer": "global_ratio"}, "without weights"),
        ({"plan": _plan(), "weights": np.ones(100, np.float32), "reducer": "per_rank_ratio"}, "global_ratio"),
        ({"plan": _plan(batch=4)}, "plan batch 4"),
        ({"plan": _plan(), "n_dataset_rows": 99}, "dataset has 99 rows"),
        ({"plan": _plan(), "weights": np.full(100, np.nan, np.float32), "reducer": "global_ratio"}, "no finite weight"),
        ({"plan": _plan(), "cond": np.full((4, 2, 1), 255, np.uint8), "vocab": {5: "\nQuality: 5"}}, "forbidden"),
        ({"plan": _plan(), "cond": np.full((4, 2, 1), 5, np.uint8)}, "one field"),
        ({"plan": _plan(), "cond": np.full((4, 2, 1), 2, np.uint8), "vocab": {5: "\nQuality: 5"}}, "no vocabulary"),
        (
            {
                "plan": _plan(),
                "cond": np.full((4, 2, 1), 5, np.uint8),
                "vocab": {5: "x"},
                "serve": ({"fields": {"quality": "5"}, "uncond_fields": {}, "beta": 2.0},),
            },
            "untagged",
        ),
        ({"plan": _plan(), "kind": "sparse"}, "row space kind 'sparse'"),
        ({"plan": _plan(), "requires": ("plan", "loss_weights")}, "requires 'loss_weights' but carries no weights"),
        ({"plan": _plan(), "requires": ("plan", "cond_prompt")}, "requires 'cond_prompt' but carries no cond"),
    ],
)
def test_refusals(tmp_path, kw, match):
    n = kw.pop("n_dataset_rows", 100)
    problems = _check(tmp_path, n_dataset_rows=n, **kw)
    assert len(problems) == 1, problems
    assert match in problems[0], problems


def test_pure_conditioning_is_accepted_without_guidance(tmp_path):
    assert (
        _check(
            tmp_path,
            plan=_plan(),
            cond=np.full((4, 2, 1), 3, np.uint8),
            vocab={3: "\nQuality: 3"},
            serve=({"fields": {"quality": "3"}, "uncond_fields": None, "beta": 1.0},),
        )
        == []
    )


def test_the_sampler_replays_from_the_resume_step():
    assert list(axis_bundle.PlanSampler(steps=3, batch=2)) == [0, 1, 2, 3, 4, 5]
    assert list(axis_bundle.PlanSampler(steps=3, batch=2, start_step=2)) == [4, 5]


class _Rows:
    def __len__(self):
        return 100

    def __getitem__(self, i):
        return {"row": i, "prompt": "pick up the cup"}


def test_positions_map_to_planned_rows_with_tags_and_weights():
    plan = _plan()
    cond = np.zeros((4, 2, 1), np.uint8)
    cond[1, 0, 0] = 5
    rows = axis_bundle.PlanRows(_Rows(), plan, cond)
    assert [rows[p]["row"] for p in range(8)] == plan.reshape(-1).tolist()
    tag = axis_bundle.BundleConditioning({5: "\nQuality: 5"})
    assert tag(rows[2])["prompt"] == "pick up the cup\nQuality: 5"
    assert tag(rows[0])["prompt"] == "pick up the cup"
    w = np.arange(100, dtype=np.float32)
    weighted = axis_bundle.PlanWeights(rows, plan, w)
    assert [float(weighted[p][LOSS_WEIGHT_KEY]) for p in range(8)] == w[plan.reshape(-1)].tolist()


def test_the_token_budget_is_checked_over_every_prompt_and_tag():
    margin = axis_bundle.check_token_budget(["put the tomato beside the fork"], ["\nQuality: 5"], 200)
    assert 0 < margin < 200
    with pytest.raises(axis_bundle.BundleError, match="over the budget"):
        axis_bundle.check_token_budget(["word " * 300], [], 200)


class _Franka:
    def __len__(self):
        return 100

    def __getitem__(self, i):
        return {
            "observation.images.third_person": np.zeros((3, 36, 64), np.float32),
            "observation.images.wrist": np.zeros((3, 36, 64), np.float32),
            "observation.state": np.full(8, i / 100, np.float32),
            "action": np.zeros((10, 8), np.float32),
            "prompt": "put the tomato beside the fork",
        }


def test_the_loader_replays_the_plan_with_its_weights(tmp_path, monkeypatch):
    plan = _plan(steps=3, batch=2)
    weights = np.arange(100, dtype=np.float32) + 1
    bdir = write_bundle(
        tmp_path / "bundles", plan=plan, weights=weights, reducer="global_ratio", requires=("plan", "loss_weights")
    )
    norm_id = save_norm_stats(tmp_path / "norm")
    roots, base = write_corpus(tmp_path / "corpus", {800: [100]})
    d = copy.deepcopy(FT)
    d.update(num_train_steps=3, batch_size=2, num_workers=0)
    d["data"].update(
        bundle=str(bdir),
        bundle_id=f"sha256:{bdir.name}",
        norm_stats_dir=str(tmp_path / "norm"),
        roots_index=str(roots),
        corpus_base=str(base),
    )
    d["identity"].update(bundle=f"sha256:{bdir.name}", norm=norm_id)
    config = axis_config.train_config_from_dict(d)
    monkeypatch.setattr(axis_roots, "build_concat_dataset", lambda data_config, horizon: _Franka())
    monkeypatch.setattr(axis_roots, "task_prompts", lambda roots, repo_id: ["put the tomato beside the fork"])
    from openpi.training import data_loader as _data_loader

    observation, actions, w = next(iter(_data_loader.create_data_loader(config)))
    assert np.asarray(actions).shape == (2, 10, 32)
    np.testing.assert_array_equal(np.asarray(w), weights[plan[0]])
    np.testing.assert_allclose(np.asarray(observation.state)[:, 0], plan[0] / 100 * 2 - 1, atol=1e-6)


def test_the_committed_fixture_was_written_by_the_signal_compiler(tmp_path):
    """axis.signals.testing.toy_bundle(weights=True, cond=True), written by AXIS-QVLA's own writer,
    reads here unchanged (format drift on either side fails this)."""
    dirs = sorted(p for p in TESTDATA.iterdir() if p.is_dir())
    assert len(dirs) == 1
    b = axis_bundle.load_bundle(dirs[0])
    assert b.plan.tolist() == [[1, 2], [6, 7], [15, 16], [17, 21]]
    assert (b.kind, b.n_rows, b.budget) == ("corpus", 23, (4, 2))
    assert b.reducer == "global_ratio"
    assert b.fields == ("quality",)
    assert b.vocab[5] == "\nQuality: 5"
    assert b.cond[:, :, 0].tolist() == [[5, 0], [3, 1], [2, 4], [0, 5]]
    assert not b.guided
    assert axis_bundle.check(b, batch_size=2, num_train_steps=4, smoke=False, n_dataset_rows=23) == []
    # Its row space is task 800's episodes of 6, 4 and 5 frames, then task 1000's one of 8: datasets
    # laid out that way are the row space, and one episode longer is not.
    layout = axis_roots.corpus_layout(
        axis_roots.ordered_roots(*write_corpus(tmp_path / "a", {800: [6, 4, 5], 1000: [8]}))
    )
    assert axis_bundle.check_corpus_layout(b, layout) == []
    # Listed in another order, the same episodes occupy the same rows.
    shuffled = [
        axis_roots.TaskLayout(800, 0, ((2, 10, 5), (0, 0, 6), (1, 6, 4)), 15),
        axis_roots.TaskLayout(1000, 15, ((0, 0, 8),), 8),
    ]
    assert axis_bundle.check_corpus_layout(b, shuffled) == []
    other = axis_roots.corpus_layout(
        axis_roots.ordered_roots(*write_corpus(tmp_path / "b", {800: [6, 5, 4], 1000: [8]}))
    )
    assert axis_bundle.check_corpus_layout(b, other) == [
        "2 episodes are not where the bundle's row space puts them; the first in task and episode order is "
        "(task, episode, first row, length) = (800, 1, 6, 5) in the datasets and (800, 1, 6, 4) in the row space"
    ]


def test_flat_row_r_is_the_forks_rth_ranges_row(tmp_path):
    layout = axis_roots.corpus_layout(axis_roots.ordered_roots(*write_corpus(tmp_path, TOY_EPISODES)))
    rows = axis_roots.plan_rows_from_roots(layout, TOY_RANGES)
    assert rows.tolist() == FORK_RANGE_ROWS
    where = {t.base + s + f: (e, f) for t in layout for e, s, n in t.episodes for f in range(n)}
    assert [where[r] for r in rows.tolist()] == FORK_EPISODE_FRAME


def test_the_ranges_walk_refuses_what_would_misplace_a_row(tmp_path):
    layout = axis_roots.corpus_layout(axis_roots.ordered_roots(*write_corpus(tmp_path / "a", TOY_EPISODES)))
    with pytest.raises(ValueError, match="1 of the ranges file's 2 episodes name no episode"):
        axis_roots.plan_rows_from_roots(layout, {"task_800--0": [[0, 1]], "task_900--0": [[0, 1]]})
    # Episode 2 records offset 9, but its rows follow episode 1's at 5: refused once it has frames.
    shifted = axis_roots.corpus_layout(
        axis_roots.ordered_roots(*write_corpus(tmp_path / "b", {800: [3, 2, 4]}, starts={800: [0, 3, 9]}))
    )
    assert axis_roots.plan_rows_from_roots(shifted, {"task_800--1": [[0, 2]]}).tolist() == [3, 4]
    with pytest.raises(ValueError, match="episode 2 has frames in the ranges file but its dataset_from_index 9"):
        axis_roots.plan_rows_from_roots(shifted, {"task_800--2": [[0, 1]]})
    assert axis_roots.plan_rows_from_roots(shifted, {"task_800--2": [[7, 9]]}).tolist() == []  # clamped away


FLAT = {"row_decode": axis_bundle.FLAT_ROW_DECODE}


@pytest.mark.parametrize(
    ("params", "inputs", "match"),
    [
        ({}, {"roots_index": "a", "ranges": "b"}, "decodes its flat rows as None"),
        ({"row_decode": "row space row r"}, {"roots_index": "a", "ranges": "b"}, "as 'row space row r'"),
        (FLAT, {"ranges": "b"}, "records no 'roots_index' digest"),
        (FLAT, {"roots_index": "a"}, "records no 'ranges' digest"),
        (FLAT, {"roots_index": "a", "ranges": "c"}, "ranges file this run reads has sha256 b"),
        (FLAT, {"roots_index": "x", "ranges": "b"}, "roots index this run reads has sha256 a"),
    ],
)
def test_a_flat_bundle_names_its_decode_rule_and_the_files_this_run_reads(tmp_path, params, inputs, match):
    b = axis_bundle.load_bundle(
        write_bundle(tmp_path, plan=_plan(n_rows=7), n_rows=7, kind="flat", params=params, inputs=inputs)
    )
    problems = axis_bundle.check_flat_inputs(b, roots_sha256="a", ranges_sha256="b")
    assert len(problems) == 1, problems
    assert match in problems[0], problems


def test_a_flat_bundle_with_the_rule_and_the_digests_passes(tmp_path):
    b = axis_bundle.load_bundle(
        write_bundle(
            tmp_path,
            plan=_plan(n_rows=7),
            n_rows=7,
            kind="flat",
            params=FLAT,
            inputs={"roots_index": "a", "ranges": "b"},
        )
    )
    assert b.episodes is None
    assert axis_bundle.check_flat_inputs(b, roots_sha256="a", ranges_sha256="b") == []
    assert axis_bundle.check_corpus_layout(b, []) == []
