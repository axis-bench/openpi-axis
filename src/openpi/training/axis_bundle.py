"""Bind an AXIS-QVLA SignalBundle to openpi's training loader.

A bundle is a directory named by its content id: `manifest.json` (canonical JSON, schema
qvla.signal_bundle/1, whose sha256 is the id) and one `<name>.npy` per array. This consumer reads
three of them -- `plan` int64 [S, B], optional `weights` float32 [N] (NaN on rows no plan reaches),
optional `cond` uint8 [S, B, F] (0 = the field is dropped) -- and, for a corpus row space, the
row space's episode table (`rowspace.episode_{task,index,from,length}`); it verifies each against
the manifest's digest (sha256 over dtype, shape and bytes). It is read with json and numpy only;
nothing is imported from AXIS-QVLA.

What a plan row means depends on the row space's kind:
  - corpus: row r is row r of the concatenated datasets (axis_roots.corpus_layout). The datasets'
    episodes, placed in the concatenation, must be the row space's episode table, entry for entry.
  - flat: row r is the r-th row of the ranges walk over the roots index
    (axis_roots.plan_rows_from_roots, FLAT_ROW_DECODE), which is how a pi0.5 finetune's 86,912
    rows are cut from its 230,088 frames. The bundle must record that rule in its program
    (`params.row_decode`) and the sha256 of the roots index and ranges file it was cut by in its
    `inputs`, and those must be the files this run reads.

Before the model is built this module refuses:
  - a config that binds no bundle (a policy server's config);
  - norm statistics whose raw bytes are not the ones the arm names (`identity.norm`);
  - a directory name that is not the manifest's digest, a manifest that is not canonical, or an
    array whose content is not the one the manifest names;
  - a capability this consumer does not implement, or one the bundle requires without the array
    that carries it (`loss_weights` without weights, `cond_prompt` without codes);
  - a reducer other than `mean` (no weights) or `global_ratio` (weights);
  - a plan whose shape is not the bundle's budget, whose batch is not the global batch, or whose
    step count is not the run's (only a smoke run trains a prefix: the first `num_train_steps`);
  - datasets that do not lay out the bundle's row space (above), or a row count that is not it;
  - a planned row without a finite weight, a planned code that is 255 or has no text, more than
    one conditioned field;
  - a worst-case prompt (any task string with any of the bundle's tags, or bare when it has none)
    over the model's token budget;
  - guidance at serve time with no untagged sample to guide away from. Pure conditioning (every
    sample tagged, no dropout) is accepted when the bundle serves without guidance.
Then the loader replays the plan: step s trains exactly the rows plan[s], in order, starting at
`plan_start_step` (a resumed run continues where its checkpoint stopped), and it ends after the
plan's last step: it yields num_train_steps - plan_start_step batches and no more. A code c != 0
appends the field's text for c to the task prompt; code 0 leaves it bare. The log line that
describes the bound bundle names, for a flat row space, the sha256 of the decoded row map (the
int64 dataset row of every flat row), so a run's log says which frames its rows are.

Online tier (openpi venv).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pathlib

import jax
import numpy as np
import torch
from typing_extensions import override

import openpi.models.model as _model
from openpi.shared import normalize as _normalize
from openpi.training import data_loader as _data_loader
from openpi.training.weighted_loss import LOSS_WEIGHT_KEY
import openpi.transforms as _transforms

SCHEMA = "qvla.signal_bundle/1"
CAPABILITIES = frozenset({"plan", "loss_weights", "cond_prompt", "guidance"})
COND_KEY = "axis_cond"
FORBIDDEN = 255
KINDS = ("corpus", "flat")
# How a flat row maps onto the datasets, in the words the finetune compiler records it in
# (`program.params.row_decode`), and the `inputs` keys of the two files that cut it.
FLAT_ROW_DECODE = "flat row r = the r-th row of plan_rows_from_roots(roots_index, ranges)"
INPUT_ROOTS = "roots_index"
INPUT_RANGES = "ranges"
_ARRAYS = {"plan": (np.int64, 2), "weights": (np.float32, 1), "cond": (np.uint8, 3)}
EPISODE_TABLE = ("episode_task", "episode_index", "episode_from", "episode_length")


class BundleError(ValueError):
    """The bundle cannot train this run; nothing has been built (scripts/train.py also removes the
    checkpoint directory the refused launch created, so the same launch can be run again)."""


def canonical_bytes(obj: object) -> bytes:
    """The bytes a bundle id is the sha256 of (sorted keys, indent 1, UTF-8, no NaN, newline)."""
    return (json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode()


def array_digest(a: np.ndarray) -> str:
    """sha256 over dtype, shape and C-order bytes: the digest a bundle manifest records per array."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}|{','.join(str(d) for d in a.shape)}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class Bundle:
    bundle_id: str
    manifest: dict
    plan: np.ndarray
    weights: np.ndarray | None
    cond: np.ndarray | None
    # A corpus row space's episode table (EPISODE_TABLE -> int64 [episodes]); None for a flat one.
    episodes: dict | None = None

    @property
    def reducer(self) -> str:
        return str(self.manifest["reducer"])

    @property
    def requires(self) -> frozenset[str]:
        return frozenset(self.manifest["requires"])

    @property
    def kind(self) -> str:
        return str(self.manifest["rowspace"]["kind"])

    @property
    def n_rows(self) -> int:
        return int(self.manifest["rowspace"]["n_rows"])

    @property
    def budget(self) -> tuple[int, int]:
        return int(self.manifest["budget"]["steps"]), int(self.manifest["budget"]["global_batch"])

    @property
    def fields(self) -> tuple[str, ...]:
        v = self.manifest["vocab"]
        return () if v is None else tuple(v["fields"])

    @property
    def vocab(self) -> dict[int, str]:
        """Code -> text of the first conditioned field ({} without conditioning)."""
        if not self.fields:
            return {}
        return {int(c): str(t) for c, t in self.manifest["vocab"]["text"][self.fields[0]]}

    @property
    def guided(self) -> bool:
        return any(s["uncond_fields"] is not None for s in self.manifest["serve"])

    @property
    def row_decode(self) -> str | None:
        program = self.manifest.get("program")
        params = program.get("params") if isinstance(program, dict) else None
        return params.get("row_decode") if isinstance(params, dict) else None

    @property
    def inputs(self) -> dict:
        inputs = self.manifest.get("inputs")
        return inputs if isinstance(inputs, dict) else {}


def _load_array(d: pathlib.Path, name: str, entry: dict) -> np.ndarray:
    f = d / f"{name}.npy"
    a = np.load(f, allow_pickle=False)
    if array_digest(a) != entry["sha256"] or a.dtype.str != entry["dtype"] or list(a.shape) != entry["shape"]:
        raise BundleError(f"{f}: its content is not the one the manifest names")
    return a


def load_bundle(path: str | pathlib.Path) -> Bundle:
    d = pathlib.Path(path)
    raw = (d / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if canonical_bytes(manifest) != raw:
        raise BundleError(f"{d}/manifest.json is not in canonical form")
    digest = hashlib.sha256(raw).hexdigest()
    if d.name != digest:
        raise BundleError(f"bundle directory {d.name} is not its manifest's digest {digest}")
    if manifest.get("schema") != SCHEMA:
        raise BundleError(f"{d}: schema {manifest.get('schema')!r} is not {SCHEMA!r}")
    arrays: dict[str, np.ndarray | None] = {"plan": None, "weights": None, "cond": None}
    for name, (dtype, ndim) in _ARRAYS.items():
        entry = manifest["arrays"].get(name)
        if entry is None:
            continue
        a = _load_array(d, name, entry)
        if a.dtype != dtype or a.ndim != ndim:
            raise BundleError(f"{d / name}.npy: {a.dtype} with {a.ndim} dims; {name} is {np.dtype(dtype)} with {ndim}")
        arrays[name] = a
    if arrays["plan"] is None:
        raise BundleError(f"{d}: a bundle always carries a plan")
    episodes = None
    if manifest["rowspace"].get("kind") == "corpus":
        episodes = {}
        for name in EPISODE_TABLE:
            entry = manifest["arrays"].get(f"rowspace.{name}")
            if entry is None:
                raise BundleError(f"{d}: a corpus row space carries its episode table; rowspace.{name} is missing")
            a = _load_array(d, f"rowspace.{name}", entry)
            if a.dtype != np.int64 or a.ndim != 1:
                raise BundleError(f"{d}: rowspace.{name} is {a.dtype} with {a.ndim} dims, not int64 with 1")
            episodes[name] = a
    return Bundle(
        bundle_id=f"sha256:{digest}",
        manifest=manifest,
        plan=arrays["plan"],
        weights=arrays["weights"],
        cond=arrays["cond"],
        episodes=episodes,
    )


def check(
    bundle: Bundle,
    *,
    batch_size: int,
    num_train_steps: int,
    smoke: bool,
    n_dataset_rows: int | None,
    expected_id: str | None = None,
) -> list[str]:
    """Every reason `bundle` cannot train this run; [] when it fits. `n_dataset_rows` is how many rows
    of the bundle's row space this run's datasets give (None when that could not be worked out:
    the reason is reported by the caller)."""
    out = []
    if expected_id is not None and bundle.bundle_id != expected_id:
        out.append(f"bundle {bundle.bundle_id} is not the arm's {expected_id}")
    if bundle.kind not in KINDS:
        out.append(f"row space kind {bundle.kind!r} is not one of {KINDS}")
    missing = sorted(bundle.requires - CAPABILITIES)
    if missing:
        out.append(f"requires {missing}, which this consumer does not implement")
    for capability, array, name in (("loss_weights", bundle.weights, "weights"), ("cond_prompt", bundle.cond, "cond")):
        if capability in bundle.requires and array is None:
            out.append(f"requires {capability!r} but carries no {name} array")
    if bundle.weights is None and bundle.reducer != "mean":
        out.append(f"reducer {bundle.reducer!r} without weights")
    if bundle.weights is not None and bundle.reducer != "global_ratio":
        out.append(f"reducer {bundle.reducer!r}; this consumer reduces weighted losses as 'global_ratio'")
    steps, batch = bundle.plan.shape
    if (steps, batch) != bundle.budget:
        out.append(f"plan shape {bundle.plan.shape} is not the bundle's budget {bundle.budget}")
    if batch != batch_size:
        out.append(f"plan batch {batch} != global batch {batch_size}")
    if steps < num_train_steps or (steps > num_train_steps and not smoke):
        out.append(
            f"plan has {steps} steps, the run trains {num_train_steps}"
            + ("" if smoke else " (only a smoke run trains a prefix)")
        )
    if n_dataset_rows is not None and n_dataset_rows != bundle.n_rows:
        out.append(f"the dataset has {n_dataset_rows} rows, the bundle's row space {bundle.n_rows}")
    used = bundle.plan[:num_train_steps]
    if used.size and (int(used.min()) < 0 or int(used.max()) >= bundle.n_rows):
        out.append(f"plan rows outside [0, {bundle.n_rows})")
    elif bundle.weights is not None:
        if bundle.weights.shape != (bundle.n_rows,):
            out.append(f"weights shape {bundle.weights.shape} is not [{bundle.n_rows}]")
        elif not np.isfinite(bundle.weights[used]).all():
            out.append("a planned row has no finite weight")
    if bundle.cond is not None:
        if bundle.cond.shape[:2] != bundle.plan.shape or bundle.cond.shape[2] != 1 or len(bundle.fields) != 1:
            out.append(
                f"cond shape {bundle.cond.shape} over fields {list(bundle.fields)}: this consumer "
                f"conditions on one field per sample"
            )
        else:
            cu = bundle.cond[:num_train_steps, :, 0]
            codes = set(np.unique(cu).tolist()) - {0}
            if FORBIDDEN in codes:
                out.append("a planned sample carries the forbidden code 255")
            if codes - set(bundle.vocab) - {FORBIDDEN}:
                out.append(f"codes {sorted(codes - set(bundle.vocab) - {FORBIDDEN})} have no vocabulary text")
            if bundle.guided and not bool((cu == 0).any()):
                out.append("the bundle serves with guidance but no planned sample is untagged")
    return out


def check_flat_inputs(bundle: Bundle, *, roots_sha256: str, ranges_sha256: str) -> list[str]:
    """For a flat row space: the decode rule and the two files' digests the bundle records are this
    loader's rule and the files this run reads ([] for a corpus row space)."""
    if bundle.kind != "flat":
        return []
    out = []
    if bundle.row_decode != FLAT_ROW_DECODE:
        out.append(
            f"the bundle decodes its flat rows as {bundle.row_decode!r}; this loader reads flat row r as "
            f"{FLAT_ROW_DECODE!r}"
        )
    for key, got, what in ((INPUT_ROOTS, roots_sha256, "roots index"), (INPUT_RANGES, ranges_sha256, "ranges file")):
        want = bundle.inputs.get(key)
        if not want:
            out.append(
                f"the bundle records no {key!r} digest in its inputs, so which frames its flat rows are "
                f"cannot be checked against the {what} this run reads"
            )
        elif want != got:
            out.append(
                f"the {what} this run reads has sha256 {got[:12]}, the bundle's flat rows were cut by {want[:12]}"
            )
    return out


def check_corpus_layout(bundle: Bundle, layout) -> list[str]:
    """For a corpus row space: every episode of the datasets, placed in the concatenation (`layout`,
    from axis_roots.corpus_layout), sits where the row space's episode table puts it, so plan row r
    is concat row r. Both tables are compared in (task, episode) order: where an episode is listed
    does not move its rows."""
    if bundle.kind != "corpus":
        return []

    def by_episode(a: np.ndarray) -> np.ndarray:
        return a[np.lexsort((a[:, 1], a[:, 0]))]

    got = np.asarray([(t.task_id, e, t.base + s, n) for t in layout for e, s, n in t.episodes], dtype=np.int64)
    got = by_episode(got.reshape(-1, 4))
    want = by_episode(np.stack([np.asarray(bundle.episodes[k], dtype=np.int64) for k in EPISODE_TABLE], axis=1))
    if got.shape != want.shape:
        return [f"the datasets hold {got.shape[0]} episodes, the bundle's row space {want.shape[0]}"]
    bad = np.flatnonzero((got != want).any(axis=1))
    if bad.size:
        i = int(bad[0])
        return [
            f"{bad.size} episodes are not where the bundle's row space puts them; the first in task and episode "
            f"order is (task, episode, first row, length) = {tuple(got[i].tolist())} in the datasets and "
            f"{tuple(want[i].tolist())} in the row space"
        ]
    return []


def verified_norm_stats(data_config) -> tuple[dict | None, list[str]]:
    """The norm statistics this run trains with, parsed from the very bytes whose sha256 is checked
    against the one the arm names (`identity.norm`), and every reason they cannot be used."""
    path, want = data_config.axis_norm_stats_path, data_config.axis_norm_stats_sha256
    if not want:
        return None, ["the config names no sha256 for its norm statistics (identity.norm)"]
    try:
        raw = pathlib.Path(path).read_bytes()
    except (OSError, TypeError) as exc:
        return None, [f"cannot read the norm statistics {path!r} ({exc.__class__.__name__})"]
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        return None, [
            f"the norm statistics {path} have sha256 {got[:12]}, the arm names {want[:12]} (identity.norm); "
            f"a run normalises with the statistics its arm names"
        ]
    return _normalize.deserialize_json(raw.decode("utf-8")), []


def check_token_budget(prompts: list[str], tags: list[str], max_token_len: int, *, state_dim: int = 32) -> int:
    """Smallest margin in tokens of any task string with any tag (the worst state: all bins 255).
    Raises when a combination does not fit."""
    from openpi.models import tokenizer as _tokenizer

    if not prompts:
        raise BundleError("no task strings to check the token budget against")
    probe = _tokenizer.PaligemmaTokenizer(max(8192, 2 * max_token_len))
    state = np.ones(state_dim, dtype=np.float32)
    margin, worst = max_token_len, ""
    for p in prompts:
        for t in ["", *tags]:
            _, mask = probe.tokenize(p + t, state=state)
            m = max_token_len - int(np.asarray(mask).sum())
            if m < margin:
                margin, worst = m, p + t
    if margin < 0:
        raise BundleError(f"{worst[:120]!r} is {-margin} tokens over the budget of {max_token_len}")
    return margin


class PlanSampler(torch.utils.data.Sampler[int]):
    """Yields plan positions start*B .. steps*B - 1 in order; position p is plan.flat[p]."""

    def __init__(self, *, steps: int, batch: int, start_step: int = 0):
        if not 0 <= start_step <= steps:
            raise BundleError(f"start step {start_step} is outside the plan's {steps} steps")
        self._start, self._end = start_step * batch, steps * batch

    def __len__(self) -> int:
        return self._end - self._start

    def __iter__(self):
        yield from range(self._start, self._end)


class PlanRows:
    """Position -> the planned row of the raw dataset, with its condition code. `rows` [S, B] are
    rows of the raw dataset: the plan itself for a corpus row space, its flat rows decoded for a
    flat one."""

    def __init__(self, dataset, rows: np.ndarray, cond: np.ndarray | None):
        self._dataset = dataset
        self._rows = np.ascontiguousarray(rows.reshape(-1))
        self._cond = None if cond is None else np.ascontiguousarray(cond[:, :, 0].reshape(-1))

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, position):
        p = int(position)
        item = dict(self._dataset[int(self._rows[p])])
        if self._cond is not None:
            item[COND_KEY] = int(self._cond[p])
        return item


class PlanWeights:
    """Adds each planned sample's loss weight after the transforms (the repack would drop it).
    `weights` index the bundle's row space, so they are read at the plan's own rows."""

    def __init__(self, dataset, plan: np.ndarray, weights: np.ndarray):
        self._dataset = dataset
        self._rows = np.ascontiguousarray(plan.reshape(-1))
        self._w = np.asarray(weights, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, position):
        p = int(position)
        w = self._w[int(self._rows[p])]
        if not np.isfinite(w):
            raise BundleError(f"plan position {p} reaches row {int(self._rows[p])}, which has no weight")
        item = dict(self._dataset[p])
        item[LOSS_WEIGHT_KEY] = np.float32(w)
        return item


@dataclasses.dataclass(frozen=True)
class BundleConditioning(_transforms.DataTransformFn):
    """Heads the transform chain: append vocab[code] to the prompt; code 0 leaves it bare."""

    vocab: dict

    @override
    def __call__(self, data: dict) -> dict:
        if COND_KEY not in data:
            raise KeyError(f"{COND_KEY!r} is missing: the dataset was not wrapped by PlanRows")
        code = int(data[COND_KEY])
        if code == 0:
            return data
        if "prompt" not in data:
            raise KeyError("a tagged sample has no prompt to append its tag to")
        return {**data, "prompt": f"{data['prompt']}{self.vocab[code]}"}


class BundleDataLoader(_data_loader.DataLoaderImpl):
    """Yields the plan's batches from the start step to its last step, then stops."""

    def __iter__(self):
        for batch in self._data_loader:
            observation = _model.Observation.from_dict(batch)
            if LOSS_WEIGHT_KEY in batch:
                yield observation, batch["actions"], batch[LOSS_WEIGHT_KEY]
            else:
                yield observation, batch["actions"]


def _dataset_rows(bundle: Bundle, data_config, layout) -> tuple[np.ndarray | None, list[str]]:
    """The raw-dataset row of every row of the bundle's row space (None for a corpus row space,
    where they coincide), and every reason that cannot be worked out or does not fit."""
    if bundle.kind != "flat":
        return None, check_corpus_layout(bundle, layout)
    raw, problems = {}, []
    for key, path in (("roots", data_config.axis_roots_index), ("ranges", data_config.axis_ranges_path)):
        try:
            raw[key] = pathlib.Path(path).read_bytes()
        except (OSError, TypeError) as exc:
            problems.append(f"cannot read the {key} file {path!r} ({exc.__class__.__name__})")
    if problems:
        return None, problems
    problems = check_flat_inputs(
        bundle,
        roots_sha256=hashlib.sha256(raw["roots"]).hexdigest(),
        ranges_sha256=hashlib.sha256(raw["ranges"]).hexdigest(),
    )
    if problems:
        return None, problems
    from openpi.training import axis_roots

    try:
        return axis_roots.plan_rows_from_roots(layout, json.loads(raw["ranges"])), []
    except ValueError as exc:
        return None, [f"the ranges file {data_config.axis_ranges_path}: {exc}"]


def flat_rows_sha256(flat_rows: np.ndarray) -> str:
    """sha256 of a flat row space's decoded row map: the dataset row of every flat row, as
    little-endian int64 in flat-row order."""
    return hashlib.sha256(np.ascontiguousarray(flat_rows, dtype="<i8").tobytes()).hexdigest()


def create_bundle_data_loader(
    data_config,
    model_config,
    action_horizon: int,
    batch_size: int,
    *,
    sharding=None,
    num_workers: int = 0,
    seed: int = 0,
    num_train_steps: int,
    plan_start_step: int = 0,
) -> BundleDataLoader:
    from openpi.training import axis_roots

    if jax.process_count() != 1:
        raise BundleError("plan replay runs in one process that drives all local devices")
    if data_config.axis_bundle is None:
        raise BundleError(
            "the config binds no bundle (data.bundle is null): it is a policy server's config; train the "
            "config `qvla train` renders for the arm"
        )
    if data_config.norm_stats is None:
        raise BundleError(f"no norm stats for asset {data_config.asset_id!r} in the config's assets directory")
    if not 0 <= plan_start_step < num_train_steps:
        raise BundleError(
            f"the run trains steps 0..{num_train_steps - 1}; there is no step {plan_start_step} to resume the plan at"
        )
    bundle = load_bundle(data_config.axis_bundle)
    layout = axis_roots.corpus_layout(
        axis_roots.ordered_roots(data_config.axis_roots_index, data_config.axis_corpus_base)
    )
    n_concat = sum(t.total for t in layout)
    norm_stats, problems = verified_norm_stats(data_config)
    flat_rows, more = _dataset_rows(bundle, data_config, layout)
    problems += more
    n_rows = n_concat if bundle.kind != "flat" else (None if flat_rows is None else len(flat_rows))
    problems += check(
        bundle,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        smoke=data_config.axis_smoke,
        n_dataset_rows=n_rows,
        expected_id=data_config.axis_bundle_id,
    )
    if problems:
        raise BundleError(f"bundle {bundle.bundle_id} refused:\n  " + "\n  ".join(problems))
    # What trains, and what every checkpoint saves, is exactly the statistics whose bytes were checked.
    data_config = dataclasses.replace(data_config, norm_stats=norm_stats)
    plan = bundle.plan[:num_train_steps]
    cond = None if bundle.cond is None else bundle.cond[:num_train_steps]
    transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    # Every task string must fit the budget: tagged with each of the bundle's tags, or bare.
    roots = axis_roots.ordered_roots(data_config.axis_roots_index, data_config.axis_corpus_base)
    margin = check_token_budget(
        axis_roots.task_prompts(roots, data_config.repo_id),
        sorted(set(bundle.vocab.values())) if cond is not None else [],
        model_config.max_token_len,
    )
    logging.info(f"axis bundle token margin: {margin}")
    if cond is not None:
        transforms = [BundleConditioning(bundle.vocab), *transforms]
    raw = axis_roots.build_concat_dataset(data_config, action_horizon)
    if len(raw) != n_concat:
        raise BundleError(f"the concatenated datasets have {len(raw)} rows, their metadata {n_concat}")
    rows = plan if flat_rows is None else flat_rows[plan]
    dataset = _data_loader.TransformedDataset(PlanRows(raw, rows, cond), transforms)
    if bundle.weights is not None:
        dataset = PlanWeights(dataset, plan, bundle.weights)
    decoded = "" if flat_rows is None else f" rows_sha256={flat_rows_sha256(flat_rows)}"
    logging.info(
        f"axis bundle {bundle.bundle_id}: steps={num_train_steps} batch={batch_size} "
        f"rows={bundle.n_rows} start_step={plan_start_step} weights={bundle.weights is not None} "
        f"cond={cond is not None} reducer={bundle.reducer}{decoded}"
    )
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        sharding=sharding,
        shuffle=False,
        sampler=PlanSampler(steps=num_train_steps, batch=batch_size, start_step=plan_start_step),
        num_batches=num_train_steps - plan_start_step,
        num_workers=num_workers,
        seed=seed,
        framework="jax",
    )
    return BundleDataLoader(data_config, loader)
