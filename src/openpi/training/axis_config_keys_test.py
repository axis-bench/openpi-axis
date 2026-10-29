"""The generated config's key set is exactly the generator's at every level, every scalar has the
generator's JSON type, and knowledge insulation is off.

`axis_config_test_configs.json` holds what the AXIS-QVLA generator (`qvla train`) renders for two
paper pi0.5 arms, with the four roots at /r/<root>: a full-weight pretrain arm exactly as
`qvla train --no-bind` writes it, and a LoRA finetune arm bound to a placeholder bundle and as a
policy server renders it (no bundle).
"""

import copy
import dataclasses
import json
import pathlib
import re

import flax.nnx as nnx
import pytest

from openpi.models import pi0_config
from openpi.training import axis_config

CONFIGS = json.loads((pathlib.Path(__file__).parent / "axis_config_test_configs.json").read_text())["configs"]
TOP = {
    "schema",
    "name",
    "exp_name",
    "seed",
    "batch_size",
    "num_train_steps",
    "ema_decay",
    "model",
    "lr_schedule",
    "optimizer",
    "freeze",
    "fsdp_devices",
    "num_workers",
    "save_interval",
    "keep_period",
    "log_interval",
    "wandb_enabled",
    "overwrite",
    "resume",
    "checkpoint_base_dir",
    "assets_base_dir",
    "weight_loader",
    "data",
    "identity",
}
DATA = {
    "repo_id",
    "asset_id",
    "image_center_crop",
    "default_prompt",
    "prompt_from_task",
    "roots_index",
    "ranges_path",
    "corpus_base",
    "bundle",
    "bundle_id",
    "norm_stats_dir",
    "smoke",
}
LEVELS = [(), ("data",), ("model",), ("lr_schedule",), ("optimizer",), ("weight_loader",), ("identity",)]
IDENTITY = {"arm", "armspec_sha256", "preset", "init", "rowspace", "bundle", "norm"}
KI = {"knowledge_insulation", "ki_fast_loss_weight"}


def _at(d: dict, level: tuple[str, ...]) -> dict:
    for key in level:
        d = d[key]
    return d


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_the_generated_configs_have_exactly_the_known_keys_and_build(name):
    d = CONFIGS[name]
    assert set(d) == TOP == set(axis_config._TOP)  # noqa: SLF001
    assert set(d["data"]) == DATA == set(axis_config._DATA)  # noqa: SLF001
    assert set(d["weight_loader"]) == {"params_path"}
    c = axis_config.train_config_from_dict(d)
    assert (c.name, c.seed, c.batch_size, c.num_train_steps, c.ema_decay) == (
        d["identity"]["arm"],
        d["seed"],
        d["batch_size"],
        d["num_train_steps"],
        d["ema_decay"],
    )
    assert dataclasses.asdict(c.model) == d["model"]
    assert dataclasses.asdict(c.lr_schedule) == d["lr_schedule"]
    assert dataclasses.asdict(c.optimizer) == d["optimizer"]
    assert c.weight_loader.params_path == d["weight_loader"]["params_path"]
    assert (c.data.bundle, c.data.bundle_id) == (d["data"]["bundle"], d["data"]["bundle_id"])
    assert (c.data.roots_index, c.data.ranges_path) == (d["data"]["roots_index"], d["data"]["ranges_path"])
    assert str(c.checkpoint_dir) == f"/r/runs/{d['name']}/{d['exp_name']}"


def test_the_pretrain_trains_every_weight_and_the_finetune_only_lora():
    pt = axis_config.train_config_from_dict(CONFIGS["pretrain_no_bind"])
    assert isinstance(pt.freeze_filter, nnx.Nothing)
    assert pt.ema_decay == 0.999
    for name in ("finetune_bound", "finetune_serve"):
        ft = axis_config.train_config_from_dict(CONFIGS[name])
        assert repr(ft.freeze_filter) == repr(ft.model.get_freeze_filter())
        assert ft.ema_decay is None
    assert CONFIGS["finetune_serve"]["data"]["bundle"] is None


@pytest.mark.parametrize("level", LEVELS, ids=lambda level: ".".join(level) or "top")
def test_an_added_key_at_any_level_is_refused(level):
    d = copy.deepcopy(CONFIGS["pretrain_no_bind"])
    _at(d, level)["extra"] = 1
    with pytest.raises(ValueError, match=r"unknown keys \['extra'\]"):
        axis_config.train_config_from_dict(d)


DROPS = [(level, key) for level in LEVELS for key in sorted(_at(CONFIGS["finetune_bound"], level))]


@pytest.mark.parametrize(("level", "key"), DROPS, ids=lambda x: x if isinstance(x, str) else ".".join(x) or "top")
def test_every_dropped_key_at_every_level_is_refused(level, key):
    d = copy.deepcopy(CONFIGS["finetune_bound"])
    del _at(d, level)[key]
    match = "schema is None" if key == "schema" else rf"missing keys \['{key}'\]"
    with pytest.raises(ValueError, match=match):
        axis_config.train_config_from_dict(d)


def test_the_type_tables_cover_exactly_the_key_sets():
    types = axis_config._TYPES  # noqa: SLF001
    groups = {"data", "identity", "model", "lr_schedule", "optimizer", "weight_loader"}
    assert set(types["config"]) == TOP - groups
    assert set(types["config.data"]) == DATA | set(axis_config._DATA_OPTIONAL)  # noqa: SLF001
    assert set(types["config.identity"]) == IDENTITY == set(axis_config._IDENTITY)  # noqa: SLF001
    for group, allowed in axis_config._GROUPS.items():  # noqa: SLF001
        assert set(types[f"config.{group}"]) == set(allowed), group
    for d in CONFIGS.values():
        assert set(d["identity"]) == IDENTITY


def _wrong(kinds: tuple) -> object:
    """A JSON value of none of `kinds` (a string "false" is not a bool, 1 is not a string)."""
    return next(v for v in ("false", 1, 1.5, True, None, ["x"]) if not any(axis_config._is(v, k) for k in kinds))  # noqa: SLF001


SCALARS = [(where, key) for where, table in axis_config._TYPES.items() for key in table if key != "schema"]  # noqa: SLF001


@pytest.mark.parametrize(("where", "key"), SCALARS, ids=lambda x: x)
def test_every_scalar_of_the_wrong_type_is_refused(where, key):
    d = copy.deepcopy(CONFIGS["finetune_bound"])
    level = tuple(where.split(".")[1:])
    _at(d, level)[key] = _wrong(axis_config._TYPES[where][key])  # noqa: SLF001
    with pytest.raises(ValueError, match=rf"^{re.escape(where)}\.{key}: "):
        axis_config.train_config_from_dict(d)


@pytest.mark.parametrize(
    ("level", "key", "value"),
    [
        (("data",), "smoke", "false"),
        (("data",), "image_center_crop", None),
        (("data",), "prompt_from_task", 1),
        ((), "num_train_steps", 20605.0),
        ((), "batch_size", True),
        ((), "ema_decay", "0.999"),
        (("model",), "action_horizon", 10.0),
    ],
)
def test_a_value_of_another_json_type_is_refused_not_coerced(level, key, value):
    d = copy.deepcopy(CONFIGS["pretrain_no_bind"])
    _at(d, level)[key] = value
    with pytest.raises(ValueError, match=r"is not (bool|int|float)"):
        axis_config.train_config_from_dict(d)


def test_an_integer_is_accepted_where_a_float_is_declared():
    d = copy.deepcopy(CONFIGS["finetune_bound"])
    d["optimizer"]["clip_gradient_norm"] = 1
    assert axis_config.train_config_from_dict(d).optimizer.clip_gradient_norm == 1


def test_the_identity_names_this_arm():
    d = copy.deepcopy(CONFIGS["finetune_bound"])
    d["identity"]["arm"] = "pi05.finetune.another_arm"
    with pytest.raises(ValueError, match="identity.arm 'pi05.finetune.another_arm' is not the config's name"):
        axis_config.train_config_from_dict(d)


def test_the_norm_identity_names_the_digest_the_loader_checks():
    d = CONFIGS["finetune_bound"]
    c = axis_config.train_config_from_dict(d)
    assert c.data.norm_stats_sha256 == d["identity"]["norm"].partition("#sha256:")[2]
    bad = copy.deepcopy(d)
    bad["identity"]["norm"] = "heldout24_qual_v2"
    with pytest.raises(ValueError, match=r"identity.norm 'heldout24_qual_v2' is not <name>#sha256:<64 hex>"):
        axis_config.train_config_from_dict(bad)


@pytest.mark.parametrize("missing", ["bundle", "bundle_id"])
def test_a_bundle_and_its_id_come_together(missing):
    d = copy.deepcopy(CONFIGS["finetune_bound"])
    d["data"][missing] = None
    with pytest.raises(ValueError, match="bundle and bundle_id are given together"):
        axis_config.train_config_from_dict(d)


def test_knowledge_insulation_is_off_and_cannot_be_turned_on():
    # The generator renders no KI field and the model config has none: knowledge_insulation is
    # false and ki_fast_loss_weight 1.0, which is what the projection reports.
    assert not KI & {f.name for f in dataclasses.fields(pi0_config.Pi0Config)}
    for d in CONFIGS.values():
        assert not KI & set(d["model"])
        p = axis_config.project(axis_config.train_config_from_dict(d))
        assert p["model.knowledge_insulation"] is False
        assert p["model.ki_fast_loss_weight"] == 1.0
    assert "knowledge_insulation" not in (pathlib.Path(pi0_config.__file__).parent / "pi0.py").read_text()
    d = copy.deepcopy(CONFIGS["finetune_bound"])
    d["model"]["knowledge_insulation"] = True
    with pytest.raises(ValueError, match="knowledge_insulation"):
        axis_config.train_config_from_dict(d)
