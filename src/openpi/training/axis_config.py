"""AXIS-QVLA training configs for this openpi line, built from the JSON `qvla train` writes.

`qvla train <arm>` renders an ArmSpec into a JSON description of a TrainConfig (schema
qvla.openpi_train_config/1). `train_config_from_dict` builds the TrainConfig from it and refuses any
key it does not know; `cli()` lets scripts/train.py and scripts/compute_norm_stats.py take
`--config-json <file>` beside openpi's named configs, and scripts/serve_policy.py takes the same
file as `--policy.config=<file>.json`. No AXIS config is registered by name: every run is generated.

`AxisFrankaDataConfig` is the one AXIS data factory: the per-task LeRobot datasets of a roots index
concatenated by task id (axis_roots), the DROID 8-D Franka transforms, and a tokenizer that raises
rather than truncate. Which rows are drawn, with what weight and with what prompt tag, comes only
from the bound SignalBundle (axis_bundle). The norm statistics are the file the config names by
directory and asset id, and the loader trains with them only if their raw bytes have the sha256
the arm names (`identity.norm`, `<name>#sha256:<digest>`).

Online tier (openpi venv).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import sys

import flax.nnx as nnx
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.axis_franka_policy as axis_franka_policy
import openpi.policies.axis_image_aug as axis_image_aug
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

SCHEMA = "qvla.openpi_train_config/1"
FREEZES = ("none", "lora", "lora_frozen_vision")
# The SigLIP tower; its projector (img/head) stays trainable.
VISION_TOWER = ".*img/(Transformer|embedding|pos_embedding).*"
_TOP = frozenset(
    {
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
)
_DATA = frozenset(
    {
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
)
# Data keys a config may leave out, and the value it then has.
_DATA_OPTIONAL = {"sim_image_aug": False, "action_units": "rad_s"}
# What the generator records about the arm a config was rendered for. Training reads one of them:
# the sha256 in `norm`, which the norm statistics it loads must have.
_IDENTITY = frozenset({"arm", "armspec_sha256", "preset", "init", "rowspace", "bundle", "norm"})
_NORM_ID = re.compile(r"^[^#]+#sha256:(?P<sha>[0-9a-f]{64})$")
# What may follow `--config-json <file>`: the launch options, which change how a run starts, never
# what it trains.
LAUNCH_FLAGS = ("--resume",)
# The JSON type of every scalar the config gives, per level: `int` is never a bool, `float` takes
# an integer too (JSON has one number type), `None` is null. A wrong type is refused rather than
# coerced: "false" is a true string, and a null crop flag would train without the crop and then
# serve without publishing the serve contract.
_TYPES: dict[str, dict[str, tuple]] = {
    "config": {
        "schema": (str,),
        "name": (str,),
        "exp_name": (str,),
        "seed": (int,),
        "batch_size": (int,),
        "num_train_steps": (int,),
        "ema_decay": (float, None),
        "freeze": (str,),
        "fsdp_devices": (int,),
        "num_workers": (int,),
        "save_interval": (int,),
        "keep_period": (int, None),
        "log_interval": (int,),
        "wandb_enabled": (bool,),
        "overwrite": (bool,),
        "resume": (bool,),
        "checkpoint_base_dir": (str,),
        "assets_base_dir": (str,),
    },
    "config.data": {
        "repo_id": (str,),
        "asset_id": (str,),
        "image_center_crop": (bool,),
        "default_prompt": (str, None),
        "prompt_from_task": (bool,),
        "roots_index": (str,),
        "ranges_path": (str,),
        "corpus_base": (str,),
        "bundle": (str, None),
        "bundle_id": (str, None),
        "norm_stats_dir": (str,),
        "smoke": (bool,),
        "sim_image_aug": (bool,),
        "action_units": (str,),
    },
    "config.model": {
        "dtype": (str,),
        "paligemma_variant": (str,),
        "action_expert_variant": (str,),
        "action_dim": (int,),
        "action_horizon": (int,),
        "max_token_len": (int,),
        "pi05": (bool,),
        "discrete_state_input": (bool,),
        "pytorch_compile_mode": (str, None),
    },
    "config.lr_schedule": {"warmup_steps": (int,), "peak_lr": (float,), "decay_steps": (int,), "decay_lr": (float,)},
    "config.optimizer": {
        "b1": (float,),
        "b2": (float,),
        "eps": (float,),
        "weight_decay": (float,),
        "clip_gradient_norm": (float,),
    },
    "config.weight_loader": {"params_path": (str,)},
    "config.identity": {
        "arm": (str,),
        "armspec_sha256": (str,),
        "preset": (str,),
        "init": (str,),
        "rowspace": (str,),
        "bundle": (str, None),
        "norm": (str,),
    },
}
# Fields held at the constant values below, projected so a TrainConfig built here compares with the
# paper-config lock.
DROPPED_INERT = {
    "model.knowledge_insulation": False,
    "model.ki_fast_loss_weight": 1.0,
    "data.eef_action": False,
    "data.joint9_action": False,
}


@dataclasses.dataclass(frozen=True)
class AxisDataConfig(_config.DataConfig):
    """DataConfig plus what the AXIS loader needs."""

    axis_roots_index: str | None = None
    # The ranges file of the row space: which frames of the roots index's datasets a flat row
    # space's rows are (axis_roots.plan_rows_from_roots). A corpus row space indexes every frame.
    axis_ranges_path: str | None = None
    axis_corpus_base: str | None = None
    axis_bundle: str | None = None
    axis_bundle_id: str | None = None
    axis_smoke: bool = False
    # The norm statistics file and the sha256 of its raw bytes the arm names: the loader refuses
    # any other bytes.
    axis_norm_stats_path: str | None = None
    axis_norm_stats_sha256: str | None = None


@dataclasses.dataclass(frozen=True)
class AxisFrankaDataConfig(_config.DataConfigFactory):
    roots_index: str = ""
    ranges_path: str = ""
    corpus_base: str = ""
    bundle: str = ""
    bundle_id: str = ""
    image_center_crop: bool = True
    default_prompt: str | None = None
    prompt_from_task: bool = True
    smoke: bool = False
    norm_stats_sha256: str = ""
    sim_image_aug: bool = False
    action_units: str = "rad_s"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> AxisDataConfig:
        if not isinstance(model_config, pi0_config.Pi0Config) or not model_config.pi05:
            raise ValueError("AxisFrankaDataConfig trains pi0.5 models")
        base = AxisDataConfig(
            prompt_from_task=self.prompt_from_task,
            action_sequence_keys=("action",),
            axis_roots_index=self.roots_index,
            axis_ranges_path=self.ranges_path or None,
            axis_corpus_base=self.corpus_base or None,
            axis_bundle=self.bundle,
            axis_bundle_id=self.bundle_id,
            axis_smoke=self.smoke,
        )
        data = dataclasses.replace(self, base_config=base).create_base_config(assets_dirs, model_config)
        norm_dir = pathlib.Path(self.assets.assets_dir or assets_dirs)
        return dataclasses.replace(
            data,
            axis_norm_stats_path=str(norm_dir / str(data.asset_id) / "norm_stats.json"),
            axis_norm_stats_sha256=self.norm_stats_sha256 or None,
            repack_transforms=_transforms.Group(
                inputs=[
                    *([axis_image_aug.SimImageAug()] if self.sim_image_aug else []),
                    _transforms.RepackTransform(
                        {
                            "base_0_rgb": "observation.images.third_person",
                            "left_wrist_0_rgb": "observation.images.wrist",
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    ),
                ]
            ),
            data_transforms=_transforms.Group(
                inputs=[axis_franka_policy.AxisFrankaInputs(center_crop=self.image_center_crop)],
                outputs=[axis_franka_policy.AxisFrankaOutputs()],
            ),
            model_transforms=_transforms.Group(
                inputs=[
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model_config.max_token_len, strict=True),
                        discrete_state_input=model_config.discrete_state_input,
                    ),
                    _transforms.PadStatesAndActions(model_config.action_dim),
                ]
            ),
        )


def _keys(d: dict, allowed: frozenset[str], where: str, optional: frozenset[str] = frozenset()) -> None:
    if not isinstance(d, dict):
        raise ValueError(f"{where}: expected an object, got {type(d).__name__}")
    unknown, missing = sorted(set(d) - allowed), sorted(allowed - set(d) - optional)
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {unknown}, missing keys {missing}")


def _is(value: object, kind: type | None) -> bool:
    if kind is None:
        return value is None
    if kind is bool:
        return isinstance(value, bool)
    if kind is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind is float:
        return isinstance(value, int | float) and not isinstance(value, bool)
    return isinstance(value, kind)


def _typed(d: dict, where: str) -> None:
    for key, kinds in _TYPES[where].items():
        if key not in d:
            continue
        if not any(_is(d[key], k) for k in kinds):
            names = " or ".join("null" if k is None else k.__name__ for k in kinds)
            raise ValueError(f"{where}.{key}: {d[key]!r} is not {names}")


def _fields(cls: type) -> frozenset[str]:
    return frozenset(f.name for f in dataclasses.fields(cls) if f.init)


# The groups passed through whole to one dataclass each. Every field must be given: a missing one
# would take openpi's default (an action horizon of 50, say) without the config saying so.
_GROUPS = {
    "model": _fields(pi0_config.Pi0Config),
    "lr_schedule": _fields(_optimizer.CosineDecaySchedule),
    "optimizer": _fields(_optimizer.AdamW),
    "weight_loader": frozenset({"params_path"}),
}


def train_config_from_dict(d: dict) -> _config.TrainConfig:
    if not isinstance(d, dict) or d.get("schema") != SCHEMA:
        raise ValueError(f"schema is {d.get('schema') if isinstance(d, dict) else None!r}, expected {SCHEMA!r}")
    _keys(d, _TOP, "config")
    data = d["data"]
    _keys(data, _DATA | frozenset(_DATA_OPTIONAL), "config.data", optional=frozenset(_DATA_OPTIONAL))
    for group, allowed in _GROUPS.items():
        _keys(d[group], allowed, f"config.{group}")
    _keys(d["identity"], _IDENTITY, "config.identity")
    _typed(d, "config")
    for group in ("data", "identity", *_GROUPS):
        _typed(d[group], f"config.{group}")
    if d["identity"]["arm"] != d["name"]:
        raise ValueError(f"config.identity.arm {d['identity']['arm']!r} is not the config's name {d['name']!r}")
    norm = _NORM_ID.fullmatch(d["identity"]["norm"])
    if norm is None:
        raise ValueError(f"config.identity.norm {d['identity']['norm']!r} is not <name>#sha256:<64 hex>")
    if (data["bundle"] is None) != (data["bundle_id"] is None):
        raise ValueError("config.data: bundle and bundle_id are given together or not at all")
    if d["freeze"] not in FREEZES:
        raise ValueError(f"freeze {d['freeze']!r} is not one of {FREEZES}")
    model = pi0_config.Pi0Config(**d["model"])
    if d["freeze"] in ("lora", "lora_frozen_vision"):
        if "lora" not in model.paligemma_variant and "lora" not in model.action_expert_variant:
            raise ValueError(f"freeze {d['freeze']!r} on a model with no LoRA variant would freeze nothing")
        freeze = model.get_freeze_filter()
        if d["freeze"] == "lora_frozen_vision":
            freeze = nnx.Any(freeze, nnx_utils.PathRegex(VISION_TOWER))
    else:
        freeze = nnx.Nothing()
    return _config.TrainConfig(
        name=d["name"],
        exp_name=d["exp_name"],
        model=model,
        weight_loader=weight_loaders.CheckpointWeightLoader(d["weight_loader"]["params_path"]),
        lr_schedule=_optimizer.CosineDecaySchedule(**d["lr_schedule"]),
        optimizer=_optimizer.AdamW(**d["optimizer"]),
        ema_decay=d["ema_decay"],
        freeze_filter=freeze,
        data=AxisFrankaDataConfig(
            repo_id=data["repo_id"],
            assets=_config.AssetsConfig(assets_dir=data["norm_stats_dir"], asset_id=data["asset_id"]),
            roots_index=data["roots_index"],
            ranges_path=data["ranges_path"],
            corpus_base=data["corpus_base"],
            bundle=data["bundle"],
            bundle_id=data["bundle_id"],
            image_center_crop=data["image_center_crop"],
            default_prompt=data["default_prompt"],
            prompt_from_task=data["prompt_from_task"],
            smoke=data["smoke"],
            norm_stats_sha256=norm["sha"],
            sim_image_aug=data.get("sim_image_aug", _DATA_OPTIONAL["sim_image_aug"]),
            action_units=data.get("action_units", _DATA_OPTIONAL["action_units"]),
        ),
        assets_base_dir=d["assets_base_dir"],
        checkpoint_base_dir=d["checkpoint_base_dir"],
        seed=d["seed"],
        batch_size=d["batch_size"],
        num_workers=d["num_workers"],
        num_train_steps=d["num_train_steps"],
        log_interval=d["log_interval"],
        save_interval=d["save_interval"],
        keep_period=d["keep_period"],
        overwrite=d["overwrite"],
        resume=d["resume"],
        wandb_enabled=d["wandb_enabled"],
        fsdp_devices=d["fsdp_devices"],
    )


def load_train_config(path: str | pathlib.Path) -> _config.TrainConfig:
    return train_config_from_dict(json.loads(pathlib.Path(path).read_text()))


def cli() -> _config.TrainConfig:
    """`--config-json <file> [--resume]` builds the generated config; anything else is openpi's own CLI.

    The config is complete, so the one flag that may follow it is a launch option: `--resume`
    continues the run in the config's checkpoint directory from the step after its last saved
    checkpoint (a directory with no saved step starts from step 0). It changes nothing the run
    trains. A smoke config starts over at every launch, so it is not resumed.
    """
    if len(sys.argv) >= 2 and sys.argv[1] == "--config-json":
        if (
            len(sys.argv) < 3
            or any(a not in LAUNCH_FLAGS for a in sys.argv[3:])
            or len(set(sys.argv[3:])) != len(sys.argv[3:])
        ):
            raise SystemExit(
                "usage: --config-json <file> [--resume]; the generated config is complete, and only a "
                "launch option follows it"
            )
        config = load_train_config(sys.argv[2])
        if "--resume" in sys.argv[3:]:
            if config.overwrite:
                raise SystemExit(f"{sys.argv[2]}: a smoke config starts over at every launch; it is not resumed")
            config = dataclasses.replace(config, resume=True)
        return config
    return _config.cli()


def project(config: _config.TrainConfig) -> dict:
    """The TrainConfig's fields in the paper-config lock's dotted openpi names (scalars only)."""
    out = {
        "seed": config.seed,
        "batch_size": config.batch_size,
        "num_train_steps": config.num_train_steps,
        "ema_decay": config.ema_decay,
        "weight_loader.params_path": config.weight_loader.params_path,
    }
    for group in ("model", "lr_schedule", "optimizer"):
        out.update(
            {
                f"{group}.{k}": v
                for k, v in dataclasses.asdict(getattr(config, group)).items()
                if isinstance(v, str | int | float | bool | type(None))
            }
        )
    out.update(
        {
            "data.repo_id": config.data.repo_id,
            "data.image_center_crop": config.data.image_center_crop,
            "data.default_prompt": config.data.default_prompt,
            **DROPPED_INERT,
        }
    )
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--project":
        raise SystemExit("usage: python -m openpi.training.axis_config --project <generated config.json>")
    print(json.dumps(project(load_train_config(sys.argv[2])), sort_keys=True))
