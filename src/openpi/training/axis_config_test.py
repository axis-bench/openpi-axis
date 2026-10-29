"""A generated AXIS config builds the TrainConfig it describes, and nothing it does not."""

import copy

import flax.nnx as nnx
import pytest

from openpi.training import axis_config

FT = {
    "schema": "qvla.openpi_train_config/1",
    "name": "pi05.finetune.example",
    "exp_name": "train",
    "seed": 42,
    "batch_size": 32,
    "num_train_steps": 13580,
    "ema_decay": None,
    "model": {
        "pi05": True,
        "action_dim": 32,
        "action_horizon": 10,
        "max_token_len": 200,
        "dtype": "bfloat16",
        "paligemma_variant": "gemma_2b_lora",
        "action_expert_variant": "gemma_300m_lora",
        "discrete_state_input": True,
        "pytorch_compile_mode": "max-autotune",
    },
    "lr_schedule": {"warmup_steps": 500, "peak_lr": 2.5e-05, "decay_steps": 13580, "decay_lr": 2.5e-05},
    "optimizer": {"b1": 0.9, "b2": 0.95, "eps": 1e-08, "weight_decay": 1e-10, "clip_gradient_norm": 1.0},
    "freeze": "lora",
    "fsdp_devices": 8,
    "num_workers": 8,
    "save_interval": 2716,
    "keep_period": 2716,
    "log_interval": 100,
    "wandb_enabled": False,
    "overwrite": False,
    "resume": False,
    "checkpoint_base_dir": "/r/runs",
    "assets_base_dir": "/r/runs/pi05.finetune.example/assets",
    "weight_loader": {"params_path": "/r/cache/checkpoints/qvla-v1-pi05/pretrain/x/20604/params"},
    "data": {
        "repo_id": "Devon018/Franka-Datasets-v2",
        "asset_id": "Devon018/Franka-Datasets-v2",
        "image_center_crop": True,
        "default_prompt": None,
        "prompt_from_task": True,
        "roots_index": "/r/corpus/qvla-v1-bench/ft_demos/splits/qual_v2.roots.json",
        "ranges_path": "/r/corpus/qvla-v1-bench/ft_demos/splits/qual_v2.ranges.json",
        "corpus_base": "/r/corpus/qvla-v1-bench",
        "bundle": "/r/corpus/qvla-v1-corpus/bundles/ab",
        "bundle_id": "sha256:" + "ab" * 32,
        "norm_stats_dir": "/r/cache/norm/heldout24_qual_v2",
        "smoke": False,
    },
    "identity": {
        "arm": "pi05.finetune.example",
        "armspec_sha256": "cd" * 32,
        "preset": "pi05.finetune.lora.paper",
        "init": "axisrobotics/qvla-v1-pi05:pretrain/x/20604#sha256:" + "ef" * 32,
        "rowspace": "heldout24_ft#sha256:" + "12" * 32,
        "bundle": "sha256:" + "ab" * 32,
        "norm": "heldout24_qual_v2#sha256:" + "34" * 32,
    },
}


def test_the_finetune_config_is_built_field_for_field():
    c = axis_config.train_config_from_dict(FT)
    assert (c.name, c.exp_name, c.seed, c.batch_size, c.num_train_steps, c.ema_decay) == (
        "pi05.finetune.example",
        "train",
        42,
        32,
        13580,
        None,
    )
    assert (c.lr_schedule.warmup_steps, c.lr_schedule.decay_steps, c.optimizer.clip_gradient_norm) == (500, 13580, 1.0)
    assert c.model.paligemma_variant == "gemma_2b_lora"
    assert c.fsdp_devices == 8
    assert c.keep_period == 2716
    assert c.weight_loader.params_path.endswith("/20604/params")
    assert repr(c.freeze_filter) == repr(c.model.get_freeze_filter())
    assert c.data.assets.assets_dir == "/r/cache/norm/heldout24_qual_v2"
    assert c.data.bundle_id == "sha256:" + "ab" * 32
    assert str(c.checkpoint_dir) == "/r/runs/pi05.finetune.example/train"


def test_the_projection_matches_the_lock_names():
    p = axis_config.project(axis_config.train_config_from_dict(FT))
    assert p["lr_schedule.warmup_steps"] == 500
    assert p["model.max_token_len"] == 200
    assert p["model.knowledge_insulation"] is False
    assert p["data.eef_action"] is False
    assert p["weight_loader.params_path"].endswith("/params")


def test_a_full_weight_config_freezes_nothing():
    d = copy.deepcopy(FT)
    d["model"].update(paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    d["freeze"] = "none"
    assert isinstance(axis_config.train_config_from_dict(d).freeze_filter, nnx.Nothing)
    d["freeze"] = "lora"
    with pytest.raises(ValueError, match="would freeze nothing"):
        axis_config.train_config_from_dict(d)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda d: d.update(extra=1), "unknown keys"),
        (lambda d: d.pop("seed"), "missing keys"),
        (lambda d: d["data"].update(roots_index_path="x"), "config.data"),
        (lambda d: d.update(schema="other"), "schema"),
        (lambda d: d.update(freeze="vision"), "freeze"),
    ],
)
def test_unknown_or_missing_keys_are_refused(edit, match):
    d = copy.deepcopy(FT)
    edit(d)
    with pytest.raises(ValueError, match=match):
        axis_config.train_config_from_dict(d)


def test_the_data_config_has_the_franka_chain_and_a_strict_tokenizer():
    c = axis_config.train_config_from_dict(FT)
    dc = c.data.create(c.assets_dirs, c.model)
    assert isinstance(dc, axis_config.AxisDataConfig)
    assert dc.axis_bundle == FT["data"]["bundle"]
    assert dc.action_sequence_keys == ("action",)
    # The norm statistics the loader checks: the file training loads, and the digest the arm names.
    assert dc.axis_norm_stats_path == f"{FT['data']['norm_stats_dir']}/{FT['data']['asset_id']}/norm_stats.json"
    assert dc.axis_norm_stats_sha256 == "34" * 32
    tok = next(t for t in dc.model_transforms.inputs if type(t).__name__ == "TokenizePrompt")
    assert tok.tokenizer._strict is True  # noqa: SLF001


def _pred(freeze):
    return nnx.filterlib.to_predicate(freeze)


def test_frozen_vision_lora_freezes_the_vision_tower_but_not_its_projector():
    d = copy.deepcopy(FT)
    d["freeze"] = "lora_frozen_vision"
    f = _pred(axis_config.train_config_from_dict(d).freeze_filter)
    assert f(("PaliGemma", "img", "Transformer", "encoderblock", "MlpBlock_0", "Dense_0", "kernel"), None)
    assert f(("PaliGemma", "img", "embedding", "kernel"), None)
    assert not f(("PaliGemma", "img", "head", "kernel"), None)
    assert f(("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"), None)
    assert not f(("PaliGemma", "llm", "layers", "attn", "q_einsum", "lora_a"), None)


def test_the_augmentation_and_action_units_are_optional_and_reach_the_data_config():
    assert axis_config.train_config_from_dict(copy.deepcopy(FT)).data.action_units == "rad_s"
    d = copy.deepcopy(FT)
    d["data"] |= {"sim_image_aug": True, "action_units": "droid_normalized_joint_velocity_command"}
    c = axis_config.train_config_from_dict(d)
    assert c.data.sim_image_aug is True
    assert c.data.action_units == "droid_normalized_joint_velocity_command"
    from openpi.policies import policy_config

    assert (
        policy_config.axis_serve_contract(c, quality_tag=None)["action_units"]
        == "droid_normalized_joint_velocity_command"
    )


def test_a_mistyped_optional_key_is_refused():
    d = copy.deepcopy(FT)
    d["data"]["sim_image_aug"] = "yes"
    with pytest.raises(ValueError, match="sim_image_aug"):
        axis_config.train_config_from_dict(d)
