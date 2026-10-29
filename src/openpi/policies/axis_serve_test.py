"""Serving a tagged arm: the conditional prompt carries "\\nQuality: q", the unconditional is bare."""

import copy
import dataclasses
import json
import pathlib
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.policies import axis_franka_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config
from openpi.shared import normalize as _normalize
from openpi.training import axis_config
from openpi.training import config as _config


def test_the_tag_and_its_unconditional_twin():
    tag = axis_franka_policy.ServeQualityTag(q=5)
    assert tag({"prompt": "put the cup on the plate"})["prompt"] == "put the cup on the plate\nQuality: 5"
    assert axis_franka_policy.ServeQualityTag(q=5, drop_all=True)({"prompt": "p"})["prompt"] == "p"
    assert tag({"state": 1}) == {"state": 1}
    with pytest.raises(ValueError, match="already carries"):
        tag({"prompt": "p\nQuality: 3"})
    with pytest.raises(ValueError, match="not a bin"):
        axis_franka_policy.ServeQualityTag(q=0)


def test_the_two_chains_differ_only_in_the_tag():
    kw = {"repack_inputs": [], "default_prompt": None, "data_config": _config.DataConfig(), "norm_stats": {}}
    cond = policy_config._input_chain(**kw, quality_tag=5)  # noqa: SLF001
    uncond = policy_config._input_chain(**kw, quality_tag=5, quality_drop_all=True)  # noqa: SLF001
    assert [type(t).__name__ for t in cond] == [type(t).__name__ for t in uncond]
    assert next(t for t in cond if isinstance(t, axis_franka_policy.ServeQualityTag)).drop_all is False
    assert next(t for t in uncond if isinstance(t, axis_franka_policy.ServeQualityTag)).drop_all is True
    assert not any(isinstance(t, axis_franka_policy.ServeQualityTag) for t in policy_config._input_chain(**kw))  # noqa: SLF001


def test_the_server_publishes_the_protocols_optional_keys():
    plain = axis_franka_policy.serve_contract(image_center_crop=True, quality_tag=None)
    assert plain == {
        "image_geometry": {"base_0_rgb": "center_crop224", "left_wrist_0_rgb": "center_crop224"},
        "state_order": ["q0", "q1", "q2", "q3", "q4", "q5", "q6", "gripper"],
        "action_units": "rad_s",
        "action_fps": 15,
        "action_dim_out": 8,
        "prompt_suffix": "",
    }
    tagged = axis_franka_policy.serve_contract(image_center_crop=False, quality_tag=5)
    assert tagged["prompt_suffix"] == "\nQuality: 5"
    assert tagged["image_geometry"]["base_0_rgb"] == "letterbox224"


def test_the_contract_is_published_for_an_axis_config_only():
    """Recognised by the data factory's type: another factory carrying an `image_center_crop`
    attribute is not an AXIS Franka config and publishes nothing."""
    from openpi.training.axis_config_test import FT

    axis = axis_config.train_config_from_dict(FT)
    assert policy_config.axis_serve_contract(axis, quality_tag=5) == axis_franka_policy.serve_contract(
        image_center_crop=True, quality_tag=5
    )

    @dataclasses.dataclass(frozen=True)
    class _Other(_config.FakeDataConfig):
        image_center_crop: bool = True

    upstream = _config.get_config("pi05_libero")
    assert policy_config.axis_serve_contract(upstream, quality_tag=None) == {}
    assert policy_config.axis_serve_contract(dataclasses.replace(upstream, data=_Other()), quality_tag=5) == {}
    with pytest.raises(TypeError, match="must be a bool"):
        axis_franka_policy.serve_contract(image_center_crop=None, quality_tag=None)


_OPENED: list[str] | None = None


def _audit(event: str, args: tuple) -> None:
    if _OPENED is not None and event in ("open", "os.listdir", "os.scandir") and args and args[0] is not None:
        _OPENED.append(str(args[0]))


sys.addaudithook(_audit)  # inert unless a test sets _OPENED


class _StubModel:
    """The weights an eval-only user downloads, without loading them: zero actions."""

    def __init__(self, config):
        self.action_horizon, self.action_dim = config.action_horizon, config.action_dim

    def sample_actions(self, rng, observation, **kwargs):
        return jnp.zeros((observation.state.shape[0], self.action_horizon, self.action_dim))


def serve_without_training_data(d: dict, tmp_path: pathlib.Path, monkeypatch, *, quality_tag: int | None = 5):
    """Build the policy a server builds from the generated serve config `d`, with every data path the
    config names (roots index, ranges file, corpus, norm-stats directory, run directories) pointing
    into a directory that does not exist, and only the checkpoint's norm stats on disk. Returns the
    policy, one inference on a raw observation, and every path opened while doing both."""
    global _OPENED  # noqa: PLW0603
    absent = tmp_path / "absent"
    d = copy.deepcopy(d)
    d["data"].update(
        roots_index=str(absent / "roots.json"),
        ranges_path=str(absent / "ranges.json"),
        corpus_base=str(absent / "corpus"),
        norm_stats_dir=str(absent / "norm"),
    )
    d.update(assets_base_dir=str(absent / "assets"), checkpoint_base_dir=str(absent / "runs"))
    path = tmp_path / "openpi_config.json"
    path.write_text(json.dumps(d))
    ckpt = tmp_path / "checkpoint" / "13579"
    (ckpt / "params").mkdir(parents=True)
    stats = _normalize.NormStats(mean=np.zeros(8), std=np.ones(8), q01=np.zeros(8), q99=np.ones(8))
    _normalize.save(ckpt / "assets" / d["data"]["asset_id"], {"state": stats, "actions": stats})
    monkeypatch.setattr(policy_config._model, "restore_params", lambda path, **kw: {})  # noqa: SLF001
    monkeypatch.setattr(pi0_config.Pi0Config, "load", lambda self, params, **kw: _StubModel(self))
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda f, **kw: f)
    obs = {
        "base_0_rgb": np.zeros((360, 640, 3), np.uint8),
        "left_wrist_0_rgb": np.zeros((360, 640, 3), np.uint8),
        "state": np.zeros(8, np.float32),
        "prompt": "put the tomato beside the fork",
    }
    _OPENED = []
    try:
        policy = policy_config.create_trained_policy(axis_config.load_train_config(path), ckpt, quality_tag=quality_tag)
        out = policy.infer(obs)
        opened = list(_OPENED)
    finally:
        _OPENED = None
    assert not absent.exists()
    return policy, out, [p for p in opened if p.startswith(str(absent))]


def test_a_serve_config_loads_and_serves_with_no_training_data_on_disk(tmp_path, monkeypatch):
    """An eval-only user has a checkpoint and nothing else: serving opens, hashes and needs none of
    the training files the config names."""
    configs = json.loads((pathlib.Path(axis_config.__file__).parent / "axis_config_test_configs.json").read_text())
    policy, out, touched = serve_without_training_data(configs["configs"]["finetune_serve"], tmp_path, monkeypatch)
    assert touched == []
    assert out["actions"].shape == (10, 8)
    assert policy.metadata["prompt_suffix"] == "\nQuality: 5"
    assert policy.metadata["image_geometry"]["base_0_rgb"] == "center_crop224"
