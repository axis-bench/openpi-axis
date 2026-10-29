from __future__ import annotations

from collections.abc import Sequence
import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
from openpi.policies import axis_franka_policy as _axis_franka_policy
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import axis_config as _axis_config
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def _input_chain(
    *,
    repack_inputs: Sequence[transforms.DataTransformFn],
    default_prompt: str | None,
    data_config: _config.DataConfig,
    norm_stats: dict[str, transforms.NormStats],
    quality_tag: int | None = None,
    quality_drop_all: bool = False,
) -> list[transforms.DataTransformFn]:
    """The inference input chain, optionally carrying a fixed quality tag.

    One function builds both guidance branches: the conditional and unconditional chains are two
    calls that differ only in `quality_drop_all`, so nothing but the prompt tag can differ between
    them. The tag sits right after `InjectDefaultPrompt`, upstream of the tokenizer.
    """
    if quality_drop_all and quality_tag is None:
        raise ValueError("quality_drop_all without a quality_tag: both branches would be identical")
    tag = (
        []
        if quality_tag is None
        else [_axis_franka_policy.ServeQualityTag(q=int(quality_tag), drop_all=bool(quality_drop_all))]
    )
    return [
        *repack_inputs,
        transforms.InjectDefaultPrompt(default_prompt),
        *tag,
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]


def axis_serve_contract(train_config: _config.TrainConfig, *, quality_tag: int | None) -> dict:
    """The protocol's optional handshake keys (image geometry, state order, action units, fps and
    width, prompt suffix) for an AXIS Franka config; {} for any other. The config is recognised by
    its data factory's type, never by an attribute another factory could also carry."""
    if not isinstance(train_config.data, _axis_config.AxisFrankaDataConfig):
        return {}
    return _axis_franka_policy.serve_contract(
        image_center_crop=train_config.data.image_center_crop,
        quality_tag=quality_tag,
        action_units=train_config.data.action_units,
    )


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    guidance_scale: float = 0.0,
    quality_tag: int | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method.
        default_prompt: The default prompt to inject when the input carries none.
        norm_stats: The norm stats to use; by default those saved in the checkpoint.
        pytorch_device: Device for PyTorch models; by default "cuda" if available.
        guidance_scale: `w` in v = v_uncond + (1 + w) (v_cond - v_uncond); beta = 1 + w. Needs a
            quality_tag (the conditional branch); 0 is plain conditional sampling.
        quality_tag: append "\\nQuality: {q}" to every prompt; with guidance, the unconditional
            branch is the bare prompt. Published in the policy metadata with beta.
    """
    repack_transforms = repack_transforms or transforms.Group()
    if guidance_scale != 0.0 and quality_tag is None:
        raise ValueError("guidance_scale != 0 needs a quality_tag for the conditional branch")
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    chain = {
        "repack_inputs": repack_transforms.inputs,
        "default_prompt": default_prompt,
        "data_config": data_config,
        "norm_stats": norm_stats,
    }
    input_transforms = _input_chain(**chain, quality_tag=quality_tag)
    uncond_transforms = (
        _input_chain(**chain, quality_tag=quality_tag, quality_drop_all=True) if quality_tag is not None else None
    )
    metadata = train_config.policy_metadata
    if quality_tag is not None:
        sample_kwargs = dict(sample_kwargs or {})
        if guidance_scale != 0.0:
            sample_kwargs["guidance_scale"] = float(guidance_scale)
        metadata = {
            **(metadata or {}),
            "cfg_conditioning": True,
            "guidance_scale": float(guidance_scale),
            "checkpoint_dir": str(checkpoint_dir),
            "config_name": str(train_config.name),
            "quality_tag": int(quality_tag),
            "beta": 1.0 + float(guidance_scale),
        }
    contract = axis_serve_contract(train_config, quality_tag=quality_tag)
    if contract:
        metadata = {**(metadata or {}), **contract}

    return _policy.Policy(
        model,
        transforms=input_transforms,
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        uncond_transforms=uncond_transforms,
        sample_kwargs=sample_kwargs,
        metadata=metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
