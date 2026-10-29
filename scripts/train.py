import dataclasses
import functools
import json
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.axis_config as _axis_config
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.training.weighted_loss as _weighted_loss


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)

    # A freeze filter that matches nothing trains a full-weight clone under a LoRA name: log the
    # split, and refuse a declared freeze that froze nothing.
    def _n_params(tree) -> int:
        return sum(int(np.prod(x.shape)) for x in jax.tree.leaves(tree))

    total_params = _n_params(train_state_shape.params.to_pure_dict())
    trainable_params = _n_params(train_state_shape.params.filter(config.trainable_filter).to_pure_dict())
    logging.info(
        f"param_counts: total={total_params} trainable={trainable_params} frozen={total_params - trainable_params}"
    )
    if not isinstance(config.freeze_filter, nnx.Nothing) and total_params == trainable_params:
        raise ValueError("freeze_filter is set but matched 0 params -- refusing to train a silent clone")

    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions]
    | tuple[_model.Observation, _model.Actions, at.Float[at.Array, "*b"]],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        loss_weights: at.Float[at.Array, "*b"] | None = None,
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        # No weights: exactly jnp.mean(chunked_loss). Weights: the global_ratio reducer.
        return _weighted_loss.combine(chunked_loss, loss_weights)

    train_rng = jax.random.fold_in(rng, state.step)
    # A bundle with loss weights yields a third batch element; every other loader yields two.
    observation, actions, *extra = batch
    loss_weights = extra[0] if extra else None

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions, loss_weights)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # The compilation cache can be moved off a small home directory with JAX_COMPILATION_CACHE_DIR.
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR") or "~/.cache/jax"
    jax.config.update("jax_compilation_cache_dir", str(epath.Path(cache_dir).expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    _check_fsdp_devices(config)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    created = not epath.Path(config.checkpoint_dir).exists()
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    try:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

        # A resumed run replays its bundle's plan from the step after its last checkpoint.
        plan_start_step = int(checkpoint_manager.latest_step()) + 1 if resuming else 0
        data_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=True,
            plan_start_step=plan_start_step,
        )
        data_iter = iter(data_loader)
        batch = next(data_iter)
        logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

        # Log images from first batch to sanity check.
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)

        train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
        jax.block_until_ready(train_state)
        logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

        if resuming:
            train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
            if int(train_state.step) != plan_start_step:
                raise ValueError(f"restored step {int(train_state.step)} but the plan resumes at {plan_start_step}")
    except Exception:
        _remove_unstarted(config, checkpoint_manager, created=created)
        raise

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        # The next step's batch; no batch is drawn after the last step. An AXIS bundle's loader
        # ends with its plan, so every batch it yields is trained and nothing past it is read.
        if step + 1 < config.num_train_steps:
            batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    _finish_run(config, checkpoint_manager, train_state)


def _check_fsdp_devices(config: _config.TrainConfig) -> None:
    """A generated config's `fsdp_devices` must divide the devices this launch sees (sharding only
    splits the global batch and the state across them; the batch itself is the config's)."""
    if isinstance(config.data, _axis_config.AxisFrankaDataConfig) and jax.device_count() % config.fsdp_devices:
        raise ValueError(
            f"fsdp_devices {config.fsdp_devices} does not divide the {jax.device_count()} visible devices; "
            f"render the run with `qvla train --fsdp-devices <a divisor of {jax.device_count()}>`"
        )


def _remove_unstarted(config: _config.TrainConfig, checkpoint_manager, *, created: bool) -> None:
    """A launch that stops before its first training step leaves no checkpoint directory behind,
    so the same command can be run again once the cause is fixed: process 0 removes the directory
    this launch created, when no step was ever saved in it. A directory that was there before the
    launch (a resumed run's) is never removed."""
    if not created or jax.process_index() != 0 or tuple(checkpoint_manager.all_steps()):
        return
    checkpoint_manager.close()
    checkpoint_dir = epath.Path(config.checkpoint_dir)
    if checkpoint_dir.exists():
        checkpoint_dir.rmtree()
    logging.info(f"Removed {checkpoint_dir}: the launch stopped before its first step")


def _finish_run(config: _config.TrainConfig, checkpoint_manager, train_state: training_utils.TrainState) -> None:
    """Once every checkpoint is on disk, process 0 records what this run trained (`_write_summary`)."""
    checkpoint_manager.wait_until_finished()
    if jax.process_index() != 0:
        return
    _write_summary(config, trained_steps=int(train_state.step), final_step=checkpoint_manager.latest_step())


def _write_summary(config: _config.TrainConfig, *, trained_steps: int, final_step: int | None) -> None:
    """A `final` link to the last checkpoint written, then `train_summary.json`, beside the step dirs.

    A later run may start from this one (an ArmSpec init `run:<arm id>`) only if the summary shows
    the whole declared budget was trained, outside a smoke run. `trained_steps` is the train state's
    own step counter and `final_step` the checkpoint manager's latest step, so neither is assumed
    from the config. Both files are replaced atomically and the summary is written last: a reader
    that finds a summary finds the link it describes.
    """
    ckpt_dir = config.checkpoint_dir
    if final_step is None:
        raise RuntimeError(f"{ckpt_dir}: the run wrote no checkpoint, so there is no final step to record")
    tmp_link = ckpt_dir / f".final.tmp-{os.getpid()}"
    tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(str(int(final_step)))
    os.replace(tmp_link, ckpt_dir / "final")
    smoke = isinstance(config.data, _axis_config.AxisFrankaDataConfig) and config.data.smoke
    summary = {
        "schema": "qvla.train_summary/1",
        "name": config.name,
        "declared_steps": config.num_train_steps,
        "trained_steps": int(trained_steps),
        "final_step": int(final_step),
        "smoke": smoke,
    }
    tmp = ckpt_dir / f".train_summary.json.tmp-{os.getpid()}"
    tmp.write_text(json.dumps(summary, sort_keys=True))
    os.replace(tmp, ckpt_dir / "train_summary.json")


if __name__ == "__main__":
    main(_axis_config.cli())
