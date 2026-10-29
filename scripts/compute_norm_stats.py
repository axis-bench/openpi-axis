"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory (for a generated AXIS config, `data.norm_stats_dir`: see
`stats_output_path`). For a generated config the statistics cover every frame of the roots
index's datasets, and an existing `norm_stats.json` is never replaced unless `--overwrite` is
given: the config names the statistics it trains with by sha256, and the published sets sit in
lock-verified repositories. An openpi config's file is replaced, as upstream does.
"""

import pathlib

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.axis_config as _axis_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str | None = None,
    config_json: str | None = None,
    max_frames: int | None = None,
    num_workers: int | None = None,
    *,
    overwrite: bool = False,
):
    """Exactly one of `config_name` (an openpi config) and `config_json` (a generated AXIS config).
    `num_workers` overrides the config's, since stats are often computed on a smaller machine.
    `overwrite` lets a generated config's existing `norm_stats.json` be replaced; without it the
    run is refused before any frame is read."""
    if (config_name is None) == (config_json is None):
        raise ValueError("pass exactly one of --config-name and --config-json")
    config = _axis_config.load_train_config(config_json) if config_json else _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    output_path = stats_output_path(config, data_config)
    existing = output_path / "norm_stats.json"
    if isinstance(config.data, _axis_config.AxisFrankaDataConfig) and existing.exists() and not overwrite:
        raise FileExistsError(
            f"{existing} exists and is not replaced: a run names the statistics it trains with by sha256, "
            f"and a published set is lock-verified. Compute your own into a norm_stats_dir of their own, or "
            f"pass --overwrite to replace this file"
        )

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers if num_workers is None else num_workers,
            max_frames,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


def stats_output_path(config: _config.TrainConfig, data_config: _config.DataConfig) -> pathlib.Path:
    """Where the statistics are written: for a generated AXIS config, `<data.norm_stats_dir>/<asset id>`,
    the directory training loads them from (DataConfigFactory.create_base_config reads
    `assets.assets_dir / asset_id`) and saves into every checkpoint the server reads; for an openpi
    config, upstream's `<assets_dirs>/<repo id>`."""
    if isinstance(config.data, _axis_config.AxisFrankaDataConfig):
        return pathlib.Path(config.data.assets.assets_dir) / data_config.asset_id
    return config.assets_dirs / data_config.repo_id


if __name__ == "__main__":
    tyro.cli(main)
