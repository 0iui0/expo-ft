"""Shared utilities for Pi robot training scripts."""

import dataclasses
import logging
from typing import Any, Dict

import etils.epath as epath
import jax
import jax.numpy as jnp
import numpy as np
import wandb

from openpi.training import config as openpi_config


def init_logging() -> None:
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


def init_wandb(checkpoint_dir: epath.Path, resuming: bool, project: str, name: str) -> None:
    wandb_id_file = checkpoint_dir / "wandb_id.txt"
    if resuming and wandb_id_file.exists():
        run_id = wandb_id_file.read_text().strip()
        wandb.init(id=run_id, resume="must", project=project, name=name)
    else:
        wandb.init(project=project, name=name)
        wandb_id_file.write_text(wandb.run.id)


def clear_batch(batch: Dict[str, Any]) -> None:
    """Recursively clear a batch dictionary to free memory."""
    if isinstance(batch, dict):
        for v in batch.values():
            if isinstance(v, dict):
                clear_batch(v)
        batch.clear()


def get_batch_info(batch: Dict[str, Any]) -> Dict[str, float]:
    """Extract basic statistics from a batch dictionary for logging."""
    return {
        "rewards_mean": float(np.mean(batch["rewards"])),
        "rewards_std": float(np.std(batch["rewards"])),
        "rewards_max": float(np.max(batch["rewards"])),
        "rewards_min": float(np.min(batch["rewards"])),
        "masks_mean": float(np.mean(batch["masks"])),
        "masks_std": float(np.std(batch["masks"])),
        "valids_mean": float(np.mean(batch["valids"])),
        "valids_std": float(np.std(batch["valids"])),
        "actions_mean": float(np.mean(batch["actions"])),
        "actions_std": float(np.std(batch["actions"])),
        "actions_max": float(np.max(batch["actions"])),
        "actions_min": float(np.min(batch["actions"])),
        "states_mean": float(np.mean(batch["state"])),
        "states_std": float(np.std(batch["state"])),
        "states_max": float(np.max(batch["state"])),
        "states_min": float(np.min(batch["state"])),
        "base_image_max": float(np.max(batch["image"]["base_0_rgb"])),
        "base_image_min": float(np.min(batch["image"]["base_0_rgb"])),
        "base_image_std": float(np.std(batch["image"]["base_0_rgb"])),
    }


def build_pi05_config(config):
    """Extract pi05 settings from agent config and build the openpi train config.

    Returns (agent_kwargs, pi05_train_config, pi05_resize_size, model_cls).
    ``agent_kwargs`` is a plain dict with pi05-specific keys removed.
    """
    agent_kwargs = dict(config)
    pi05_config_name = agent_kwargs.pop("pi05_config_name")
    pi05_resize_size = agent_kwargs.pop("pi05_resize_size")
    pi05_weight_loader_path = agent_kwargs.pop("pi05_weight_loader_path", "") or None
    pi05_assets_dir = agent_kwargs.pop("pi05_assets_dir", "") or None
    pi05_asset_id = agent_kwargs.pop("pi05_asset_id", "") or None
    model_cls = agent_kwargs.pop("model_cls")

    pi05_train_config = openpi_config.get_config(
        pi05_config_name, weight_loader_path=pi05_weight_loader_path
    )
    if pi05_assets_dir or pi05_asset_id:
        from openpi.training.config import AssetsConfig
        new_assets = AssetsConfig(
            assets_dir=pi05_assets_dir or pi05_train_config.data.assets.assets_dir,
            asset_id=pi05_asset_id or pi05_train_config.data.assets.asset_id,
        )
        pi05_train_config = dataclasses.replace(
            pi05_train_config,
            data=dataclasses.replace(pi05_train_config.data, assets=new_assets),
        )
    return agent_kwargs, pi05_train_config, pi05_resize_size, model_cls


def _concat_leaves(x, y):
    """tree_map'd concatenation along axis 0 that tolerates None leaves."""
    if x is None:
        return y
    if y is None:
        return x
    return jnp.concatenate([jnp.asarray(x), jnp.asarray(y)], axis=0)


def _shuffle_batch(key, batch):
    """Shuffle a (possibly nested) batch along axis=0 with one shared permutation."""
    leaves, treedef = jax.tree_util.tree_flatten(batch)
    n = leaves[0].shape[0]
    perm = jax.random.permutation(key, n)
    shuffled = [jnp.asarray(x)[perm] for x in leaves]
    return jax.tree_util.tree_unflatten(treedef, shuffled)


def combine_batches(online_batch, offline_batch, rng):
    """Combine online and offline batches: concatenate, then shuffle to mix.

    Batches must already have the desired sizes (proportional to offline_ratio).
    """
    combined = jax.tree_util.tree_map(_concat_leaves, online_batch, offline_batch)
    return _shuffle_batch(rng, combined)


def build_gr00t_config(config):
    """Extract GR00T settings from agent config, return flattened agent_kwargs.

    Unlike ``build_pi05_config``, there is no OpenPI config to load — GR00T
    loads its own model, processor, and statistics directly from the checkpoint
    directory.

    Returns (agent_kwargs, gr00t_model_path, gr00t_embodiment_tag, model_cls).
    ``agent_kwargs`` is a plain dict with GR00T-specific keys removed.
    """
    agent_kwargs = dict(config)
    gr00t_model_path = agent_kwargs.pop("gr00t_model_path")
    gr00t_embodiment_tag = agent_kwargs.pop("gr00t_embodiment_tag")
    freeze_gr00t_backbone = agent_kwargs.pop("freeze_gr00t_backbone", False)
    model_cls = agent_kwargs.pop("model_cls")
    return agent_kwargs, gr00t_model_path, gr00t_embodiment_tag, model_cls
