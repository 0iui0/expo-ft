"""Base agent (algorithm) interface for the expo training framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import logging

import etils.epath as epath
import jax.numpy as jnp
import orbax.checkpoint as ocp

from expo_ft.data.dataset import DatasetDict
from expo_ft.types import PRNGKey


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    item_handlers = {
        "agent": ocp.PyTreeCheckpointHandler(),
        "params": ocp.PyTreeCheckpointHandler(),
    }
    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers=item_handlers,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=100,
            keep_period=keep_period,
            create=False,
            # Async checkpointing spawns orbax asyncio worker threads at manager
            # creation. With async enabled, these race with the SFT checkpoint
            # restore in build_pi05 (restore_params -> orbax batched_device_put),
            # causing a flaky std::bad_alloc / std::system_error during startup.
            # Disabling async here makes the manager's handler infrastructure
            # synchronous so it doesn't conflict with the initial restore. Saves
            # become blocking (acceptable: infrequent, and checkpoint_model=False
            # for dry-runs). Validated: 3/3 clean restores vs 5/5 crashes.
            enable_async_checkpointing=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


class AgentLearner(ABC):
    """Base class for training algorithms (learners).

    Mirrors ``expo_ft.agents.vla.Model``: subclasses must implement the
    methods used by train / eval loops. Concrete learners are typically
    Flax ``struct.PyTreeNode`` types that also carry ``rng`` and network state.
    """

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs) -> "AgentLearner":
        raise NotImplementedError

    @abstractmethod
    def sample_actions(
        self,
        observations: Dict[str, Any],
        *,
        only_base_actions: bool = False,
    ) -> Tuple[jnp.ndarray, "AgentLearner", Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        agent: "AgentLearner",
        batch: DatasetDict,
        utd_ratio: int,
        actor_batch: Optional[DatasetDict] = None,
    ) -> Tuple["AgentLearner", Dict[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def update_actor(
        self, batch: DatasetDict
    ) -> Tuple["AgentLearner", Dict[str, float]]:
        raise NotImplementedError
