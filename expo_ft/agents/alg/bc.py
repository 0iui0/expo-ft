"""BCLearner: imitation-only agent (no critic). Sample one action from Pi05; update only main actor."""

import os
from typing import Any, Callable, Dict, Optional, Tuple, Type
import dataclasses

import logging
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import struct
from flax.training.train_state import TrainState

import numpy as np

import openpi.shared.array_typing as at
import openpi.training.sharding as _sharding
import openpi.training.utils as training_utils

from expo_ft.agents.alg.agent import AgentLearner, initialize_checkpoint_dir
from expo_ft.agents.alg.batch_utils import prepare_critic_batch
from expo_ft.data.dataset import DatasetDict
from expo_ft.utils.augmentation import make_data_augmentation_fn


def _split_params_bc(agent: Any) -> tuple[Any, dict[str, at.Params]]:
    """Split params for BCLearner (actor only)."""
    with at.disable_typechecking():
        if agent.actor_train_state.ema_params is not None:
            actor_params = agent.actor_train_state.ema_params
            actor_train_state = dataclasses.replace(agent.actor_train_state, ema_params=None)
        else:
            actor_params = agent.actor_train_state.params
            actor_train_state = dataclasses.replace(agent.actor_train_state, params={})
    agent = dataclasses.replace(agent, actor_train_state=actor_train_state)
    return agent, {"actor_params": actor_params}


def _merge_params_bc(agent: Any, params: dict[str, at.Params]) -> Any:
    """Merge params for BCLearner."""
    with at.disable_typechecking():
        if agent.actor_train_state.params:
            actor_train_state = dataclasses.replace(agent.actor_train_state, ema_params=params["actor_params"])
        else:
            actor_train_state = dataclasses.replace(agent.actor_train_state, params=params["actor_params"])
    return dataclasses.replace(agent, actor_train_state=actor_train_state)


def restore_checkpoint(checkpoint_manager, agent, step: int | None = None):
    agent, params = _split_params_bc(agent)
    restored = checkpoint_manager.restore(
        step,
        items={
            "agent": agent,
            "params": params,
        },
    )
    return _merge_params_bc(restored["agent"], restored["params"])


def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    agent: Any,
    step: int,
):
    agent, params = _split_params_bc(agent)
    checkpoint_manager.save(step, {"agent": agent, "params": params})

def load_agent(seed, example_observation, example_action, example_state,
               actor, actor_train_state, target_actor_params, agent_kwargs, metadata,
               mesh, data_sharding, replicated_sharding, resume, replan_steps,
               default_prompt, **kwargs):
    """Create a BCLearner from a pre-built VLA actor and remaining config kwargs."""
    agent_kwargs.update(
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        resume=resume,
        replan_steps=replan_steps,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        default_prompt=default_prompt,
        **metadata,
    )
    return BCLearner.create(seed, example_observation, example_action, example_state, **agent_kwargs)


class BCLearner(AgentLearner, struct.PyTreeNode):
    """Imitation-only agent: no critic. Sample one action from main (Pi05) actor; update only the main actor."""

    rng: jax.random.PRNGKey
    data_augmentation_fn: Callable = struct.field(pytree_node=False)
    actor: Any = struct.field(pytree_node=False)
    actor_train_state: training_utils.TrainState
    target_actor_params: at.Params
    actor_tau: float
    action_dim: int = struct.field(pytree_node=False)
    state_dim: int = struct.field(pytree_node=False)
    full_action_dim: int = struct.field(pytree_node=False)
    replan_steps: int = struct.field(pytree_node=False)
    action_horizon: int = struct.field(pytree_node=False)
    resize_size: Optional[int] = struct.field(pytree_node=False)
    default_prompt: Optional[str] = struct.field(pytree_node=False)
    data_sharding: Optional[jax.sharding.NamedSharding] = struct.field(pytree_node=False)
    replicated_sharding: Optional[jax.sharding.NamedSharding] = struct.field(pytree_node=False)
    freeze_encoder: Optional[bool] = struct.field(pytree_node=False)
    actor_success_only: bool = struct.field(pytree_node=False)
    _infer_cache: Optional[dict] = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space,
        action_space,
        states,
        # Pre-built VLA actor (constructed by build_pi05 or similar factory)
        actor: Any = None,
        actor_train_state: Any = None,
        target_actor_params: Any = None,
        # VLA metadata (extracted by factory from backbone config)
        action_horizon: int = 1,
        freeze_encoder: bool = False,
        # Learner params
        replan_steps: int = 1,
        data_sharding: Optional[jax.sharding.NamedSharding] = None,
        replicated_sharding: Optional[jax.sharding.NamedSharding] = None,
        default_prompt: Optional[str] = None,
        resize_size: Optional[int] = None,
        actor_success_only: bool = False,
        use_full_augmentation: bool = True,
        actor_tau: float = 0.001,
        **kwargs,
    ):
        action_dim = action_space.shape[-1]
        state_dim = states.shape[-1]
        full_action_dim = replan_steps * action_dim

        rng = jax.random.PRNGKey(seed)

        return cls(
            rng=rng,
            data_augmentation_fn=make_data_augmentation_fn(use_full_augmentation),
            actor=actor,
            actor_train_state=actor_train_state,
            target_actor_params=target_actor_params,
            actor_tau=actor_tau,
            action_dim=action_dim,
            state_dim=state_dim,
            full_action_dim=full_action_dim,
            replan_steps=replan_steps,
            action_horizon=action_horizon,
            resize_size=resize_size,
            default_prompt=default_prompt,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            freeze_encoder=freeze_encoder,
            actor_success_only=actor_success_only,
        ).cache_infer_params()

    def _place_aug_key(self, key):
        """Place augmentation RNG key on the same mesh as sharded image batches."""
        if self.replicated_sharding is not None:
            return jax.device_put(key, self.replicated_sharding)
        if self.data_sharding is not None:
            replicated = jax.sharding.NamedSharding(
                self.data_sharding.mesh, jax.sharding.PartitionSpec()
            )
            return jax.device_put(key, replicated)
        return key

    def cache_infer_params(self):
        """Copy actor params onto infer_sharding for rollout sampling."""
        s = self.actor.infer_sharding
        return self.replace(_infer_cache={
            "actor_train_state": jax.device_put(self.actor_train_state, s),
        })

    def sample_actions(self, observations, only_base_actions=False):
        """Sample a single action from the main actor; no critic, no selection."""
        rng = self.rng
        key, rng = jax.random.split(rng)
        c = self._infer_cache or {}
        _actor_train_state = c.get("actor_train_state") or self.actor_train_state
        transformed_inputs = self.actor.process_raw_inputs(observations, self.action_dim, self.resize_size)
        transformed_actions, sample_time = self.actor.sample_actions(
            transformed_inputs,
            train_state=_actor_train_state,
            rng=key,
            train=False,
            num_samples=1,
        )
        raw_actions = self.actor.process_transformed_outputs(
            transformed_actions, state=transformed_inputs["state"]
        )
        action = raw_actions[0]
        sample_info = {"sample_time": sample_time, "selected_action_type": "main"}
        return jnp.array(action), self.replace(rng=rng), sample_info

    def update_actor(self, batch: DatasetDict) -> Tuple[AgentLearner, Dict[str, float]]:
        actor_batch = self.actor.prepare_batch_for_actor(batch)
        def ensure_sharding(x):
            if isinstance(x, jnp.ndarray):
                return jax.device_put(x, self.data_sharding)
            return x
        actor_batch = jax.tree_util.tree_map(ensure_sharding, actor_batch)
        rng = self.rng
        key, rng = jax.random.split(rng, 2)
        with _sharding.set_mesh(self.actor.mesh):
            new_train_state, info = self.actor.train_step(key, self.actor_train_state, actor_batch)
        new_train_state_params = self.actor.get_params(new_train_state)
        target_actor_params = optax.incremental_update(new_train_state_params, self.target_actor_params, self.actor_tau)
        new_agent = self.replace(actor_train_state=new_train_state, target_actor_params=target_actor_params, rng=rng)
        return new_agent, info

    def update(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None):
        """Update only the main actor; no critic. Rebuilds inference cache after."""
        if actor_batch is None:
            raise ValueError(
                "BCLearner.update expected actor_batch, but got None. "
                "BCLearner training requires a human-intervention actor_batch on every update."
            )
        actor_batch = actor_batch.copy()
        rng, key = jax.random.split(agent.rng)
        key = self._place_aug_key(key)
        actor_batch["image"] = self.data_augmentation_fn(key, actor_batch["image"])
        agent = agent.replace(rng=rng)
        actor_batch = prepare_critic_batch(actor_batch, self.actor.model_config.action_dim, self.action_dim, self.state_dim, self.action_horizon, self.replan_steps)
        agent, actor_info = agent.update_actor(actor_batch)
        return agent.cache_infer_params(), actor_info
