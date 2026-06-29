"""EXPOLearner variant that supports PyTorch-based actors like GR00T N1.7.

Key difference from EXPOLearner: the update loop runs in eager mode (not jitted),
allowing the actor.train_step to be a PyTorch function. Critic updates also run
eagerly, which is slightly slower but compatible with PyTorch actor updates.

This enables mixing JAX (critic/residual actor/batch encoder) with PyTorch (VLA actor).
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from expo_ft.agents.alg.agent import AgentLearner
from expo_ft.agents.alg.expo_ft import EXPOLearner, _split_params, _merge_params
from expo_ft.data.dataset import DatasetDict
from expo_ft.data.gr00t_replay_buffer import prepare_gr00t_critic_batch


def load_agent(
    seed,
    example_observation,
    example_action,
    example_state,
    actor,
    actor_train_state,
    target_actor_params,
    agent_kwargs,
    metadata,
    mesh,
    data_sharding,
    replicated_sharding,
    resume,
    replan_steps,
    default_prompt,
    residual_action_xyzg,
):
    """Create an EXPOLearnerGR00T from a pre-built GR00T actor and remaining config kwargs."""
    agent_kwargs.update(
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        mesh=mesh,
        resume=resume,
        replan_steps=replan_steps,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        default_prompt=default_prompt,
        residual_action_xyzg=residual_action_xyzg,
        **metadata,
    )
    return EXPOLearnerGR00T.create(seed, example_observation, example_action, example_state, **agent_kwargs)


def restore_checkpoint(checkpoint_manager: ocp.CheckpointManager, agent):
    """Restore agent from checkpoint."""
    agent, params = _split_params(agent)
    restored = checkpoint_manager.restore(
        checkpoint_manager.latest_step(),
        items={"agent": agent, "params": params},
    )
    return _merge_params(restored["agent"], restored["params"])


def save_checkpoint(checkpoint_manager: ocp.CheckpointManager, agent, step: int):
    """Save agent checkpoint."""
    agent, params = _split_params(agent)
    checkpoint_manager.save(step, items={"agent": agent, "params": params})


class EXPOLearnerGR00T(EXPOLearner):
    """EXPOLearner variant that supports PyTorch-based actors like GR00T N1.7.

    Inherits all sharding, critic, residual actor logic from EXPOLearner.
    Overrides only ``update()`` to run without JAX jit, allowing
    actor.train_step to be a PyTorch function that cannot be jitted.

    The PyTorchTrainState (opaque pytree leaf) is compatible with EXPOLearner's
    pytree structure, so most methods work unchanged.
    """

    def update(
        self,
        agent,
        batch: DatasetDict,
        utd_ratio: int,
        actor_batch: DatasetDict = None,
    ) -> Tuple["EXPOLearnerGR00T", Dict[str, float]]:
        """Update critic and actor in eager mode (no JAX jit).

        Runs the full update loop without jit compilation:
        1. Data augmentation (eager JAX)
        2. Critic minibatch updates (eager JAX)
        3. Actor update (Python, may call PyTorch)
        4. Residual actor + temperature updates (eager JAX)

        Args:
            agent: Current agent state (EXPOLearnerGR00T instance).
            batch: Training batch dict.
            utd_ratio: Update-to-data ratio.
            actor_batch: Optional separate batch for actor (success-only mode).

        Returns:
            (new_agent, info_dict)
        """
        # Drop stale inference copies
        agent = agent.replace(_infer_cache=None)

        # --- Data augmentation ---
        # TODO(gr00t): per-view random crop on the critic images. The OpenPI
        # augmentation fn hardcodes base_0_rgb/left_wrist_0_rgb and assumes
        # [-1,1] float, so it is intentionally a no-op here until ported to the
        # GR00T view keys (hand_view/table_view) and uint8->float inputs.
        rng = agent.rng

        batch = batch.copy()

        # Prepare critic batch: concat camera views, normalize uint8->float,
        # truncate action chunks to replan_steps. Uses GR00T view keys.
        batch = prepare_gr00t_critic_batch(
            batch,
            camera_keys=self.actor._video_keys,
            padded_dim=self.actor.model_config.action_dim,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            replan_steps=self.replan_steps,
        )

        new_agent = agent.replace(rng=rng)

        # --- Split into minibatches for UTD ---
        total_bs = batch["actions"].shape[0]
        assert total_bs % utd_ratio == 0, (
            f"Batch size ({total_bs}) must be a multiple of utd_ratio ({utd_ratio})"
        )
        minibatch_size = total_bs // utd_ratio

        def reshape_minibatch(x):
            if hasattr(x, "shape") and len(x.shape) >= 1:
                return x.reshape((utd_ratio, minibatch_size) + x.shape[1:])
            return x

        minibatches = jax.tree_util.tree_map(reshape_minibatch, batch)

        # --- Critic update loop (eager, not jax.lax.scan) ---
        for i in range(utd_ratio):
            mb = jax.tree_util.tree_map(
                lambda x, idx=i: x[idx] if hasattr(x, "shape") and x is not None else x,
                minibatches,
            )
            new_agent, critic_step_info = new_agent.update_critic(mb)

        # Use last minibatch for actor updates
        last_minibatch = jax.tree_util.tree_map(
            lambda x: x[-1] if hasattr(x, "shape") and x is not None else x,
            minibatches,
        )

        # --- Actor update (Python, may call PyTorch) ---
        if self.actor_success_only and actor_batch is not None:
            actor_batch = actor_batch.copy()
            actor_batch = prepare_gr00t_critic_batch(
                actor_batch,
                camera_keys=self.actor._video_keys,
                padded_dim=self.actor.model_config.action_dim,
                action_dim=self.action_dim,
                state_dim=self.state_dim,
                action_horizon=self.action_horizon,
                replan_steps=self.replan_steps,
            )
            new_agent, actor_info = new_agent.update_actor(actor_batch)
        else:
            new_agent, actor_info = new_agent.update_actor(last_minibatch)

        actor_info = dict(actor_info)

        # --- Residual actor + temperature updates ---
        if self.n_edit_samples > 0:
            new_agent, r_actor_info = new_agent.update_residual_actor(last_minibatch)
            new_agent, temp_info = new_agent.update_temperature(r_actor_info["entropy"])
            actor_info = {**actor_info, **r_actor_info, **temp_info}

        # Combine all info
        info = {**actor_info, **critic_step_info}

        # Cache inference params for rollout
        return new_agent.cache_infer_params(), info

    def update_actor(self, batch: DatasetDict) -> Tuple[AgentLearner, Dict[str, float]]:
        """Actor update without JAX mesh/sharding context.

        Overrides EXPOLearner.update_actor to skip the `_sharding.set_mesh` context
        manager and JAX device_put sharding, which are unnecessary for the PyTorch
        actor (GR00T runs on torch devices, not JAX shards). The actor batch is
        passed through unchanged; Gr00tAgent.prepare_batch_for_actor already
        produces torch tensors on the correct device.

        The JAX rng `key` is unused by the PyTorch train_step (GR00T uses its own
        torch RNG), but kept for interface compatibility.
        """
        actor_batch = self.actor.prepare_batch_for_actor(batch)

        rng = self.rng
        key, rng = jax.random.split(rng, 2)

        new_train_state, info = self.actor.train_step(
            key, self.actor_train_state, actor_batch
        )

        # No base-VLA target EMA on the GR00T path: EXPO's OTF / next-action
        # sampling uses the online actor directly, and only the critic carries a
        # Polyak target. target_actor_params stays None (see build_gr00t).
        new_agent = self.replace(actor_train_state=new_train_state, rng=rng)
        return new_agent, info
