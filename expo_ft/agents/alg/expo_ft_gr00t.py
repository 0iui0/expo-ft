"""EXPOLearner variant that supports PyTorch-based actors like GR00T N1.7.

The update loop runs eager (no jit) so ``actor.train_step`` can be a PyTorch
function. The critic / residual actor / temperature / image encoder stay JAX
(inherited from EXPOLearner). Only the actor-coupled methods are overridden to
speak GR00T's data contract (view keys, torch inference, env-space actions):

- ``update``             : eager; hoists next-action sampling OUT of the UTD
                           critic loop (decision B: one VLA forward per update).
- ``sample_actions``     : rollout OTF -- N base chunks (one batched GR00T
                           forward, native per-sample noise) + residual edits,
                           argmax-Q selection.
- ``sample_batch_actions``: same OTF, vectorized over a batch of next-states,
                           for the TD target.
- ``update_critic``      : consumes precomputed next-actions (no internal VLA
                           call), so the critic loop never touches the actor.

update_residual_actor / update_temperature / cache_infer_params are inherited
unchanged (they are key-agnostic JAX over the prepared critic batch).
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from expo_ft.agents.alg.agent import AgentLearner
from expo_ft.agents.alg.expo_ft import (
    EXPOLearner,
    _merge_params,
    _split_params,
    batch_encode,
    compute_q,
)
from expo_ft.data.dataset import DatasetDict
from expo_ft.data.gr00t_replay_buffer import prepare_gr00t_critic_batch
from expo_ft.networks import subsample_image_ensemble


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
    """Create an EXPOLearnerGR00T from a pre-built GR00T actor + config kwargs."""
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
    return EXPOLearnerGR00T.create(
        seed, example_observation, example_action, example_state, **agent_kwargs
    )


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
    """EXPOLearner variant for PyTorch actors (GR00T N1.7).

    Inherits sharding, critic, residual actor, temperature and checkpoint-split
    logic from EXPOLearner. Overrides the actor-coupled methods (see module
    docstring) to use GR00T's data contract and eager-mode PyTorch inference.
    """

    # ------------------------------------------------------------------
    # Rollout inference: OTF (N base + residual edits) argmax-Q selection
    # ------------------------------------------------------------------
    def sample_actions(self, observations, only_base_actions=False):
        rng = self.rng
        c = self._infer_cache or {}
        _actor_train_state = c.get("actor_train_state") or self.actor_train_state
        _batch_encoder_params = c.get("batch_encoder_params") or self.batch_encoder.params
        _residual_actor_params = c.get("residual_actor_params") or self.residual_actor.params
        _target_critic_params = c.get("target_critic_params") or self.target_critic.params

        # N base chunks from GR00T (one batched forward, native per-sample noise).
        transformed_inputs = self.actor.process_raw_inputs(
            observations, self.action_dim, self.resize_size
        )
        key, rng = jax.random.split(rng)
        transformed_actions, sample_time = self.actor.sample_actions(
            transformed_inputs, _actor_train_state, key, train=False, num_samples=self.N
        )
        raw_actions = jnp.asarray(self.actor.process_transformed_outputs(transformed_actions))

        if only_base_actions:
            action = raw_actions[0].reshape(self.action_horizon, self.action_dim)
            return (
                jnp.array(action),
                self.replace(rng=rng),
                {"sample_time": sample_time, "selected_action_type": "main"},
            )

        # Critic obs/state from the RAW observation (not from GR00T).
        critic_obs_np, critic_state_np = self.actor.critic_inputs_from_observation(observations)
        critic_obs = jnp.asarray(critic_obs_np.astype(np.float32) / 255.0)
        critic_state = jnp.asarray(critic_state_np)
        encoded = batch_encode(
            self.batch_encoder.apply_fn, _batch_encoder_params, critic_obs, stop_gradient=True
        )

        base_actions = raw_actions[:, : self.replan_steps, :].reshape(self.N, self.full_action_dim)

        if self.n_edit_samples > 0:
            key, rng = jax.random.split(rng, 2)
            r_obs = jnp.repeat(encoded, self.n_edit_samples, axis=0)
            r_states = jnp.repeat(critic_state, self.n_edit_samples, axis=0)
            base_for_edit = base_actions[: self.n_edit_samples]
            r_samples, _, rng = self._sample_residual(
                key, _residual_actor_params, r_obs, r_states, base_for_edit
            )
            # full-horizon edited chunks (first replan_steps replaced by base+edit)
            r_modified = r_samples.reshape(self.n_edit_samples, self.replan_steps, self.action_dim)
            full_edited = raw_actions[: self.n_edit_samples].at[:, : self.replan_steps, :].set(
                r_modified
            )
            all_chunks = jnp.concatenate([raw_actions, full_edited], axis=0)
            candidates = jnp.concatenate([base_actions, r_samples], axis=0)
        else:
            all_chunks = raw_actions
            candidates = base_actions

        n_cand = candidates.shape[0]
        enc_rep = jnp.repeat(encoded, n_cand, axis=0)
        state_rep = jnp.repeat(critic_state, n_cand, axis=0)
        key, rng = jax.random.split(rng)
        target_params = subsample_image_ensemble(
            key, _target_critic_params, self.num_min_qs, self.num_qs
        )
        qs = compute_q(
            self.target_critic.apply_fn, target_params, enc_rep, candidates, state_rep,
            self.num_min_qs,
        )
        idx = jnp.argmax(qs)
        action = all_chunks[idx].reshape(self.action_horizon, self.action_dim)
        rng, _ = jax.random.split(rng, 2)
        return jnp.array(action), self.replace(rng=rng), {"sample_time": sample_time}

    # ------------------------------------------------------------------
    # TD-target next-action sampling (hoisted out of the UTD loop)
    # ------------------------------------------------------------------
    def sample_batch_actions(self, batch):
        """OTF-select the next action for each next-state in ``batch``.

        Returns ``(next_actions (B, full_action_dim), info, new_rng)``. Called
        ONCE per update (decision B) so the VLA forward is not repeated UTD
        times. Per-state loop is correct; batching/chunking is a B6 concern.
        """
        rng = self.rng
        n_next = batch["next_state"].shape[0]
        selected = []
        for i in range(n_next):
            images = {v: np.asarray(batch["next_image"][v][i]) for v in self.actor._video_keys}
            flat_state = np.asarray(batch["next_state"][i])
            obs = self.actor.build_obs_dict(images, flat_state, batch["prompt"][i])
            tin = self.actor.process_raw_inputs(obs, self.action_dim, self.resize_size)

            key, rng = jax.random.split(rng)
            tacts, _ = self.actor.sample_actions(
                tin, self.actor_train_state, key, train=False, num_samples=self.N
            )
            raw_base = self.actor.process_transformed_outputs(tacts)
            base_actions = raw_base[:, : self.replan_steps, :].reshape(self.N, self.full_action_dim)

            critic_obs_np, critic_state_np = self.actor.critic_inputs_from_observation(obs)
            critic_obs = jnp.asarray(critic_obs_np.astype(np.float32) / 255.0)
            critic_state = jnp.asarray(critic_state_np)
            encoded = batch_encode(
                self.batch_encoder.apply_fn, self.batch_encoder.params, critic_obs,
                stop_gradient=True,
            )

            if self.n_edit_samples > 0:
                key, rng = jax.random.split(rng, 2)
                r_obs = jnp.repeat(encoded, self.n_edit_samples, axis=0)
                r_states = jnp.repeat(critic_state, self.n_edit_samples, axis=0)
                r_samples, _, rng = self._sample_residual(
                    key, self.residual_actor.params, r_obs, r_states, base_actions[: self.n_edit_samples]
                )
                candidates = jnp.concatenate([base_actions, r_samples], axis=0)
            else:
                candidates = base_actions

            n_cand = candidates.shape[0]
            enc_rep = jnp.repeat(encoded, n_cand, axis=0)
            state_rep = jnp.repeat(critic_state, n_cand, axis=0)
            key, rng = jax.random.split(rng)
            target_params = subsample_image_ensemble(
                key, self.target_critic.params, self.num_min_qs, self.num_qs
            )
            qs = compute_q(
                self.target_critic.apply_fn, target_params, enc_rep, candidates, state_rep,
                self.num_min_qs,
            )
            selected.append(candidates[jnp.argmax(qs)])
        rng, _ = jax.random.split(rng, 2)
        return jnp.stack(selected), {}, rng

    # ------------------------------------------------------------------
    # Critic update (consumes precomputed next-actions; no actor call inside)
    # ------------------------------------------------------------------
    def update_critic(self, batch: DatasetDict, next_actions) -> Tuple[AgentLearner, dict]:
        rng = self.rng
        next_actions = jnp.asarray(next_actions)

        key, rng = jax.random.split(rng)
        target_params = subsample_image_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )
        key, rng = jax.random.split(rng)

        next_observations = batch_encode(
            self.batch_encoder.apply_fn, self.batch_encoder.params,
            batch["next_observations"], stop_gradient=True,
        )
        next_qs = self.target_critic.apply_fn(
            {"params": target_params}, next_observations, next_actions, False,
            p=batch["next_critic_states"], sample_num=self.num_min_qs,
        )
        next_q_nan_mask = jnp.isnan(next_qs)
        next_qs = jnp.where(next_q_nan_mask, 0.0, next_qs)
        next_q = next_qs.min(axis=0)
        target_q = batch["rewards"] + (self.discount ** self.replan_steps) * batch["masks"] * next_q

        key, rng = jax.random.split(rng)
        params_dict = {"critic": self.critic.params}
        if not self.freeze_critic_encoder:
            params_dict["batch_encoder"] = self.batch_encoder.params

        def critic_loss_fn(params_dict):
            if self.freeze_critic_encoder:
                observations = batch_encode(
                    self.batch_encoder.apply_fn, self.batch_encoder.params,
                    batch["observations"], stop_gradient=True,
                )
            else:
                observations = batch_encode(
                    self.batch_encoder.apply_fn, params_dict["batch_encoder"], batch["observations"]
                )
            qs = self.critic.apply_fn(
                {"params": params_dict["critic"]}, observations, batch["actions"], True,
                p=batch["critic_states"], rngs={"dropout": key},
            )
            critic_loss = (((qs - target_q) ** 2) * batch["valids"]).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean(),
                "q_min": qs.min(),
                "q_max": qs.max(),
                "target_q_mean": target_q.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(params_dict)
        critic = self.critic.apply_gradients(grads=grads["critic"])
        batch_encoder = (
            self.batch_encoder
            if self.freeze_critic_encoder
            else self.batch_encoder.apply_gradients(grads=grads["batch_encoder"])
        )
        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)
        return self.replace(
            critic=critic, target_critic=target_critic, batch_encoder=batch_encoder, rng=rng
        ), info

    # ------------------------------------------------------------------
    # Actor update (BC on success data); no base-VLA target EMA
    # ------------------------------------------------------------------
    def update_actor(self, batch: DatasetDict) -> Tuple[AgentLearner, dict]:
        actor_batch = self.actor.prepare_batch_for_actor(batch)
        rng = self.rng
        key, rng = jax.random.split(rng, 2)
        new_train_state, info = self.actor.train_step(
            key, self.actor_train_state, actor_batch
        )
        # No base-VLA target EMA on the GR00T path: EXPO's OTF / next-action
        # sampling uses the online actor; only the critic carries a Polyak target.
        return self.replace(actor_train_state=new_train_state, rng=rng), info

    # ------------------------------------------------------------------
    # Eager update: prep -> hoist next-actions -> UTD critic -> actor -> residual
    # ------------------------------------------------------------------
    def update(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None):
        agent = agent.replace(_infer_cache=None)
        batch = batch.copy()

        batch = prepare_gr00t_critic_batch(
            batch,
            camera_keys=self.actor._video_keys,
            padded_dim=self.actor.model_config.action_dim,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            replan_steps=self.replan_steps,
        )

        # HOIST (decision B): sample next actions ONCE for the whole batch.
        next_actions, _, new_rng = agent.sample_batch_actions(batch)
        agent = agent.replace(rng=new_rng)

        # prompt was only needed for next-action sampling; drop before UTD reshape.
        batch.pop("prompt", None)

        total_bs = batch["actions"].shape[0]
        assert total_bs % utd_ratio == 0, (
            f"Batch size ({total_bs}) must be a multiple of utd_ratio ({utd_ratio})"
        )
        minibatch_size = total_bs // utd_ratio

        def reshape_mb(x):
            if hasattr(x, "shape") and getattr(x, "shape", None):
                return x.reshape((utd_ratio, minibatch_size) + x.shape[1:])
            return x

        def sel(x, idx):
            return x[idx] if (hasattr(x, "shape") and x is not None) else x

        minibatches = jax.tree_util.tree_map(reshape_mb, batch)
        na_minibatches = next_actions.reshape((utd_ratio, minibatch_size) + next_actions.shape[1:])

        new_agent = agent
        critic_info = {}
        for i in range(utd_ratio):
            mb = jax.tree_util.tree_map(lambda x, _i=i: sel(x, _i), minibatches)
            new_agent, critic_info = new_agent.update_critic(mb, na_minibatches[i])

        last_minibatch = jax.tree_util.tree_map(lambda x: sel(x, -1), minibatches)

        if self.actor_success_only and actor_batch is not None:
            actor_batch = prepare_gr00t_critic_batch(
                actor_batch.copy(),
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

        if self.n_edit_samples > 0:
            new_agent, r_actor_info = new_agent.update_residual_actor(last_minibatch)
            new_agent, temp_info = new_agent.update_temperature(r_actor_info["entropy"])
            actor_info = {**actor_info, **r_actor_info, **temp_info}

        return new_agent.cache_infer_params(), {**actor_info, **critic_info}
