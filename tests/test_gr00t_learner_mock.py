"""End-to-end mock test of EXPOLearnerGR00T (no real GR00T model needed).

A tiny FakeGr00tAgent (small torch MLP satisfying the Model interface) drives the
full eager learner loop on CPU, validating B1-B3:
  - PyTorchTrainState (B1) round-trips through train_step.
  - prepare_gr00t_critic_batch + uint8->float (B2) feeds the JAX critic.
  - sample_batch_actions (hoisted) + update_critic + update_actor +
    update_residual_actor + sample_actions rollout (B3) run without crashing and
    produce correctly-shaped, finite outputs.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
import torch

import jax  # noqa: F401  (forces CPU platform via env in the runner)


# ---------------------------------------------------------------------------
# Fake GR00T actor
# ---------------------------------------------------------------------------
class _FakeVLA(torch.nn.Module):
    def __init__(self, action_dim: int, horizon: int):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.head = torch.nn.Linear(1, action_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred = self.get_action(batch)["action_pred"]  # (B, horizon, dim)
        target = batch["actions"]
        return {"loss": ((pred - target) ** 2).mean()}

    def get_action(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # per-sample noise => tiling the input N x yields N diverse chunks
        b = inputs["state"].shape[0]
        noise = torch.randn(b, self.horizon, self.action_dim)
        return {"action_pred": self.head(torch.ones(b, 1)).unsqueeze(1) + 0.1 * noise}


class FakeGr00tAgent:
    """Minimal GR00T-actor stand-in implementing the EXPOLearnerGR00T contract."""

    def __init__(self, action_dim, state_dim, horizon, video_keys, image_size):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon
        self._video_keys = list(video_keys)
        self._image_size = image_size
        self._state_keys = ["s"]  # single key spanning the full state
        self.model = _FakeVLA(action_dim, horizon)
        self.model_config = SimpleNamespace(action_dim=action_dim, action_horizon=horizon)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.mesh = None
        self.infer_sharding = None

    # -- Model interface -------------------------------------------------
    def get_params(self, train_state):
        return train_state.get_best_params()

    def init_target_params(self, rng, *, resume=False):
        return None

    def process_raw_inputs(self, obs, action_dim, resize_size, normalize=True):
        flat = np.concatenate([np.asarray(obs[f"state.{k}"]).reshape(-1) for k in self._state_keys])
        return {"state": torch.from_numpy(flat.astype(np.float32)).unsqueeze(0)}

    def process_transformed_outputs(self, tacts, unnormalize=True):
        arr = np.asarray(tacts)  # already env-space in the fake
        return arr[:, : self.horizon, : self.action_dim]

    def sample_actions(self, transformed_inputs, train_state, rng, train=False, num_samples=1):
        s = transformed_inputs["state"]
        if num_samples > 1:
            s = s.repeat(num_samples, 1)
        with torch.inference_mode():
            pred = self.model.get_action({"state": s})["action_pred"].float()
        return pred.cpu().numpy(), 1.0

    def sample_training_actions(self, *args, **kwargs):
        return self.sample_actions(*args, **kwargs)

    def prepare_batch_for_actor(self, batch):
        b = batch["full_actions"].shape[0]
        actions = np.asarray(batch["full_actions"]).reshape(b, self.horizon, self.action_dim)
        return {
            "state": torch.zeros(b, self.state_dim),
            "actions": torch.from_numpy(actions.astype(np.float32)),
        }

    def train_step(self, rng_key, train_state, batch):
        model = train_state.model
        model.train()
        train_state.optimizer.zero_grad()
        loss = model.forward(batch)["loss"]
        loss.backward()
        train_state.optimizer.step()
        return train_state.replace(step=train_state.step + 1), {"actor_loss": float(loss.detach())}

    # -- GR00T-specific helpers used by the learner ----------------------
    def build_obs_dict(self, images, flat_state, prompt):
        obs = {f"state.{k}": np.asarray(flat_state, dtype=np.float32) for k in self._state_keys}
        for v in self._video_keys:
            obs[f"video.{v}"] = np.asarray(images[v])
        obs["prompt"] = prompt
        return obs

    def critic_inputs_from_observation(self, obs, image_size=None):
        views = [np.asarray(obs[f"video.{v}"]) for v in self._video_keys]
        critic_obs = np.concatenate(views, axis=-1)[np.newaxis].astype(np.uint8)
        flat = np.asarray(obs["state.s"]).reshape(-1)
        return critic_obs, flat[np.newaxis].astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def learner():
    from expo_ft.agents.alg.expo_ft_gr00t import EXPOLearnerGR00T
    from expo_ft.agents.vla.gr00t_train_state import PyTorchTrainState

    action_dim, state_dim, horizon = 4, 4, 3
    video_keys = ["hand_view", "table_view"]
    image_size = (32, 32)
    replan = 2

    actor = FakeGr00tAgent(action_dim, state_dim, horizon, video_keys, image_size)
    train_state = PyTorchTrainState.create(actor.model, actor.optimizer, ema_decay=None)

    example_obs = np.zeros((32, 32, 6), dtype=np.uint8)
    example_action = np.zeros((action_dim,), dtype=np.float32)  # ENV action, not full chunk
    example_state = np.zeros((state_dim,), dtype=np.float32)

    # EXPOLearner.create needs a real mesh (openpi fsdp_sharding dereferences
    # mesh.shape). The GR00T actor is PyTorch; only the JAX critic is sharded,
    # so a single-device mesh is correct.
    import openpi.training.sharding as ops

    mesh = jax.sharding.Mesh(np.array([jax.devices()[0]]).reshape(1, 1), (ops.BATCH_AXIS, ops.FSDP_AXIS))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(ops.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    agent = EXPOLearnerGR00T.create(
        seed=0,
        observation_space=example_obs,
        action_space=example_action,
        states=example_state,
        actor=actor,
        actor_train_state=train_state,
        target_actor_params=None,
        action_horizon=horizon,
        replan_steps=replan,
        N=2,
        n_edit_samples=2,
        num_qs=4,
        num_min_qs=2,
        hidden_dims=(8, 8),
        latent_dim_image=8,
        latent_dim_state=8,
        encoder_stage_sizes=(1, 1, 1, 1),
        encoder_num_filters=4,
        actor_success_only=False,
        use_full_augmentation=False,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        default_prompt="test task",
    )
    return {
        "agent": agent,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "horizon": horizon,
        "replan": replan,
        "image_size": image_size,
        "video_keys": video_keys,
    }


def _raw_batch(b, horizon, action_dim, state_dim, video_keys, image_size):
    h, w = image_size
    batch: Dict[str, Any] = {
        "image": {v: np.random.randint(0, 256, (b, h, w, 3), dtype=np.uint8) for v in video_keys},
        "next_image": {v: np.random.randint(0, 256, (b, h, w, 3), dtype=np.uint8) for v in video_keys},
        "image_mask": {v: np.ones((b,), dtype=bool) for v in video_keys},
        "next_image_mask": {v: np.ones((b,), dtype=bool) for v in video_keys},
        "state": np.random.randn(b, state_dim).astype(np.float32),
        "next_state": np.random.randn(b, state_dim).astype(np.float32),
        "actions": np.random.randn(b, horizon, action_dim).astype(np.float32),
        "rewards": np.zeros((b,), dtype=np.float32),
        "masks": np.ones((b,), dtype=np.float32),
        "dones": np.zeros((b,), dtype=bool),
        "valids": np.ones((b,), dtype=np.float32),
        "prompt": ["test task"] * b,
    }
    return batch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_update_one_step(learner):
    import jax.numpy as jnp

    ld = learner
    agent = ld["agent"]
    utd = 2
    batch = _raw_batch(utd, ld["horizon"], ld["action_dim"], ld["state_dim"], ld["video_keys"], ld["image_size"])
    batch = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in batch.items()}

    new_agent, info = agent.update(agent, batch, utd)

    for k in ("critic_loss", "actor_loss", "residual_actor_loss"):
        assert k in info, f"missing {k} in update info"
        assert np.isfinite(float(info[k])), f"{k} not finite: {info[k]}"
    assert new_agent is not None


def test_sample_actions_rollout(learner):
    ld = learner
    agent = ld["agent"].cache_infer_params()
    obs = {
        "video.hand_view": np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8),
        "video.table_view": np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8),
        "state.s": np.random.randn(ld["state_dim"]).astype(np.float32),
        "prompt": "test task",
    }
    action, new_agent, info = agent.sample_actions(obs)
    assert action.shape == (ld["horizon"], ld["action_dim"])
    assert np.all(np.isfinite(np.asarray(action)))
    assert "sample_time" in info


def test_sample_actions_only_base(learner):
    ld = learner
    agent = ld["agent"].cache_infer_params()
    obs = {
        "video.hand_view": np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8),
        "video.table_view": np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8),
        "state.s": np.random.randn(ld["state_dim"]).astype(np.float32),
        "prompt": "test task",
    }
    action, _, info = agent.sample_actions(obs, only_base_actions=True)
    assert action.shape == (ld["horizon"], ld["action_dim"])
    assert info["selected_action_type"] == "main"
