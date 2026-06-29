"""Integration tests for the GR00T-EXPO bridge components.

Marks:
- ``not gpu`` — runs on CPU (no GR00T model required)
- ``gpu`` — requires a CUDA GPU and a GR00T checkpoint
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _gr00t_checkpoint() -> Path | None:
    p = os.environ.get("GR00T_CKPT")
    return Path(p) if p and Path(p).exists() else None


# =============================================================================
# PyTorchTrainState tests (CPU)
# =============================================================================


class TestPyTorchTrainState:
    """Verify the opaque JAX pytree leaf that holds the live GR00T model."""

    @pytest.fixture(autouse=True)
    def _imports(self):
        import jax  # noqa: F401
        from expo_ft.agents.vla.gr00t_train_state import PyTorchTrainState

        self.PyTorchTrainState = PyTorchTrainState
        self.jax = jax

    def test_create_from_model(self):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        state = self.PyTorchTrainState.create(model, opt, ema_decay=0.999)

        assert state.step == 0
        assert state.model is model
        assert state.ema_params is not None
        assert len(state.trainable_names) > 0
        assert set(state.ema_params.keys()) == set(state.trainable_names)

    def test_jax_pytree_leaf(self):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        state = self.PyTorchTrainState.create(model, opt)

        leaves, treedef = self.jax.tree_util.tree_flatten(state)
        assert len(leaves) == 0
        restored = self.jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(restored, self.PyTorchTrainState)
        assert restored.model is model

    def test_opaque_in_jax_pytree(self):
        import jax.numpy as jnp

        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        state = self.PyTorchTrainState.create(model, opt)

        container = {"step": jnp.array(0), "actor_state": state}
        mapped = self.jax.tree_util.tree_map(lambda x: x, container)
        assert "actor_state" in mapped
        assert isinstance(mapped["actor_state"], self.PyTorchTrainState)

    def test_ema_update(self):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        state = self.PyTorchTrainState.create(model, opt, ema_decay=0.999)

        old = {n: p.detach().clone() for n, p in model.named_parameters()}
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1)
        updated = state.update_ema()

        for n, p in model.named_parameters():
            expected = 0.999 * old[n] + 0.001 * p.detach()
            np.testing.assert_allclose(updated.ema_params[n], expected, atol=1e-6)

    def test_get_best_params_returns_ema_when_available(self):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        state = self.PyTorchTrainState.create(model, opt, ema_decay=0.999)
        assert state.get_best_params() is state.ema_params

    def test_get_best_params_falls_back_to_trainable(self):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        state = self.PyTorchTrainState.create(model, opt, ema_decay=None)
        best = state.get_best_params()
        assert set(best.keys()) == set(state.trainable_names)

    def test_load_params_into_model_round_trip(self):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        state = self.PyTorchTrainState.create(model, opt)
        x = torch.randn(1, 4)

        out_before = model(x)
        snapshot = {n: p.detach().clone() for n, p in state.trainable_params().items()}
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5)
        state.load_params_into_model(snapshot)
        out_after = model(x)

        assert torch.equal(out_before, out_after), "load_params_into_model round-trip changed output"

    def test_replace(self):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        state = self.PyTorchTrainState.create(model, opt, ema_decay=0.999)

        new = state.replace(step=10)
        assert new.step == 10
        assert new.model is model
        assert new.ema_params is state.ema_params


# =============================================================================
# Gr00tReplayBuffer tests (CPU)
# =============================================================================


class TestGr00tReplayBuffer:
    """Verify buffer insert/sample mechanics."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from expo_ft.data.gr00t_replay_buffer import Gr00tReplayBuffer

        self.buffer = Gr00tReplayBuffer(
            camera_keys=["hand_view", "table_view"],
            video_horizon=2,
            image_size=(256, 256),
            env_state_dim=16,
            env_action_dim=16,
            env_action_horizon=10,
            capacity=100,
            task_description="pick up the cube",
            replan_steps=2,
            discount=0.99,
        )
        self.buffer.seed(42)

    def _make_transition(self, **overrides) -> dict:
        t = {
            "image": {
                "hand_view": np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),
                "table_view": np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),
            },
            "state": np.random.randn(16).astype(np.float32),
            "actions": np.random.randn(10, 16).astype(np.float32),
            "rewards": 0.0,
            "masks": 1.0,
            "dones": False,
            "is_hil": False,
            "is_success": False,
        }
        t.update(**overrides)
        return t

    def test_insert_and_count(self):
        assert len(self.buffer) == 0
        for i in range(50):
            self.buffer.insert(self._make_transition(dones=(i == 49)))
        assert len(self.buffer) == 50
        assert self.buffer.count_episodes_chronological() == 1

    def test_sample_jax_returns_correct_keys(self):
        for _ in range(10):
            self.buffer.insert(self._make_transition())
        batch = self.buffer.sample_jax(4)
        assert batch is not None
        assert "image" in batch
        assert "hand_view" in batch["image"]
        assert "table_view" in batch["image"]
        assert "state" in batch
        assert "actions" in batch
        assert "next_image" in batch
        assert "next_state" in batch
        assert batch["image"]["hand_view"].shape[0] == 4

    def test_success_only_sampling(self):
        for i in range(20):
            self.buffer.insert(self._make_transition(is_success=(i >= 10)))
        batch = self.buffer.sample_jax(4, success_only=True)
        assert batch is not None

    def test_hil_only_sampling(self):
        for i in range(20):
            self.buffer.insert(self._make_transition(is_hil=(i >= 15)))
        batch = self.buffer.sample_jax(4, hil_only=True)
        assert batch is not None

    def test_convert_to_critic_format(self):
        for _ in range(5):
            self.buffer.insert(self._make_transition())

        example = {
            "image": {
                k: self.buffer.dataset_dict[f"image_{k}"][0][np.newaxis]
                for k in self.buffer._camera_keys
            },
            "state": self.buffer.dataset_dict["state"][0][np.newaxis],
            "actions": self.buffer.dataset_dict["actions"][0][np.newaxis],
        }
        obs, state, act = self.buffer.convert_to_critic_format(example)
        assert obs.ndim == 4
        assert state.shape[-1] == 16
        assert act.shape[-1] == 16

    def test_clear(self):
        for _ in range(10):
            self.buffer.insert(self._make_transition())
        assert len(self.buffer) == 10
        self.buffer.clear()
        assert len(self.buffer) == 0

    def test_mark_episode_success(self):
        self.buffer.insert(self._make_transition(dones=False))
        self.buffer.insert(self._make_transition(dones=True))
        self.buffer.mark_episode_success(0, 2)
        assert self.buffer.dataset_dict["is_success"][0]
        assert self.buffer.dataset_dict["is_success"][1]

    def test_prepare_gr00t_critic_batch(self):
        from expo_ft.data.gr00t_replay_buffer import prepare_gr00t_critic_batch

        for _ in range(10):
            self.buffer.insert(self._make_transition())
        batch = self.buffer.sample_jax(4)

        prepared = prepare_gr00t_critic_batch(
            batch,
            camera_keys=["hand_view", "table_view"],
            padded_dim=132,
            action_dim=16,
            state_dim=16,
            action_horizon=10,
            replan_steps=2,
        )
        assert "observations" in prepared
        assert "full_actions" in prepared
        assert prepared["observations"].shape[-1] == 6
        # uint8 -> float32 normalization for the JAX critic encoder
        import jax.numpy as jnp
        assert prepared["observations"].dtype == jnp.float32
        # actions truncated to replan_steps * action_dim
        assert prepared["actions"].shape == (4, 2 * 16)


# =============================================================================
# Gr00tAgent -- interface contract tests (offline, no real model)
# =============================================================================


class TestGr00tAgentInterface:
    """Verify Gr00tAgent implements all Model abstract methods."""

    def test_has_required_attributes(self):
        from expo_ft.agents.vla.gr00t_agent import Gr00tAgent

        # All required attributes are set in __init__ as instance attributes.
        # Verify they appear in the __init__ source or code signature.
        import inspect
        sig = inspect.signature(Gr00tAgent.__init__)
        init_params = set(sig.parameters.keys())
        # Check that key attributes are set in __init__ body
        src = inspect.getsource(Gr00tAgent.__init__)
        for attr in ("model_config", "action_dim", "state_dim", "mesh", "infer_sharding"):
            assert f"self.{attr}" in src or attr in init_params, (
                f"Missing attribute: {attr}"
            )

    def test_has_required_methods(self):
        from expo_ft.agents.vla.gr00t_agent import Gr00tAgent

        for method in (
            "initialize",
            "get_params",
            "init_target_params",
            "process_raw_inputs",
            "process_transformed_outputs",
            "sample_actions",
            "sample_training_actions",
            "prepare_batch_for_actor",
            "train_step",
        ):
            assert hasattr(Gr00tAgent, method), f"Missing method: {method}"


# =============================================================================
# Gr00tAgent end-to-end (GPU, requires checkpoint)
# =============================================================================


@pytest.mark.gpu
class TestGr00tAgentReal:
    """End-to-end test with a real GR00T checkpoint on GPU."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        ckpt = _gr00t_checkpoint()
        if ckpt is None:
            pytest.skip("GR00T_CKPT env var not set or path missing")
        if not _has_cuda():
            pytest.skip("CUDA not available")

        from expo_ft.agents.vla.gr00t_agent import Gr00tAgent
        from expo_ft.agents.vla.gr00t_train_state import PyTorchTrainState

        agent, train_state, _ = Gr00tAgent.initialize(
            model_path=ckpt,
            embodiment_tag="NEW_EMBODIMENT",
            device="cuda:0",
            dtype=torch.bfloat16,
            lr=1e-5,
            weight_decay=1e-5,
            ema_decay=0.999,
        )
        self.agent = agent
        self.train_state = train_state
        self.PyTorchTrainState = PyTorchTrainState

    def test_initialize_creates_agent(self):
        assert self.agent.model is not None
        assert self.agent.processor is not None
        assert isinstance(self.train_state, self.PyTorchTrainState)
        assert self.agent.action_dim > 0

    def test_get_params_returns_dict(self):
        params = self.agent.get_params(self.train_state)
        assert isinstance(params, dict)
        assert all(isinstance(v, torch.Tensor) for v in params.values())

    def test_init_target_params_returns_copy(self):
        target = self.agent.init_target_params(None)
        assert isinstance(target, dict)
        assert len(target) == len(self.train_state.params)

    def test_model_config_has_action_dim(self):
        assert self.agent.model_config.action_dim > 0
        assert self.agent.model_config.action_horizon > 0

    def test_process_raw_inputs_pipeline(self):
        obs = {
            "video.hand_view": np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),
            "video.table_view": np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),
            "state.eef_9d": np.random.randn(9).astype(np.float32),
            "state.joint_pos": np.random.randn(6).astype(np.float32),
            "state.gripper_pos": np.random.randn(1).astype(np.float32),
            "prompt": "pick up the cube",
        }
        processed = self.agent.process_raw_inputs(obs, action_dim=16, resize_size=256)
        assert isinstance(processed, dict)
        for key in ("state", "embodiment_id", "action_mask", "pixel_values"):
            assert key in processed, f"Missing key: {key}"

    def test_sample_actions_pipeline(self):
        import jax

        obs = {
            "video.hand_view": np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),
            "video.table_view": np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),
            "state.eef_9d": np.random.randn(9).astype(np.float32),
            "state.joint_pos": np.random.randn(6).astype(np.float32),
            "state.gripper_pos": np.random.randn(1).astype(np.float32),
            "prompt": "pick up the cube",
        }
        transformed = self.agent.process_raw_inputs(obs, action_dim=16, resize_size=256)
        actions, ms = self.agent.sample_actions(
            transformed, self.train_state, jax.random.PRNGKey(0)
        )
        assert isinstance(actions, np.ndarray)
        assert actions.shape[-1] == self.agent.action_dim
        assert ms >= 0

    def test_process_transformed_outputs(self):
        dummy = np.random.randn(
            1, self.agent.model_config.action_horizon, self.agent.action_dim
        ).astype(np.float32)
        decoded = self.agent.process_transformed_outputs(dummy, unnormalize=True)
        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == dummy.shape
