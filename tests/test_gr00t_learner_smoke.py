"""GPU integration smoke test: jax critic + torch GR00T actor coexist in ONE
``agent.update()`` on a single GPU.

This is the end-to-end coexistence gate for the unified venv (jax cuda13 +
torch cu130 + gr00t).  It bypasses the env server + batch processor (thin glue)
and feeds a synthetic replay buffer directly, proving the heavy path -- torch
3B VLA forward + jax critic UTD updates + torch actor step -- runs without the
single-GPU sharding / RELATIVE-decode / dead-config regressions fixed in b8.

Requires a CUDA GPU and a GR00T CR5AF checkpoint via the ``GR00T_CKPT`` env var
(a ``checkpoint-<step>/`` dir with safetensors + ``processor_config.json``)::

    GR00T_CKPT=/tmp/cr5af_finetune/cr5af-grasp/checkpoint-20000 \
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python -m pytest tests/test_gr00t_learner_smoke.py -s -v

Marked ``gpu`` so it is excluded from the default CPU run; it skips cleanly
when CUDA or ``GR00T_CKPT`` is unavailable.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _gr00t_checkpoint() -> str | None:
    p = os.environ.get("GR00T_CKPT")
    return p if p and os.path.isdir(p) else None


def test_gr00t_learner_smoke():
    ckpt = _gr00t_checkpoint()
    if ckpt is None:
        pytest.skip("GR00T_CKPT env var not set or path missing")
    if not _has_cuda():
        pytest.skip("CUDA not available")

    import jax
    import openpi.training.sharding as _sharding

    from configs.model.expo_ft_gr00t_config import get_config as get_model_config
    from configs.task.cr5af import get_config as get_task_config
    from expo_ft.agents.alg.expo_ft_gr00t import load_agent
    from expo_ft.agents.vla.gr00t_agent import build_gr00t
    from expo_ft.data.gr00t_replay_buffer import Gr00tReplayBuffer

    SEED = 42
    REPLAN = 8
    BATCH = 2
    UTD = 2
    N_TRANS = 40

    # Single-device sharding (no mesh). The self-built jax hits
    # "device_assignment cannot be None" when eager jax.grad compiles over
    # arrays sharded via a (1,1) NamedSharding mesh; SingleDeviceSharding
    # avoids it.  We also patch fsdp_sharding (called inside EXPOLearner.create
    # with a mesh) so this test is SELF-CONTAINED: it does not rely on the
    # vendored openpi/sharding.py mesh=None fast path, which is not tracked by
    # git (expo_ft/agents/vla/openpi/ is gitignored).
    DEV0 = jax.devices()[0]
    DATA_SHARDING = jax.sharding.SingleDeviceSharding(DEV0)
    REPLICATED_SHARDING = jax.sharding.SingleDeviceSharding(DEV0)
    MESH = None

    def _fsdp_noop(pytree, mesh, *, min_size_mbytes=4, log=False):
        return jax.tree_util.tree_map(
            lambda x: jax.sharding.SingleDeviceSharding(DEV0), pytree
        )

    _sharding.fsdp_sharding = _fsdp_noop

    # Valid 16-dim state/action: eef_9d (xyz + identity rot6d) + joint_pos(6)
    # + gripper(1).  All-zero rot6d is geometrically invalid and raises an SVD
    # non-convergence error in apply_action's Rotation.from_matrix -- real
    # CR5AF actions are valid poses, so this is synthetic-data hygiene only.
    EEF9D = np.array([0, 0, 0.5, 1, 0, 0, 0, 1, 0], dtype=np.float32)  # identity rot6d
    STATE16 = np.concatenate(
        [EEF9D, np.zeros(6, dtype=np.float32), np.zeros(1, dtype=np.float32)]
    )
    ACT16 = STATE16.copy()

    # 1. build_gr00t: load the 3B torch actor from the SFT checkpoint.
    cfg = get_model_config()
    cfg.gr00t_model_path = ckpt
    cfg.gr00t_target_sample_chunk = 2  # small sub-batch for the smoke run
    cfg.N = 2
    cfg.n_edit_samples = 0  # skip the residual/temperature branch
    # Shrink the jax critic for fast compile (coexistence test, not fidelity).
    cfg.latent_dim_image = 64
    cfg.latent_dim_state = 32
    cfg.hidden_dims = (128, 128)
    cfg.encoder_stage_sizes = (2, 2, 2, 2)
    cfg.encoder_num_filters = 32
    cfg.num_qs = 2
    cfg.num_min_qs = 2

    task = get_task_config()
    actor, actor_train_state, target_actor_params, agent_kwargs_in, metadata = build_gr00t(
        cfg, SEED, MESH, DATA_SHARDING, REPLICATED_SHARDING, False, "grasp the housing",
    )
    assert actor.processor.use_relative_action, (
        "CR5AF embodiment must use RELATIVE actions (eef_9d/joint_pos)"
    )

    # 2. replay buffer + synthetic transitions.
    mod = actor._get_modality_cfg()
    video_horizon = len(mod["video"].delta_indices)
    env_action_horizon = len(mod["action"].delta_indices)
    H, W = 256, 256
    rb = Gr00tReplayBuffer(
        camera_keys=actor._video_keys,
        video_horizon=video_horizon,
        image_size=(H, W),
        env_state_dim=actor.state_dim,
        env_action_dim=actor.action_dim,
        env_action_horizon=env_action_horizon,
        capacity=64,
        task_description="grasp the housing",
        replan_steps=REPLAN,
        discount=cfg.discount,
    )
    rb.seed(SEED)
    rng = np.random.RandomState(SEED)
    for i in range(N_TRANS):
        img = (rng.rand(H, W, 3) * 255).astype(np.uint8)
        rb.insert(
            dict(
                image={v: img.copy() for v in actor._video_keys},
                state=STATE16.copy(),
                actions=ACT16.copy(),
                rewards=float(i == N_TRANS - 1),
                masks=1.0,
                dones=bool((i + 1) % 20 == 0),
                is_hil=False,
                is_success=False,
            )
        )
    # Mark the first 20 slots as success so success_only sampling has eligible
    # samples for the actor batch.
    rb.dataset_dict["is_success"][:20] = True

    # 3. build the agent (jax critic on the GPU).
    agent_example = {
        "image": {k: rb.dataset_dict[f"image_{k}"][0][np.newaxis] for k in actor._video_keys},
        "state": rb.dataset_dict["state"][0][np.newaxis],
        "actions": rb.dataset_dict["actions"][0][np.newaxis],
    }
    ex_obs, ex_state, ex_action = rb.convert_to_critic_format(agent_example)
    actor.action_dim = ex_action.squeeze().shape[-1]
    actor.state_dim = ex_state.squeeze().shape[-1]

    agent_kwargs = dict(agent_kwargs_in)
    agent = load_agent(
        seed=SEED,
        example_observation=ex_obs.squeeze(),
        example_action=ex_action.squeeze(),
        example_state=ex_state.squeeze(),
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        agent_kwargs=agent_kwargs,
        metadata=metadata,
        mesh=MESH,
        data_sharding=DATA_SHARDING,
        replicated_sharding=REPLICATED_SHARDING,
        resume=False,
        replan_steps=REPLAN,
        default_prompt="grasp the housing",
        residual_action_xyzg=task.residual_action_xyzg,
    )

    # Dead-config fix (b8): build_gr00t must plumb config hyperparameters into
    # the learner instead of returning {}.
    assert agent.N == cfg.N, f"dead-config: agent.N={agent.N} != cfg.N={cfg.N}"
    assert agent.num_qs == cfg.num_qs, "dead-config: num_qs mismatch"
    assert agent.actor_success_only == cfg.actor_success_only, (
        "dead-config: actor_success_only mismatch"
    )

    # 4. sample a critic batch and a success-only actor batch.
    batch = rb.sample_jax(BATCH * UTD)
    actor_batch = rb.sample_jax(BATCH, success_only=True)
    assert actor_batch is not None, "no success-eligible samples for the actor batch"

    # 5. agent.update: torch VLA forward + jax critic UTD + torch actor step.
    agent = agent.replace(rng=jax.device_put(agent.rng, jax.devices()[0]))
    _agent, info = agent.update(agent, batch, UTD, actor_batch)

    loss_keys = [
        k for k in info
        if "loss" in k.lower() or k in ("q", "q_min", "q_max", "target_q_mean")
    ]
    assert loss_keys, f"no loss/q keys in update info: {list(info)}"
    for k in loss_keys:
        val = float(info[k])
        assert np.isfinite(val), f"{k} not finite: {val}"
