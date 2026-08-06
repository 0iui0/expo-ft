"""GPU smoke test for the PI0.5 EXPO-FT RL agent (JAX TrainState path).

The GR00T RL path is battle-tested, but PI0.5 takes a DIFFERENT branch
everywhere: ``_split_params``/``_merge_params`` JAX path, ``Pi05Agent.initialize``
target params, ``cache_infer_params``, and the ``PiReplayBuffer`` (not the
Gr00t one). None of this has run on the real robot. This test exercises it
end-to-end WITHOUT the env server: load the SFT checkpoint -> build the
EXPO learner -> one ``agent.update()`` on real offline data.

Two phases, so a failure localises cleanly:
  A. build   -- build_pi05 + EXPOLearner.create + checkpoint load + shapes.
  B. update  -- insert real shaft_insert RL transitions, sample a critic +
                success-only actor batch, run one ``update()``; assert finite.

Requires a CUDA GPU and the PI0.5 SFT checkpoint via ``PI05_CKPT``::

    PI05_CKPT=/data/openpi_checkpoints/expo_pi05_droid_lora_finetune_sft_cartesian_state/pi05_shaft_insert_sft/19999 \
    SHAFT_RL_DATA=/datasets/shaft_insert_single_rl \
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python -m pytest tests/test_pi05_learner_smoke.py -s -v

Marked ``gpu`` so it is excluded from the default CPU run; skips cleanly when
CUDA / ``PI05_CKPT`` is unavailable.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

try:
    import pytest

    pytestmark = pytest.mark.gpu
except ImportError:  # run as a plain script: provide a no-op shim
    class _PytestShim:
        class mark:
            @staticmethod
            def gpu(f):
                return f

        @staticmethod
        def skip(_reason):
            raise SystemExit(f"SKIP: {_reason}")

    pytest = _PytestShim()  # type: ignore

_PI05_CKPT_DEFAULT = (
    "/data/openpi_checkpoints/expo_pi05_droid_lora_finetune_sft_cartesian_state/"
    "pi05_shaft_insert_sft/19999"
)
_SHAFT_RL_DATA_DEFAULT = "/datasets/shaft_insert_single"


def _has_cuda() -> bool:
    try:
        import jax

        return jax.devices()[0].platform == "gpu"
    except Exception:
        return False


def _ckpt() -> str | None:
    p = os.environ.get("PI05_CKPT", _PI05_CKPT_DEFAULT)
    return p if p and os.path.isdir(p) else None


def _rl_data() -> str | None:
    p = os.environ.get("SHAFT_RL_DATA", _SHAFT_RL_DATA_DEFAULT)
    return p if p and os.path.isdir(p) else None


def test_pi05_learner_smoke():
    ckpt = _ckpt()
    if ckpt is None:
        pytest.skip("PI05_CKPT path missing")
    if not _has_cuda():
        pytest.skip("CUDA not available")

    import jax
    import openpi.training.sharding as openpi_sharding

    from configs.model.expo_ft_pi_config import get_config as get_model_config
    from configs.task.cr5af_pi05 import get_config as get_task_config
    from expo_ft.agents.alg.expo_ft import load_agent
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.data.replay_buffer import create_replay_buffer

    SEED = 42
    REPLAN = 8
    # The OOM was NOT batch-driven: it was the jitted _update_jit receiving the ~3.3B Pi0
    # param set TWICE (self + agent) with no donation (~38.5GB of args, batch-independent).
    # That is fixed in update()/_update_jit (single self arg + donate_argnums=0), so a
    # representative batch fits. Keep it small — this is a coexistence smoke test, not a run.
    BATCH = 2
    UTD = 1
    N_TRANS_LOAD = 2  # a couple of real episodes is enough to exercise update

    # Mirror train_pi_robo's sharding setup verbatim: a real single-device mesh
    # + the real fsdp_sharding. (The GR00T smoke test patched fsdp_sharding to a
    # noop and passed mesh=None, but PI0.5's pi05_init_train_state builds
    # NamedSharding(mesh, ...) which requires a real mesh.)
    MESH = openpi_sharding.make_mesh(1)
    DATA_SHARDING = jax.sharding.NamedSharding(
        MESH, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    REPLICATED_SHARDING = jax.sharding.NamedSharding(
        MESH, jax.sharding.PartitionSpec()
    )
    DEV0 = jax.devices()[0]

    # ── config ────────────────────────────────────────────────────────────
    # Shrink the jax critic for fast compile (coexistence test, not fidelity).
    # Keep freeze_pi05_encoder=True (the real PI0.5 default) so the heavy path
    # is realistic but the VLM encoder is not trained.
    cfg = get_model_config()
    # CheckpointWeightLoader expects the <step>/params orbax subdir (has
    # _METADATA), NOT the <step>/ dir itself.
    cfg.pi05_weight_loader_path = os.path.join(ckpt, "params")
    # norm_stats live at <ckpt>/assets/cr5af/shaft_insert/norm_stats.json.
    # expo_ft_pi_config leaves these empty; set them so the buffer transform
    # can normalise. (Same fix the real RL launch needs.)
    cfg.pi05_assets_dir = os.path.join(ckpt, "assets")
    cfg.pi05_asset_id = "cr5af/shaft_insert"
    cfg.latent_dim_image = 64
    cfg.latent_dim_state = 32
    cfg.hidden_dims = (128, 128)
    cfg.encoder_stage_sizes = (2, 2, 2, 2)
    cfg.encoder_num_filters = 32
    cfg.num_qs = 2
    cfg.num_min_qs = 2
    cfg.N = 1
    cfg.n_edit_samples = 1  # exercise the residual + temperature branch

    task = get_task_config()
    # PI0.5 native action space is 10D [xyz(3), rot6d(6), gripper(1)] absolute
    # next-pose (matches the SFT norm_stats), NOT the GR00T 16D velocity action in
    # the cr5af task config. Size the buffer/critic to 10D so dims match the model.
    example_action = np.zeros((1, 10), dtype=np.float32)

    # ── Phase A: build the agent (checkpoint load + JAX construction) ──────
    rb = create_replay_buffer(
        config=cfg,
        example_action=example_action,
        capacity=128,
        task_description=task.language_instruction,
        replan_steps=REPLAN,
        seed=SEED,
    )

    # Fabricate a minimal example to size the critic (convert_to_critic_format
    # is pure numpy -- no transform -- so zeros are fine here).
    padded = rb._action_dim
    ah = rb._action_horizon
    agent_example = {
        "base_image": np.zeros((1, 224, 224, 3), dtype=np.uint8),
        "left_wrist_image": np.zeros((1, 224, 224, 3), dtype=np.uint8),
        "state": np.zeros((1, padded), dtype=np.float32),
        "actions": np.zeros((1, ah, rb._raw_action_dim), dtype=np.float32),
    }
    ex_obs, ex_state, ex_action = rb.convert_to_critic_format(agent_example)

    actor, actor_train_state, target_actor_params, agent_kwargs_in, metadata = build_pi05(
        cfg, SEED, MESH, DATA_SHARDING, REPLICATED_SHARDING, False,
        task.language_instruction,
    )
    actor.action_dim = ex_action.squeeze().shape[-1]
    actor.state_dim = ex_state.squeeze().shape[-1]

    agent = load_agent(
        seed=SEED,
        example_observation=ex_obs.squeeze(),
        example_action=ex_action.squeeze(),
        example_state=ex_state.squeeze(),
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        agent_kwargs=dict(agent_kwargs_in),
        metadata=metadata,
        mesh=MESH,
        data_sharding=DATA_SHARDING,
        replicated_sharding=REPLICATED_SHARDING,
        resume=False,
        replan_steps=REPLAN,
        default_prompt=task.language_instruction,
        # translation-only RL: mask the residual actor's rotation dims [3:6].
        residual_action_xyzg=True,
    )
    agent = agent.cache_infer_params()
    print(
        f"[A] built PI0.5 agent: action_dim={agent.action_dim} "
        f"state_dim={agent.state_dim} full_action_dim={agent.full_action_dim} "
        f"N={agent.N} n_edit={agent.n_edit_samples} num_qs={agent.num_qs}"
    )
    assert agent.action_dim == example_action.shape[-1], (
        f"action_dim {agent.action_dim} != env action {example_action.shape[-1]}"
    )

    # ── Phase B: one update on real offline data ──────────────────────────
    rl_data = _rl_data()
    if rl_data is None:
        pytest.skip("SHAFT_RL_DATA path missing (Phase B needs real transitions)")

    from expo_ft.env.droid_utils import process_cr5af_npz_pi05

    dataset = process_cr5af_npz_pi05(
        rl_data, task.language_instruction, num_data=N_TRANS_LOAD
    )
    rb.insert_dataset(dataset)
    assert len(rb) >= REPLAN, f"buffer too small after insert: {len(rb)} < {REPLAN}"

    # sample_jax returns the raw n-step batch; the training loop (via get_iterator
    # for the critic, _sample_success_actor_batch for the actor) then applies
    # _convert_to_openpi_format to build the "image"/"next_image" dicts update()
    # expects. Mirror that here.
    batch = rb._convert_to_openpi_format(
        rb.sample_jax(BATCH * UTD, data_sharding=DATA_SHARDING)
    )
    actor_raw = rb.sample_jax(BATCH, data_sharding=DATA_SHARDING, success_only=True)
    assert actor_raw is not None, "no success-eligible samples for the actor batch"
    actor_batch = rb._convert_to_openpi_format(actor_raw)

    agent = agent.replace(rng=jax.device_put(agent.rng, DEV0))
    _agent, info = agent.update(agent, batch, UTD, actor_batch)

    loss_keys = [
        k for k in info
        if "loss" in k.lower() or k in ("q", "q_min", "q_max", "target_q_mean")
    ]
    assert loss_keys, f"no loss/q keys in update info: {list(info)}"
    print("[B] update info:", {k: float(info[k]) for k in loss_keys})
    for k in loss_keys:
        val = float(info[k])
        assert np.isfinite(val), f"{k} not finite: {val}"
