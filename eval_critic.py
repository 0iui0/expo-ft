#! /usr/bin/env python
"""Robot-free offline critic RELIABILITY eval for PI0.5 EXPO-FT.

Loads a warmed critic checkpoint (from pretrain_critic.py) and probes, on the
offline demos, whether the Q function is usable for route ② selection — WITHOUT
the robot. It runs the exact route ② inference path
(``agent.sample_actions(obs, only_base_actions=False)``): sample N base VLA
candidates + n_edit residual edits, score all with the target critic, argmax.

Three signals per state (see the printed VERDICT):
  1. Candidate discrimination: spread (max-min, std) of Q across the 16 candidates.
     ~0 => the critic gives every candidate the same value and cannot select.
  2. Residual usefulness: fraction of states where the argmax is a residual-edited
     candidate whose Q beats every base sample => the route ② improvement mechanism
     is actually finding higher-Q actions than the frozen base prior.
  3. Trajectory monotonicity: Q along a demo episode should rise toward the +1
     terminal (early-half mean vs late-half mean).

HONEST LIMIT (same caveat as pretrain_critic.py): the demos are all successes, so
this cannot verify good-vs-bad discrimination — the critic never saw a failure.
It verifies the *mechanism* (does Q vary across candidates / along a trajectory),
not final judgement quality. That only sharpens with online failures in route ②,
which needs the real robot.

Run on the free GPU only, no thor, no robot:

    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 JAXTYPING_DISABLE=1 \
    .venv/bin/python eval_critic.py \
        --config configs/model/expo_ft_pi_config.py \
        --config_task configs/task/cr5af_pi05.py \
        --dataset_path /data/datasets/shaft_insert_single \
        --run_name pi05_critic_pretrain --output_dir ./logs \
        --num_data 128 --eval_states 32 --traj_episodes 2 --traj_stride 10
"""
import os
import logging

# Same sm_120 XLA autotune workaround as the trainers; must precede `import jax`.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
import csv

import numpy as np
from absl import app, flags
from ml_collections import config_flags

import jax
import etils.epath as epath

from expo_ft.agents import initialize_checkpoint_dir
from expo_ft.data.replay_buffer import create_replay_buffer
from expo_ft.env.droid_utils import process_droid_dataset, process_cr5af_npz_pi05
from expo_ft.utils.train_utils import init_logging

import openpi.training.sharding as openpi_sharding

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string("run_name", "pi05_critic_pretrain", "Run name (checkpoints live under output_dir/run_name/checkpoints).")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("num_data", 0, "Max number of offline demo episodes to load (0 = all).")
flags.DEFINE_integer("eval_states", 32, "Number of random demo states for the discrimination probe.")
flags.DEFINE_integer("traj_episodes", 2, "Number of demo episodes for the trajectory-monotonicity probe.")
flags.DEFINE_integer("traj_stride", 10, "Stride when walking a trajectory (subsample transitions).")
flags.DEFINE_string("output_dir", "./logs", "Directory holding run_name/checkpoints.")
flags.DEFINE_integer("fsdp_devices", 1, "Number of FSDP devices for sharding.")
flags.DEFINE_integer("replan_steps", 8, "Number of replan steps (must match training).")
flags.DEFINE_string("dataset_path", "", "Path to the dataset.")

config_flags.DEFINE_config_file(
    "config",
    "configs/model/expo_ft_pi_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "config_task",
    "configs/task/cr5af_pi05.py",
    "File path to the task configuration.",
    lock_config=False,
)


def _episode_bounds(dataset):
    """Return list of (start, end) transition indices per episode (end exclusive)."""
    dones = np.array([float(t["dones"]) for t in dataset])
    ends = list(np.flatnonzero(dones > 0.5) + 1)
    if not ends or ends[-1] != len(dataset):
        ends.append(len(dataset))
    starts = [0] + ends[:-1]
    return list(zip(starts, ends))


def main(_):
    init_logging()
    jax.config.update(
        "jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser())
    )

    update_devices = jax.devices()
    num_update = len(update_devices)
    mesh = jax.sharding.Mesh(
        np.array(update_devices).reshape(num_update // FLAGS.fsdp_devices, FLAGS.fsdp_devices),
        (openpi_sharding.BATCH_AXIS, openpi_sharding.FSDP_AXIS),
    )
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    log_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name)
    checkpoint_dir_path = epath.Path(os.path.join(log_dir, "checkpoints"))
    # resume=True, overwrite=False: load the warmed critic, never wipe checkpoints.
    checkpoint_manager, resuming = initialize_checkpoint_dir(
        checkpoint_dir_path, keep_period=None, overwrite=False, resume=True,
    )
    if not resuming:
        raise RuntimeError(f"No checkpoint found under {checkpoint_dir_path}; run pretrain_critic.py first.")

    task_description = FLAGS.config_task.language_instruction
    if FLAGS.config_task.env_type in ('droid', 'sim'):
        if getattr(FLAGS.config, "use_pi05", False):
            dataset = process_cr5af_npz_pi05(FLAGS.dataset_path, task_description, num_data=FLAGS.num_data or None)
        else:
            dataset = process_droid_dataset(FLAGS.dataset_path, FLAGS.config_task, num_data=FLAGS.num_data)
        example_action = dataset[0]['actions'][np.newaxis]
    else:
        raise ValueError(f"Unsupported dataset type: {FLAGS.config_task.env_type}")

    if FLAGS.config.model_cls != "EXPOLearner":
        raise ValueError(f"eval_critic only supports EXPOLearner, got {FLAGS.config.model_cls}")
    from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint
    from expo_ft.agents.vla.pi05 import build_pi05

    FLAGS.config.freeze_base_actor = True
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        FLAGS.config, FLAGS.seed, mesh, data_sharding, replicated_sharding, True, task_description,
    )

    # capacity=1 stub buffer: only used for the shape example passed to load_agent
    # (agent init needs example obs/state/action dims). NOT seeded — eval feeds raw
    # demo observations straight into sample_actions, so no 128-episode seed needed.
    replay_buffer = create_replay_buffer(
        config=FLAGS.config, example_action=example_action, capacity=1,
        task_description=task_description, replan_steps=FLAGS.replan_steps, seed=FLAGS.seed,
    )
    agent_example_observation, agent_example_state, agent_example_action = replay_buffer.convert_to_critic_format({
        "base_image": replay_buffer.dataset_dict['base_image'][0][np.newaxis],
        "left_wrist_image": replay_buffer.dataset_dict['left_wrist_image'][0][np.newaxis],
        "state": replay_buffer.dataset_dict['state'][0][np.newaxis],
        "actions": replay_buffer.dataset_dict['actions'][0][np.newaxis],
    })
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]
    agent = load_agent(
        seed=FLAGS.seed,
        example_observation=agent_example_observation.squeeze(),
        example_action=agent_example_action.squeeze(),
        example_state=agent_example_state.squeeze(),
        actor=actor, actor_train_state=actor_train_state, target_actor_params=target_actor_params,
        agent_kwargs=agent_kwargs, metadata=vla_metadata, mesh=mesh,
        data_sharding=data_sharding, replicated_sharding=replicated_sharding,
        resume=True, replan_steps=FLAGS.replan_steps, default_prompt=task_description,
        residual_action_xyzg=FLAGS.config_task.residual_action_xyzg,
    )
    agent = restore_checkpoint(checkpoint_manager, agent)
    loaded_step = max(checkpoint_manager.all_steps())
    logging.info("Loaded warmed critic from step %d", loaded_step)
    # Cache infer params onto the inference device (matches the online rollout path).
    agent = agent.cache_infer_params()

    n_edit = int(FLAGS.config.n_edit_samples)
    N = int(FLAGS.config.N)

    def score_state(obs_raw):
        """Run the route ② selection path on one raw demo observation."""
        nonlocal agent
        obs = dict(obs_raw)  # shallow copy: process_raw_inputs adds an 'actions' key
        _action, agent, info = agent.sample_actions(obs, only_base_actions=False)
        qs = np.asarray(info["qs"]).reshape(-1)
        return qs, int(info.get("selected_idx", 0)), float(info.get("q_selected", qs.max()))

    # ---- Probe 1 + 2: candidate discrimination + residual usefulness ----
    rng_np = np.random.RandomState(FLAGS.seed)
    idxs = rng_np.choice(len(dataset), size=min(FLAGS.eval_states, len(dataset)), replace=False)
    rows = []
    logging.info("Discrimination probe over %d random states (N=%d base + %d edit candidates)...", len(idxs), N, n_edit)
    for j, i in enumerate(idxs):
        qs, sel, q_sel = score_state(dataset[int(i)]["observations"])
        base_qs = qs[:N]
        edit_qs = qs[N:] if qs.shape[0] > N else np.array([])
        edit_won = sel >= N and edit_qs.size > 0 and q_sel > base_qs.max()
        rows.append({
            "idx": int(i), "q_min": float(qs.min()), "q_max": float(qs.max()),
            "q_range": float(qs.max() - qs.min()), "q_std": float(qs.std()),
            "base_std": float(base_qs.std()), "base_best": float(base_qs.max()),
            "edit_best": float(edit_qs.max()) if edit_qs.size else float("nan"),
            "selected_idx": sel, "q_selected": q_sel, "edit_won": int(edit_won),
        })
        if j < 5 or j % 10 == 0:
            logging.info("  state %d: range=%.4f std=%.4f sel=%d edit_won=%d", int(i), rows[-1]["q_range"], rows[-1]["q_std"], sel, int(edit_won))

    # ---- Probe 3: trajectory monotonicity ----
    episodes = _episode_bounds(dataset)[:FLAGS.traj_episodes]
    traj_rows = []
    for ep_k, (s, e) in enumerate(episodes):
        ts = list(range(s, e, FLAGS.traj_stride))
        logging.info("Trajectory probe ep %d: %d points over [%d,%d)", ep_k, len(ts), s, e)
        for t in ts:
            qs, sel, q_sel = score_state(dataset[t]["observations"])
            frac = (t - s) / max(1, (e - 1 - s))  # 0 at start, 1 near terminal
            traj_rows.append({"episode": ep_k, "t": t, "frac": float(frac),
                              "q_mean": float(qs.mean()), "q_max": float(qs.max()), "q_selected": q_sel})

    # ---- Write CSVs ----
    disc_csv = os.path.join(log_dir, "eval_discrimination.csv")
    with open(disc_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    traj_csv = os.path.join(log_dir, "eval_trajectory.csv")
    with open(traj_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
        w.writeheader(); w.writerows(traj_rows)

    # ---- VERDICT ----
    q_range = np.array([r["q_range"] for r in rows])
    q_std = np.array([r["q_std"] for r in rows])
    base_std = np.array([r["base_std"] for r in rows])
    edit_won_frac = np.mean([r["edit_won"] for r in rows])
    tr = np.array([[r["frac"], r["q_selected"]] for r in traj_rows])
    early = tr[tr[:, 0] < 0.5][:, 1]
    late = tr[tr[:, 0] >= 0.5][:, 1]
    corr = float(np.corrcoef(tr[:, 0], tr[:, 1])[0, 1]) if tr.shape[0] > 2 else float("nan")
    q_valley = float(tr[:, 1].min())  # a deep mid-trajectory dip toward the cold floor (~-0.9)
    valley_frac = float(tr[tr[:, 1].argmin(), 0])  # means value hasn't propagated through the middle

    print("\n" + "=" * 68)
    print(f"  CRITIC RELIABILITY (checkpoint step {loaded_step}, {len(rows)} states)")
    print("=" * 68)
    print(f"  1) Candidate discrimination  (Q spread across {N}+{n_edit} candidates)")
    print(f"       mean Q-range  = {q_range.mean():.4f}   (median {np.median(q_range):.4f})")
    print(f"       mean Q-std    = {q_std.mean():.4f}")
    print(f"       mean base-std = {base_std.mean():.4f}   (spread among base VLA samples)")
    print(f"     -> {'OK: critic distinguishes candidates' if q_range.mean() > 0.05 else 'WEAK: Q nearly flat across candidates'}")
    print(f"  2) Residual usefulness")
    print(f"       edit-won fraction = {edit_won_frac:.2f}   (residual beat all base samples)")
    print(f"     -> {'OK: residual finds higher-Q actions' if edit_won_frac > 0.15 else 'LOW: residual rarely beats base (expected pre-online)'}")
    print(f"  3) Trajectory monotonicity  (Q should rise toward +1 terminal)")
    print(f"       early-half mean Q = {early.mean():.4f}   late-half mean Q = {late.mean():.4f}")
    print(f"       corr(time, Q)     = {corr:.3f}")
    print(f"       min Q             = {q_valley:.4f} at frac {valley_frac:.2f}  (cold floor ~= -0.9)")
    if corr > 0.2:
        print(f"     -> OK: Q climbs monotonically along successful demos")
    elif q_valley < -0.5:
        print(f"     -> PARTIAL: endpoints warm but a mid-trajectory valley near the cold")
        print(f"        floor => value not yet propagated through the middle (warm-start only)")
    else:
        print(f"     -> FLAT: no clear time trend (corr~0), Q roughly constant along demos")
    print("=" * 68)
    print("  CAVEAT: all demos are successes -> this verifies the *mechanism*")
    print("  (Q varies across candidates / along trajectory), NOT good-vs-bad")
    print("  judgement. True discrimination needs online failures (route 2, robot).")
    print("=" * 68)
    print(f"  CSVs: {disc_csv}\n        {traj_csv}\n")


if __name__ == "__main__":
    app.run(main)
