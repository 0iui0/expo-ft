#! /usr/bin/env python
import os
import logging

# RTX 5090 (sm_120) + selfbuilt jaxlib: XLA conv autotune selects a cuDNN algo that
# segfaults during compile of the critic's ResNetV2 encoder (the real (3,4,6,3)/64
# config; the shrunk smoke-test encoder does not trigger it). Disabling autotune
# makes XLA use a default algo — slightly slower, but crash-free. Must be set before
# jax/jaxlib initialise, so it is set here, before `import jax`. Overridable via env.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
import time
import threading
import csv
from collections import deque

import numpy as np
import tqdm
from absl import app, flags

from ml_collections import config_flags

import jax
import etils.epath as epath

import wandb
from expo_ft.agents import initialize_checkpoint_dir, save_replay_buffer_transition
from expo_ft.data.replay_buffer import create_replay_buffer
from expo_ft.data.batch_processor import BatchProcessor
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.env.droid_utils import process_droid_dataset, process_cr5af_npz_pi05
from expo_ft.utils.log_utils import EpisodeState, TrainingStats
from expo_ft.utils.visualization import capture_step_data, log_episode_visualizations
from expo_ft.utils.train_utils import get_batch_info, init_logging, init_wandb

import openpi.training.sharding as openpi_sharding

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "expo-ft", "wandb project name.")
flags.DEFINE_string("run_name", None, "Optional wandb run name.")
flags.DEFINE_float("offline_ratio", 0.0, "Offline batch fraction; 0 inserts dataset into online replay buffer.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_float("ep_timeout_secs", 120.0, "Pause update thread if no episode finishes within this many seconds. 0 to disable.")
flags.DEFINE_integer("batch_size", 64, "Mini batch size.")
flags.DEFINE_integer("max_steps", 100_000, "Number of training steps.")
flags.DEFINE_integer("num_data", 0, "Max number of offline demo episodes to load (0 = all).")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint during training.")
flags.DEFINE_integer("checkpoint_interval", 0, "Save agent checkpoint every N steps. When 0 and checkpoint_model=True, no interval saving (save at end only).")
flags.DEFINE_boolean("checkpoint_buffer", False, "Save agent replay buffer on evaluation.")
flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")
flags.DEFINE_integer("keep_period", None, "Keep checkpoints every N steps.")
flags.DEFINE_boolean("overwrite", False, "Overwrite existing checkpoint directory.")
flags.DEFINE_boolean("resume", False, "Resume training from checkpoint.")
flags.DEFINE_string("output_dir", "./logs", "Directory for logs and checkpoints.")
flags.DEFINE_integer("fsdp_devices", 1, "Number of FSDP devices for sharding.")

flags.DEFINE_string("client_host", "localhost", "Host for environment operations server.")
flags.DEFINE_integer("client_port", 8102, "Port for environment operations server.")

flags.DEFINE_integer("replan_steps", 8, "Number of replan steps for evaluation.")

# Chunk-boundary blend (#4): the VLA infers a 16-step chunk but we execute only
# replan_steps (8) then re-infer. The new chunk's first waypoints are absolute
# targets planned from a lagging observation, so they pull BACKWARD toward the
# arm's current pose -> a visible "退一步" every replan (~1s). Fix: temporally
# ensemble the seam. sample_actions (only_base_actions) returns the FULL 16-step
# chunk; we keep the discarded tail a[replan:] and blend it into the next chunk's
# first blend_steps waypoints (aligned b0<->a8), tapering the weight to 0. This
# uses already-computed actions (no extra inference) and cancels the reversal.
# Only xyz (dims 0:3) is blended; rot6d/gripper pass through. blend_w0=0 disables.
flags.DEFINE_integer("boundary_blend_steps", 4, "Chunk-seam blend length (# waypoints tapered).")
flags.DEFINE_float("boundary_blend_w0", 0.6, "Chunk-seam blend initial weight on prev-chunk tail (0=off).")

# Rollout uses only the base VLA action (1 sample, no OTF Q-select, no residual
# edit). The OTF critic + residual actor are random at startup (no offline
# warm-start), so argmax-Q over N candidates + a random residual corrupts the
# SFT prior -> off-course motion / e-stop. only_base_actions=True matches the
# proven SFT open-loop deploy (1 base sample) and is safe. The base actor still
# trains from RL updates; flip to False once the critic is warm-started.
flags.DEFINE_boolean("only_base_actions", True, "Rollout action selection: True=1 base VLA sample (safe, SFT-like); False=full OTF (N candidates + residual + Q-select).")

flags.DEFINE_string("dataset_path", "", "Path to the dataset.")
config_flags.DEFINE_config_file(
    "config",
    "configs/model/expo_ft_pi_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_task",
    "configs/task/pick.py",
    "File path to the task configuration.",
    lock_config=False,
)

def main(_):
    init_logging()
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0

    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path("~/.cache/jax").expanduser()),
    )

    num_gpus = jax.device_count()
    if num_gpus < 2:
        raise ValueError(
            f"At least 2 GPUs required (1 for sampling, rest for updates), got {num_gpus}"
        )
    sample_device = jax.devices()[0]
    update_devices = jax.devices()[1:]
    num_update = len(update_devices)
    if num_update % FLAGS.fsdp_devices != 0:
        raise ValueError(
            f"Number of update devices ({num_update}) must be divisible by "
            f"fsdp_devices ({FLAGS.fsdp_devices})"
        )
    mesh = jax.sharding.Mesh(
        np.array(update_devices).reshape(num_update // FLAGS.fsdp_devices, FLAGS.fsdp_devices),
        (openpi_sharding.BATCH_AXIS, openpi_sharding.FSDP_AXIS),
    )
    logging.info("Device layout: sampling on %s, updates on %s", sample_device, update_devices)

    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    
    log_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name)
    os.makedirs(log_dir, exist_ok=True)
    train_video_dir = os.path.join(log_dir, "train_videos")
    os.makedirs(train_video_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_dir_path = epath.Path(checkpoint_dir)
    checkpoint_manager, resuming = initialize_checkpoint_dir(
        checkpoint_dir_path,
        keep_period=FLAGS.keep_period,
        overwrite=FLAGS.overwrite,
        resume=FLAGS.resume,
    )

    init_wandb(checkpoint_dir_path, resuming, FLAGS.project_name, FLAGS.run_name)
    wandb.config.update(FLAGS.flag_values_dict(), allow_val_change=resuming)

    if FLAGS.config_task.env_type in ('droid', 'sim'):
        if getattr(FLAGS.config, "use_pi05", False):
            # PI0.5 prior was SFT'd on CR5AF npz demos; load the same episodes as
            # 10-D absolute next-pose transitions (xyz_m + rot6d + gripper).
            dataset = process_cr5af_npz_pi05(
                FLAGS.dataset_path,
                FLAGS.config_task.language_instruction,
                num_data=FLAGS.num_data or None,
            )
        else:
            dataset = process_droid_dataset(
                FLAGS.dataset_path,
                FLAGS.config_task,
                num_data=FLAGS.num_data,
            )
        example_action = dataset[0]['actions'][np.newaxis]
    else:
        raise ValueError(f"Unsupported dataset type: {FLAGS.config_task.env_type}")
    
    # Create training environment wrapper directly
    train_env_creation_request = {
        "example_action": example_action,
        "env_usage": "train",
        "video_dir": train_video_dir,
    }

    logging.info("Creating environment...")
    env = EnvClientWrapper(
        env_creation_request=train_env_creation_request,
        host=FLAGS.client_host,
        port=FLAGS.client_port
    )
    env.reset()
    logging.info(f"Created training environment {env.env_id}")

    model_cls = FLAGS.config.model_cls
    # BCLearner uses human-intervention chunks for the actor batch only (no critic).
    use_dagger_hil_sampling = model_cls == "BCLearner"
    if model_cls == "BCLearner":
        from expo_ft.agents.alg.bc import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "EXPOLearner":
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint, save_checkpoint
    else:
        raise ValueError(f"Unsupported model class: {model_cls}")

    from expo_ft.agents.vla.pi05 import build_pi05
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        FLAGS.config, FLAGS.seed, mesh, data_sharding, replicated_sharding,
        resuming, env.task_description,
    )

    rb_args = dict(
        config=FLAGS.config,
        example_action=example_action,
        capacity=FLAGS.max_steps,
        task_description=env.task_description,
        replan_steps=FLAGS.replan_steps,
        seed=FLAGS.seed,
    )
    replay_buffer = create_replay_buffer(**rb_args)
    offline_replay_buffer = create_replay_buffer(**rb_args)

    actor_success_only = getattr(FLAGS.config, "actor_success_only", False)
    batch_processor = BatchProcessor(
        replay_buffer=replay_buffer,
        offline_replay_buffer=offline_replay_buffer,
        data_sharding=data_sharding,
        batch_size=FLAGS.batch_size,
        utd_ratio=FLAGS.utd_ratio,
        offline_ratio=FLAGS.offline_ratio,
        actor_success_only=actor_success_only,
        use_dagger_hil_sampling=use_dagger_hil_sampling,
        dataset=dataset,
    )

    agent_example_observation, agent_example_state, agent_example_action = offline_replay_buffer.convert_to_critic_format(
    {
        "base_image": offline_replay_buffer.dataset_dict['base_image'][0][np.newaxis],
        "left_wrist_image": offline_replay_buffer.dataset_dict['left_wrist_image'][0][np.newaxis],
        "state": offline_replay_buffer.dataset_dict['state'][0][np.newaxis],
        "actions": offline_replay_buffer.dataset_dict['actions'][0][np.newaxis],
    })
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]
    agent = load_agent(
        seed=FLAGS.seed,
        example_observation=agent_example_observation.squeeze(),
        example_action=agent_example_action.squeeze(),
        example_state=agent_example_state.squeeze(),
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        agent_kwargs=agent_kwargs,
        metadata=vla_metadata,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        resume=resuming,
        replan_steps=FLAGS.replan_steps,
        default_prompt=env.task_description,
        residual_action_xyzg=FLAGS.config_task.residual_action_xyzg,
    )
    
    start_step = 0
    if resuming:
        agent = restore_checkpoint(checkpoint_manager, agent)
        steps = tuple(checkpoint_manager.all_steps())
        latest_step = max(steps) if steps else None
        if latest_step is not None:
            start_step = latest_step
            logging.info("Resuming from step %d", start_step)
        batch_processor.restore(checkpoint_dir_path, up_to_step=latest_step)

    episode_log = EpisodeState()
    training_log = TrainingStats(
        ep_count=replay_buffer.count_episodes_chronological() if resuming else 0,
    )
    logging.info("Resuming: ep_count set to %d (episodes in replay buffer).", training_log.ep_count)

    batch_processor.on_episode_start()

    dt = 1.0 / FLAGS.config_task.control_hz
    done = False
    env.reset()
    start_step_time = time.time()
    env.step(FLAGS.config_task.example_action.squeeze().tolist())
    action_plan = deque()
    action_type = "policy"
    # Tail (a[replan_steps:]) of the last VLA chunk, kept for the boundary blend.
    # None = no valid predecessor (first chunk, or after human/reset): blend skipped.
    prev_tail = None
    last_q = None  # scalar critic Q of the last sampled action (for thor preview overlay)
    combine_rng = jax.random.PRNGKey(FLAGS.seed + 100)

    # --- Async update thread setup ---
    # Actor (main thread) samples on device[0], learner (background thread)
    # updates on device[1:]. Params published via atomic reference swap.
    _actor_agent = agent.cache_infer_params()
    sample_sharding = jax.sharding.SingleDeviceSharding(sample_device)
    _actor_agent.actor.replicated_sharding = sample_sharding
    _published = [None]
    _publish_lock = threading.Lock()
    _buffer_lock = threading.Lock()
    _stop_event = threading.Event()
    _can_update = threading.Event()
    _episode_done = threading.Event()
    _update_count = [0]
    _ckpt_request = [None]  # main thread sets step number; update thread saves and clears
    _ckpt_done = threading.Event()
    # Single wandb writer: the update thread stashes its metrics here and the
    # main rollout thread flushes them at step=i. Two threads each calling
    # wandb.log() with independent explicit steps races the monotonic step
    # counter and silently drops the background thread's rows — that is why
    # training/critic_loss never landed in wandb.
    _log_lock = threading.Lock()
    _pending_train_log = {}

    def _update_worker():
        nonlocal combine_rng
        learner_agent = agent
        last_episode_time = time.time()
        update_time = deque(maxlen=10)

        _can_update.wait()
        while not _stop_event.is_set():
            try:
                if FLAGS.ep_timeout_secs > 0 and time.time() - last_episode_time > FLAGS.ep_timeout_secs:
                    logging.info("No episode finished for %.1fs, pausing updates.", FLAGS.ep_timeout_secs)
                    with _log_lock:
                        _pending_train_log["training/update_paused"] = 1
                    _episode_done.wait()
                    _episode_done.clear()
                    last_episode_time = time.time()
                    logging.info("Episode signal received, resuming updates.")
                    with _log_lock:
                        _pending_train_log["training/update_paused"] = 0

                if _episode_done.is_set():
                    _episode_done.clear()
                    last_episode_time = time.time()

                with _buffer_lock:
                    batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)
                batch_info = get_batch_info(batch)

                t0 = time.time()
                learner_agent = learner_agent.replace(
                    rng=jax.device_put(learner_agent.rng, replicated_sharding)
                )
                learner_agent, update_info = learner_agent.update(
                    learner_agent, batch, FLAGS.utd_ratio, actor_batch
                )

                with _publish_lock:
                    _published[0] = learner_agent

                update_time.append(time.time() - t0)
                _update_count[0] += 1
                log_dict = {f"training/{k}": v for k, v in update_info.items()}
                log_dict["batch_info"] = batch_info
                log_dict["training/num_updates"] = _update_count[0]
                if _update_count[0] % 10 == 0 and len(update_time) == update_time.maxlen:
                    log_dict["training/update_time_avg_ms"] = float(np.mean(update_time)) * 1000.0
                with _log_lock:
                    _pending_train_log.update(log_dict)

                ckpt_step = _ckpt_request[0]
                if ckpt_step is not None:
                    _ckpt_request[0] = None
                    try:
                        save_checkpoint(checkpoint_manager, learner_agent, ckpt_step)
                        logging.info("Saved agent checkpoint at step %d (from update thread)", ckpt_step)
                    except Exception as e:
                        logging.error("Could not save model checkpoint: %s", e)
                    _ckpt_done.set()
            except Exception:
                logging.exception("Update thread crashed at update %d", _update_count[0])
                break
        # Handle checkpoint request after stop signal
        ckpt_step = _ckpt_request[0]
        if ckpt_step is not None:
            _ckpt_request[0] = None
            try:
                save_checkpoint(checkpoint_manager, learner_agent, ckpt_step)
                logging.info("Saved agent checkpoint at step %d (from update thread)", ckpt_step)
            except Exception as e:
                logging.error("Could not save checkpoint: %s", e)
            _ckpt_done.set()
        logging.info("Update thread exiting (updates=%d).", _update_count[0])

    _update_thread = threading.Thread(target=_update_worker, daemon=True)
    _update_thread.start()

    if resuming and training_log.ep_count >= 10 and replay_buffer._size >= FLAGS.batch_size:
        _can_update.set()
        logging.info("Resuming: replay buffer already warm, update thread starting immediately.")

    # Raw per-step metrics CSV for offline analysis (Q curve etc.). Written only
    # by this main thread — same single-writer discipline as wandb — so rows stay
    # ordered by env step. Append mode is resume-friendly; header on empty file.
    _metrics_csv_path = os.path.join(log_dir, "train_metrics.csv")
    _metrics_cols = [
        "step", "num_updates", "critic_loss", "q", "q_min", "q_max",
        "target_q_mean", "next_q_nan_ratio", "critic_grad_norm",
        "temperature", "entropy", "q_selected", "loop_time_ms",
    ]
    _metrics_csv = open(_metrics_csv_path, "a", newline="")
    _metrics_writer = csv.DictWriter(_metrics_csv, fieldnames=_metrics_cols, extrasaction="ignore")
    if _metrics_csv.tell() == 0:
        _metrics_writer.writeheader()
        _metrics_csv.flush()

    for i in tqdm.tqdm(
        range(start_step, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        loop_start = time.time()
        step_metrics = {}

        with _publish_lock:
            new_agent = _published[0]
            _published[0] = None
        if new_agent is not None:
            # Free the old GPU0 inference cache BEFORE building the new one. On
            # dual-GPU, cache_infer_params device_puts the ~3.3B actor GPU1->GPU0;
            # building it while the previous cache is still held by _actor_agent
            # makes old + new coexist on the sampling card and OOMs. Dropping the old
            # cache first (single-threaded here) lets jax free those GPU0 buffers
            # before the new copy is allocated. Costs ~0.4s/transfer, once per update.
            _actor_agent = _actor_agent.replace(_infer_cache=None)
            # Keep the sampler on its OWN rng. The update thread donates its
            # agent's buffers on the next update() (donate_argnums); the published
            # agent's rng is among them (the device_put at ~L332 is a no-op on the
            # already-replicated key, so it is NOT a defensive copy). The sampler
            # holds rng as a live reference and uses it steps later in
            # sample_actions (after thor round-trips), so adopting the donated rng
            # crashes with "Array has been deleted uint32[2]". Carry our own
            # independent, already-advanced rng across the swap instead.
            new_agent = new_agent.replace(rng=_actor_agent.rng)
            _actor_agent = new_agent.cache_infer_params()

        observation = env.get_observation()
        done, success, reward, mask = env.get_info_for_step()

        # Skip model inference while human is controlling.
        if not action_plan and action_type != "human":
            sample_start = time.time()
            action_chunk, _actor_agent, new_si = _actor_agent.sample_actions(
                observation, only_base_actions=FLAGS.only_base_actions)
            if i == start_step:
                _a = np.asarray(action_chunk[0])
                logging.info("[ACTION-DIAG] step0 first action: xyz(m)=%s grip=%s finite=%s",
                             _a[:3].tolist(), float(_a[9]) if _a.shape[0] > 9 else None,
                             bool(np.isfinite(_a).all()))
            episode_log.sample_info_history.append(new_si)
            episode_log.step_data_history.append(capture_step_data(new_si))
            last_q = new_si.get("q_selected", last_q)
            training_log.record_sample_time(time.time() - sample_start, step_metrics)
            # Chunk-seam blend: cancel the "退一步" reversal by ensembling the new
            # chunk's first waypoints with the discarded tail of the previous chunk
            # (which continues forward). b_k aligns with prev_tail[k] (=a_{replan+k}).
            chunk = np.asarray(action_chunk, dtype=np.float64)  # (action_horizon, action_dim)
            exec_chunk = chunk[:FLAGS.replan_steps].copy()
            if prev_tail is not None and FLAGS.boundary_blend_w0 > 0.0:
                n = min(FLAGS.boundary_blend_steps, len(prev_tail), len(exec_chunk))
                for k in range(n):
                    w = FLAGS.boundary_blend_w0 * (1.0 - k / FLAGS.boundary_blend_steps)
                    exec_chunk[k, 0:3] = (1.0 - w) * exec_chunk[k, 0:3] + w * prev_tail[k, 0:3]
            prev_tail = chunk[FLAGS.replan_steps:].copy()  # a[replan:] for next seam
            action_plan.extend(exec_chunk)
        else:
            episode_log.sample_info_history.append(episode_log.sample_info_history[-1] if episode_log.sample_info_history else None)
            episode_log.step_data_history.append(episode_log.step_data_history[-1] if episode_log.step_data_history else {})

        elapsed = time.time() - start_step_time
        if elapsed < dt:
            time.sleep(dt - elapsed)

        has_action = bool(action_plan)
        action = action_plan.popleft() if has_action else np.zeros_like(example_action.squeeze())
        real_action, action_type = env.step(action.tolist(), q_value=last_q)
        start_step_time = time.time()

        episode_log.record_step(observation, len(action_plan), action_type, real_action, reward)

        if action_type == "human":
            action_plan.clear()
            prev_tail = None  # human moved the arm; last chunk's tail is stale

        if has_action or action_type == "human":
            transition_dict = dict(
                observations=observation,
                actions=real_action,
                rewards=reward,
                masks=mask,
                dones=done,
                is_hil=(action_type == "human"),
            )
            with _buffer_lock:
                batch_processor.insert_transition(transition_dict)

        if done:
            with _buffer_lock:
                batch_processor.on_episode_done(success)
            _episode_done.set()
            env.reset()

            training_log.on_episode_done(episode_log, success, step_metrics)

            # ── Online RL visualizations (Q dist, sampling, edit) ─────────
            try:
                vis_images = log_episode_visualizations(
                    episode_log, success,
                    save_dir=os.path.join(log_dir, "visualizations"),
                )
                for tag, path in vis_images:
                    step_metrics[tag] = wandb.Image(path)
            except Exception as e:
                logging.warning("Visualization failed (non-fatal): %s", e)

            episode_log.reset()
            with _buffer_lock:
                batch_processor.on_episode_start()

            observation = env.get_observation()
            done = False
            action_type = "policy"
            action_plan.clear()
            prev_tail = None  # new episode; no predecessor chunk to blend

            # don't wait 10 episode to start updating as compiling is slow
            if not _can_update.is_set() and replay_buffer._size >= FLAGS.batch_size:
                _can_update.set()
                logging.info("Replay buffer ready (ep_count=%d), starting update thread.", training_log.ep_count)

        if FLAGS.checkpoint_model and FLAGS.checkpoint_interval > 0 and i > 0 and i % FLAGS.checkpoint_interval == 0:
            _ckpt_done.clear()
            _ckpt_request[0] = i

        if FLAGS.checkpoint_buffer and (has_action or action_type == "human"):
            try:
                save_replay_buffer_transition(checkpoint_dir_path, transition_dict, step=i)
            except Exception:
                logging.exception("Could not save agent buffer.")

        with _log_lock:
            if _pending_train_log:
                step_metrics.update(_pending_train_log)
                _pending_train_log.clear()
        if last_q is not None:
            step_metrics["training/q_selected"] = float(last_q)
        step_metrics["training/loop_time_ms"] = (time.time() - loop_start) * 1000.0
        wandb.log(step_metrics, step=i)

        # Mirror the analysis-relevant scalars to CSV (coerce jax/numpy 0-d to float).
        _row = {"step": i}
        for _k, _v in step_metrics.items():
            _kk = _k[len("training/"):] if _k.startswith("training/") else _k
            if _kk not in _metrics_cols:
                continue
            try:
                _arr = np.asarray(_v)
                if _arr.ndim == 0:
                    _row[_kk] = float(_arr)
            except (TypeError, ValueError):
                pass
        _metrics_writer.writerow(_row)
        _metrics_csv.flush()

    if FLAGS.checkpoint_model:
        _ckpt_done.clear()
        _ckpt_request[0] = FLAGS.max_steps
    _stop_event.set()
    _can_update.set()
    _episode_done.set()
    _update_thread.join()
    _metrics_csv.close()

    if FLAGS.checkpoint_model:
        logging.info("Waiting for checkpoint manager to finish")
        checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    app.run(main)