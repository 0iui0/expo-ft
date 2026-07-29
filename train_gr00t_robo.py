#! /usr/bin/env python
"""EXPO-FT online RL training with GR00T N1.7 VLA actor.

Usage:
    python train_gr00t_robo.py \
        --config configs/model/expo_ft_gr00t_config.py \
        --config_task configs/task/cr5af.py \
        --dataset_path /datasets/cr5af_grasp_housing \
        --gr00t_model_path /path/to/finetuned/checkpoint
"""
import os
import logging
import time
from collections import deque

import numpy as np
import tqdm
from absl import app, flags

from ml_collections import config_flags

import jax
import etils.epath as epath
import openpi.training.sharding as _sharding

import wandb
from expo_ft.agents import initialize_checkpoint_dir
from expo_ft.data.gr00t_replay_buffer import create_gr00t_replay_buffer
from expo_ft.data.gr00t_batch_processor import Gr00tBatchProcessor
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.env.droid_utils import process_droid_dataset
from expo_ft.utils.log_utils import EpisodeState, TrainingStats
from expo_ft.utils.train_utils import get_batch_info, init_logging, init_wandb

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "expo-ft-gr00t", "wandb project name.")
flags.DEFINE_string("run_name", None, "Optional wandb run name.")
flags.DEFINE_float("offline_ratio", 0.0, "Offline batch fraction; 0 inserts dataset into online replay buffer.")
flags.DEFINE_string("deploy_inference", "", "Use deploy inference server at host:port (e.g. 'localhost:5555') instead of local GR00T inference.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_enum("update_type", "episode", ["episode", "step", "batch"], "When to run gradient updates.")
flags.DEFINE_integer("num_updates", 1, "Number of gradient updates per trigger.")
flags.DEFINE_integer("num_batch", 1, "Number of episodes per update batch.")
flags.DEFINE_integer("batch_size", 64, "Mini batch size.")
flags.DEFINE_integer("max_steps", 100_000, "Number of training steps.")
flags.DEFINE_integer("num_data", 0, "Max number of offline demo episodes to load (0 = all).")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint.")
flags.DEFINE_integer("checkpoint_interval", 0, "Save checkpoint every N steps.")
flags.DEFINE_boolean("checkpoint_buffer", False, "Save replay buffer on evaluation.")
flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")
flags.DEFINE_integer("keep_period", None, "Keep checkpoints every N steps.")
flags.DEFINE_boolean("overwrite", False, "Overwrite existing checkpoint directory.")
flags.DEFINE_boolean("resume", False, "Resume training from checkpoint.")
flags.DEFINE_string("output_dir", "./logs", "Directory for logs and checkpoints.")
flags.DEFINE_integer("fsdp_devices", 1, "Number of FSDP devices for sharding.")

flags.DEFINE_string("client_host", "localhost", "Host for environment operations server.")
flags.DEFINE_integer("client_port", 8102, "Port for environment operations server.")

flags.DEFINE_integer("replan_steps", 8, "Number of replan steps for evaluation.")

# GR00T-specific flags
flags.DEFINE_string("dataset_path", "", "Path to the offline dataset.")
flags.DEFINE_string("gr00t_model_path", "", "Path to the finetuned GR00T checkpoint.")

config_flags.DEFINE_config_file(
    "config",
    "configs/model/expo_ft_gr00t_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_task",
    "configs/task/cr5af.py",
    "File path to the task configuration.",
    lock_config=False,
)


def main(_):
    global _deploy_socket
    init_logging()

    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path("~/.cache/jax").expanduser()),
    )

    # Override config model path if provided as flag
    if FLAGS.gr00t_model_path:
        FLAGS.config.gr00t_model_path = FLAGS.gr00t_model_path

    # Mesh / sharding. Mirror the pi05 path (train_pi_robo.py) but fall back to
    # SingleDeviceSharding on a single GPU: a (1,1) NamedSharding mesh makes the
    # GR00T learner's eager jax.grad raise "device_assignment cannot be None" on
    # this jaxlib build. fsdp_sharding (called inside EXPOLearner.create) treats
    # mesh=None as single-device replication.
    if jax.device_count() == 1:
        mesh = None
        _dev = jax.devices()[0]
        data_sharding = jax.sharding.SingleDeviceSharding(_dev)
        replicated_sharding = jax.sharding.SingleDeviceSharding(_dev)
    else:
        mesh = _sharding.make_mesh(FLAGS.fsdp_devices)
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(_sharding.DATA_AXIS)
        )
        replicated_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec()
        )

    log_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name or "gr00t")
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

    # --- Load offline dataset ---
    if FLAGS.config_task.env_type in ("droid", "sim"):
        dataset = process_droid_dataset(
            FLAGS.dataset_path,
            FLAGS.config_task,
            num_data=FLAGS.num_data,
        )
        example_action = dataset[0]["actions"][np.newaxis]
    else:
        raise ValueError(f"Unsupported dataset type: {FLAGS.config_task.env_type}")

    # --- Build GR00T actor FIRST (model load is slow; env connect is deferred) ---
    from expo_ft.agents.vla.gr00t_agent import build_gr00t
    from expo_ft.agents.alg.expo_ft_gr00t import EXPOLearnerGR00T, load_agent, restore_checkpoint, save_checkpoint

    task_description = getattr(FLAGS.config_task, "language_instruction",
                                "grasp motor shaft and insert into bushing")
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_gr00t(
        FLAGS.config, FLAGS.seed, mesh, data_sharding, replicated_sharding,
        resuming, task_description,
    )

    # --- Create environment (connects to env server — retry loop until up) ---
    train_env_creation_request = {
        "example_action": example_action,
        "env_usage": "train",
        "video_dir": train_video_dir,
    }
    logging.info("Creating environment (will wait for env server at %s:%d)...",
                 FLAGS.client_host, FLAGS.client_port)
    env = EnvClientWrapper(
        env_creation_request=train_env_creation_request,
        host=FLAGS.client_host,
        port=FLAGS.client_port,
    )
    env.reset()
    logging.info("Created training environment %s", env.env_id)

    # --- Create replay buffers ---
    rb_args = dict(
        config=FLAGS.config,
        example_action=example_action,
        capacity=FLAGS.max_steps,
        task_description=env.task_description,
        replan_steps=FLAGS.replan_steps,
        seed=FLAGS.seed,
    )
    replay_buffer = create_gr00t_replay_buffer(**rb_args)
    offline_replay_buffer = create_gr00t_replay_buffer(**rb_args)

    actor_success_only = getattr(FLAGS.config, "actor_success_only", False)
    batch_processor = Gr00tBatchProcessor(
        replay_buffer=replay_buffer,
        offline_replay_buffer=offline_replay_buffer,
        batch_size=FLAGS.batch_size,
        utd_ratio=FLAGS.utd_ratio,
        offline_ratio=FLAGS.offline_ratio,
        actor_success_only=actor_success_only,
        dataset=dataset,
    )
    # Snapshot save/restore: the buffer already holds the offline dataset;
    # everything inserted after this point is online (HIL + policy) data.
    _online_snapshot_start = int(replay_buffer._insert_index)

    # Restore online data from a previous session's snapshot (if it exists).
    # This means HIL data survives training crashes — no more lost episodes.
    _snapshot_path = os.path.join(checkpoint_dir, "buffers", "online_snapshot.npz")
    if os.path.exists(_snapshot_path):
        n = replay_buffer.restore_online_from(_snapshot_path)
        if n > 0:
            logging.info("Restored %d online transitions from snapshot.", n)
            # Recalculate online_start after restore
            _online_snapshot_start = int(replay_buffer._insert_index)

    # --- Create EXPOLearnerGR00T ---
    # Build example observation/action/state for agent init
    # Use format compatible with GR00T replay buffer layout
    agent_example = {
        "image": {k: offline_replay_buffer.dataset_dict[f"image_{k}"][0][np.newaxis] for k in actor._video_keys},
        "state": offline_replay_buffer.dataset_dict["state"][0][np.newaxis],
        "actions": offline_replay_buffer.dataset_dict["actions"][0][np.newaxis],
    }
    # Ground-truth action dim from buffer storage
    env_action_dim = offline_replay_buffer._env_action_dim
    env_state_dim = offline_replay_buffer._env_state_dim

    # Convert to critic format for agent init
    agent_example_obs, agent_example_state, agent_example_action = (
        offline_replay_buffer.convert_to_critic_format(agent_example)
    )
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]

    agent = load_agent(
        seed=FLAGS.seed,
        example_observation=agent_example_obs.squeeze(),
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
        agent = agent.cache_infer_params()
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
    logging.info("Resuming: ep_count set to %d", training_log.ep_count)

    # If --deploy_inference is set, connect to the deploy inference server
    # (Gr00tPolicy on GPU0, port 5555) so rollout actions exactly match deploy.
    _deploy_socket = None
    if FLAGS.deploy_inference:
        import zmq
        _deploy_socket = zmq.Context().socket(zmq.REQ)
        _deploy_socket.connect(f"tcp://{FLAGS.deploy_inference}")
        _deploy_socket.setsockopt(zmq.RCVTIMEO, 5000)
        _deploy_socket.setsockopt(zmq.SNDTIMEO, 1000)
        logging.info("Using deploy inference server at %s", FLAGS.deploy_inference)

    batch_processor.on_episode_start()

    dt = 1.0 / FLAGS.config_task.control_hz
    done = False
    env.reset()
    start_step_time = time.time()
    action_plan = deque()
    action_type = "policy"
    episodes_since_update = 0
    combine_rng = jax.random.PRNGKey(FLAGS.seed + 100)

    def run_agent_updates(num_updates: int, metrics: dict):
        nonlocal agent, combine_rng
        for _ in range(num_updates):
            update_start = time.time()
            batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)

            metrics["batch_info"] = get_batch_info(batch)

            agent = agent.replace(rng=jax.device_put(agent.rng, jax.devices()[0]))
            agent, update_info = agent.update(agent, batch, FLAGS.utd_ratio, actor_batch)
            training_log.record_update_time(time.time() - update_start, metrics)
            for k, v in update_info.items():
                metrics[f"training/{k}"] = v

    for i in tqdm.tqdm(
        range(start_step, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        loop_start = time.time()
        step_metrics = {}

        observation = env.get_observation()
        done, success, reward, mask = env.get_info_for_step()

        # Convert env observation to GR00T format (video.* / state.* keys)
        # for the GR00T actor and critic_inputs_from_observation.
        gr00t_obs = actor.convert_env_obs_to_gr00t(
            observation,
            side_camera_id=FLAGS.config_task.side_camera_id,
            wrist_camera_id=FLAGS.config_task.wrist_camera_id,
        )

        if not action_plan and action_type != "human":
            sample_start = time.time()
            if _deploy_socket is not None:
                # Use deploy's exact inference pipeline via ZMQ.
                # Build observation in the format deploy_cr5af_gripper expects.
                deploy_obs = {
                    "video": {
                        view: np.stack([
                            np.asarray(gr00t_obs[f"video.{view}"])] * 2, axis=0
                        )[None, ...]  # (1, 2, H, W, C)
                        for view in actor._video_keys
                    },
                    "state": {
                        k: np.asarray(gr00t_obs[f"state.{k}"]).reshape(-1)[None, None, :]
                        for k in actor._state_keys
                    },
                    "language": {"annotation.human.task_description": [[task_description]]},
                }
                import msgpack, msgpack_numpy
                msgpack_numpy.patch()
                _deploy_socket.send(msgpack.packb({"observation": deploy_obs}))
                raw = _deploy_socket.recv()
                response = msgpack.unpackb(raw)
                action_dict = {
                    k: np.array(v, dtype=np.float32).squeeze(0) for k, v in response.items()
                }
                # eef_9d from deploy is (T,9) in mm; graft into action format
                abs_eef = action_dict.get("eef_9d", np.zeros((10, 9), dtype=np.float32))
                abs_j = action_dict.get("joint_pos", np.zeros((10, 6), dtype=np.float32))
                abs_g = action_dict.get("gripper_pos", np.zeros((10, 1), dtype=np.float32))
                action_chunk = np.concatenate([abs_eef, abs_j, abs_g], axis=-1)  # (T, 16), mm
                new_si = {}
            else:
                action_chunk, agent, new_si = agent.sample_actions(gr00t_obs)
            episode_log.sample_info_history.append(new_si)
            training_log.record_sample_time(time.time() - sample_start, step_metrics)
            action_plan.extend(action_chunk[:FLAGS.replan_steps])
        else:
            episode_log.sample_info_history.append(
                episode_log.sample_info_history[-1] if episode_log.sample_info_history else None
            )

        elapsed = time.time() - start_step_time
        if elapsed < dt:
            time.sleep(dt - elapsed)

        has_action = bool(action_plan)
        action = action_plan.popleft() if has_action else np.zeros_like(
            FLAGS.config_task.example_action.squeeze()
        )
        real_action, action_type = env.step(action.tolist())
        start_step_time = time.time()

        episode_log.record_step(gr00t_obs, len(action_plan), action_type, real_action, reward)

        if action_type == "human":
            action_plan.clear()

        if has_action or action_type == "human":
            # Build transition in Gr00tReplayBuffer.insert() format:
            # {"image": {view: (H,W,3) uint8}, "state": (D,) float32, ...}
            rb_image = {
                view: gr00t_obs[f"video.{view}"]
                for view in actor._video_keys
            }
            rb_state = np.concatenate(
                [gr00t_obs[f"state.{k}"].reshape(-1) for k in actor._state_keys]
            ).astype(np.float32)
            transition_dict = dict(
                image=rb_image,
                state=rb_state,
                actions=real_action,
                rewards=reward,
                masks=mask,
                dones=done,
                is_hil=(action_type == "human"),
            )
            batch_processor.insert_transition(transition_dict)

        can_update = training_log.ep_count >= 10 and i >= FLAGS.batch_size
        if FLAGS.update_type == "step" and can_update:
            run_agent_updates(FLAGS.num_updates, step_metrics)

        if done:
            logging.info(
                "Episode done at step %d: success=%s, ep_count=%d, "
                "online_episodes=%d, update_threshold=%d",
                i, success, training_log.ep_count + 1,
                replay_buffer.count_episodes_chronological(), 10,
            )
            batch_processor.on_episode_done(success)
            env.reset()

            # Persist online buffer snapshot so HIL data survives crashes.
            # Saved every episode; size ~ few hundred transitions → fast.
            try:
                snapshot_dir = os.path.join(checkpoint_dir, "buffers")
                os.makedirs(snapshot_dir, exist_ok=True)
                replay_buffer.save_snapshot(
                    os.path.join(snapshot_dir, "online_snapshot.npz"),
                    online_start=_online_snapshot_start,
                )
            except Exception as e:
                logging.warning("Buffer snapshot save failed: %s", e)

            if FLAGS.update_type == "episode" and can_update:
                for _ in tqdm.tqdm(range(FLAGS.num_updates)):
                    run_agent_updates(1, step_metrics)
            elif FLAGS.update_type == "batch" and can_update:
                episodes_since_update += 1
                if episodes_since_update >= FLAGS.num_batch:
                    for _ in tqdm.tqdm(range(FLAGS.num_updates)):
                        run_agent_updates(1, step_metrics)
                    episodes_since_update = 0

            training_log.on_episode_done(episode_log, success, step_metrics)
            episode_log.reset()
            batch_processor.on_episode_start()

            observation = env.get_observation()
            done = False
            action_type = "policy"
            action_plan.clear()

        if FLAGS.checkpoint_model and FLAGS.checkpoint_interval > 0 and i > 0 and i % FLAGS.checkpoint_interval == 0:
            try:
                save_checkpoint(checkpoint_manager, agent, i)
                logging.info("Saved agent checkpoint at step %d", i)
            except Exception as e:
                logging.error("Could not save model checkpoint: %s", e)

        step_metrics["training/loop_time_ms"] = (time.time() - loop_start) * 1000.0
        wandb.log(step_metrics, step=i)

    if FLAGS.checkpoint_model:
        try:
            save_checkpoint(checkpoint_manager, agent, FLAGS.max_steps)
            logging.info("Saved final agent checkpoint at step %d", FLAGS.max_steps)
        except Exception as e:
            logging.error("Could not save final checkpoint: %s", e)
        logging.info("Waiting for checkpoint manager to finish")
        checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    app.run(main)
