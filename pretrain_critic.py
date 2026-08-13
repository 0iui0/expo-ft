#! /usr/bin/env python
"""Robot-free offline critic warm-up for PI0.5 EXPO-FT.

Warms the OTF critic (and residual actor) on the offline success demos with the
base Pi0.5 VLA FROZEN (`freeze_base_actor=True`), so route ② can later resume a
checkpoint whose Q already discriminates progress instead of a cold, flat critic.

No env, no thor, no robot: the prompt comes from the task config
(`config_task.language_instruction`), the data from the offline demos. Run on the
free GPU only, e.g.:

    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 JAXTYPING_DISABLE=1 \
    .venv/bin/python pretrain_critic.py \
        --config configs/model/expo_ft_pi_config.py \
        --config_task configs/task/cr5af_pi05.py \
        --dataset_path <demos> --run_name pi05_critic_pretrain \
        --pretrain_steps 3000 --checkpoint_interval 500 --output_dir ./logs

Caveat: the demos are all successes (terminal +1). TD warms the critic to a
stable, high Q but cannot teach it to discriminate good from bad without failures
— that sharpening happens online in route ②. Watch target_q_mean climb off ~-0.9
and plateau; pick a checkpoint from there.
"""
import os
import logging

# Same sm_120 XLA autotune workaround as the trainers; must precede `import jax`.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
import csv

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

import jax
import etils.epath as epath

import wandb
from expo_ft.agents import initialize_checkpoint_dir
from expo_ft.data.replay_buffer import create_replay_buffer
from expo_ft.data.batch_processor import BatchProcessor
from expo_ft.env.droid_utils import process_droid_dataset, process_cr5af_npz_pi05
from expo_ft.utils.train_utils import get_batch_info, init_logging, init_wandb

import openpi.training.sharding as openpi_sharding

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "expo-ft", "wandb project name.")
flags.DEFINE_string("run_name", None, "Optional wandb run name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 16, "Mini batch size. 64 OOMs a single 32GB GPU: update_critic samples next actions through the frozen 3.3B Pi0.5 (batch*N flow passes). Keep small — this is single-GPU (no sample/update split).")
flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")
flags.DEFINE_integer("pretrain_steps", 3000, "Number of offline critic-warmup updates.")
flags.DEFINE_integer("num_data", 0, "Max number of offline demo episodes to load (0 = all).")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_integer("checkpoint_interval", 500, "Save agent checkpoint every N steps (0 = end only).")
flags.DEFINE_integer("keep_period", None, "Keep checkpoints every N steps.")
flags.DEFINE_boolean("overwrite", False, "Overwrite existing checkpoint directory.")
flags.DEFINE_boolean("resume", False, "Resume warm-up from checkpoint.")
flags.DEFINE_string("output_dir", "./logs", "Directory for logs and checkpoints.")
flags.DEFINE_integer("fsdp_devices", 1, "Number of FSDP devices for sharding.")
flags.DEFINE_integer("replan_steps", 8, "Number of replan steps (TD n-step / chunk size).")
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


def main(_):
    init_logging()

    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path("~/.cache/jax").expanduser()),
    )

    # No sample/update split: every visible device is part of the update mesh.
    update_devices = jax.devices()
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
    logging.info("Device layout: offline critic warm-up on %s", update_devices)

    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    log_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name)
    os.makedirs(log_dir, exist_ok=True)
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

    # Prompt from config (no thor round-trip). Data from the offline demos.
    task_description = FLAGS.config_task.language_instruction
    if FLAGS.config_task.env_type in ('droid', 'sim'):
        if getattr(FLAGS.config, "use_pi05", False):
            dataset = process_cr5af_npz_pi05(
                FLAGS.dataset_path,
                task_description,
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

    if FLAGS.config.model_cls != "EXPOLearner":
        raise ValueError(f"pretrain_critic only supports EXPOLearner, got {FLAGS.config.model_cls}")
    from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint, save_checkpoint

    from expo_ft.agents.vla.pi05 import build_pi05
    # Freeze the base VLA for the whole warm-up: only the critic + residual adapt,
    # so the saved actor_params stay byte-identical to the SFT prior.
    FLAGS.config.freeze_base_actor = True
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        FLAGS.config, FLAGS.seed, mesh, data_sharding, replicated_sharding,
        resuming, task_description,
    )

    rb_args = dict(
        config=FLAGS.config,
        example_action=example_action,
        capacity=max(FLAGS.pretrain_steps, len(dataset)),
        task_description=task_description,
        replan_steps=FLAGS.replan_steps,
        seed=FLAGS.seed,
    )
    replay_buffer = create_replay_buffer(**rb_args)
    # offline_ratio=0 -> the offline buffer is never sampled, so allocate a 1-slot
    # stub instead of a second full-capacity buffer. Each full buffer is ~90GB of
    # images at 265 episodes (443KB/slot x ~213k transitions); the duplicate was
    # the host-RAM blowup. BatchProcessor still requires the arg to exist.
    offline_replay_buffer = create_replay_buffer(**{**rb_args, "capacity": 1})

    # offline_ratio=0 seeds the demos into the online replay buffer; next_batch then
    # samples them directly (functionally offline — all data is demos). Base frozen,
    # so no actor batch is needed (actor_success_only=False skips that sampling).
    batch_processor = BatchProcessor(
        replay_buffer=replay_buffer,
        offline_replay_buffer=offline_replay_buffer,
        data_sharding=data_sharding,
        batch_size=FLAGS.batch_size,
        utd_ratio=FLAGS.utd_ratio,
        offline_ratio=0.0,
        actor_success_only=False,
        use_dagger_hil_sampling=False,
        dataset=dataset,
    )

    agent_example_observation, agent_example_state, agent_example_action = replay_buffer.convert_to_critic_format(
        {
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
        default_prompt=task_description,
        residual_action_xyzg=FLAGS.config_task.residual_action_xyzg,
    )

    start_step = 0
    if resuming:
        agent = restore_checkpoint(checkpoint_manager, agent)
        steps = tuple(checkpoint_manager.all_steps())
        if steps:
            start_step = max(steps)
            logging.info("Resuming warm-up from step %d", start_step)

    # Raw per-step metrics CSV for offline analysis of the Q warm-up curve.
    _metrics_cols = [
        "step", "critic_loss", "q", "q_min", "q_max", "target_q_mean",
        "next_q_nan_ratio", "critic_grad_norm", "temperature", "entropy",
    ]
    _metrics_csv = open(os.path.join(log_dir, "train_metrics.csv"), "a", newline="")
    _metrics_writer = csv.DictWriter(_metrics_csv, fieldnames=_metrics_cols, extrasaction="ignore")
    if _metrics_csv.tell() == 0:
        _metrics_writer.writeheader()
        _metrics_csv.flush()

    combine_rng = jax.random.PRNGKey(FLAGS.seed + 100)
    logging.info("Starting offline critic warm-up: %d steps (base actor frozen).", FLAGS.pretrain_steps)
    for i in tqdm.tqdm(
        range(start_step, FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)
        step_metrics = {"batch_info": get_batch_info(batch)}
        agent = agent.replace(rng=jax.device_put(agent.rng, replicated_sharding))
        agent, update_info = agent.update(agent, batch, FLAGS.utd_ratio, actor_batch)
        for k, v in update_info.items():
            step_metrics[f"training/{k}"] = v
        wandb.log(step_metrics, step=i)

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

        if FLAGS.checkpoint_interval > 0 and i > start_step and i % FLAGS.checkpoint_interval == 0:
            try:
                save_checkpoint(checkpoint_manager, agent, i)
                logging.info("Saved warm-up checkpoint at step %d", i)
            except Exception as e:
                logging.error("Could not save checkpoint at step %d: %s", i, e)

    try:
        save_checkpoint(checkpoint_manager, agent, FLAGS.pretrain_steps)
        logging.info("Saved final warm-up checkpoint at step %d", FLAGS.pretrain_steps)
    except Exception as e:
        logging.error("Could not save final checkpoint: %s", e)
    checkpoint_manager.wait_until_finished()
    _metrics_csv.close()
    logging.info("Critic warm-up done. Resume route ② with --resume from %s", checkpoint_dir)


if __name__ == "__main__":
    app.run(main)
