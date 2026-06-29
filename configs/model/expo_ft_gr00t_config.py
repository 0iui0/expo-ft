"""EXPO-FT training config for GR00T N1.7 VLA actor.

This config is analogous to ``expo_ft_pi_config.py`` but uses the GR00T N1.7
model (via ``Gr00tAgent``) as the VLA actor instead of Pi05.

Key differences:
- ``model_cls = "EXPOLearnerGR00T"`` — runs the update loop eagerly
  (no JAX jit) so the PyTorch-based GR00T actor can participate.
- All ``pi05_*`` fields are replaced by ``gr00t_*`` fields.
- No OpenPI config or assets needed — GR00T loads its own processor/statistics.
"""

from configs.model import sac_config


def get_config():
    config = sac_config.get_config()

    # Use the EXPOLearnerGR00T variant (eager-mode updates for PyTorch actor)
    config.model_cls = "EXPOLearnerGR00T"

    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm = True

    config.N = 8
    config.n_edit_samples = 8

    config.adjust_target_entropy = False
    config.entropy_scale = 1.0
    config.edit_scale = 0.2
    config.actor_drop = 0.0
    config.actor_lr = 3e-5  # GR00T finetune lr is typically lower than pi05
    config.critic_lr = 3e-4

    config.latent_dim_image = 512
    config.latent_dim_state = 64
    config.include_state = True
    config.encoder_stage_sizes = (3, 4, 6, 3)
    config.encoder_num_filters = 64
    config.hidden_dims = (256, 256, 256)

    config.encode_batch_split = 1
    config.batch_split = 1

    # --- GR00T-specific fields ---

    # Path to the finetuned GR00T checkpoint directory
    config.gr00t_model_path = ""

    # Embodiment tag registered by the modality config (e.g. "NEW_EMBODIMENT")
    config.gr00t_embodiment_tag = "NEW_EMBODIMENT"

    # If True, freeze GR00T backbone (faster training, lower VRAM)
    config.freeze_gr00t_backbone = False

    config.freeze_critic_encoder = False

    config.actor_success_only = True

    # GR00T handles its own image augmentation internally via the processor;
    # the EXPO data-augmentation pipeline is disabled to avoid double-processing.
    # Note: this applies to the ACTOR path only. Critic augmentation is applied
    # inside EXPOLearnerGR00T.update_critic using the same augmentation function.
    config.use_full_augmentation = True

    # --- GR00T modality keys (must match the embodiment's modality config) ---
    # Camera view keys in canonical order (used for critic input concatenation
    # via ``critic_inputs_from_observation`` and replay buffer storage).
    config.gr00t_camera_keys = ["hand_view", "table_view"]

    # State modality keys in canonical order (used to split/assemble flat state).
    config.gr00t_state_keys = ["eef_9d", "joint_pos", "gripper_pos"]

    # Action modality keys in canonical order.
    config.gr00t_action_keys = ["eef_9d", "joint_pos", "gripper_pos"]

    # --- Latency / VRAM tuning ---
    # Gradient checkpointing: recomputes activations during backward to reduce
    # peak VRAM ~25-30% at the cost of ~15% more wall time per training step.
    config.use_gradient_checkpointing = False

    # --- Async safety ---
    # Model access lock: when using async learner/actor mode, serializes
    # optimizer.step() with inference forward() to prevent parameter reads
    # during writes. Disabled by default (brief write windows are acceptable
    # for most setups). Enable if you see NaN losses or policy collapse.
    config.use_model_lock = False

    return config
