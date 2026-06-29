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
    config.use_full_augmentation = False

    return config
