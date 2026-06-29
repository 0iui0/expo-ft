"""CR5AF task config: Doosan CR5AF arm with TopHand dexterous hand.

This task config is compatible with the ``Gr00tReplayBuffer`` and
``Gr00tAgent`` pipeline. The environment server produces observations
in GR00T flat-key format (``video.hand_view``, ``state.eef_9d``, etc.).
"""

import ml_collections
import numpy as np


def get_config():
    config = ml_collections.ConfigDict()

    config.env_type = "droid"

    # Cartesian velocity action space.
    config.action_space = "cartesian_velocity"
    config.gripper_action_space = "velocity"

    # Workspace bounds and reset joints (override per task).
    config.bounds = None
    config.reset_joints = None

    config.reset_random = False
    config.randomize_low = np.array([0.0] * 16)
    config.randomize_high = np.array([0.0] * 16)

    # CR5AF cameras: wrist D405 (hand_view) + third-person D455 (table_view)
    config.side_camera_id = "table_view"
    config.wrist_camera_id = "hand_view"

    # Image size (matches GR00T processor defaults)
    config.image_size = (256, 256)
    config.control_hz = 8

    config.camera_kwargs = {
        "hand_camera": {"image": True, "depth": False, "left_only": True},
        "static_camera": {"image": True, "depth": False, "left_only": True},
    }

    # CR5AF action: eef_9d(9) + joint_pos(6) + gripper_pos(1) = 16D
    config.example_action = np.zeros((1, 16), dtype=np.float32)

    config.residual_action_xyzg = False

    return config
