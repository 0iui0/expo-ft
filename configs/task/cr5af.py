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
    config.env_name = "cr5af_shaft_insert"
    config.language_instruction = "grasp motor shaft and insert into bushing"

    # ── env class ──────────────────────────────────────────────────────────
    try:
        from client.envs.cr5af_gripper_env import CR5AFGripperEnv
        config.env = CR5AFGripperEnv
    except Exception as e:
        raise ImportError(
            "Failed to import CR5AFGripperEnv. Check that "
            "client/envs/cr5af_gripper_env.py is on PYTHONPATH and all "
            "dependencies are installed.  Original error: " + str(e)
        ) from e

    # ── robot connection ───────────────────────────────────────────────────
    config.robot_ip = "192.168.5.1"
    config.speed = 50.0

    # SpaceMouse HIL override velocity limits (mm/s and deg/s)
    config.collect_max_lin_vel = 50.0
    config.collect_max_rot_vel = 10.0
    # HIL takeover: mm per normalised unit per step (record_demo feel ~20mm/step),
    # and SpaceMouse dead-zone on normalised axes.
    config.hil_action_scale = 20.0
    config.hil_dead_zone = 0.15

    # Cartesian velocity action space.
    config.action_space = "cartesian_velocity"
    config.gripper_action_space = "velocity"

    # Workspace bounds — calibrated for THIS desk (matching deploy_cr5af_gripper.py
    # --workspace-min 369 -245 110 --workspace-max 820 299 442).
    config.bounds = [[369, -245, 110], [820, 299, 442]]
    config.reset_joints = None

    config.reset_random = False
    config.randomize_low = np.array([0.0] * 16)
    config.randomize_high = np.array([0.0] * 16)

    # CR5AF cameras: wrist D405 (hand_view) + third-person D455 (table_view)
    config.side_camera_id = "table_view"
    config.wrist_camera_id = "hand_view"

    # Image size — preserved at 256×256. The GR00T processor resizes internally;
    # a different aspect ratio here changes pixel values in non-obvious ways and
    # was observed to break the policy's directional output.
    config.image_size = (256, 256)
    config.control_hz = 8
    config.preview = True   # show labelled camera preview (D455|D405) in run_client

    config.camera_kwargs = {
        "hand_camera": {"image": True, "depth": False, "left_only": True},
        "static_camera": {"image": True, "depth": False, "left_only": True},
    }

    # CR5AF action: eef_9d(9) + joint_pos(6) + gripper_pos(1) = 16D
    config.example_action = np.zeros((1, 16), dtype=np.float32)

    config.residual_action_xyzg = False

    return config
