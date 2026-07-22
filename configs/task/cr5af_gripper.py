"""CR5AF + DH PGE gripper task config for expo-ft online RL.

The env server (``client/run_client.py``) loads this config and creates
``CR5AFGripperEnv`` with these parameters.  The training machine also uses
this config for ``example_action`` shape / ``residual_action_xyzg``.
"""
import numpy as np


def get_config():
    from ml_collections import ConfigDict

    config = ConfigDict()

    # ── env class (set on the robot side; training side ignores try/except) ──
    try:
        from client.envs.cr5af_gripper_env import CR5AFGripperEnv

        config.env = CR5AFGripperEnv
    except Exception:
        pass

    config.env_type = "droid"
    config.env_name = "cr5af_shaft_insert"

    config.language_instruction = "grasp motor shaft and insert into bushing"

    # ── robot connection ───────────────────────────────────────────────────
    config.robot_ip = "192.168.5.1"
    config.speed = 50.0  # motion speed percentage (0-100)
    config.translation_only = False

    # ── cameras (fill in the RealSense serials for hand + table cameras) ──
    config.camera_serial_hand = ""
    config.camera_serial_table = ""

    # ── observation ────────────────────────────────────────────────────────
    config.image_size = (256, 256)

    # ── action: 16-dim (eef_9d + joint_pos + gripper_pos) ──────────────────
    config.example_action = np.zeros((1, 16), dtype=np.float32)
    config.residual_action_xyzg = False

    return config
