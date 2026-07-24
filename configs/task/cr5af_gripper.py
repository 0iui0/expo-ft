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
    # PI0.5 path uses the DROID-format adapter (7D cartesian obs/action); the
    # underlying CR5AF+gripper hardware is driven by CR5AFGripperEnv.
    try:
        from client.envs.cr5af_gripper_droid_env import CR5AFGripperDroidEnv

        config.env = CR5AFGripperDroidEnv
    except Exception:
        pass

    config.env_type = "droid"
    config.env_name = "cr5af_shaft_insert"

    config.language_instruction = "grasp motor shaft and insert into bushing"

    # ── robot connection ───────────────────────────────────────────────────
    config.robot_ip = "192.168.5.1"
    config.speed = 50.0  # motion speed percentage (0-100)
    config.translation_only = False

    # ── control rate ───────────────────────────────────────────────────────
    # Rollout loop rate (train_pi_robo.py) and env.step ServoP velocity scaling
    # must agree. 30 Hz matches the recording rate so SFT data and deploy share
    # one time-scale; env.step uses dt = 1/control_hz.
    config.control_hz = 30

    # ── cameras ──────────────────────────────────────────────────────────
    config.camera_serial_hand = "260322277798"   # D405 wrist camera
    config.camera_serial_table = "333422302713"  # D455 table camera

    # ── observation ────────────────────────────────────────────────────────
    config.image_size = (256, 256)

    # ── action: 7-dim DROID cartesian (xyz_m + euler_rad + gripper) ─────────
    # PI0.5 path. The DROID adapter env converts this to the 16-dim CR5AF
    # absolute target. action_dim/state_dim for EXPOLearner come from the
    # offline dataset (7), not from here; example_action is only the reset step.
    config.example_action = np.zeros((1, 7), dtype=np.float32)
    config.residual_action_xyzg = False

    return config
