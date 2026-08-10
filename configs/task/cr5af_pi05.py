"""CR5AF + DH PGE gripper task config for PI0.5 online RL (translation-only).

Differs from ``cr5af_gripper.py`` (the 16-D GR00T/velocity task) in two ways:

1. **10-D absolute next-pose action** ``[xyz_m(3), rot6d(6), gripper(1)]`` — the
   PI0.5 SFT prior's native action space (matches the cr5af/shaft_insert
   norm_stats). The GR00T 16-D velocity action does NOT apply here.
2. **translation_only + residual_action_xyzg** — the residual actor's rotation
   dims [3:6] are masked, and the env locks orientation in step(). This is the
   known-good CR5AF path (see the wrist-singularity fix); 6-DOF is deferred.

The env server (``client/run_client.py``) loads this on the robot side; the
training side uses it for ``example_action`` shape and the xyzg mask.
"""
import numpy as np


def get_config():
    from ml_collections import ConfigDict

    config = ConfigDict()

    # ── env class (set on the robot side; training side ignores try/except) ──
    # PI0.5 needs DROID-format observations; CR5AFGripperDroidEnv wraps the real
    # CR5AFGripperEnv and emits DROID keys (exterior_image_1_left, wrist_image_left,
    # cartesian_position, gripper_position) + converts 10-D actions to 16-D targets.
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
    config.speed = 30.0  # motion speed percentage (0-100); lowered from 50 for joint-speed margin
    config.translation_only = True  # lock orientation in env.step()
    config.control_hz = 8  # RL control loop rate → env step() dt = 1/control_hz
    # TCP/Cartesian control (ServoP). joint_space (ServoJ+IK) disabled — it
    # compounded the blind-RT problem (IK failed with jointNear=0). The real
    # safety gate is the RT fail-safe in step(): no motion without a valid RT frame.
    config.joint_space = False
    config.max_joint_vel = 90.0  # unused in TCP mode; kept for reference

    # ── cameras ──────────────────────────────────────────────────────────
    config.camera_serial_hand = "260322277798"   # D405 wrist camera
    config.camera_serial_table = "333422302713"  # D455 table camera

    # ── observation ────────────────────────────────────────────────────────
    config.image_size = (256, 256)

    # ── action: 10-D absolute next pose [xyz_m(3), rot6d(6), gripper(1)] ───
    config.example_action = np.zeros((1, 10), dtype=np.float32)
    # Mask the residual actor's rotation dims [3:6]; only xyz+gripper are learned.
    config.residual_action_xyzg = True

    return config
