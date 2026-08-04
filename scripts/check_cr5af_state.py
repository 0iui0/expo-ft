"""Diagnostic: print the CR5AF robot's reported pose vs the SFT training state.

No policy server, no motion. Instantiates the same env the deploy client uses,
reads one observation, and compares the reported cartesian_position against the
checkpoint norm-stats state mean. If the reported xyz is near-origin while the
training mean is ~[0.644, 0.001, 0.183] m, the pose read (unit/frame) is wrong
and the policy is being fed out-of-distribution state.

    ~/workspaces/hil-serl/.venv/bin/python scripts/check_cr5af_state.py \
        --robot-ip 192.168.5.1
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import tyro  # noqa: E402

from client.envs.cr5af_gripper_droid_env import CR5AFGripperDroidEnv  # noqa: E402
from configs.task import cr5af_gripper  # noqa: E402

# Training state xyz mean from the checkpoint norm_stats (assets/cr5af/shaft_insert).
EXPECTED_STATE_MEAN_XYZ_M = np.array([0.6439, 0.0014, 0.1825])


@dataclasses.dataclass
class Args:
    robot_ip: str = "192.168.5.1"
    task: str = "grasp motor shaft and insert into bushing"


def main(args: Args) -> None:
    cfg = cr5af_gripper.get_config()
    env = CR5AFGripperDroidEnv(
        language_instruction=args.task,
        robot_ip=args.robot_ip,
        camera_serial_hand=cfg.camera_serial_hand,
        camera_serial_table=cfg.camera_serial_table,
        image_size=tuple(cfg.image_size),
    )
    try:
        raw = env._env.get_observation()          # wrapped GR00T-format obs
        obs = env.get_observation()               # DROID-adapted obs
        eef = np.asarray(raw["state.eef_9d"], dtype=np.float64).flatten()
        cart = np.asarray(obs["cartesian_position"], dtype=np.float64).flatten()
        grip = np.asarray(obs["gripper_position"], dtype=np.float64).flatten()

        print("\n=== CR5AF pose diagnostic ===")
        print(f"raw state.eef_9d           : {np.round(eef, 4)}")
        print(f"  -> eef xyz (raw units)   : {np.round(eef[:3], 4)}")
        print(f"adapted cartesian_position : {np.round(cart, 4)}")
        print(f"  -> xyz (m, fed to policy): {np.round(cart[:3], 5)}")
        print(f"gripper_position           : {np.round(grip, 4)}")
        print(f"\nexpected training xyz mean (m): {EXPECTED_STATE_MEAN_XYZ_M}")
        err = np.linalg.norm(cart[:3] - EXPECTED_STATE_MEAN_XYZ_M)
        print(f"||reported - mean|| (m)       : {err:.4f}")
        if err > 0.3:
            print("\n  VERDICT: OUT OF DISTRIBUTION — reported pose is far from training.")
            print("  Likely a unit/frame bug in the pose read (or arm parked off-workspace).")
            print("  Do NOT run the policy on hardware until this matches.")
        else:
            print("\n  VERDICT: pose is in the training range; state read looks correct.")
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
