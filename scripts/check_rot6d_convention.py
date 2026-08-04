"""Which rot6d convention matches the SFT training distribution?

Read-only. Grabs the robot's current orientation, encodes it BOTH ways
(from_rotvec — the fix — and from_euler XYZ — the original), and reports which
lands inside the checkpoint norm_stats rot6d distribution. Decisively settles
whether the training recorder used axis-angle or Euler.

    ~/workspaces/hil-serl/.venv/bin/python scripts/check_rot6d_convention.py \
        --robot-ip 192.168.5.1
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import tyro  # noqa: E402

from client.envs.cr5af_gripper_env import CR5AFGripperEnv, _matrix_to_rot6d  # noqa: E402
from configs.task import cr5af_gripper  # noqa: E402

# norm_stats state (10D) — xyz(3) + rot6d(6) + gripper(1)
NS_MEAN = np.array([0.6439, 0.0014, 0.1825,
                    0.4593, -0.2616, -0.4808, 0.2611, -0.7975, 0.2709, 0.4289])
NS_STD = np.array([0.0803, 0.1042, 0.0469,
                   0.0939, 0.2242, 0.656, 0.2241, 0.1292, 0.3943, 0.4949])
MEAN_ROT6D = NS_MEAN[3:9]
STD_ROT6D = NS_STD[3:9]


@dataclasses.dataclass
class Args:
    robot_ip: str = "192.168.5.1"


def main(args: Args) -> None:
    cfg = cr5af_gripper.get_config()
    env = CR5AFGripperEnv(
        robot_ip=args.robot_ip,
        camera_serial_hand="", camera_serial_table="",  # skip cameras, faster
        image_size=tuple(cfg.image_size),
    )
    try:
        time.sleep(1.0)  # let RT feed populate
        with env._lock:
            quat = env._pos[3:].copy()          # built via from_rotvec now
            xyz_m = env._pos[:3].copy()
        # Recover the robot's raw tool_vector orientation triple (deg).
        v = R.from_quat(quat).as_rotvec(degrees=True)

        rot6d_rotvec = _matrix_to_rot6d(R.from_rotvec(v, degrees=True).as_matrix())
        rot6d_euler = _matrix_to_rot6d(R.from_euler("XYZ", v, degrees=True).as_matrix())

        def zscore(r6):
            return np.abs((r6 - MEAN_ROT6D) / STD_ROT6D)

        print("\n=== rot6d convention check ===")
        print(f"robot xyz (m)           : {np.round(xyz_m, 4)}  (train mean {NS_MEAN[:3]})")
        print(f"robot tool triple (deg) : {np.round(v, 2)}")
        print(f"\ntraining mean rot6d     : {np.round(MEAN_ROT6D, 3)}")
        print(f"  from_rotvec (the fix) : {np.round(rot6d_rotvec, 3)}  |z|={np.round(zscore(rot6d_rotvec),1)} sum={zscore(rot6d_rotvec).sum():.1f}")
        print(f"  from_euler (original) : {np.round(rot6d_euler, 3)}  |z|={np.round(zscore(rot6d_euler),1)} sum={zscore(rot6d_euler).sum():.1f}")

        d_rv = np.linalg.norm(rot6d_rotvec - MEAN_ROT6D)
        d_eu = np.linalg.norm(rot6d_euler - MEAN_ROT6D)
        print(f"\nL2 to training mean rot6d: rotvec={d_rv:.3f}  euler={d_eu:.3f}")
        winner = "from_rotvec (current fix is CORRECT)" if d_rv < d_eu else "from_euler (REVERT the fix)"
        print(f"VERDICT: training convention is likely {winner}")
        print("\n(Only meaningful if the arm is at a normal task pose. If BOTH sums")
        print(" are large, the current pose is off-distribution — check start pose/setup.)")
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
