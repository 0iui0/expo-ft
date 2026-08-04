"""Find the InverseKin command format the CR5AF controller accepts.

Read-only: InverseKin only *computes* a joint solution, it never moves the arm.
The first joint-space run got ErrorID -5 with an empty {} solution. This probes
several command-string variants at the CURRENT pose and prints each raw reply,
so we can see which one returns ``0,{j1,...,j6},...`` instead of ``-5,{},...``.

Variants cover the two suspects: (a) the missing ``user=0,tool=0`` params that
the official Dobot v4 ROS driver includes, and (b) the current jointNear joint6
sitting beyond ±180 (RT feed reads it at ~-194°).

    ~/workspaces/hil-serl/.venv/bin/python scripts/probe_inverse_kin.py \
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

from client.envs.cr5af_gripper_env import CR5AFGripperEnv  # noqa: E402
from configs.task import cr5af_gripper  # noqa: E402


@dataclasses.dataclass
class Args:
    robot_ip: str = "192.168.5.1"


def main(args: Args) -> None:
    cfg = cr5af_gripper.get_config()
    env = CR5AFGripperEnv(
        robot_ip=args.robot_ip,
        camera_serial_hand="", camera_serial_table="",  # skip cameras
        image_size=tuple(cfg.image_size),
    )
    try:
        time.sleep(1.0)  # let RT feed populate
        with env._lock:
            pos = env._pos.copy()
            q = np.degrees(env._q.copy())
        x, y, z = pos[:3] * 1000.0
        rx, ry, rz = R.from_quat(pos[3:]).as_rotvec(degrees=True)

        j_raw = "{" + ",".join(f"{v:.4f}" for v in q) + "}"
        q_wrap = ((q + 180.0) % 360.0) - 180.0                 # each joint -> [-180,180)
        j_wrap = "{" + ",".join(f"{v:.4f}" for v in q_wrap) + "}"
        pose = f"{x:.3f},{y:.3f},{z:.3f},{rx:.4f},{ry:.4f},{rz:.4f}"

        print(f"\ncurrent xyz(mm)   : [{x:.1f} {y:.1f} {z:.1f}]")
        print(f"current rvec(deg) : [{rx:.2f} {ry:.2f} {rz:.2f}]")
        print(f"current joints    : {np.round(q, 2)}  (joint6={q[5]:.2f})")
        print(f"wrapped joints    : {np.round(q_wrap, 2)}  (joint6={q_wrap[5]:.2f})\n")

        variants = {
            "1 baseline (mine)          ": f"InverseKin({pose},useJointNear=1,jointNear={j_raw})",
            "2 +user/tool (ref driver)  ": f"InverseKin({pose},user=0,tool=0,useJointNear=1,jointNear={j_raw})",
            "3 no jointNear bias         ": f"InverseKin({pose})",
            "4 no bias +user/tool        ": f"InverseKin({pose},user=0,tool=0)",
            "5 wrapped jointNear +u/t    ": f"InverseKin({pose},user=0,tool=0,useJointNear=1,jointNear={j_wrap})",
            "6 quoted jointNear +u/t     ": f'InverseKin({pose},user=0,tool=0,useJointNear=1,jointNear="{j_raw}")',
        }
        for label, cmd in variants.items():
            resp = env._send_cmd(cmd, read_response=True, timeout=2.0)
            print(f"[{label}] -> {resp!r}")
            time.sleep(0.15)
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
