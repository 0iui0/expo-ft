"""PI0.5 real-robot rollout client for the CR5AF arm + DH PGE gripper (Route 2).

Connects to an openpi websocket policy server (``serve_policy.py`` serving the
pi05 ``shaft_insert`` SFT checkpoint) and drives the hardware through
``CR5AFGripperDroidEnv``. The server reconstructs absolute cartesian actions
from delta-xyz internally (``Policy.infer`` threads the observation state), so
this client only forwards observations, executes the returned action chunk, and
enforces a workspace clamp + NaN guard for safety.

Server (on the GPU host that holds the checkpoint)::

    XLA_PYTHON_CLIENT_PREALLOCATE=false JAXTYPING_DISABLE=1 CUDA_VISIBLE_DEVICES=0 \\
    .venv/bin/python expo_ft/agents/vla/openpi/scripts/serve_policy.py \\
      --port 8000 --default-prompt "grasp motor shaft and insert into bushing" \\
      policy:checkpoint \\
        --policy.config expo_pi05_droid_lora_finetune_sft_cartesian_state \\
        --policy.dir /data/openpi_checkpoints/expo_pi05_droid_lora_finetune_sft_cartesian_state/pi05_shaft_insert_sft/19999

Client (on the host that reaches the robot + RealSense), from the repo root::

    python client/deploy_pi05_cr5af.py \\
        --server-ip 192.168.16.155 --port 8000 --robot-ip 192.168.5.1 \\
        --task "grasp motor shaft and insert into bushing" \\
        --workspace-min 369 -245 110 --workspace-max 820 299 442 \\
        --speed 20 --translation-only

Full 6-DOF requires ``--joint-space`` (InverseKin + ServoJ): the task is
performed at the wrist singularity (joint5≈90°), where Cartesian ServoP
orientation servo races joint6 past its limit and the controller freezes the
whole command. Probe IK latency first with ``--joint-space --dry-run``.

Run ``--dry-run`` first: one inference on a live observation, no robot motion.
"""
from __future__ import annotations

import collections
import dataclasses
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

# Ensure the project root (parent of client/) is on sys.path so configs.* and
# client.* import regardless of how this script is launched.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import tyro  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

from client.envs.cr5af_gripper_droid_env import CR5AFGripperDroidEnv  # noqa: E402
from configs.task import cr5af_gripper  # noqa: E402

logger = logging.getLogger("deploy_pi05_cr5af")


def _to_server_obs(obs: dict) -> dict:
    """Remap the env's flat DROID keys to the ``observation/*`` namespace the
    served policy expects. ``serve_policy.py`` builds the policy without the data
    config's repack transform (``create_trained_policy`` defaults ``repack_transforms``
    to empty), so ``DroidInputs`` reads ``observation/...`` keys directly."""
    return {
        "observation/exterior_image_1_left": obs["exterior_image_1_left"],
        "observation/wrist_image_left": obs["wrist_image_left"],
        "observation/cartesian_position": obs["cartesian_position"],  # (9,) xyz_m + rot6d
        "observation/gripper_position": obs["gripper_position"],      # (1,)
        "prompt": obs["prompt"],
    }


@dataclasses.dataclass
class Args:
    server_ip: str = "192.168.16.155"          # policy server host
    port: int = 8000                           # policy server port
    robot_ip: str = "192.168.5.1"              # CR5AF arm
    task: str = "grasp motor shaft and insert into bushing"
    # Cartesian workspace bounds in mm (matches the GR00T deploy envelope).
    workspace_min: Tuple[float, float, float] = (369.0, -245.0, 110.0)
    workspace_max: Tuple[float, float, float] = (820.0, 299.0, 442.0)
    replan_steps: int = 8                      # actions consumed per inference
    control_hz: float = 30.0
    max_steps: int = 400
    translation_only: bool = False             # zero rotation velocity (safer first runs)
    joint_space: bool = False                  # InverseKin+ServoJ (needed at wrist singularity)
    max_joint_vel: float = 120.0               # per-joint deg/s cap for ServoJ deltas
    max_rot_vel: float = 8.0                   # ServoP orientation deg/s cap (low: joint6 limit near singularity)
    speed: float = 20.0                        # motion speed percentage (low default)
    dry_run: bool = False                      # one inference, no robot motion


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    task_cfg = cr5af_gripper.get_config()

    env = CR5AFGripperDroidEnv(
        language_instruction=args.task,
        robot_ip=args.robot_ip,
        control_hz=args.control_hz,
        translation_only=args.translation_only,
        joint_space=args.joint_space,
        max_joint_vel=args.max_joint_vel,
        max_rot_vel=args.max_rot_vel,
        speed=args.speed,
        camera_serial_hand=task_cfg.camera_serial_hand,
        camera_serial_table=task_cfg.camera_serial_table,
        image_size=tuple(task_cfg.image_size),
    )
    client = websocket_client_policy.WebsocketClientPolicy(host=args.server_ip, port=args.port)
    logger.info("connected; server metadata: %s", client.get_server_metadata())

    ws_min_m = np.asarray(args.workspace_min, dtype=np.float64) / 1000.0
    ws_max_m = np.asarray(args.workspace_max, dtype=np.float64) / 1000.0

    def infer_chunk() -> np.ndarray:
        chunk = np.asarray(client.infer(_to_server_obs(env.get_observation()))["actions"], dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != 10:
            raise RuntimeError(f"unexpected action shape {chunk.shape}, expected (H, 10)")
        return chunk

    if args.dry_run:
        chunk = infer_chunk()
        finite = bool(np.isfinite(chunk).all())
        cur_xyz = np.asarray(env.get_observation()["cartesian_position"][:3], dtype=np.float64)
        first_delta_mm = np.linalg.norm(chunk[0, :3] - cur_xyz) * 1000.0
        logger.info("DRY-RUN: chunk %s, finite=%s, first-step |Δxyz|=%.1f mm, xyz range(m)=[%s, %s]",
                    chunk.shape, finite, first_delta_mm, chunk[:, :3].min(0), chunk[:, :3].max(0))
        if args.joint_space:
            # Verify blocking InverseKin fits the control period before any motion.
            env.probe_ik_latency(n=30)
        env.close()
        return

    env.reset()
    dt = 1.0 / args.control_hz
    plan: collections.deque = collections.deque()
    prev_action = None
    try:
        for step in range(args.max_steps):
            t0 = time.time()
            if not plan:
                plan.extend(infer_chunk()[:args.replan_steps])
            a = np.asarray(plan.popleft(), dtype=np.float64)
            if not np.isfinite(a).all():
                logger.warning("step %d: non-finite action, holding pose", step)
                if prev_action is None:
                    continue
                a = prev_action
            a = a.copy()
            a[:3] = np.clip(a[:3], ws_min_m, ws_max_m)  # workspace clamp (meters)
            env.step(a)
            prev_action = a
            if step % 15 == 0:
                cur = np.asarray(env.get_observation()["cartesian_position"][:3], dtype=np.float64)
                logger.info("step %3d: target xyz(m)=%s cur=%s |err|=%.1fmm grip=%.0f",
                            step, np.round(a[:3], 3), np.round(cur, 3),
                            float(np.linalg.norm(a[:3] - cur) * 1000.0), a[9])
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        logger.info("interrupted by user")
    finally:
        try:
            env.stop()  # StopRobot: halt motion, exit ServoP mode
            env._gripper_open()
        except Exception:
            logger.warning("cleanup (stop/open) failed", exc_info=True)
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
