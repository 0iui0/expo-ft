"""Rollout server for RL training.

Serves as a websocket server to handle environment operations requested by the training server.
Supports operations: create_env, reset, step, get_observation, get_info_for_step.
"""

import asyncio
import dataclasses
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import websockets
import websockets.asyncio.server as _server
import msgpack_numpy

# Ensure the project root (parent of client/) is on sys.path so that
# configs.* and client.* are importable regardless of how the server
# process is launched (nohup, systemd, etc.).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import tyro

def load_task_config(config_path: Optional[str]):
    """Load task config from module path, similar to config_flags.DEFINE_config_file."""
    if config_path is None:
        return None

    # Convert file path to module path if needed (e.g., "configs/task/pick.py" -> "configs.task.pick")
    if '/' in config_path or '.py' in config_path:
        config_path = config_path.replace('.py', '').replace('/', '.')

    try:
        module = __import__(config_path, fromlist=['get_config'])
        return module.get_config()
    except Exception as e:
        raise ImportError(f"Failed to load task config from '{config_path}': {e}")


@dataclasses.dataclass
class Args:
    """Configuration arguments for rollout server."""

    server_host: str = "0.0.0.0"
    server_port: int = 8102
    config_task_path: str = "configs/task/cr5af.py"


_env_storage: Dict[str, Any] = {}
_config_task_path: Optional[str] = None
_task_config: Optional[Any] = None

# Preview tasks, keyed by env_id, so the cv2 renderer can be cleanly cancelled
# when an env is replaced (stale-cleanup). Runs in the asyncio main thread.
_preview_tasks: dict = {}


async def _preview_loop(env: Any, env_id: str):
    """Render the cv2 preview at ~15 Hz in the asyncio main thread (cv2-safe).

    Runs independently of the step() control path so imshow/waitKey latency
    never jitters the robot motion timing.
    """
    del env_id  # unused; kept for future task-cancellation key
    while getattr(env, "_running", False):
        try:
            env._render_preview()
        except Exception:
            pass
        await asyncio.sleep(1.0 / 15)


# Human-in-the-loop: SpaceMouse teleop matching record_demo_gripper.py.
# Reference logic (HidrawSpaceMouse):
#   - Right button (buttons[1]) = deadman: hold to move
#   - Left button (buttons[0])  = gripper toggle (edge-triggered)
#   - dead_zone = 0.15 on normalised axes (raw / 350 → [-1, 1])
#   - tdelta = [tx*scale, ty*scale, -tz*scale]   (mm per step)
#   - rdelta = [-roll*rot_scale, pitch*rot_scale, -yaw*rot_scale]  (deg per step)
_spacemouse_policy: Optional[Any] = None
_hil_gripper_state: float = 1.0   # 0.0=closed, 1.0=open (toggled by left button)
_hil_prev_left_btn: bool = False
_HIL_DEAD_ZONE = 0.15


def _get_human_override_action(task_config: Optional[Any] = None) -> tuple:
    """Return (action_7d or None, is_human).

    action_7d = [tx, ty, tz, rx, ry, rz, grip] where tx..tz are normalised
    [-1, 1] from HidrawSpaceMouse (easyhid) and grip is 0.0 or 1.0.

    Matches record_demo_gripper.py exactly: right-button deadman, dead_zone
    threshold, left-button gripper toggle. The device already normalises and
    orders axes as [tx,ty,tz,roll,pitch,yaw] — no extra /350 needed.
    """
    global _spacemouse_policy, _hil_gripper_state, _hil_prev_left_btn
    try:
        if _spacemouse_policy is None:
            from client.real_utils.spacemouse import HidrawSpaceMouse
            _spacemouse_policy = HidrawSpaceMouse()

        action_6d, buttons = _spacemouse_policy.get_action()
        # action_6d = [tx, ty, tz, roll, pitch, yaw] already normalised [-1, 1]

        # Right button (buttons[1]) = deadman: only while held does the human own
        # the arm. On release, control returns to the policy.
        deadman = len(buttons) > 1 and bool(buttons[1])
        if not deadman:
            return (None, False)

        # Left button (buttons[0]) = gripper toggle (edge-triggered) — processed
        # whenever the deadman is held, BEFORE the dead-zone check so it works
        # even while the SpaceMouse is momentarily still.
        left_btn = len(buttons) > 0 and bool(buttons[0])
        if left_btn and not _hil_prev_left_btn:
            _hil_gripper_state = 0.0 if _hil_gripper_state > 0.5 else 1.0
        _hil_prev_left_btn = left_btn

        # Deadman held → human owns the arm this step, even if the SpaceMouse is
        # momentarily still. Return a hold (zero-delta) action with is_human=True
        # so the policy NEVER resumes mid-takeover (that caused jitter / partial
        # takeover as policy and human alternated control).
        dead_zone = float(getattr(task_config, "hil_dead_zone", _HIL_DEAD_ZONE)) if task_config is not None else _HIL_DEAD_ZONE
        if float(np.max(np.abs(action_6d[:6]))) < dead_zone:
            tx = ty = tz = 0.0  # hold position
        else:
            tx, ty, tz = float(action_6d[0]), float(action_6d[1]), float(action_6d[2])

        # Rotation held in HIL; pass [tx, ty, tz, 0,0,0, grip]
        action_7d = np.array([tx, ty, tz, 0.0, 0.0, 0.0, _hil_gripper_state], dtype=np.float64)
        return (action_7d, True)
    except Exception as e:
        logging.getLogger(__name__).warning("Spacemouse unavailable (%s), using policy action.", e)
        return None, False


async def _handle_environment_request(websocket: _server.ServerConnection):
    """Handle robomimic operation requests from training server."""
    global _task_config
    logger = logging.getLogger(__name__)
    packer = msgpack_numpy.Packer()
    
    try:
        while True:
            try:
                request = msgpack_numpy.unpackb(await websocket.recv())
                operation = request.get("operation")
                
                if operation == "create_env":
                    task_config = load_task_config(_config_task_path)
                    _task_config = task_config
                    env_name = task_config.env_name
                    env_usage = request["env_usage"]
                    env_id = f"{env_name}_{env_usage}"

                    # Clean up stale env from previous client (release robot sockets)
                    if env_id in _env_storage:
                        logger.info(f"Closing stale environment {env_id}...")
                        if env_id in _preview_tasks:
                            _preview_tasks.pop(env_id).cancel()
                        try:
                            _env_storage[env_id].close()
                        except Exception:
                            pass
                        del _env_storage[env_id]

                    logger.info(f"Creating environment {env_id}...")
                    env_kwargs = dict(task_config)
                    env_kwargs["video_dir"] = request.get("video_dir") or ""
                    env = task_config.env(**env_kwargs)
                    _env_storage[env_id] = env
                    logger.info(f"Environment {env_id} created successfully")
                    # Start a low-rate preview task if the config enables it.
                    # Runs in the asyncio main thread (cv2-safe) so it cannot
                    # jitter the step() control path.
                    if getattr(task_config, "preview", False):
                        _preview_tasks[env_id] = asyncio.create_task(
                            _preview_loop(env, env_id))
                    
                    task_description = task_config.language_instruction
                    response = {"status": "success", "env_id": env_id, "task_description": task_description}
                    await websocket.send(packer.pack(response))
                    logger.info(f"Sent create_env response for {env_id}")
                    
                elif operation == "reset":
                    env_id = request["env_id"]
                    env = _env_storage.get(env_id)
                    
                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        obs = env.reset()
                        response = {
                            "status": "success",
                            "observation": obs,
                            "done": False,
                        }
                    await websocket.send(packer.pack(response))
                    
                elif operation == "step":
                    env_id = request["env_id"]
                    sent_action = np.array(request["action"])
                    env = _env_storage.get(env_id)
                    
                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        sent_action = sent_action.astype(np.float64)
                        if not np.isfinite(sent_action).all():
                            logger.warning(
                                "Action contains NaN/Inf; replacing with zeros. "
                                "Check policy inputs (observations, encoder), training stability, or checkpoint."
                            )
                            sent_action = np.where(np.isfinite(sent_action), sent_action, 0.0)
                        real_action = sent_action.copy()
                        action_type = "policy"
                        is_human = False
                        if _task_config is not None and _task_config.env_type == "droid":
                            sm_action, is_human = _get_human_override_action(_task_config)
                            if is_human and sm_action is not None:
                                # Reference-style delta (matches record_demo_gripper.py):
                                #   tdelta = [tx*scale, ty*scale, -tz*scale]
                                #   target = cur_xyz + tdelta
                                # action_6d values are normalised [-1, 1]; grip is 0/1.
                                obs = env.get_observation()
                                cur_xyz = obs["state.eef_9d"][:3]       # mm
                                cur_rot6d = obs["state.eef_9d"][3:9]
                                cur_joints = obs["state.joint_pos"]
                                # Per-step scale (mm/unit/step). record_demo uses
                                # 8mm/step @ 30Hz; at 8Hz control we scale up so the
                                # takeover feels similar. Tunable via config.
                                action_scale = float(getattr(_task_config, "hil_action_scale", 20.0))
                                tx, ty, tz = sm_action[0], sm_action[1], sm_action[2]
                                tdelta = np.array([
                                    tx * action_scale,
                                    ty * action_scale,
                                    -tz * action_scale,   # match reference: -tz
                                ])
                                new_xyz = cur_xyz + tdelta
                                # Rotation: hold current
                                new_eef_9d = np.concatenate([new_xyz, cur_rot6d]).astype(np.float64)
                                # Gripper: action_7d[6] is already 0.0/1.0 from toggle
                                new_grip = float(sm_action[6])
                                real_action = np.concatenate([
                                    new_eef_9d, cur_joints, [new_grip]
                                ]).astype(np.float64)
                                action_type = "human"
                        sent_is_invalid = np.allclose(sent_action, -1.0)
                        if is_human:
                            env._hil_mode = True  # skip EMA + dead-zone for direct feel
                        if is_human or not sent_is_invalid:
                            step_result = env.step(real_action)
                            executed_action = np.array(
                                step_result["executed_action"],
                                dtype=np.float64,
                            )
                        if is_human:
                            env._hil_mode = False
                        else:
                            executed_action = real_action

                        response = {
                            "status": "success",
                            "action": executed_action.tolist(),
                            "action_type": action_type,
                        }
                    await websocket.send(packer.pack(response))

                elif operation == "get_observation":
                    env_id = request["env_id"]
                    env = _env_storage.get(env_id)

                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        obs = env.get_observation()
                        response = {
                            "status": "success",
                            "observation": obs,
                        }
                    await websocket.send(packer.pack(response))

                elif operation == "get_info_for_step":
                    env_id = request["env_id"]
                    env = _env_storage.get(env_id)
                    
                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        done, success, reward, mask = env.get_info_for_step()
                        response = {
                            "status": "success",
                            "done": bool(done),
                            "success": bool(success),
                            "reward": float(reward),
                            "mask": float(mask),
                        }
                    await websocket.send(packer.pack(response))

                else:
                    response = {"status": "error", "message": f"Unknown operation: {operation}"}
                    await websocket.send(packer.pack(response))
            
            except websockets.exceptions.ConnectionClosed:
                logger.debug(f"Connection closed by client {websocket.remote_address}")
                break
            except Exception as e:
                logger.error(f"Error handling request: {e}", exc_info=True)
                try:
                    response = {"status": "error", "message": str(e)}
                    await websocket.send(packer.pack(response))
                except websockets.exceptions.ConnectionClosed:
                    logger.debug("Connection closed while sending error response")
                    break
                
    except websockets.exceptions.ConnectionClosed:
        logger.debug(f"Connection closed: {websocket.remote_address}")
    except Exception as e:
        logger.error(f"Unexpected error in request handler: {e}", exc_info=True)


async def _run_server(host: str, port: int, config_task_path: Optional[str]):
    """Run the websocket server for environment operations."""
    global _config_task_path
    _config_task_path = config_task_path
    logger = logging.getLogger(__name__)

    async with _server.serve(
        _handle_environment_request, 
        host, 
        port, 
        compression=None, 
        max_size=None,
        # This server handles potentially long blocking work (env init/step).
        # Disable keepalive pings to avoid ping timeouts while busy.
        ping_interval=None,
        ping_timeout=None,
        close_timeout=100,
    ) as server:
        logger.info(f"Environment operations server started on {host}:{port}")
        await server.serve_forever()

async def main_async(args: Args) -> None:
    """Main async entry point."""
    await _run_server(args.server_host, args.server_port, args.config_task_path)

def main(args: Args) -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger("websockets.server").setLevel(logging.WARNING)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)

