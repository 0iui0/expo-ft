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
    config_task_path: str = "configs/task/pick.py"


_env_storage: Dict[str, Any] = {}
_config_task_path: Optional[str] = None
_task_config: Optional[Any] = None

# HIL (spacemouse takeover) is now handled INSIDE the env's step() (it has the
# current state to apply the human delta). The env returns action_type
# ("human"/"policy"); this server just forwards it. See cr5af_gripper_env.step.


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
                    
                    logger.info(f"Creating environment {env_id}...")
                    env_kwargs = dict(task_config)
                    env_kwargs["video_dir"] = request.get("video_dir") or ""
                    env = task_config.env(**env_kwargs)
                    _env_storage[env_id] = env
                    logger.info(f"Environment {env_id} created successfully")
                    
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
                        sent_is_invalid = np.allclose(sent_action, -1.0)
                        # Forward the learner-computed scalar Q (if any) for the
                        # live preview overlay; env-agnostic (no-op if unsupported).
                        q_value = request.get("q_value")
                        if q_value is not None and hasattr(env, "set_preview_q"):
                            env.set_preview_q(q_value)
                        # The env does HIL (spacemouse takeover) internally and
                        # returns action_type ("human"/"policy"). No override here.
                        if not sent_is_invalid:
                            step_result = env.step(sent_action)
                            executed_action = np.array(
                                step_result["executed_action"],
                                dtype=np.float64,
                            )
                            action_type = step_result.get("action_type", "policy")
                        else:
                            executed_action = sent_action
                            action_type = "policy"

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

