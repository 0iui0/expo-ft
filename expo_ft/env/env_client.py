"""Client for interacting with environment rollout server via WebSocket."""

import logging
import time
from typing import Dict, Tuple, Any

import numpy as np
import websockets.sync.client
import websockets.exceptions
import msgpack_numpy

# Handshake/connection errors that can occur over SSH reverse tunnels (connection
# accepted but closed before/during HTTP response). Direct connection often avoids these.
_CONNECT_RETRY_EXC: tuple = (
    ConnectionRefusedError,
    websockets.exceptions.InvalidMessage,
    EOFError,
    OSError,
    ConnectionResetError,
)


class EnvClient:
    """Client for environment operations server via WebSocket.

    Uses a single persistent connection for all operations so eval loops are fast.
    Reconnects automatically on connection close or errors.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8102):
        """Initialize the environment client."""
        self.host = host
        self.port = port
        self._uri = f"ws://{host}:{port}"
        self._conn = None
        self._packer = msgpack_numpy.Packer()

    def _connect(self):
        """Open a new WebSocket connection (used on first use or after close/error)."""
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    close_timeout=100,
                    ping_interval=None,
                    ping_timeout=None,
                )
                return conn
            except _CONNECT_RETRY_EXC as e:
                logging.info(
                    "Waiting for operations client (%s: %s)...",
                    type(e).__name__,
                    e,
                )
                time.sleep(5)

    def _get_connection(self):
        """Return the current connection, opening one if needed (persistent for eval speed)."""
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def _close_connection(self):
        """Drop the current connection so the next call will reconnect."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _call_operation(self, operation: str, request: dict) -> dict:
        last_exc = None
        for attempt in range(3):
            try:
                conn = self._get_connection()
                request_data = {"operation": operation, **request}
                conn.send(self._packer.pack(request_data))
                raw = conn.recv()
                response = msgpack_numpy.unpackb(raw)
                if response.get("status", response.get("stats")) == "error":
                    raise RuntimeError(
                        f"Environment operation {operation} failed: {response.get('message')}"
                    )
                return response
            except _CONNECT_RETRY_EXC as e:
                last_exc = e
                self._close_connection()
                if attempt < 2:
                    logging.debug(
                        "EnvClient %s attempt %s failed (%s), retrying...",
                        operation,
                        attempt + 1,
                        e,
                    )
                    time.sleep(2)
            except websockets.exceptions.ConnectionClosed as e:
                last_exc = e
                self._close_connection()
                if attempt < 2:
                    logging.debug(
                        "EnvClient %s connection closed (attempt %s), retrying...",
                        operation,
                        attempt + 1,
                    )
                    time.sleep(2)
            except Exception as e:
                last_exc = e
                self._close_connection()
                raise RuntimeError(f"EnvClient {operation} failed: {e}") from e
        raise RuntimeError(f"EnvClient {operation} failed after retries: {last_exc}") from last_exc
    
    def _prepare_request(self, request: dict) -> dict:
        """Prepare request dict by converting numpy arrays to lists for serialization."""
        prepared = {}
        for k, v in request.items():
            if isinstance(v, np.ndarray):
                prepared[k] = v.tolist()
            else:
                prepared[k] = v
        return prepared
    
    def create_env(self, request: dict) -> Tuple[str, str]:
        """Create a environment."""
        prepared_request = self._prepare_request(request)
        response = self._call_operation("create_env", prepared_request)
        return response["env_id"], response["task_description"]
    
    def reset(self, env_id: str) -> Tuple[Dict[str, Any], bool]:
        """Reset a environment."""
        response = self._call_operation("reset", {"env_id": env_id})
        observation = response["observation"]
        return observation, response["done"]
    
    def step(self, env_id: str, action: np.ndarray, q_value=None) -> Tuple[np.ndarray, str]:
        """Step the environment. Returns (real_executed_action, action_type)."""
        request = {"env_id": env_id, "action": action}
        if q_value is not None:
            request["q_value"] = float(q_value)
        response = self._call_operation("step", request)
        real_action = np.array(response.get("action", action))
        action_type = response.get("action_type", "policy")
        return real_action, action_type

    def get_observation(self, env_id: str) -> dict:
        """Get the observation of the environment."""
        response = self._call_operation("get_observation", {"env_id": env_id})
        return response["observation"]

    def get_info_for_step(self, env_id: str) -> Tuple[bool, bool, float, float]:
        """Evaluate termination after a step: (done, success, reward, continuation_mask)."""
        response = self._call_operation("get_info_for_step", {"env_id": env_id})
        return response["done"], response["success"], response["reward"], response["mask"]

class EnvClientWrapper:
    """Wrapper that provides a gym-like interface for EnvClient."""
    
    def __init__(self, env_creation_request: dict, host: str = "localhost", port: int = 8102):
        """Initialize the wrapper."""
        self.host = host
        self.port = port
        self.client = EnvClient(host=host, port=port)
        self.env_id, self.task_description = self.client.create_env(env_creation_request)
        self.env_creation_request = env_creation_request

    def _call(self, op_name: str, thunk):
        """Run an op; if server returns error, recreate env + reset, then retry once."""
        status = "normal"
        while True:
            if status == "normal":
                try:
                    return thunk()
                except RuntimeError as e:
                    logging.warning(f"{op_name} failed ({e}); will recreate env and retry...")
                    status = "recover"
                    time.sleep(10)
            else:
                try:
                    self.client = EnvClient(host=self.host, port=self.port)
                    self.env_id, self.task_description = self.client.create_env(self.env_creation_request)
                    if op_name != "reset":
                        self.client.reset(self.env_id)
                    self.client.reset(self.env_id)
                    time.sleep(60)
                    status = "normal"
                except RuntimeError as e:
                    logging.warning(f"{op_name} recovery failed ({e}); retrying recovery...")
                    time.sleep(10)
    
    def reset(self):
        """Reset the environment and return observation."""
        observation, _ = self._call("reset", lambda: self.client.reset(self.env_id))
        return observation
    
    def step(self, action, q_value=None):
        """Step the environment.

        Args:
            action: Action to take (policy output).
            q_value: Optional scalar critic Q of the taken action, forwarded to
                the env server for the live preview overlay (not used to control).

        Returns:
            Tuple of (real_executed_action, action_type) where action_type is "policy" or "human".
        """
        return self._call("step", lambda: self.client.step(self.env_id, action, q_value=q_value))

    def get_observation(self):
        """Get the observation of the environment."""
        return self._call("get_observation", lambda: self.client.get_observation(self.env_id))

    def get_info_for_step(self):
        """Evaluate termination after a step: (done, success, reward, continuation_mask)."""
        return self._call("get_info_for_step", lambda: self.client.get_info_for_step(self.env_id))
    
    
