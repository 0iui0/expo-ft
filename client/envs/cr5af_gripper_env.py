"""CR5AF + DH PGE gripper env for expo-ft online RL (WebSocket server).

Implements the ``client/run_client.py`` WebSocket interface:
- ``reset()`` → GR00T observation dict
- ``step(action: np.ndarray)`` → ``{"executed_action": ...}``
- ``get_observation()`` → GR00T observation dict
- ``get_info_for_step()`` → ``(done, success, reward, mask)``

Robot control is self-contained (no cross-workspace imports).  The CR5AF arm
is driven via raw TCP (ServoP for cartesian servo, RT feed for state) cribbed
from the proven ``cr5af_server.py`` in hil-serl.  The DH PGE gripper uses
RunScript (DobotStudio DHGrip plugin) — binary open/close.

Action: 16-dim ABSOLUTE position targets (policy-decoded eef_9d + joint_pos
+ gripper_pos).  The env computes the delta from the current RT state and
issues ServoP for arm + RunScript for gripper.

Start on thor::

    cd ~/workspaces/expo-ft
    ~/workspaces/hil-serl/.venv/bin/python client/run_client.py \
        --config_task_path configs/task/cr5af_gripper.py
"""
from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

# ── Dobot TCP constants ─────────────────────────────────────────────────────
MM_TO_M = 0.001
M_TO_MM = 1000.0
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# RT data struct offsets (1440-byte frames from port 30004)
RT_Q_ACTUAL = 432
RT_TOOL_VECTOR = 624
RT_TCP_SPEED = 672
RT_ACTUAL_QUAT = 1384
RT_ROBOT_MODE = 24
RT_TEST_VALUE = 48
RT_FRAME_MAGIC = 0x123456789ABCDEF

# ── action layout ────────────────────────────────────────────────────────────
EEF9D_SLICE = slice(0, 9)
JOINT_SLICE = slice(9, 15)
GRIPPER_IDX = 15


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """rot6d (two 3-vectors) → 3x3 rotation matrix (Gram-Schmidt)."""
    a = rot6d[:3] / np.linalg.norm(rot6d[:3])
    b = rot6d[3:6] - np.dot(a, rot6d[3:6]) * a
    b /= np.linalg.norm(b)
    c = np.cross(a, b)
    return np.column_stack([a, b, c])


def _matrix_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → rot6d (first two columns)."""
    return np.concatenate([mat[:, 0], mat[:, 1]])


class CR5AFGripperEnv:
    """Minimal env for the CR5AF arm + DH PGE gripper via Dobot TCP/IP.

    The constructor accepts all ``**kwargs`` from a task config, so it works
    directly with ``client/run_client.py``.
    """

    def __init__(
        self,
        robot_ip: str = "192.168.5.1",
        command_port: int = 29999,
        rt_port: int = 30004,
        image_size: Tuple[int, int] = (256, 256),
        camera_serial_hand: str = "",
        camera_serial_table: str = "",
        speed: float = 50.0,
        translation_only: bool = False,
        language_instruction: str = "grasp motor shaft and insert into bushing",
        video_dir: str = "",
        **kwargs,
    ):
        self._robot_ip = robot_ip
        self._command_port = command_port
        self._rt_port = rt_port
        self._image_size = image_size
        self._speed_pct = speed
        self._translation_only = translation_only
        self._language_instruction = language_instruction
        self._servo_gain = 250  # lower = softer (default 250, range 200-1000)

        # ── thread-safe state cache (SI units) ─────────────────────────────
        self._lock = threading.Lock()
        self._pos = np.zeros(7, dtype=np.float64)  # xyz + quat
        self._q = np.zeros(6, dtype=np.float64)     # joint angles (rad)
        self._eef_9d = np.zeros(9, dtype=np.float32)  # current eef_9d
        self._joint_pos = np.zeros(6, dtype=np.float32)
        self._gripper_pos = 1.0  # 1=open, 0=closed
        self._robot_mode: int = 0
        self._connected = False

        # ── RT feed (port 30004) ───────────────────────────────────────────
        self._rt_sock: Optional[socket.socket] = None
        self._running = True
        self._connect_rt()

        # ── command socket (port 29999) ────────────────────────────────────
        self._cmd_sock: Optional[socket.socket] = None
        self._cmd_lock = threading.Lock()
        self._connect_cmd()
        self._enable_robot()

        # ── cameras (RealSense) ────────────────────────────────────────────
        self._cam_hand = None
        self._cam_table = None
        try:
            import pyrealsense2 as rs
            # Auto-detect if serials are empty (like record_demo_gripper.py)
            if not camera_serial_hand or not camera_serial_table:
                ctx = rs.context()
                for dev in ctx.devices:
                    name = dev.get_info(rs.camera_info.name)
                    sn = dev.get_info(rs.camera_info.serial_number)
                    if "455" in name and not camera_serial_table:
                        camera_serial_table = sn
                    elif "405" in name and not camera_serial_hand:
                        camera_serial_hand = sn
                logger.info("Auto-detected cameras: hand=%s table=%s",
                            camera_serial_hand, camera_serial_table)
            if camera_serial_hand:
                self._cam_hand = self._init_realsense(rs, camera_serial_hand, image_size)
            if camera_serial_table:
                self._cam_table = self._init_realsense(rs, camera_serial_table, image_size)
        except ImportError:
            logger.warning("pyrealsense2 not available — camera frames will be empty.")

        # ── episode state ──────────────────────────────────────────────────
        self._steps_since_reset = 0
        self.done = False
        self.success = False
        self.reward = 0.0
        self._video_dir = video_dir

    @staticmethod
    def _init_realsense(rs, serial, image_size):
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, image_size[1], image_size[0],
                          rs.format.rgb8, 30)
        pipe.start(cfg)
        # let auto-exposure settle
        for _ in range(15):
            pipe.wait_for_frames()
        return pipe

    # ═══════════════════════════════════════════════════════════════════════
    # RT feed (port 30004) — cribbed from cr5af_server.py
    # ═══════════════════════════════════════════════════════════════════════

    def _connect_rt(self):
        self._rt_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._rt_sock.settimeout(5.0)
        try:
            self._rt_sock.connect((self._robot_ip, self._rt_port))
            self._connected = True
            logger.info("RT feed connected %s:%d", self._robot_ip, self._rt_port)
        except Exception as e:
            logger.error("RT feed connection failed: %s", e)
            return
        t = threading.Thread(target=self._rt_loop, daemon=True)
        t.start()

    def _rt_loop(self):
        buf = bytearray()
        while self._running:
            if self._rt_sock is None:
                time.sleep(0.5)
                continue
            try:
                chunk = self._rt_sock.recv(4096)
                if not chunk:
                    raise ConnectionError("RT connection closed")
                buf.extend(chunk)
                # parse complete 1440-byte frames
                while len(buf) >= 1440:
                    frame = bytes(buf[:1440])
                    del buf[:1440]
                    self._parse_rt_frame(frame)
            except Exception as e:
                logger.warning("RT recv error: %s, reconnecting...", e)
                try:
                    self._rt_sock.close()
                except Exception:
                    pass
                self._rt_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._rt_sock.settimeout(5.0)
                try:
                    self._rt_sock.connect((self._robot_ip, self._rt_port))
                except Exception:
                    time.sleep(1.0)

    def _parse_rt_frame(self, data: bytes):
        fields = struct.unpack_from("<H", data, 0)
        if fields[0] != 1440:
            return
        magic = struct.unpack_from("<Q", data, RT_TEST_VALUE)[0]
        if magic != RT_FRAME_MAGIC:
            return

        tv = list(struct.unpack_from("<6d", data, RT_TOOL_VECTOR))
        xyz = np.array(tv[:3]) * MM_TO_M
        rxyz_rad = np.array(tv[3:6]) * DEG2RAD
        rot = R.from_euler("XYZ", rxyz_rad)
        quat = rot.as_quat()  # xyzw

        q = np.array(list(struct.unpack_from("<6d", data, RT_Q_ACTUAL))) * DEG2RAD

        with self._lock:
            self._pos = np.concatenate([xyz, quat])
            self._q = q.copy()
            self._eef_9d = np.concatenate([xyz, _matrix_to_rot6d(rot.as_matrix())]).astype(np.float32)
            self._joint_pos = self._q.astype(np.float32)
            self._robot_mode = struct.unpack_from("<Q", data, RT_ROBOT_MODE)[0]

    # ═══════════════════════════════════════════════════════════════════════
    # Command socket (port 29999)
    # ═══════════════════════════════════════════════════════════════════════

    def _connect_cmd(self):
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._cmd_sock.settimeout(5.0)
        self._cmd_sock.connect((self._robot_ip, self._command_port))
        self._cmd_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._drain_cmd()
        logger.info("Command socket connected %s:%d", self._robot_ip, self._command_port)

    def _drain_cmd(self):
        if self._cmd_sock is None:
            return
        self._cmd_sock.setblocking(False)
        try:
            while True:
                self._cmd_sock.recv(4096)
        except BlockingIOError:
            pass
        self._cmd_sock.setblocking(True)

    def _send_cmd(self, cmd: str, read_response: bool = False, timeout: float = 5.0) -> str:
        with self._cmd_lock:
            try:
                self._drain_cmd()
                self._cmd_sock.sendall(cmd.encode("utf-8"))
                if read_response:
                    self._cmd_sock.settimeout(timeout)
                    resp = bytearray()
                    while True:
                        c = self._cmd_sock.recv(1)
                        if not c or c == b";":
                            break
                        resp.extend(c)
                    return resp.decode("utf-8").strip()
                return ""
            except Exception as e:
                logger.warning("cmd error: %s", e)
                self._reconnect_cmd()
                return ""

    def _reconnect_cmd(self):
        try:
            if self._cmd_sock:
                self._cmd_sock.close()
        except Exception:
            pass
        try:
            self._connect_cmd()
        except Exception as e:
            logger.error("cmd reconnect failed: %s", e)

    def _enable_robot(self):
        for cmd in ("EnableRobot()", f"SpeedFactor({int(self._speed_pct)})",
                     f"AccL({int(self._speed_pct)})"):
            self._send_cmd(cmd, read_response=False)
            time.sleep(0.3)

    # ═══════════════════════════════════════════════════════════════════════
    # Robot motion
    # ═══════════════════════════════════════════════════════════════════════

    def _servop(self, x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg):
        """Fire-and-forget ServoP (cartesian velocity, non-blocking)."""
        g = self._servo_gain
        cmd = (f"ServoP({x_mm:.3f},{y_mm:.3f},{z_mm:.3f},"
               f"{rx_deg:.3f},{ry_deg:.3f},{rz_deg:.3f},gain={g})")
        with self._cmd_lock:
            try:
                self._drain_cmd()
                self._cmd_sock.sendall(cmd.encode("utf-8"))
            except Exception as e:
                logger.warning("servop error: %s", e)

    def _runscript(self, project: str):
        """Trigger a DobotStudio project via RunScript."""
        self._send_cmd(f'RunScript("{project}")', read_response=False)

    def _robot_mode_check(self) -> int:
        """Return RobotMode (7=RUNNING)."""
        r = self._send_cmd("RobotMode()", read_response=True, timeout=2.0)
        try:
            return int(r.split(",")[1] if "," in r else r)
        except Exception:
            return -1

    def _wait_idle(self, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._robot_mode_check() != 7:
                return
            time.sleep(0.1)

    # ═══════════════════════════════════════════════════════════════════════
    # Gripper (RunScript mode — binary open/close)
    # ═══════════════════════════════════════════════════════════════════════

    def _gripper_open(self):
        self._wait_idle()
        self._runscript("grip_open")
        self._wait_idle()
        self._gripper_pos = 1.0

    def _gripper_close(self):
        self._wait_idle()
        self._runscript("grip_close")
        self._wait_idle()
        self._gripper_pos = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # Observations
    # ═══════════════════════════════════════════════════════════════════════

    def _read_camera(self, cam: Optional[Any]) -> np.ndarray:
        if cam is None:
            return np.zeros((*self._image_size, 3), dtype=np.uint8)
        frames = cam.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data())
        return img

    def get_observation(self) -> Dict[str, Any]:
        with self._lock:
            eef = self._eef_9d.copy()
            joints = self._joint_pos.copy()
            grip = self._gripper_pos

        return {
            "video.hand_view": self._read_camera(self._cam_hand),
            "video.table_view": self._read_camera(self._cam_table),
            "state.eef_9d": eef,
            "state.joint_pos": joints,
            "state.gripper_pos": np.array([grip], dtype=np.float32),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Env protocol (run_client.py interface)
    # ═══════════════════════════════════════════════════════════════════════

    def reset(self) -> Dict[str, Any]:
        self._steps_since_reset = 0
        self.done = False
        self.success = False
        self.reward = 0.0

        # Open gripper
        try:
            self._gripper_open()
        except Exception:
            logger.warning("gripper open failed on reset")

        time.sleep(0.5)
        return self.get_observation()

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        action = np.asarray(action, dtype=np.float64).ravel()
        assert action.shape == (16,), f"action must be (16,), got {action.shape}"

        with self._lock:
            cur_eef = self._eef_9d.astype(np.float64).copy()
            cur_grip = self._gripper_pos

        # eef delta: target - current
        targ_eef = action[EEF9D_SLICE]
        eef_delta = targ_eef - cur_eef

        # position delta (xyz) in mm
        pos_delta_mm = eef_delta[:3] * M_TO_MM

        # rotation delta: R_targ @ R_cur^T → axis-angle velocity
        R_cur = _rot6d_to_matrix(cur_eef[3:9])
        R_targ = _rot6d_to_matrix(targ_eef[3:9])
        R_delta = R_targ @ R_cur.T
        rxyz_delta_deg = R.from_matrix(R_delta).as_euler("XYZ", degrees=True)

        # scale to velocity
        dt = 0.1  # step period in seconds (10 Hz control)
        vel_mm = pos_delta_mm / dt
        vel_deg = rxyz_delta_deg / dt

        # Clamp velocities
        max_vel_mm = 50.0  # m/s equivalent safety cap
        vel_mm = np.clip(vel_mm, -max_vel_mm, max_vel_mm)
        vel_deg = np.clip(vel_deg, -30.0, 30.0)

        if self._translation_only:
            vel_deg = np.zeros(3)

        self._servop(vel_mm[0], vel_mm[1], vel_mm[2],
                     vel_deg[0], vel_deg[1], vel_deg[2])

        # gripper
        grip_target = float(action[GRIPPER_IDX])
        time.sleep(0.02)
        if grip_target < 0.5 and cur_grip >= 0.5:
            self._gripper_close()
        elif grip_target >= 0.5 and cur_grip < 0.5:
            self._gripper_open()

        self._steps_since_reset += 1
        return {"executed_action": action.astype(np.float64)}

    def get_info_for_step(self) -> Tuple[bool, bool, float, float]:
        # Manual success via keyboard (from client.real_utils.detector)
        try:
            from client.real_utils.detector import success_detector_manual
            manual = success_detector_manual()
            if manual == "success":
                self.done, self.success = True, True
            elif manual == "reset":
                self.done, self.success = True, False
            else:
                self.done, self.success = False, False
        except Exception:
            self.done, self.success = False, False

        reward = 1.0 if self.success else 0.0
        mask = 0.0 if self.done else 1.0
        return self.done, self.success, reward, mask

    def close(self):
        self._running = False
        for s in (self._rt_sock, self._cmd_sock):
            if s:
                try:
                    s.close()
                except Exception:
                    pass
        for cam in (self._cam_hand, self._cam_table):
            if cam:
                try:
                    cam.stop()
                except Exception:
                    pass
        logger.info("CR5AF gripper env closed.")
