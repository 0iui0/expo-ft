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

# ── Workspace bounds (matching deploy_cr5af_gripper.py SafetyChecker defaults).
WORKSPACE_MIN_XYZ = np.array([200.0, -400.0, 0.0], dtype=np.float64)    # mm
WORKSPACE_MAX_XYZ = np.array([800.0, 200.0, 600.0], dtype=np.float64)   # mm
MAX_TRANSLATION_DELTA = 10.0   # mm per step (deploy default)
MAX_ROTATION_DELTA = 5.0       # deg per step (deploy default)
MAX_CONSECUTIVE_VIOLATIONS = 10


class SafetyChecker:
    """Clip per-step deltas and enforce workspace limits.

    Ported verbatim from deploy_cr5af_gripper.py — works entirely on deltas
    (delta_xyz_mm, delta_rxyz_deg) to avoid confusion between absolute positions
    and incremental movements.
    """

    def __init__(
        self,
        max_translation_delta: float = MAX_TRANSLATION_DELTA,
        max_rotation_delta: float = MAX_ROTATION_DELTA,
        workspace_min_xyz=WORKSPACE_MIN_XYZ,
        workspace_max_xyz=WORKSPACE_MAX_XYZ,
        max_consecutive_violations: int = MAX_CONSECUTIVE_VIOLATIONS,
    ):
        self.max_translation_delta = max_translation_delta
        self.max_rotation_delta = max_rotation_delta
        self.workspace_min = np.array(workspace_min_xyz, dtype=np.float64)
        self.workspace_max = np.array(workspace_max_xyz, dtype=np.float64)
        self.max_consecutive_violations = max_consecutive_violations
        self.violation_count = 0

    def check_numerics(self, arr: np.ndarray) -> bool:
        """Return True if NaN or inf detected."""
        return np.any(np.isnan(arr)) or np.any(np.isinf(arr))

    def clip_delta(self, delta_xyz, delta_rxyz):
        """Clip translation (magnitude) and rotation (per-axis) independently."""
        trans_norm = np.linalg.norm(delta_xyz)
        if trans_norm > self.max_translation_delta:
            delta_xyz = delta_xyz / trans_norm * self.max_translation_delta
        delta_rxyz = np.clip(delta_rxyz, -self.max_rotation_delta, self.max_rotation_delta)
        return delta_xyz, delta_rxyz

    def check_workspace(self, target_xyz: np.ndarray) -> bool:
        """Return True if target xyz is within workspace."""
        return bool(np.all(target_xyz >= self.workspace_min)
                    and np.all(target_xyz <= self.workspace_max))

    def check_and_clip(self, delta_xyz, delta_rxyz, current_xyz):
        """Apply all safety layers. Returns (clipped_delta_xyz, clipped_delta_rxyz, ok).
        ok=False means too many consecutive violations → stop.
        """
        if self.check_numerics(delta_xyz) or self.check_numerics(delta_rxyz):
            self.violation_count += 1
            return delta_xyz * 0, delta_rxyz * 0, False

        delta_xyz, delta_rxyz = self.clip_delta(delta_xyz, delta_rxyz)

        # Workspace: check if current + delta is within bounds
        target_xyz = current_xyz + delta_xyz
        if not self.check_workspace(target_xyz):
            # Clamp delta so target stays in workspace
            clamped = np.clip(target_xyz, self.workspace_min, self.workspace_max)
            delta_xyz = clamped - current_xyz
            self.violation_count += 1
        else:
            self.violation_count = 0

        if self.violation_count >= self.max_consecutive_violations:
            return delta_xyz, delta_rxyz, False

        return delta_xyz, delta_rxyz, True


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """rot6d (two 3-vectors) → 3x3 rotation matrix (Gram-Schmidt)."""
    a_norm = np.linalg.norm(rot6d[:3])
    if a_norm < 1e-8:
        return np.eye(3, dtype=rot6d.dtype)
    a = rot6d[:3] / a_norm
    b = rot6d[3:6] - np.dot(a, rot6d[3:6]) * a
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-8:
        return np.eye(3, dtype=rot6d.dtype)
    b /= b_norm
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
        control_hz: float = 8.0,
        language_instruction: str = "grasp motor shaft and insert into bushing",
        video_dir: str = "",
        preview: bool = False,
        **kwargs,
    ):
        self._robot_ip = robot_ip
        self._command_port = command_port
        self._rt_port = rt_port
        self._image_size = image_size
        self._speed_pct = speed
        self._control_hz = control_hz
        self._translation_only = translation_only
        self._language_instruction = language_instruction
        self._servo_gain = 250  # lower = softer (default 250, range 200-1000)

        # ── thread-safe state cache (SI units) ─────────────────────────────
        self._lock = threading.Lock()
        self._pos = np.zeros(7, dtype=np.float64)  # xyz + quat
        self._q = np.zeros(6, dtype=np.float64)     # joint angles (rad)
        self._eef_9d = np.zeros(9, dtype=np.float32)  # current eef_9d (xyz in mm, rot6d)
        self._tcp_rxyz_deg = np.zeros(3, dtype=np.float64)  # current orientation rotvec (deg)
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

        # ── DH PGE gripper via the DHGrip plugin (RunScript fire-and-forget).
        # Uses the same RunScript mechanism as record_demo_gripper.py's
        # PluginGripper._fire(): checks RobotMode first, skips if busy, never
        # blocks the control loop. Crucial difference from the old blocking
        # _wait_idle + _runscript pattern that froze the controller.
        self._gripper: Optional[Any] = None
        try:
            from client.real_utils.dh_gripper import PluginGripper
            self._gripper = PluginGripper(
                lambda cmd, timeout=3.0: self._send_cmd(cmd, read_response=True, timeout=timeout))
            self._gripper.initialize(full=False)
            logger.info("PluginGripper (RunScript fire-and-forget) initialised")
        except Exception as e:
            logger.warning("PluginGripper init failed (%s)", e)
            self._gripper = None

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
                try:
                    self._cam_hand = self._init_realsense(rs, camera_serial_hand, image_size)
                except Exception as e:
                    logger.warning("D405 hand camera init failed (%s), using black frames.", e)
            if camera_serial_table:
                try:
                    self._cam_table = self._init_realsense(rs, camera_serial_table, image_size)
                except Exception as e:
                    logger.warning("D455 table camera init failed (%s), using black frames.", e)
        except ImportError:
            logger.warning("pyrealsense2 not available — camera frames will be empty.")

        # ── episode state ──────────────────────────────────────────────────
        self._steps_since_reset = 0
        self.done = False
        self.success = False
        self.reward = 0.0
        self._video_dir = video_dir
        self._preview = bool(preview)

        # Safety checker (ported from deploy_cr5af_gripper.py): clip per-step
        # deltas, keep targets inside the calibrated workspace.
        self._safety = SafetyChecker()

        # Camera frame cache (owned by the preview thread when preview is on,
        # so get_observation never races a concurrent RealSense read).
        self._cam_lock = threading.Lock()
        self._last_hand: Optional[np.ndarray] = None
        self._last_table: Optional[np.ndarray] = None
        self._last_delta_mm = np.zeros(3, dtype=np.float64)
        if self._preview:
            self._preview_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self._preview_thread.start()

        # Fixed-rate control loop: reads latest action target + current robot
        # state, sends ServoP at control_hz. Decouples robot command timing
        # from the RL training-loop inference latency.
        self._ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._ctrl_thread.start()

        # HIL takeover flag: when set by the env server, step() skips EMA
        # smoothing and the 1.5mm dead-zone so the SpaceMouse feels direct.
        self._hil_mode = False

        # Latest action target (set by step(), consumed by the fixed-rate
        # _control_loop thread so ServoP timing is independent of RL step rate).
        self._latest_action: Optional[np.ndarray] = None

        # EMA-smoothed translation target (alpha=0.6, matches deploy) to kill
        # flow-matching diffusion noise jitter on the ServoP target.
        self._ema_target: Optional[np.ndarray] = None

    @staticmethod
    def _init_realsense(rs, serial, image_size):
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        # Capture at native 640x480 (256x256 is below RealSense min resolution),
        # resize to image_size in _read_camera.
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        profile = pipe.start(cfg)
        # Let auto-exposure settle
        for _ in range(30):
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
        # NOTE: xyz in MILLIMETERS, joint angles in DEGREES, and tv[3:6] is the
        # ROTVEC (axis-angle) orientation in degrees — all matching
        # deploy_cr5af_gripper.py and the SFT training data. Converting to SI
        # (m / rad) or treating tv[3:6] as euler XYZ would descale the policy's
        # state input and corrupt the rotation.
        xyz_mm = np.array(tv[:3])
        tcp_rxyz_deg = np.array(tv[3:6])
        rot = R.from_rotvec(tcp_rxyz_deg, degrees=True)
        quat = rot.as_quat()  # xyzw

        q_deg = np.array(list(struct.unpack_from("<6d", data, RT_Q_ACTUAL)))

        with self._lock:
            self._pos = np.concatenate([xyz_mm, quat])
            self._q = q_deg.copy()
            self._tcp_rxyz_deg = tcp_rxyz_deg.copy()  # ServoP rotation base (rotvec, deg)
            self._eef_9d = np.concatenate([xyz_mm, _matrix_to_rot6d(rot.as_matrix())]).astype(np.float32)
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
        """Fire-and-forget ServoP — no gain, no RobotMode check (the ~10ms
        blocking RobotMode call added timing jitter to the 8 Hz control loop).
        Gripper RunScript projects own their busy-guard through PluginGripper."""
        cmd = (f"ServoP({x_mm:.3f},{y_mm:.3f},{z_mm:.3f},"
               f"{rx_deg:.3f},{ry_deg:.3f},{rz_deg:.3f})")
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
        if self._gripper is not None:
            try:
                self._gripper.open()
            except Exception as e:
                logger.warning("gripper open failed: %s", e)
        else:
            self._gripper_pos = 1.0

    def _gripper_close(self):
        if self._gripper is not None:
            try:
                self._gripper.close()
            except Exception as e:
                logger.warning("gripper close failed: %s", e)
        else:
            self._gripper_pos = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # Observations
    # ═══════════════════════════════════════════════════════════════════════

    def _read_camera(self, cam: Optional[Any], native: bool = False) -> np.ndarray:
        if cam is None:
            sz = (480, 640) if native else self._image_size
            return np.zeros((*sz, 3), dtype=np.uint8)
        frames = cam.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data())
        if not native:
            h, w = self._image_size
            if img.shape[0] != h or img.shape[1] != w:
                import cv2
                img = cv2.resize(img, (w, h))
        return img

    def get_observation(self) -> Dict[str, Any]:
        with self._lock:
            eef = self._eef_9d.copy()
            joints = self._joint_pos.copy()
        grip = (self._gripper.get_position() if self._gripper is not None
                else self._gripper_pos)

        # When the preview thread owns the cameras, reuse its latest cached
        # native frames (avoids a concurrent RealSense read) and resize to the
        # policy image_size. Live-read and resize otherwise.
        h_pol, w_pol = self._image_size
        if self._preview:
            with self._cam_lock:
                hand = self._last_hand.copy() if self._last_hand is not None else self._read_camera(self._cam_hand)
                table = self._last_table.copy() if self._last_table is not None else self._read_camera(self._cam_table)
            # The thread caches native (640×480) frames; resize for the policy.
            import cv2
            if hand.shape[0] != h_pol or hand.shape[1] != w_pol:
                hand = cv2.resize(hand, (w_pol, h_pol))
            if table.shape[0] != h_pol or table.shape[1] != w_pol:
                table = cv2.resize(table, (w_pol, h_pol))
        else:
            hand = self._read_camera(self._cam_hand)
            table = self._read_camera(self._cam_table)

        return {
            "video.hand_view": hand,
            "video.table_view": table,
            "state.eef_9d": eef,
            "state.joint_pos": joints,
            "state.gripper_pos": np.array([grip], dtype=np.float32),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Env protocol (run_client.py interface)
    # ═══════════════════════════════════════════════════════════════════════

    def reset(self) -> Dict[str, Any]:
        self._steps_since_reset = 0
        self._ema_target = None  # re-snap EMA to current pose on new episode
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
        """Non-blocking: cache the latest action target and return immediately.

        A separate control-loop thread runs at ``control_hz``, reading the cached
        target + current robot state, computing deltas, enforcing safety, and
        sending ServoP commands at a fixed rate.  This decouples the RL training
        step cadence (which includes variable-duration GPU inference) from the
        robot command rate.
        """
        action = np.asarray(action, dtype=np.float64).ravel()
        assert action.shape == (16,), f"action must be (16,), got {action.shape}"
        with self._cmd_lock:  # reuse for latest-action cache (held briefly)
            if self._latest_action is None:
                self._latest_action = action.copy()
            else:
                self._latest_action[:] = action
        return {"executed_action": action.astype(np.float64)}

    def _control_loop(self):
        """Fixed-rate (control_hz) loop: read cached target + current state,
        compute delta, enforce safety, send ServoP, handle gripper.

        When the SpaceMouse deadman (right button) is held, the loop switches to
        HIL mode: it reads the SpaceMouse directly at 8 Hz (matching record_demo's
        30 Hz pattern) and computes translation deltas inline — no action cache,
        no EMA lag, no run_client indirection.  This prevents the 1-2 Hz
        undersampling that caused severe HIL jitter.
        """
        dt = 1.0 / self._control_hz            # 125 ms (8 Hz) for policy
        dt_hil = 1.0 / max(self._control_hz, 15.0)  # ~67 ms (~15 Hz) for HIL
        _sm: Any = None      # lazy HidrawSpaceMouse (easyhid)
        _hil_grip = 1.0       # 0=closed, 1=open (toggled by left button during deadman)
        _hil_prev_left = False
        _was_hil = False      # track deadman release → snap EMA on transition
        _last_grip_cmd = 0.0  # cooldown: don't fire gripper commands back-to-back
        while self._running:
            t0 = time.time()
            try:
                with self._lock:
                    cur_eef = self._eef_9d.astype(np.float64).copy()
                    cur_rxyz_deg = self._tcp_rxyz_deg.copy()
                cur_grip = (self._gripper.get_position() if self._gripper is not None
                            else self._gripper_pos)

                # ── HIL: SpaceMouse deadman → direct control ─────────────
                is_hil = False
                sm_action = np.zeros(6, dtype=np.float32)
                sm_buttons = [0, 0]
                try:
                    if _sm is None:
                        from client.real_utils.spacemouse import HidrawSpaceMouse
                        _sm = HidrawSpaceMouse()
                    sm_action, sm_buttons = _sm.get_action()
                    deadman = len(sm_buttons) > 1 and bool(sm_buttons[1])
                    if deadman:
                        is_hil = True
                        # gripper toggle (left button, edge-triggered)
                        left = bool(sm_buttons[0])
                        if left and not _hil_prev_left:
                            _hil_grip = 0.0 if _hil_grip > 0.5 else 1.0
                        _hil_prev_left = left
                except Exception:
                    pass  # no SpaceMouse — stay in policy mode

                if is_hil:
                    # SpaceMouse → translation delta (record_demo logic)
                    tx, ty, tz = float(sm_action[0]), float(sm_action[1]), float(sm_action[2])
                    hil_scale = 10.0  # mm / normalised-unit / step at ~15 Hz
                    if float(np.max(np.abs(sm_action[:6]))) >= 0.15:
                        pos_delta = np.array([tx * hil_scale, ty * hil_scale, -tz * hil_scale])
                    else:
                        pos_delta = np.zeros(3, dtype=np.float64)
                    target_xyz = cur_eef[:3] + pos_delta
                    target_xyz = np.clip(target_xyz, WORKSPACE_MIN_XYZ, WORKSPACE_MAX_XYZ)
                    if self._safety.check_workspace(target_xyz):
                        if float(np.linalg.norm(pos_delta)) > 0.0:
                            self._servop(target_xyz[0], target_xyz[1], target_xyz[2],
                                         cur_rxyz_deg[0], cur_rxyz_deg[1], cur_rxyz_deg[2])
                    # gripper — cooldown prevents re-triggering while a
                    # (possibly slow) DHGripControl project is still running.
                    if self._gripper is not None and time.time() - _last_grip_cmd > 3.0:
                        if _hil_grip < 0.5 and cur_grip >= 0.5:
                            self._gripper_close()
                            _last_grip_cmd = time.time()
                        elif _hil_grip > 0.5 and cur_grip < 0.5:
                            self._gripper_open()
                            _last_grip_cmd = time.time()
                    if self._preview:
                        with self._cam_lock:
                            self._last_delta_mm = pos_delta.copy()
                    self._steps_since_reset += 1
                    _was_hil = True
                    # rate-control at HIL rate (faster for smoother tracking)
                    elapsed = time.time() - t0
                    if elapsed < dt_hil:
                        time.sleep(dt_hil - elapsed)
                    continue  # skip policy path below

                # ── Transition HIL→policy: snap to current pose so the
                # policy's absolute target doesn't cause a sudden jump.
                if _was_hil:
                    self._ema_target = None  # reset EMA to current position
                    _was_hil = False
                    # Drop through to policy mode (one hold-step, then normal)

                # ── Policy mode: use cached action from step() ───────────
                with self._cmd_lock:
                    action = (self._latest_action.copy() if self._latest_action is not None
                              else None)
                if action is None:
                    time.sleep(dt)
                    continue

                if not np.isfinite(action).all():
                    action = np.where(np.isfinite(action), action, 0.0)

                # ── motion computation ─────────────────────────────────
                targ_eef = action[EEF9D_SLICE].copy()
                cur_xyz_mm = cur_eef[:3]

                delta_xyz = targ_eef[:3] - cur_xyz_mm
                R_cur = _rot6d_to_matrix(cur_eef[3:9])
                R_targ = _rot6d_to_matrix(targ_eef[3:9])
                R_delta = R_targ @ R_cur.T
                delta_rxyz_deg = R.from_matrix(R_delta).as_rotvec(degrees=True)
                dmag = float(np.linalg.norm(delta_rxyz_deg))
                if dmag > MAX_ROTATION_DELTA and dmag > 1e-6:
                    delta_rxyz_deg = delta_rxyz_deg / dmag * MAX_ROTATION_DELTA
                if dmag > 30.0:
                    delta_rxyz_deg = np.zeros(3, dtype=np.float64)

                safe_delta_xyz, safe_delta_rxyz, ok = self._safety.check_and_clip(
                    delta_xyz, delta_rxyz_deg, cur_xyz_mm)
                if not ok:
                    safe_delta_xyz = np.zeros(3, dtype=np.float64)
                    safe_delta_rxyz = np.zeros(3, dtype=np.float64)

                target_xyz_mm = cur_xyz_mm + safe_delta_xyz
                target_rxyz_deg = cur_rxyz_deg + safe_delta_rxyz
                if self._translation_only:
                    target_rxyz_deg = cur_rxyz_deg

                if self._hil_mode:
                    if self._safety.check_workspace(target_xyz_mm):
                        if float(np.linalg.norm(target_xyz_mm - cur_xyz_mm)) > 0.0:
                            self._servop(target_xyz_mm[0], target_xyz_mm[1], target_xyz_mm[2],
                                         target_rxyz_deg[0], target_rxyz_deg[1], target_rxyz_deg[2])
                else:
                    if self._ema_target is None:
                        self._ema_target = target_xyz_mm.copy()
                    else:
                        self._ema_target = 0.6 * target_xyz_mm + 0.4 * self._ema_target
                    target_xyz_mm = self._ema_target
                    if self._safety.check_workspace(target_xyz_mm):
                        if float(np.linalg.norm(target_xyz_mm - cur_xyz_mm)) >= 1.5:
                            self._servop(target_xyz_mm[0], target_xyz_mm[1], target_xyz_mm[2],
                                         target_rxyz_deg[0], target_rxyz_deg[1], target_rxyz_deg[2])

                if self._preview:
                    with self._cam_lock:
                        self._last_delta_mm = safe_delta_xyz.copy()

                grip_target = float(action[GRIPPER_IDX])
                if self._gripper is not None and time.time() - _last_grip_cmd > 3.0:
                    if grip_target < 0.3 and cur_grip >= 0.5:
                        self._gripper_close()
                        _last_grip_cmd = time.time()
                    elif grip_target > 0.7 and cur_grip < 0.5:
                        self._gripper_open()
                        _last_grip_cmd = time.time()

                self._steps_since_reset += 1
            except Exception as e:
                logger.error("control-loop error: %s", e, exc_info=True)

            # Rate-control: sleep the remainder of this step period
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def _camera_loop(self):
        """Background thread: ONLY read native-resolution RealSense frames into
        the cache (no resize, no cv2 — Qt in a daemon thread would segfault).
        Rendering & resize happen on the main thread (_render_preview /
        get_observation).
        """
        while self._running:
            try:
                hand = self._read_camera(self._cam_hand, native=True)
                table = self._read_camera(self._cam_table, native=True)
                with self._cam_lock:
                    self._last_hand = hand
                    self._last_table = table
            except Exception as e:
                logger.warning("camera read error: %s", e)
            time.sleep(1.0 / 15)  # ~15 Hz

    def _render_preview(self):
        """Render the labelled preview (D455 | D405) from the cached native frames.

        MUST be called on the main thread (asyncio loop, i.e. from step()) so
        cv2.imshow / waitKey are safe with the Qt backend.

        The cached frames are native-resolution RGB (640×480). We convert to BGR
        (cv2's expected format), downscale to 320×240 to match record_demo's
        preview, then upscale 2× so the window is readable.
        """
        import cv2
        try:
            with self._cam_lock:
                hand = None if self._last_hand is None else self._last_hand.copy()
                table = None if self._last_table is None else self._last_table.copy()
                delta = self._last_delta_mm.copy()
            if hand is None or table is None:
                return
            with self._lock:
                cur_xyz = self._eef_9d[:3].copy()
                grip = self._gripper_pos
            # Convert RGB (RealSense rgb8) → BGR for cv2.imshow
            hand = cv2.cvtColor(hand, cv2.COLOR_RGB2BGR)
            table = cv2.cvtColor(table, cv2.COLOR_RGB2BGR)
            th, tw = 240, 320
            preview = np.hstack([
                cv2.resize(table, (tw * 2, th * 2)),
                cv2.resize(hand, (tw * 2, th * 2)),
            ])
            label = (f"D455(table) | D405(hand)"
                     f"  xyz=[{cur_xyz[0]:.0f} {cur_xyz[1]:.0f} {cur_xyz[2]:.0f}]mm"
                     f"  d=[{delta[0]:+.1f} {delta[1]:+.1f} {delta[2]:+.1f}]mm"
                     f"  grip={'CLOSE' if grip < 0.5 else 'OPEN'}")
            cv2.putText(preview, label, (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("CR5AF env (D455 | D405)", preview)
            cv2.waitKey(1)
        except Exception as e:
            logger.warning("preview error: %s", e)

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
        if self._gripper is not None:
            try:
                self._gripper.close_conn()
            except Exception:
                pass
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
