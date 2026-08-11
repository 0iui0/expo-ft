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

# Workspace safety bounds (mm) — clamp ServoP xyz targets so the arm can't be
# driven out of the proven box (matches deploy_cr5af_gripper / the recorder).
WORKSPACE_MIN_MM = np.array([369.0, -245.0, 110.0], dtype=np.float64)
WORKSPACE_MAX_MM = np.array([820.0, 299.0, 442.0], dtype=np.float64)

# CR5AF joint soft limits (deg), read off the controller pendant: only J3 is
# narrow (±160°); all others are ±360°. reset() checks the parked joints against
# these — a joint parked past its limit makes the controller refuse EVERY servo
# command (ServoJ/ServoP alike) with ErrorID -5, so we fail fast instead of
# streaming rejected commands. +2° matches the pendant's limit tolerance band.
JOINT_LIMITS_DEG = np.array([360.0, 360.0, 160.0, 360.0, 360.0, 360.0])
JOINT_LIMIT_TOL_DEG = 2.0


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
        joint_space: bool = False,
        max_joint_vel: float = 120.0,
        max_rot_vel: float = 8.0,
        control_hz: float = 30.0,
        use_spacemouse: bool = True,
        sm_lin_scale: float = 0.01,  # m per unit per step (recorder action_scale=8mm; 10mm @ 8Hz ≈ 80mm/s)
        language_instruction: str = "grasp motor shaft and insert into bushing",
        video_dir: str = "",
        dry_run: bool = False,
        **kwargs,
    ):
        self._robot_ip = robot_ip
        self._command_port = command_port
        self._rt_port = rt_port
        self._image_size = image_size
        self._speed_pct = speed
        self._translation_only = translation_only
        # Held orientation for translation_only mode: captured ONCE per episode and
        # reused. The LIVE RT axis-angle triple flips sign near |rot|=180° (the arm
        # sits at rx≈-179°), so re-reading it each step makes ServoP see a 358° jump
        # → planner rotates the long way → joint4/6 speed spikes past the 234°/s
        # limit (e-stop) and the wrist visibly rotates. A one-time capture is
        # flip-free: the target is byte-constant every step.
        self._held_rot_deg = None
        # dry_run: connect RT telemetry + cameras (so observations flow) but skip
        # the command socket, _enable_robot, and all motion (ServoP/ServoJ/gripper).
        # The robot never moves. Used to validate the online obs/sample/step path
        # without actuating the arm.
        self._dry_run = dry_run
        # Joint-space control (InverseKin + ServoJ) avoids the wrist-singularity
        # (joint5≈90°) Cartesian planner throttle that freezes ServoP. max_joint_vel
        # (deg/s) is the per-joint safety cap applied to each ServoJ delta.
        self._joint_space = joint_space
        self._max_joint_vel = max_joint_vel
        # ServoP orientation rate cap (deg/s). Near this pose the Cartesian->joint
        # map amplifies angular rate onto joint6 ~15x, so 15 deg/s tripped the
        # joint6 planning-speed alarm (code 53, ~233 deg/s vs 234 limit). Keep this
        # low so joint6 stays under the limit.
        self._max_rot_vel = max_rot_vel
        self._language_instruction = language_instruction
        self._dt = 1.0 / control_hz  # step period for ServoP velocity scaling

        # ── thread-safe state cache (SI units) ─────────────────────────────
        self._lock = threading.Lock()
        self._pos = np.zeros(7, dtype=np.float64)  # xyz + quat
        self._tcp_rxyz_deg = np.zeros(3, dtype=np.float64)  # native tool axis-angle (deg)
        self._q = np.zeros(6, dtype=np.float64)     # joint angles (rad)
        self._eef_9d = np.zeros(9, dtype=np.float32)  # current eef_9d
        self._joint_pos = np.zeros(6, dtype=np.float32)
        self._gripper_pos = 1.0  # 1=open, 0=closed
        self._robot_mode: int = 0
        self._connected = False
        # RT validity gate. Until the RT thread parses a real frame (and it stays
        # fresh), step() REFUSES to move — no servo, no gripper. Operating blind
        # (zero RT state) computes targets near the origin → wild motion → e-stop.
        self._rt_valid = False
        self._rt_last_ts = 0.0
        self._rt_first_logged = False
        # HIL state: BTN_1 (right) = deadman (hold to teleop the arm); BTN_0
        # (left) = gripper toggle (rising edge flips open/close). Matches the
        # record_demo_gripper convention.
        self._hil_grip = 1.0          # HIL gripper intent (1=open, 0=closed)
        self._prev_btn0 = False

        # ── RT feed (port 30004) ───────────────────────────────────────────
        self._rt_sock: Optional[socket.socket] = None
        self._running = True
        self._connect_rt()

        # ── command socket (port 29999) ────────────────────────────────────
        self._cmd_sock: Optional[socket.socket] = None
        self._cmd_lock = threading.Lock()
        if not self._dry_run:
            self._connect_cmd()
            self._enable_robot()
            self._gripper_init()  # DHGripInit — REQUIRED before grip_open/close actuate
        # SpaceMouse for HIL (human takeover). Optional — if absent, policy only.
        self._sm_lin_scale = sm_lin_scale
        self._spacemouse = None
        if use_spacemouse and not self._dry_run:
            try:
                from client.real_utils.spacemouse_hil import HidrawSpaceMouse
                self._spacemouse = HidrawSpaceMouse()
                logger.info("[SPACEMOUSE] connected (HIL enabled)")
            except Exception as e:
                logger.warning("[SPACEMOUSE] unavailable (%s) — HIL disabled, policy only", e)
                self._spacemouse = None

        # ── HIL teleop state (recorder-faithful) ────────────────────────────
        # Deadman held -> the 30 Hz streamer reads the spacemouse directly and
        # drives target = nominal + per-cycle increment, exactly like
        # record_demo_gripper (no 8 Hz staircase, no goal-pursuit deadband).
        # step() only flags the takeover + gripper + bookkeeping.
        self._hil_deadman = False
        self._hil_nominal_mm = None       # teleoperated nominal xyz (mm); None = not in HIL
        self._sm_hil_scale_mm = 8.0       # recorder --action-scale: mm per unit per 30 Hz cycle
        self._sm_dead_zone = 0.15         # recorder --dead-zone: reject tremor below this
        # Median zero-offset calibration (recorder L737-743): the device rests at
        # a small per-axis bias; subtract it on every read so idle -> exactly 0 and
        # pushes aren't skewed. Requires the operator NOT touch it at env create.
        self._sm_zero = np.zeros(6, dtype=np.float64)
        if self._spacemouse is not None:
            samples = []
            for _ in range(30):
                samples.append(self._spacemouse.read()[0].astype(np.float64))
                time.sleep(1.0 / 30.0)
            self._sm_zero = np.median(np.asarray(samples), axis=0)
            logger.info("[SPACEMOUSE] zero-offset (median of %d still samples) = %s",
                        len(samples), np.round(self._sm_zero, 3).tolist())

        # Background 30 Hz ServoP streamer — keeps servo mode engaged between the
        # 8 Hz policy steps. At 8 Hz alone the controller drops servo + re-solves
        # IK each step -> jitter. step() sets _servop_target; this thread streams
        # it at 30 Hz. Idle (None) until the first step sets a target.
        self._servop_target = None
        self._servop_thread = None
        # Gripper actuation pause: RunScript(grip_*) needs the controller IDLE
        # (RobotMode != 7), but this streamer keeps re-sending ServoP every cycle
        # (even at the goal, to stop dither) -> RobotMode stays RUNNING -> the
        # gripper RunScript is rejected with -5 (busy) and the grip is lost. While
        # a gripper op is in flight, the streamer skips ServoP so the controller
        # goes idle and the RunScript lands. (record_demo_gripper gets this for
        # free: its delta-threshold skips ServoP when the arm is still.)
        self._gripper_busy = False
        # Streamer arrival deadband (mm). Within it, the streamer holds the goal
        # (fixed) so a vibrating end-effector can't dither the command. Pursuit
        # speed is per-target (max_mms from step()), carried in _servop_target.
        self._servop_deadband_mm = 0.5
        if not self._dry_run:
            self._servop_thread = threading.Thread(target=self._servop_loop, daemon=True)
            self._servop_thread.start()

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
        magic_bytes = struct.pack("<Q", RT_FRAME_MAGIC)  # 8-byte frame sync anchor
        _recv = _parsed = _dropped = 0
        _last_log = time.time()
        while self._running:
            if self._rt_sock is None:
                time.sleep(0.5)
                continue
            try:
                chunk = self._rt_sock.recv(4096)
                if not chunk:
                    raise ConnectionError("RT connection closed")
                buf.extend(chunk)
                _recv += len(chunk)
                # parse 1440-byte frames, SYNCED on the magic at offset
                # RT_TEST_VALUE. TCP is a byte stream; a recv can start mid-frame
                # and a single misaligned 1440-slice stays misaligned forever
                # (every frame fails the magic check -> RT goes stale -> blind
                # env). Re-sync by finding the magic before each frame.
                while len(buf) >= 1440:
                    idx = buf.find(magic_bytes)
                    if idx == -1:
                        del buf[:-7]  # keep partial magic, wait for more
                        break
                    frame_start = idx - RT_TEST_VALUE
                    if frame_start < 0:
                        del buf[:idx + 8]  # magic too early; skip (false positive)
                        continue
                    if len(buf) < frame_start + 1440:
                        del buf[:frame_start]  # discard preamble, wait for full frame
                        break
                    if frame_start > 0:
                        del buf[:frame_start]
                        _dropped += frame_start
                    frame = bytes(buf[:1440])
                    del buf[:1440]
                    self._parse_rt_frame(frame)
                    _parsed += 1
                now = time.time()
                if now - _last_log > 5.0:
                    _last_log = now
                    logger.info("[RT-STATS] recv=%d parsed=%d dropped=%d buf=%d valid=%s age=%.1fs",
                                _recv, _parsed, _dropped, len(buf),
                                self._rt_valid, now - (self._rt_last_ts or now))
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
        # Dobot tool_vector[3:6] is an axis-angle (rotation vector) in degrees.
        # This MUST match the training recorder (record_demo_gripper.rxyz_to_rot6d
        # uses R.from_rotvec); interpreting it as Euler XYZ injects a ~71 deg
        # orientation error and drives the policy out of distribution.
        rot = R.from_rotvec(np.array(tv[3:6]), degrees=True)
        quat = rot.as_quat()  # xyzw

        q = np.array(list(struct.unpack_from("<6d", data, RT_Q_ACTUAL))) * DEG2RAD

        with self._lock:
            self._pos = np.concatenate([xyz, quat])
            self._tcp_rxyz_deg = np.array(tv[3:6], dtype=np.float64)
            self._q = q.copy()
            self._eef_9d = np.concatenate([xyz, _matrix_to_rot6d(rot.as_matrix())]).astype(np.float32)
            self._joint_pos = self._q.astype(np.float32)
            self._robot_mode = struct.unpack_from("<Q", data, RT_ROBOT_MODE)[0]
        # Mark RT valid + fresh. step() gates ALL motion on this — operating
        # blind (zero RT) computes targets near the origin → wild motion → e-stop.
        self._rt_valid = True
        self._rt_last_ts = time.time()
        if not self._rt_first_logged:
            self._rt_first_logged = True
            logger.info("[RT-FIRST] valid frame: joints(deg)=%s xyz_mm=%s rxyz_deg=%s",
                        np.round(np.degrees(q), 1).tolist(),
                        np.round(np.asarray(tv[:3]), 1).tolist(),
                        np.round(np.asarray(tv[3:6]), 1).tolist())

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
                if not read_response:
                    return ""
                # Dobot echoes the command in every reply ("ErrorID,{...},Name(...)").
                # This socket also carries fire-and-forget commands (ServoP/ServoJ/
                # RunScript) whose un-read replies can arrive mid-read; skip any
                # reply whose echoed name != the command just sent so requests and
                # responses stay paired (else e.g. a stray RunScript reply is read
                # as the InverseKin result -> empty {} -> spurious "IK failed").
                name = cmd.split("(", 1)[0]
                self._cmd_sock.settimeout(timeout)
                deadline = time.time() + timeout
                while time.time() < deadline:
                    resp = bytearray()
                    while True:
                        c = self._cmd_sock.recv(1)
                        if not c or c == b";":
                            break
                        resp.extend(c)
                    text = resp.decode("utf-8").strip()
                    if not text or (name + "(") in text:
                        return text
                    # stale reply from an earlier command — skip and keep reading
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
        # ClearError + ResetRobot first: a prior servo-limit alarm (e.g. code 53)
        # latches the controller, after which InverseKin/ServoJ return errors
        # until cleared. Mirrors the proven record_demo_gripper bring-up.
        for cmd in ("ClearError()", "ResetRobot()", "EnableRobot()",
                     f"SpeedFactor({int(self._speed_pct)})",
                     f"AccL({int(self._speed_pct)})"):
            self._send_cmd(cmd, read_response=True, timeout=3.0)
            time.sleep(0.3)

    # ═══════════════════════════════════════════════════════════════════════
    # Robot motion
    # ═══════════════════════════════════════════════════════════════════════

    def _servop(self, x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg):
        """Fire-and-forget ServoP to an ABSOLUTE pose (mm, deg). No optional
        params (``gain=`` earns ErrorID -5 on this firmware) and — critically —
        NO reply read: the proven recorder streams ServoP continuously without
        reading replies. A blocking per-command reply read stalls the stream, so
        the controller drops servo mode and cold-solves each target's IK to the
        canonical branch (joint6 wraps ~360° off the current ~-198°), crossing a
        joint soft limit -> ErrorID -5 on every command. Streaming at a tight
        cadence keeps servo engaged and seeded at the current joints."""
        if self._dry_run:
            return
        cmd = (f"ServoP({x_mm:.3f},{y_mm:.3f},{z_mm:.3f},"
               f"{rx_deg:.3f},{ry_deg:.3f},{rz_deg:.3f})")
        with self._cmd_lock:
            try:
                self._drain_cmd()
                self._cmd_sock.sendall(cmd.encode("utf-8"))
            except Exception as e:
                logger.warning("servop error: %s", e)

    def _servop_loop(self):
        """Background 30 Hz ServoP streamer — recorder-faithful pursuit.

        The proven teleop recorder (record_demo_gripper.py) is smooth because,
        every 30 Hz cycle, it commands ``target = measured_pose + this_cycle's
        increment`` at the *intended* speed, and sends NOTHING when idle. Its
        target is therefore always exactly one increment ahead of where the arm
        IS — impossible to overshoot, no dither at rest.

        Our earlier streamer broke both: it chased an ABSOLUTE 8 Hz goal at a
        fixed 120 mm/s. The policy caps motion at 50 mm/s (~6 mm/step), so the
        streamer rushed the arm past the goal in ~1.5 cycles, overshot on
        inertia, then corrected backward — a back-twitch every step ("先往左再
        往右"). At rest it re-commanded ``measured + step`` each cycle, feeding
        the long end-effector's vibration back in (the moment-arm wobble).

        Fix (recorder-faithful):
          * Pursue at the SAME speed step() used to clamp the goal (``v_mm_s``,
            carried in the target tuple) so the ramp lands on the goal exactly
            as the next 8 Hz goal arrives — constant velocity, never rushing,
            never overshooting.
          * Within the arrival deadband, command the FIXED goal (not the live
            measured pose) so a vibrating arm can't dither the command.
        """
        period = 1.0 / 30.0
        deadband_mm = self._servop_deadband_mm
        _diag_last = 0.0          # [SERVO-DIAG] throttle
        _diag_prev_rem = None     # previous remaining vector (overshoot detect)
        while self._running:
            # A gripper RunScript is in flight — stop servo-ing so the controller
            # goes idle (RobotMode != 7) and the RunScript isn't rejected with -5.
            if self._gripper_busy:
                time.sleep(period)
                continue
            # ── HIL: deadman sensed + teleop driven HERE at 30 Hz ────────────
            # Sensing the deadman (and capturing the nominal) in this 30 Hz loop
            # rather than the ~5 Hz step() cuts takeover latency to ~1 cycle. If it
            # were sensed in step(), the policy kept driving for a full inference
            # period after the grab and the arm lurched to the pending policy goal
            # first (the "jump on takeover"). Teleop then mirrors record_demo_gripper
            # exactly: target = nominal + per-cycle increment, no goal-pursuit, no
            # deadband-hold -> no staircase jitter.
            if self._spacemouse is not None:
                axes, btns = self._read_sm_axes()
                dead = bool(btns[1])
                if dead and not self._hil_deadman:          # rising edge: snap nominal
                    with self._lock:
                        self._hil_nominal_mm = self._pos[:3].astype(np.float64) * M_TO_MM
                        if self._held_rot_deg is None:
                            self._held_rot_deg = self._tcp_rxyz_deg.copy()
                    self._hil_deadman = True
                    logger.info("[HIL] deadman ENGAGED (30Hz streamer) from %s",
                                np.round(self._hil_nominal_mm, 1).tolist())
                elif (not dead) and self._hil_deadman:      # falling edge: hold + resume
                    with self._lock:
                        hold_mm = self._pos[:3].astype(np.float64) * M_TO_MM
                        hold_rot = (self._held_rot_deg.copy() if self._held_rot_deg is not None
                                    else self._tcp_rxyz_deg.copy())
                        self._hil_nominal_mm = None
                    # Hold current pose so policy resume doesn't snap to a stale target.
                    self._servop_target = (hold_mm, hold_rot, 50.0)
                    self._hil_deadman = False
                    logger.info("[HIL] deadman RELEASED (30Hz streamer) — holding, policy resumes")

                if self._hil_deadman and self._hil_nominal_mm is not None:
                    if float(np.max(np.abs(axes[:3]))) > self._sm_dead_zone:
                        d = np.array([axes[0], axes[1], -axes[2]], dtype=np.float64) * self._sm_hil_scale_mm
                        with self._lock:
                            if self._hil_nominal_mm is not None:
                                self._hil_nominal_mm = np.clip(
                                    self._hil_nominal_mm + d, WORKSPACE_MIN_MM, WORKSPACE_MAX_MM)
                    with self._lock:
                        nom = None if self._hil_nominal_mm is None else self._hil_nominal_mm.copy()
                    if nom is not None and self._held_rot_deg is not None:
                        rot = self._held_rot_deg
                        self._servop(nom[0], nom[1], nom[2], rot[0], rot[1], rot[2])
                    time.sleep(period)
                    continue
            t = self._servop_target
            if t is not None:
                tgt_xyz_mm, rot_deg, v_mm_s = t
                v_per_cycle = v_mm_s * period  # max mm/cycle = intended speed
                with self._lock:
                    cur_xyz_mm = self._pos[:3].astype(np.float64) * M_TO_MM
                remaining = tgt_xyz_mm - cur_xyz_mm
                if float(np.max(np.abs(remaining))) < deadband_mm:
                    cmd_xyz = tgt_xyz_mm  # arrived: hold the GOAL (fixed, no dither)
                else:
                    step_mm = np.clip(remaining, -v_per_cycle, v_per_cycle)
                    cmd_xyz = cur_xyz_mm + step_mm  # ramp toward goal at intended speed
                # ── [SERVO-DIAG] behavior-neutral confirmation: with the fix,
                # OVERSHOOT flips should be absent (the arm no longer rushes
                # past the goal). Remove after the run confirms smoothness.
                _now = time.time()
                if _now - _diag_last > 0.15:
                    _diag_last = _now
                    flip = ""
                    if _diag_prev_rem is not None:
                        sign_flip = (np.sign(remaining) * np.sign(_diag_prev_rem) < 0)
                        moving = np.abs(remaining) > deadband_mm
                        if np.any(sign_flip & moving):
                            ax = "".join("xyz"[k] for k in np.where(sign_flip & moving)[0])
                            flip = f" OVERSHOOT[{ax}]"
                    _diag_prev_rem = remaining.copy()
                    logger.info("[SERVO-DIAG] goal=%s meas=%s rem=%s cmd=%s v=%.0f%s",
                                np.round(tgt_xyz_mm, 1).tolist(),
                                np.round(cur_xyz_mm, 1).tolist(),
                                np.round(remaining, 2).tolist(),
                                np.round(cmd_xyz, 1).tolist(), v_mm_s, flip)
                self._servop(cmd_xyz[0], cmd_xyz[1], cmd_xyz[2],
                             rot_deg[0], rot_deg[1], rot_deg[2])
            time.sleep(period)

    def _runscript(self, project: str):
        """Trigger a DobotStudio project via RunScript.

        Project name is UNQUOTED — matches the proven record_demo_gripper
        (RunScript(grip_open), not RunScript(\"grip_open\")); the controller
        rejects the quoted form."""
        if self._dry_run:
            return
        resp = self._send_cmd(f"RunScript({project})", read_response=True, timeout=5.0)
        err = resp.split(",", 1)[0].strip() if resp else "(no reply)"
        logger.info("[GRIPPER] RunScript(%s) -> ErrorID=%s", project, err)

    # ── joint-space control (InverseKin + ServoJ) ───────────────────────────

    @staticmethod
    def _parse_kin(resp: str) -> Optional[np.ndarray]:
        """Parse a Dobot ``ErrorID,{v1,...,v6},FuncName(...)`` reply into a
        6-vector (deg). Returns None on a non-zero ErrorID or malformed reply."""
        if not resp:
            return None
        try:
            if resp.split(",", 1)[0].strip() != "0":
                return None
            lb, rb = resp.find("{"), resp.find("}")
            if lb < 0 or rb < 0:
                return None
            vals = [float(v) for v in resp[lb + 1:rb].split(",")]
            return np.array(vals, dtype=np.float64) if len(vals) == 6 else None
        except Exception:
            return None

    def _inverse_kin(self, x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg,
                     jnear_deg: np.ndarray) -> Optional[np.ndarray]:
        """Cartesian pose (mm, deg axis-angle) -> joint solution (deg), biased to
        ``jnear_deg`` (current config) so the branch never flips. Blocking
        round-trip on the command socket. Returns None if the controller fails."""
        jn = "{" + ",".join(f"{v:.4f}" for v in jnear_deg) + "}"
        cmd = (f"InverseKin({x_mm:.3f},{y_mm:.3f},{z_mm:.3f},"
               f"{rx_deg:.4f},{ry_deg:.4f},{rz_deg:.4f},"
               f"useJointNear=1,jointNear={jn})")
        resp = self._send_cmd(cmd, read_response=True, timeout=1.0)
        j = self._parse_kin(resp)
        if j is None:
            logger.warning("InverseKin bad reply: %r", resp)
        return j

    def _servoj(self, j_deg: np.ndarray):
        """ServoJ joint-space servo (deg). Reads the reply so the socket stays
        synchronous with the per-step InverseKin and any throttle/limit ErrorID
        (e.g. 53) surfaces in the log instead of silently freezing the arm."""
        cmd = "ServoJ(" + ",".join(f"{v:.3f}" for v in j_deg) + ")"
        resp = self._send_cmd(cmd, read_response=True, timeout=1.0)
        if resp and resp.split(",", 1)[0].strip() != "0":
            logger.warning("servoj rejected: %s", resp)

    def _servo_joint(self, target_xyz_mm, target_rot_deg, cur_q_deg, dt):
        """Realize a Cartesian target in joint space: InverseKin (biased to the
        current config) -> per-joint unwrap + rate clamp -> ServoJ. Sidesteps the
        Cartesian wrist singularity that throttles ServoP. Holds if IK fails."""
        if self._dry_run:
            return
        j_targ = self._inverse_kin(target_xyz_mm[0], target_xyz_mm[1], target_xyz_mm[2],
                                   target_rot_deg[0], target_rot_deg[1], target_rot_deg[2],
                                   cur_q_deg)
        if j_targ is None:
            logger.warning("InverseKin failed; holding joints")
            return
        # Unwrap each joint delta into [-180, 180] (kills the joint6 ±180 flip),
        # then cap by the per-joint velocity limit.
        dj = (j_targ - cur_q_deg + 180.0) % 360.0 - 180.0
        max_dj = self._max_joint_vel * dt
        dj = np.clip(dj, -max_dj, max_dj)
        self._servoj(cur_q_deg + dj)

    def probe_ik_latency(self, n: int = 20) -> Tuple[float, float, int]:
        """Time ``n`` InverseKin round-trips at the current pose (no motion).
        Decides whether blocking IK fits the control period before any run."""
        with self._lock:
            pos = self._pos.copy()
            q_deg = np.degrees(self._q.copy())
        xyz_mm = pos[:3] * M_TO_MM
        rot_deg = R.from_quat(pos[3:]).as_rotvec(degrees=True)
        dts, ok = [], 0
        for _ in range(n):
            t0 = time.time()
            j = self._inverse_kin(xyz_mm[0], xyz_mm[1], xyz_mm[2],
                                  rot_deg[0], rot_deg[1], rot_deg[2], q_deg)
            dts.append((time.time() - t0) * 1000.0)
            if j is not None:
                ok += 1
        arr = np.asarray(dts)
        logger.info("InverseKin latency over %d calls: mean=%.1fms max=%.1fms ok=%d/%d "
                    "(control period=%.1fms)", n, arr.mean(), arr.max(), ok, n, self._dt * 1000.0)
        return float(arr.mean()), float(arr.max()), ok

    def stop(self):
        """Stop motion and exit servo mode (StopRobot)."""
        self._send_cmd("StopRobot()", read_response=False)

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

    def _gripper_init(self):
        """Run the DHGripInit project once (blocking homing move, ~5s).

        REQUIRED before grip_open/grip_close actuate — the DHGrip plugin must be
        initialized. Mirrors record_demo_gripper's gripper.initialize()
        (RunScript(grip_init) -> DHGripInit)."""
        self._gripper_busy = True
        try:
            self._wait_idle()
            self._runscript("grip_init")
            self._wait_idle()
        finally:
            self._gripper_busy = False
        logger.info("[GRIPPER] initialized (grip_init / DHGripInit done)")

    def _gripper_open(self):
        self._gripper_busy = True  # pause the ServoP streamer so RobotMode goes idle
        try:
            self._wait_idle()
            self._runscript("grip_open")
            self._wait_idle()
            self._gripper_pos = 1.0
        finally:
            self._gripper_busy = False

    def _gripper_close(self):
        self._gripper_busy = True  # pause the ServoP streamer so RobotMode goes idle
        try:
            self._wait_idle()
            self._runscript("grip_close")
            self._wait_idle()
            self._gripper_pos = 0.0
        finally:
            self._gripper_busy = False

    # ═══════════════════════════════════════════════════════════════════════
    # Observations
    # ═══════════════════════════════════════════════════════════════════════

    def _read_camera(self, cam: Optional[Any]) -> np.ndarray:
        if cam is None:
            return np.zeros((*self._image_size, 3), dtype=np.uint8)
        frames = cam.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data())
        h, w = self._image_size
        if img.shape[0] != h or img.shape[1] != w:
            import cv2
            img = cv2.resize(img, (w, h))
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
        self._held_rot_deg = None  # re-capture held orientation this episode
        logger.info("[ENV-CFG] translation_only=%s joint_space=%s speed=%.0f "
                    "control_hz=%.1f dry_run=%s",
                    self._translation_only, self._joint_space, self._speed_pct,
                    1.0 / self._dt, self._dry_run)

        # Joint-limit self-check: if the arm is parked past a soft limit, every
        # servo command is refused with -5 (and ClearError can't clear the alarm
        # while the joint stays out of range). Fail fast, naming the joint(s), so
        # the operator jogs it back instead of watching a run silently freeze.
        # Skipped in dry_run (no motion -> limits irrelevant).
        if not self._dry_run:
            with self._lock:
                q_deg = np.degrees(self._q.copy())
            over = np.abs(q_deg) > (JOINT_LIMITS_DEG + JOINT_LIMIT_TOL_DEG)
            if over.any():
                bad = ", ".join(f"J{i + 1}={q_deg[i]:.1f}°(limit ±{JOINT_LIMITS_DEG[i]:.0f}°)"
                                for i in np.where(over)[0])
                raise RuntimeError(
                    f"joint(s) parked past soft limit: {bad}. Jog them back within "
                    f"range on the pendant before deploying — servo commands would "
                    f"be refused with ErrorID -5.")

        # Open gripper
        try:
            self._gripper_open()
        except Exception:
            logger.warning("gripper open failed on reset")

        time.sleep(0.5)
        return self.get_observation()

    def _read_sm_axes(self):
        """Spacemouse (axes6, buttons) with the calibrated zero-offset removed
        (recorder L821: ``action[:6] -= zero_offset``)."""
        axes, btns = self._spacemouse.read()
        return axes.astype(np.float64) - self._sm_zero, btns

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        action = np.asarray(action, dtype=np.float64).ravel()
        assert action.shape == (16,), f"action must be (16,), got {action.shape}"

        with self._lock:
            cur_eef = self._eef_9d.astype(np.float64).copy()
            cur_pos = self._pos.copy()  # [xyz_m(3), quat(4)]
            cur_tcp_rxyz = self._tcp_rxyz_deg.copy()  # native tool triple (deg)
            cur_q_deg = np.degrees(self._q.copy())  # current joints (deg)
            cur_grip = self._gripper_pos

        # ── fail-safe: never move blind ─────────────────────────────────────
        # If no valid RT frame has arrived (or it's stale, or the joints read as
        # all-zero), the state cache is zeros/stale → targets compute near the
        # origin → wild motion → e-stop. Hold instead: no servo, no gripper.
        rt_age = time.time() - self._rt_last_ts if self._rt_last_ts else 1e9
        if (not self._rt_valid) or rt_age > 3.0 or np.allclose(cur_q_deg, 0.0):
            why = ("no valid RT frame" if not self._rt_valid
                   else f"RT stale {rt_age:.1f}s" if rt_age > 3.0
                   else "RT joints all-zero (blind)")
            logger.warning("[HOLD-BLIND] %s — refusing to move (no servo/gripper)", why)
            self._rt_valid = False  # require a fresh frame before resuming
            return {"executed_action": action.astype(np.float64), "action_type": "policy"}

        # ── HIL: spacemouse takeover (matches record_demo_gripper) ──────────
        # BTN_1 (right) = DEADMAN: hold to teleop the arm. Released -> policy
        # resumes. Held but not pushing -> HOLD (no motion). Immediate takeover
        # on press (the user's request).
        # BTN_0 (left) = gripper toggle (rising edge flips open/close).
        # Translation: [tx, ty, -tz]*scale (tz negated, recorder line 965).
        action_type = "policy"
        if self._spacemouse is not None:
            _, sm_btns = self._read_sm_axes()
            btn0 = bool(sm_btns[0])
            # BTN_0 rising edge -> toggle HIL gripper (slow RunScript; 8 Hz is fine)
            if btn0 and not self._prev_btn0:
                self._hil_grip = 0.0 if self._hil_grip >= 0.5 else 1.0
                logger.info("[HIL] gripper toggle -> %s",
                            "open" if self._hil_grip >= 0.5 else "close")
            self._prev_btn0 = btn0
            # BTN_1 = deadman. It is sensed AND handled entirely in the 30 Hz
            # streamer (nominal capture, teleop increment, release hold) so takeover
            # latency is ~1 cycle, not a full inference period. If it were sensed
            # here (this loop runs at the ~5 Hz inference rate) the policy would keep
            # driving 0.2-0.5 s after the grab -> the arm lurches to the pending
            # policy goal before HIL engages (the "jump on takeover"). step() only
            # MIRRORS the streamer's flag into the action/buffer + gripper.
            if self._hil_deadman:
                action_type = "human"
                action[GRIPPER_IDX] = self._hil_grip
                with self._lock:
                    nom = None if self._hil_nominal_mm is None else self._hil_nominal_mm.copy()
                if nom is not None:  # executed action = teleop nominal (meters)
                    action[0:3] = nom * MM_TO_M

        # ServoP takes an ABSOLUTE Cartesian target pose (mm, deg), NOT a
        # velocity. (Confirmed on hardware: feeding velocities makes the robot
        # IK-solve the raw numbers as a pose -> "预处理逆解算无解".) Mirror the
        # proven cr5af_server teleop: target = current pose + clamped delta.
        targ_eef = action[EEF9D_SLICE]
        dt = self._dt  # step period in seconds (1/control_hz)

        # ── translation: clamp per-step delta to the policy cap (50 mm/s).
        # HIL motion does NOT pass through here — it is driven directly by the
        # 30 Hz streamer (recorder-faithful); this path is policy-only.
        cur_xyz_mm = cur_pos[:3] * M_TO_MM
        targ_xyz_mm = targ_eef[:3] * M_TO_MM
        max_mms = 50.0  # policy translation cap (mm/s); HIL motion bypasses this path
        pos_delta_mm = np.clip(targ_xyz_mm - cur_xyz_mm, -max_mms * dt, max_mms * dt)
        target_xyz_mm = cur_xyz_mm + pos_delta_mm
        # Workspace safety: never let the target leave the proven box.
        target_xyz_mm = np.clip(target_xyz_mm, WORKSPACE_MIN_MM, WORKSPACE_MAX_MM)

        # ── orientation target (deg, native tool axis-angle) ─────────────────
        if self._translation_only:
            # Hold orientation by CAPTURING the tool triple once per episode and
            # reusing it. The LIVE RT axis-angle triple flips sign near |rot|=180°
            # (the arm sits at rx≈-179°): re-reading it each step makes ServoP see
            # a 358° jump → the planner rotates the long way → joint4/6 planning
            # speed spikes past the 234°/s limit (e-stop) and the wrist visibly
            # rotates. A one-time capture is flip-free (constant target every step).
            if self._held_rot_deg is None:
                self._held_rot_deg = cur_tcp_rxyz.copy()
                logger.info("[HOLD-ORIENT] captured held orientation rxyz_deg=%s",
                            self._held_rot_deg.tolist())
            target_rot_deg = self._held_rot_deg
        elif self._joint_space:
            # Absolute policy orientation target. In joint-space mode the
            # per-joint ServoJ clamp below is the hard rate limit near the wrist
            # singularity, so no Cartesian rate cap here (it can't tame joint6).
            target_rot_deg = R.from_matrix(_rot6d_to_matrix(targ_eef[3:9])).as_rotvec(degrees=True)
        else:
            # ServoP path: compose the delta in SO(3) and rate-cap it by its
            # rotation-vector magnitude (a true angular-speed cap preserving the
            # axis — per-component euler clipping would distort it). NOTE: this
            # cannot tame joint6 at the wrist singularity — use joint_space there.
            R_cur = _rot6d_to_matrix(cur_eef[3:9])
            R_targ = _rot6d_to_matrix(targ_eef[3:9])
            delta_rotvec_deg = R.from_matrix(R_targ @ R_cur.T).as_rotvec(degrees=True)
            ang = float(np.linalg.norm(delta_rotvec_deg))
            max_ang = self._max_rot_vel * dt
            if ang > max_ang:
                delta_rotvec_deg = delta_rotvec_deg * (max_ang / ang)
            R_delta = R.from_rotvec(delta_rotvec_deg, degrees=True).as_matrix()
            # ServoP orientation is the native tool_vector axis-angle (degrees);
            # invert the from_rotvec state-encode with as_rotvec.
            target_rot_deg = R.from_matrix(R_delta @ R_cur).as_rotvec(degrees=True)

        # ── dispatch: joint-space (IK + ServoJ) or Cartesian (ServoP) ────────
        # Skipped during HIL (the 30 Hz streamer drives the arm directly) and for
        # the empty-plan zeros sentinel (targ_eef xyz==0 -> would drive toward the
        # origin, outside the workspace). Both cases must NOT push a servo target.
        zeros_sentinel = bool(np.allclose(targ_eef[:3], 0.0))
        if action_type != "human" and not zeros_sentinel:
            if self._joint_space:
                self._servo_joint(target_xyz_mm, target_rot_deg, cur_q_deg, dt)
            else:
                # Hand the target to the 30 Hz background ServoP streamer. Streaming
                # at a tight cadence keeps servo mode engaged between the 8 Hz policy
                # steps; at 8 Hz alone the controller drops servo, re-solves IK each
                # step, and the arm jitters (the _servop docstring notes this).
                # Carry max_mms so the streamer pursues at the SAME speed step()
                # clamped the goal to — the ramp lands on the goal exactly as the
                # next 8 Hz goal arrives, so the arm never rushes ahead and overshoots.
                self._servop_target = (target_xyz_mm.copy(), target_rot_deg.copy(), float(max_mms))

        # gripper
        grip_target = float(action[GRIPPER_IDX])
        time.sleep(0.02)
        if grip_target < 0.5 and cur_grip >= 0.5:
            self._gripper_close()
        elif grip_target >= 0.5 and cur_grip < 0.5:
            self._gripper_open()

        self._steps_since_reset += 1
        return {"executed_action": action.astype(np.float64), "action_type": action_type}

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
