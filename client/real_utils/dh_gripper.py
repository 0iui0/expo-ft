#!/usr/bin/env python3
"""DH PGE gripper driver over the Dobot controller's tool-RS485 Modbus-RTU master.

The DH PGE gripper is wired to the CR5AF controller's tool port (RS485 + 24 V),
NOT to a Jetson serial port. The controller is its only gateway. We drive it by
issuing the controller's Modbus master commands (``ModbusRTUCreate`` /
``SetHoldRegs`` / ``GetHoldRegs``) as dashboard-port (29999) string commands —
the same wire the DobotStudio ``DHGrip`` plugin uses internally. We reuse the
data-collection script's existing dashboard command channel, so no extra socket
or cabling is needed.

Register map: ``/home/tophand/dh-docs/register_map.md`` (Modbus-RTU, FC 0x06
write / 0x03 read, slave 1). A background thread polls the current position and
grip status so the control loop reads cached values without blocking on RS485.

Convention for the GR00T / expo-ft state vector: gripper position is a single
scalar in ``[0, 1]`` with **0 = closed (holding), 1 = open (release)**. The DH
PGE ``POSITION`` register is 0-1000 per-mille of stroke; by DH default 0 = fully
open and 1000 = fully closed, so ``invert=True`` (the default) maps the raw
value to the contract convention. Confirm polarity on hardware and flip with
``invert=False`` if your gripper is wired the other way.
"""

from __future__ import annotations

import threading
import time

# ─── DH PGE register map (slave 1) ───────────────────────────────────────────
REG_INIT = 0x0100            # write: 0x01 home / 0xA5 full recalibration
REG_FORCE = 0x0101           # r/w: 20-100 (%)
REG_POSITION = 0x0103        # r/w: 0-1000 (per-mille of full stroke)
REG_SPEED = 0x0104           # r/w: 1-100 (%)
REG_INIT_STATE = 0x0200      # read: 0 not-init / 1 init / 2 initializing
REG_GRIP_STATUS = 0x0201     # read: 0 moving / 1 reached / 2 gripped / 3 dropped
REG_CURRENT_POSITION = 0x0202  # read: 0-1000 current position

STROKE_MAX = 1000            # per-mille


def parse_reply(reply: str) -> tuple[int, list[int]]:
    """Parse a dashboard modbus reply ``"Err,{vals...},Cmd(...)"``.

    Returns ``(errcode, values)``. Examples:
      ``"0,{1},ModbusRTUCreate(1,115200)"``     -> (0, [1])
      ``"0,{2,500},GetHoldRegs(1,513,2)"``       -> (0, [2, 500])
      ``"0,{},SetHoldRegs(1,259,1,500)"``        -> (0, [])
      ``"-1,{},ModbusRTUCreate(1,115200)"``      -> (-1, [])
    """
    s = (reply or "").strip().rstrip(";").strip()
    if not s:
        return -1, []
    try:
        err = int(s.split(",", 1)[0])
    except ValueError:
        err = -1
    # value list = content between the first '{' and the next '}'
    vals: list[int] = []
    lb = s.find("{")
    rb = s.find("}", lb + 1) if lb != -1 else -1
    if lb != -1 and rb != -1:
        inner = s[lb + 1:rb].strip()
        if inner:
            for tok in inner.split(","):
                tok = tok.strip()
                if tok:
                    try:
                        vals.append(int(float(tok)))
                    except ValueError:
                        pass
    return err, vals


class DHGripper:
    """DH PGE gripper driven through the Dobot controller's RS485 Modbus master.

    ``send`` is a callable ``(cmd: str, timeout: float = 3.0) -> str`` that
    issues a dashboard-port string command and returns its reply. Pass
    ``CR5AFConnection.dashboard_cmd`` from the data-collection script.
    """

    def __init__(self, send, slave_id: int = 1, baud: int = 115200,
                 force: int = 30, speed: int = 50, invert: bool = True):
        self.send = send
        self.slave_id = slave_id
        self.baud = baud
        self.invert = invert
        self._force = force
        self._speed = speed

        self.idx: int | None = None
        self._lock = threading.Lock()
        self._pos_norm = 1.0          # cached contract position [0,1], default open
        self._status: int | None = None
        self._running = False
        self._poll_thread: threading.Thread | None = None

        self._configure_tool_port()
        self._create_master()

    # ─── connection setup ───────────────────────────────────────────────────
    def _configure_tool_port(self) -> None:
        # All three are best-effort: not every firmware needs SetToolMode, and
        # SetToolPower may already be on. ModbusRTUCreate below is the real gate.
        for cmd in (
            f"SetToolMode(1,0)",            # tool port -> RS485 mode
            f"SetTool485({self.baud},N,1)", # baud / no parity / 1 stop bit
            "SetToolPower(1)",              # 24 V tool power for the gripper
        ):
            try:
                self.send(cmd, 2.0)
            except Exception as e:  # noqa: BLE001
                print(f"[dh] config warn ({cmd}): {e}", flush=True)

    def _create_master(self) -> None:
        reply = self.send(f"ModbusRTUCreate({self.slave_id},{self.baud})", 3.0)
        err, vals = parse_reply(reply)
        if err != 0 or not vals:
            low = (reply or "").lower()
            if "not tcp" in low or "control mode" in low:
                raise RuntimeError(
                    "controller is NOT in TCP/IP mode — switch it to TCP/IP remote "
                    f"control in DobotStudio, then re-run. (reply: {reply!r})")
            raise RuntimeError(
                f"ModbusRTUCreate failed (slave={self.slave_id}, baud={self.baud}): "
                f"{reply!r} — check tool-port wiring / 24 V / DHGrip plugin config")
        self.idx = int(vals[0])

    # ─── init ───────────────────────────────────────────────────────────────
    def initialize(self, full: bool = False, timeout: float = 12.0) -> bool:
        """Home / recalibrate the gripper, then apply force & speed defaults."""
        self._write(REG_INIT, 0xA5 if full else 0x01)
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = self._read(REG_INIT_STATE)[0]
            if st == 1:
                self._write(REG_FORCE, self._force)
                self._write(REG_SPEED, self._speed)
                return True
            time.sleep(0.1)
        raise RuntimeError("gripper initialize timed out (INIT_STATE != 1)")

    # ─── low-level register access ──────────────────────────────────────────
    def _write(self, addr: int, value: int) -> str:
        reply = self.send(f"SetHoldRegs({self.idx},{addr},1,{int(value)})", 2.0)
        err, _ = parse_reply(reply)
        if err != 0:
            raise RuntimeError(f"SetHoldRegs({addr:#06x},{value}) failed: {reply!r}")
        return reply

    def _read(self, addr: int, count: int = 1) -> list[int]:
        reply = self.send(f"GetHoldRegs({self.idx},{addr},{count})", 2.0)
        err, vals = parse_reply(reply)
        if err != 0 or len(vals) < count:
            raise RuntimeError(f"GetHoldRegs({addr:#06x},{count}) failed: {reply!r}")
        return vals

    # ─── contract-space position (0 = closed, 1 = open) ─────────────────────
    def _raw_to_norm(self, raw: int) -> float:
        raw = max(0, min(STROKE_MAX, int(raw)))
        return (STROKE_MAX - raw) / STROKE_MAX if self.invert else raw / STROKE_MAX

    def _norm_to_raw(self, pos_norm: float) -> int:
        pos_norm = max(0.0, min(1.0, float(pos_norm)))
        raw = round(pos_norm * STROKE_MAX)
        return STROKE_MAX - raw if self.invert else raw

    def set_position(self, pos_norm: float) -> None:
        """Command target position in contract units [0,1] (0=close, 1=open)."""
        self._write(REG_POSITION, self._norm_to_raw(pos_norm))

    def set_force(self, pct: int) -> None:
        self._force = max(20, min(100, int(pct)))
        self._write(REG_FORCE, self._force)

    def set_speed(self, pct: int) -> None:
        self._speed = max(1, min(100, int(pct)))
        self._write(REG_SPEED, self._speed)

    def get_position(self) -> float:
        """Cached current position in contract units [0,1] (non-blocking)."""
        with self._lock:
            return self._pos_norm

    def get_status(self) -> int | None:
        with self._lock:
            return self._status

    def open(self) -> None:
        self.set_position(1.0)

    def close(self) -> None:
        self.set_position(0.0)

    # ─── background poll ────────────────────────────────────────────────────
    def start_poll(self, hz: float = 25.0) -> None:
        if self._running:
            return
        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, args=(1.0 / hz,), daemon=True)
        self._poll_thread.start()

    def _poll_once(self) -> None:
        # One round-trip reads GRIP_STATUS + CURRENT_POSITION (contiguous regs).
        try:
            status, raw = self._read(REG_GRIP_STATUS, count=2)
            with self._lock:
                self._status = int(status)
                self._pos_norm = self._raw_to_norm(raw)
        except Exception:  # noqa: BLE001 — keep last cached value on transient RS485 errors
            pass

    def _poll_loop(self, dt: float) -> None:
        while self._running:
            self._poll_once()
            time.sleep(dt)

    def stop_poll(self) -> None:
        self._running = False
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=1.0)

    def close_conn(self) -> None:
        self.stop_poll()
        if self.idx is not None:
            try:
                self.send(f"ModbusClose({self.idx})", 2.0)
            except Exception:  # noqa: BLE001
                pass
            self.idx = None

    def __enter__(self) -> "DHGripper":
        return self

    def __exit__(self, *_exc) -> None:
        self.close_conn()


class PluginGripper:
    """DH PGE via the DobotStudio DHGrip plugin, triggered with ``RunScript``.

    On this controller firmware the wrist-aviation tool-flange RS485 is only
    reachable by the controller's INTERNAL gripper service (the DHGrip plugin) —
    an external ``ModbusRTUCreate`` master cannot see the slave (verified: every
    baud/parity/slave combo times out with -30004 while the plugin controls the
    gripper fine). So instead of driving Modbus ourselves, we trigger named
    DobotStudio projects that contain the plugin's open/close action.

    Requirements on the controller (DobotStudio):
      1. DHGrip plugin ENABLED,
      2. saved projects containing the plugin's Lua calls, e.g.
           grip_init  -> DHGripInit({id=1})
           grip_open  -> DHGripControl(100, force, speed, {id=1, isBlock=true})
           grip_close -> DHGripControl(0,   force, speed, {id=1, isBlock=true})
         (DHGrip width is 0-100: 0 = closed, 100 = open.)
         Project names default to grip_init / grip_open / grip_close.

    Consequence: gripper state is BINARY (open=1.0 / closed=0.0). RunScript runs
    a whole saved project and returns no value, so there is no continuous width
    feedback over this path (the plugin's DHGripWidthGet is only callable from
    inside a controller Lua project). ``get_position()`` returns the last
    commanded intent — matching the original TopHand recording's binary
    ``gripper_states`` convention. expo-ft consumes it unchanged (gripper_pos
    stays the last 1-D scalar in the 16-D vector, 0=closed / 1=open, ABSOLUTE).

    ``send`` is the same ``dashboard_cmd``-style callable ``DHGripper`` takes.
    """

    def __init__(self, send, open_project: str = "grip_open",
                 close_project: str = "grip_close", init_project: str = "grip_init",
                 project_timeout: float = 12.0, invert: bool = True):
        # ``invert`` kept for interface symmetry with DHGripper; unused here
        # because open/close are named projects, not raw positions.
        self.send = send
        self.open_project = open_project
        self.close_project = close_project
        self.init_project = init_project
        self.project_timeout = project_timeout
        self._pos_norm = 1.0     # 1.0 = open (matches record_demo start-open)
        self._status = 1         # 1 = reached (no live feedback on this path)

    def _robot_mode(self):
        r = self.send("RobotMode()", 2.0)
        err, vals = parse_reply(r)
        return vals[0] if (err == 0 and vals) else None

    def _wait_idle(self, timeout: float) -> None:
        """Block until no project is running (RobotMode != 7 RUNNING)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._robot_mode() != 7:
                return
            time.sleep(0.05)

    def _run(self, project: str, wait: bool = True) -> None:
        # RunScript is ASYNC: the controller starts the project (RobotMode -> 7)
        # and returns immediately; a second RunScript while one is running replies
        # -5 (busy). So wait for idle BEFORE issuing, and (if wait) after, so the
        # gripper action has actually completed before we move on.
        self._wait_idle(self.project_timeout)
        reply = self.send(f"RunScript({project})", 5.0)
        err, _ = parse_reply(reply)
        if err != 0:
            raise RuntimeError(
                f"RunScript({project}) failed: {reply!r} — is the project saved on "
                "the controller and the DHGrip plugin enabled? (-5 = project busy)")
        if wait:
            self._wait_idle(self.project_timeout)

    def initialize(self, full: bool = False) -> bool:
        """Run the init project (DHGripInit) once and wait for it to finish.

        DHGripInit is a blocking homing move (~5 s, RobotMode=7 while running);
        _run waits for RobotMode to leave 7 so the first open/close won't hit -5.
        """
        if self.init_project:
            self._run(self.init_project)
        return True

    def _busy(self) -> bool:
        return self._robot_mode() == 7

    def _fire(self, project: str) -> None:
        """Issue RunScript WITHOUT waiting for it to finish (loop-safe).

        Used by open()/close() during the control loop so a gripper command never
        freezes the teleop/ServoP (and admittance) stream. Silently skips if a
        gripper project is already running (RobotMode==7) — this avoids a -5 busy
        error; the G-key debounce plus gripper move time make the skip rare, and
        the next toggle retries. initialize() uses the blocking _run() instead.
        """
        if self._busy():
            return
        reply = self.send(f"RunScript({project})", 5.0)
        err, _ = parse_reply(reply)
        if err != 0 and err != -5:
            raise RuntimeError(
                f"RunScript({project}) failed: {reply!r} — is the project saved on "
                "the controller and the DHGrip plugin enabled?")

    def open(self) -> None:
        self._fire(self.open_project)
        self._pos_norm = 1.0

    def close(self) -> None:
        self._fire(self.close_project)
        self._pos_norm = 0.0

    def set_position(self, pos_norm: float) -> None:
        """Binary path: >=0.5 opens, <0.5 closes (no continuous positioning)."""
        if float(pos_norm) >= 0.5:
            self.open()
        else:
            self.close()

    def get_position(self) -> float:
        return self._pos_norm

    def get_status(self):
        return self._status

    # No-op lifecycle hooks so the record script can treat it like DHGripper.
    def start_poll(self, hz: float = 25.0) -> None:
        pass

    def stop_poll(self) -> None:
        pass

    def close_conn(self) -> None:
        pass

    def __enter__(self) -> "PluginGripper":
        return self

    def __exit__(self, *_exc) -> None:
        self.close_conn()
