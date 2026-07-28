"""SpaceMouse via hidraw (easyhid) — ported verbatim from record_demo_gripper.py.

Using easyhid/hidraw (not pyspacemouse) so button and axis behaviour matches the
proven collection script exactly: right button = deadman, left button = toggle,
action already normalised to ~[-1, 1].
"""
import threading
import time
from typing import Tuple

import numpy as np


class HidrawSpaceMouse:
    """Reads SpaceMouse via hidraw (easyhid) — works on Bluetooth and USB.

    Background thread polls hidraw for HID reports, caches latest state.
    action = [tx, ty, tz, roll, pitch, yaw] (normalized ~[-1, 1])
    buttons = [BTN_0, BTN_1]
    """

    _SUPPORTED_IDS = [(0x256F, 0xC63A), (0x256F, 0xC62E)]

    def __init__(self, device_path=""):
        self._device = None
        self._axes = [0.0] * 6
        self._buttons = [0, 0]
        self._running = True

        from easyhid import Enumeration

        hid = Enumeration()
        found_dev = None
        for d in hid.find():
            for vid, pid in self._SUPPORTED_IDS:
                if d.vendor_id == vid and d.product_id == pid:
                    found_dev = d
                    break
            if found_dev:
                break

        if found_dev is None:
            raise RuntimeError("No SpaceMouse found via hidraw")

        found_dev.open()
        found_dev.set_nonblocking(True)
        self._device = found_dev
        self._bytes_to_read = 13
        self._thread = threading.Thread(target=self._hidraw_loop, daemon=True)
        self._thread.start()

    def _hidraw_loop(self):
        def _to_int16(lo, hi):
            val = lo | (hi << 8)
            return val - 65536 if val >= 32768 else val

        while self._running:
            try:
                data = self._device.read(self._bytes_to_read)
                if not data:
                    data = self._device.read(self._bytes_to_read, timeout_ms=50)
                if data and len(data) >= 3:
                    channel = data[0]
                    if channel == 1 and len(data) >= 13:
                        self._axes[0] = _to_int16(data[1], data[2]) / 350.0
                        self._axes[1] = _to_int16(data[3], data[4]) / -350.0
                        self._axes[2] = _to_int16(data[5], data[6]) / 350.0
                        self._axes[3] = _to_int16(data[7], data[8]) / -350.0
                        self._axes[4] = _to_int16(data[9], data[10]) / -350.0
                        self._axes[5] = _to_int16(data[11], data[12]) / 350.0
                    elif channel == 3 and len(data) >= 2:
                        btn_byte = data[1]
                        self._buttons[0] = 1 if (btn_byte & 0x01) else 0
                        self._buttons[1] = 1 if (btn_byte & 0x02) else 0
            except Exception:
                pass
            time.sleep(0.001)

    def get_action(self):
        """Return (action_6d, buttons). action_6d = [tx,ty,tz,roll,pitch,yaw] ~[-1,1]."""
        a = self._axes
        action = np.array([a[1], -a[0], a[2], a[4], a[3], a[5]], dtype=np.float32)
        return action, self._buttons[:]

    def close(self):
        self._running = False
        if self._device:
            try:
                self._device.close()
            except Exception:
                pass


# Backward-compat alias: run_client / collect_data use `.spacemouse` built from this.
SpaceMouseExpert = HidrawSpaceMouse


class SpaceMousePolicy:
    def __init__(self, max_lin_vel=1, max_rot_vel=1):
        self.movement_enabled = False
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.spacemouse = HidrawSpaceMouse()
        # One-shot flags for keyboard A/B (e.g. calibration); set by GUI, read and cleared by get_info()
        self._virtual_success = False
        self._virtual_failure = False

    def get_action(self):
        return self.spacemouse.get_action()

    def forward(self, obs, include_info=False):
        """Forward pass: 7D action [lin_vel(3), rot_vel(3), gripper(1)] for collect_data.

        action_6d from the device is already normalised [-1, 1] (tx,ty,tz,roll,pitch,yaw),
        so velocities scale directly by the configured max — no extra /350.
        """
        action_6d, buttons = self.spacemouse.get_action()

        if not self.movement_enabled and np.linalg.norm(action_6d) > 0.0001:
            self.movement_enabled = True

        lin_vel = action_6d[:3] * self.max_lin_vel
        rot_vel = action_6d[3:6] * self.max_rot_vel

        gripper_vel = 1.0 if (len(buttons) > 0 and buttons[0]) else -1.0
        action = np.concatenate([lin_vel, rot_vel, [gripper_vel]])
        action = np.clip(action, -1.0, 1.0)
        if include_info:
            return action, {}
        return action

    def set_virtual_success(self, value=True):
        self._virtual_success = bool(value)

    def set_virtual_failure(self, value=True):
        self._virtual_failure = bool(value)

    def get_info(self):
        success = self._virtual_success
        failure = self._virtual_failure
        self._virtual_success = False
        self._virtual_failure = False
        return {
            "movement_enabled": self.movement_enabled,
            "success": success,
            "failure": failure,
            "controller_on": True,
        }

    def reset_state(self):
        self.movement_enabled = False
        self._virtual_success = False
        self._virtual_failure = False


def test_spacemouse():
    """Interactive test: print action and buttons at ~15Hz. Ctrl+C to stop."""
    sm = HidrawSpaceMouse()
    with np.printoptions(precision=3, suppress=True):
        while True:
            action, buttons = sm.get_action()
            print(f"action: {action}, buttons: {buttons}")
            time.sleep(1 / 15)


def main():
    test_spacemouse()


if __name__ == "__main__":
    main()
