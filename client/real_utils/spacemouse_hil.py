"""SpaceMouse reader via hidraw (easyhid) — ported from record_demo_gripper.py.

Background thread polls the HID device for reports, caches the latest axes +
button state. ``read()`` returns ``(axes6, buttons)`` where axes6 is the
normalized translation+rotation vector (~[-1, 1]) and buttons is [BTN_0, BTN_1].

Works on the same Bluetooth/USB SpaceMouse the proven recorder uses. Replaces
the pyspacemouse-based SpaceMousePolicy (which failed to find the device).
"""
import threading
import time
import numpy as np


class HidrawSpaceMouse:
    """Reads a SpaceMouse via hidraw (easyhid). Non-blocking; background thread."""

    # 3Dconnexion SpaceMouse Wireless / Pro.
    _SUPPORTED_IDS = [(0x256F, 0xC63A), (0x256F, 0xC62E)]

    def __init__(self):
        self._device = None
        self._axes = [0.0] * 6
        self._buttons = [0, 0]
        self._running = True

        from easyhid import Enumeration

        hid = Enumeration()
        found = None
        for d in hid.find():
            for vid, pid in self._SUPPORTED_IDS:
                if d.vendor_id == vid and d.product_id == pid:
                    found = d
                    break
            if found:
                break
        if found is None:
            raise RuntimeError("No SpaceMouse found via hidraw")
        found.open()
        found.set_nonblocking(True)
        self._device = found
        self._thread = threading.Thread(target=self._hidraw_loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _to_int16(lo, hi):
        val = lo | (hi << 8)
        return val - 65536 if val >= 32768 else val

    def _hidraw_loop(self):
        while self._running:
            try:
                data = self._device.read(13)
                if not data:
                    data = self._device.read(13, timeout_ms=50)
                if data and len(data) >= 3:
                    channel = data[0]
                    if channel == 1 and len(data) >= 13:
                        self._axes[0] = self._to_int16(data[1], data[2]) / 350.0
                        self._axes[1] = self._to_int16(data[3], data[4]) / -350.0
                        self._axes[2] = self._to_int16(data[5], data[6]) / 350.0
                        self._axes[3] = self._to_int16(data[7], data[8]) / -350.0
                        self._axes[4] = self._to_int16(data[9], data[10]) / -350.0
                        self._axes[5] = self._to_int16(data[11], data[12]) / 350.0
                    elif channel == 3 and len(data) >= 2:
                        b = data[1]
                        self._buttons[0] = 1 if (b & 0x01) else 0
                        self._buttons[1] = 1 if (b & 0x02) else 0
            except Exception:
                pass
            time.sleep(0.001)

    def read(self):
        """Return (axes6, buttons2). axes6 = [tx, ty, tz, rx, ry, rz] ~[-1,1]."""
        a = self._axes
        # Match the recorder's get_action axis remap.
        axes = np.array([a[1], -a[0], a[2], a[4], a[3], a[5]], dtype=np.float32)
        return axes, self._buttons[:]

    def close(self):
        self._running = False
        if self._device:
            try:
                self._device.close()
            except Exception:
                pass
