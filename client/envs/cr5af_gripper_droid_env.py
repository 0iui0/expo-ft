"""DROID-format adapter env for PI0.5 over the CR5AF+gripper hardware.

``CR5AFGripperEnv`` (the GR00T-path env) emits GR00T-style observations
(``video.<view>``, ``state.eef_9d``) and consumes 16-dim absolute targets
[xyz_mm, rot6d, joint_deg, gripper].  The PI0.5 / openpi DROID pipeline instead
expects LeRobot-DROID keys (``exterior_image_1_left``, ``wrist_image_left``,
``cartesian_position``, ``gripper_position``) and 7-dim cartesian actions
[xyz_m, euler_rad, gripper] -- the same convention written by
``scripts/convert_cr5af_npz_to_lerobot.py``.

This wrapper translates between the two so ``Pi05Agent`` runs unchanged:
    * get_observation() -> DROID-key dict (6D cartesian = xyz_m + euler_rad)
    * step(action_7d)    -> converts to 16D absolute target, delegates to the
                            wrapped env (which computes the ServoP velocity delta).

Joint dims in the 16D target are zero-filled; the wrapped env's ServoP controller
ignores them (cartesian servo only), so this is safe.
"""

from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation

from client.envs.cr5af_gripper_env import CR5AFGripperEnv


def _matrix_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> rot6d (first two columns, concatenated)."""
    return np.concatenate([mat[:, 0], mat[:, 1]]).astype(np.float64)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    a = rot6d[:3] / max(np.linalg.norm(rot6d[:3]), 1e-8)
    b = rot6d[3:6] - np.dot(a, rot6d[3:6]) * a
    b = b / max(np.linalg.norm(b), 1e-8)
    c = np.cross(a, b)
    return np.column_stack([a, b, c])


def _eef9d_to_cartesian(eef_9d: np.ndarray) -> np.ndarray:
    """9D eef [xyz_mm(3), rot6d(6)] -> 6D cartesian [xyz_m(3), euler_rad(3)]."""
    eef = np.asarray(eef_9d, dtype=np.float64).flatten()
    xyz_m = eef[:3] * 0.001
    euler = Rotation.from_matrix(_rot6d_to_matrix(eef[3:9])).as_euler("XYZ", degrees=False)
    return np.concatenate([xyz_m, euler]).astype(np.float32)


def _cartesian7d_to_eef16(action_7d: np.ndarray) -> np.ndarray:
    """7D action [xyz_m, euler_rad, gripper] -> 16D target [xyz_mm, rot6d, joint0, gripper]."""
    a = np.asarray(action_7d, dtype=np.float64).flatten()
    xyz_mm = a[:3] * 1000.0
    rot6d = _matrix_to_rot6d(Rotation.from_euler("XYZ", a[3:6]).as_matrix())
    joints = np.zeros(6, dtype=np.float64)  # unused by ServoP cartesian controller
    grip = np.array([a[6]], dtype=np.float64)
    return np.concatenate([xyz_mm, rot6d, joints, grip]).astype(np.float64)


class CR5AFGripperDroidEnv:
    """DROID-format adapter wrapping :class:`CR5AFGripperEnv` for PI0.5."""

    def __init__(self, language_instruction: str = "grasp motor shaft and insert into bushing", **kwargs):
        self._env = CR5AFGripperEnv(language_instruction=language_instruction, **kwargs)
        self._prompt = language_instruction

    # ── observation: CR5AF (GR00T-format) -> DROID keys ──────────────────────
    def get_observation(self) -> Dict[str, Any]:
        obs = self._env.get_observation()
        table = np.asarray(obs["video.table_view"], dtype=np.uint8)
        hand = np.asarray(obs["video.hand_view"], dtype=np.uint8)
        cartesian = _eef9d_to_cartesian(obs["state.eef_9d"])
        gripper = np.asarray(obs["state.gripper_pos"], dtype=np.float32).reshape(-1)
        return {
            "exterior_image_1_left": table,   # base / table camera
            "exterior_image_2_left": table,   # unused by DroidInputs; dup to satisfy repack
            "wrist_image_left": hand,         # wrist camera
            "cartesian_position": cartesian,  # (6,) xyz_m + euler_rad
            "gripper_position": gripper,      # (1,)
            "prompt": self._prompt,
        }

    # ── action: 7D DROID cartesian -> 16D CR5AF absolute target ─────────────
    def step(self, action: np.ndarray) -> Dict[str, Any]:
        return self._env.step(_cartesian7d_to_eef16(action))

    # ── delegate everything else to the wrapped env ─────────────────────────
    def __getattr__(self, name: str) -> Any:
        # Called only when the attribute is not found on this wrapper.
        return getattr(self._env, name)
