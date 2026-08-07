"""DROID-format adapter env for PI0.5 over the CR5AF+gripper hardware.

``CR5AFGripperEnv`` (the GR00T-path env) emits GR00T-style observations
(``video.<view>``, ``state.eef_9d``) and consumes 16-dim absolute targets
[xyz_mm, rot6d, joint_deg, gripper].  The PI0.5 / openpi DROID pipeline instead
expects LeRobot-DROID keys (``exterior_image_1_left``, ``wrist_image_left``,
``cartesian_position``, ``gripper_position``) and 10-dim cartesian actions
[xyz_m, rot6d, gripper] -- the same convention written by
``scripts/convert_cr5af_npz_to_lerobot.py``.

This wrapper translates between the two so ``Pi05Agent`` runs unchanged:
    * get_observation() -> DROID-key dict (9D cartesian = xyz_m + rot6d)
    * step(action_10d)   -> converts to 16D absolute target, delegates to the
                            wrapped env (which computes the ServoP velocity delta).

Rotation stays native rot6d end to end (no euler round-trip): the raw eef and
the 16D target both carry rot6d, so obs/action conversions are pure unit scaling.
Joint dims in the 16D target are zero-filled; the wrapped env's ServoP controller
ignores them (cartesian servo only), so this is safe.
"""

from typing import Any, Dict

import numpy as np

from client.envs.cr5af_gripper_env import CR5AFGripperEnv


def _eef9d_to_cartesian(eef_9d: np.ndarray) -> np.ndarray:
    """9D eef [xyz_m(3), rot6d(6)] -> 9D DROID cartesian (identity).

    ``CR5AFGripperEnv`` already emits eef xyz in meters (its RT feed applies
    ``MM_TO_M``), matching the DROID / SFT convention, so this is a passthrough.
    """
    return np.asarray(eef_9d, dtype=np.float32).flatten()[:9]


def _cartesian10d_to_eef16(action_10d: np.ndarray) -> np.ndarray:
    """10D action [xyz_m(3), rot6d(6), gripper] -> 16D target
    [eef_9d(xyz_m, rot6d), joint(6)=0, gripper].

    xyz stays in meters: the wrapped env's ``step`` subtracts the current eef
    (also meters) before scaling the delta to a ServoP velocity in mm/s.
    """
    a = np.asarray(action_10d, dtype=np.float64).flatten()
    xyz = a[:3]
    rot6d = a[3:9]
    joints = np.zeros(6, dtype=np.float64)  # unused by ServoP cartesian controller
    grip = np.array([a[9]], dtype=np.float64)
    return np.concatenate([xyz, rot6d, joints, grip]).astype(np.float64)


class CR5AFGripperDroidEnv:
    """DROID-format adapter wrapping :class:`CR5AFGripperEnv` for PI0.5."""

    def __init__(self, language_instruction: str = "grasp motor shaft and insert into bushing", **kwargs):
        self._env = CR5AFGripperEnv(language_instruction=language_instruction, **kwargs)
        self._prompt = language_instruction

    # ── observation: CR5AF (GR00T-format) -> DROID keys ──────────────────────
    def get_observation(self) -> Dict[str, Any]:
        obs = self._env.get_observation()
        table = np.asarray(obs["video.table_view"], dtype=np.uint8)[..., ::-1]  # BGR->RGB
        hand = np.asarray(obs["video.hand_view"], dtype=np.uint8)[..., ::-1]    # BGR->RGB
        cartesian = _eef9d_to_cartesian(obs["state.eef_9d"])
        gripper = np.asarray(obs["state.gripper_pos"], dtype=np.float32).reshape(-1)
        return {
            "exterior_image_1_left": table,   # base / table camera
            "exterior_image_2_left": table,   # unused by DroidInputs; dup to satisfy repack
            "wrist_image_left": hand,         # wrist camera
            "cartesian_position": cartesian,  # (9,) xyz_m + rot6d
            "gripper_position": gripper,      # (1,)
            "prompt": self._prompt,
        }

    # ── action: 10D DROID cartesian -> 16D CR5AF absolute target ────────────
    def step(self, action: np.ndarray) -> Dict[str, Any]:
        self._env.step(_cartesian10d_to_eef16(action))
        # Return the 10D DROID action as executed_action (NOT the wrapped env's
        # 16D eef target): the replay buffer / norm_stats are in the 10D DROID
        # action space [xyz_m, rot6d, gripper], so an inserted transition must
        # carry the 10D action or Normalize broadcasts (16,) vs (10,) and crashes.
        return {"executed_action": np.asarray(action, dtype=np.float64)}

    # ── delegate everything else to the wrapped env ─────────────────────────
    def __getattr__(self, name: str) -> Any:
        # Called only when the attribute is not found on this wrapper.
        return getattr(self._env, name)
