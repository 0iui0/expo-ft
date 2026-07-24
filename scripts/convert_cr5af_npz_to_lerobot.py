"""Convert CR5AF+gripper recordings (npz) to LeRobot DROID format for PI0.5 SFT.

Each recording episode is a directory containing ``data.npz`` + ``meta.json`` with:
    state            (T, 16) float32  = [xyz(3, mm), rot6d(6), joint(6, deg), gripper(1)]
    images_hand      (T, 240, 320, 3) uint8  (D405 wrist, RGB)
    images_table     (T, 240, 320, 3) uint8  (D455 table, RGB)
    gripper_states   (T,) float32      (== state[..., 15])
    timestamps       (T,) float64
    meta.json: {"task", "hz", "result", ...}

Output LeRobot dataset matches ``LeRobotDROIDDataConfig(use_cartesian_state=True,
output_action_dim=7)`` (config name ``expo_pi05_droid_lora_finetune_sft_cartesian_state``):

    exterior_image_1_left  (T, H, W, 3)  <- images_table   (base camera)
    exterior_image_2_left  (T, H, W, 3)  <- images_table   (unused by DroidInputs, dup to satisfy repack)
    wrist_image_left       (T, H, W, 3)  <- images_hand    (wrist camera)
    cartesian_position     (T, 6) float32 = [xyz_m(3), euler_rad(3)]
    gripper_position       (T, 1) float32
    actions                (T, 7) float32 = ABSOLUTE next pose [xyz_m, euler_rad, gripper]
    task                   str            (language instruction)

Action convention: ABSOLUTE next-frame target (not delta). The env consumes absolute
16-dim targets, so the policy predicts absolute 7-dim cartesian targets; the deploy
adapter converts m->mm and euler->rot6d. State/action share meters+radians so the
PI0.5 DROID prior's unit scale transfers; per-dataset norm stats handle the rest.

Usage:
    uv run scripts/convert_cr5af_npz_to_lerobot.py \\
        --data_dir /path/to/shaft_insert_single \\
        --repo_name cr5af/shaft_insert
"""

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# datasets 3.x + numpy 2.x: LeRobot maps a shape-(1,) float feature to a scalar
# ``datasets.Value``, whose ``encode_example`` calls ``float(value)`` -- numpy>=2
# rejects (1,) ndarrays. LeRobot's ``validate_frame`` simultaneously requires a
# (1,) ndarray for shape-(1,) features, so 0-d won't pass either. Unwrap size-1
# ndarrays before the default encoder so (1,) gripper values encode as scalars.
import datasets.features.features as _dsf  # noqa: E402

_orig_value_encode = _dsf.Value.encode_example


def _value_encode_unwrap(self, value):
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return _orig_value_encode(self, value)


_dsf.Value.encode_example = _value_encode_unwrap


# Image storage resolution. Recordings are 240x320; keep native to avoid distortion.
# Model transforms resize to 224x224 internally, so storage resolution is flexible.
IMG_H, IMG_W = 240, 320


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """rot6d (two 3-vectors) -> 3x3 rotation matrix (Gram-Schmidt). Mirrors the env."""
    a = rot6d[:3] / max(np.linalg.norm(rot6d[:3]), 1e-8)
    b = rot6d[3:6] - np.dot(a, rot6d[3:6]) * a
    b = b / max(np.linalg.norm(b), 1e-8)
    c = np.cross(a, b)
    return np.column_stack([a, b, c])


def _state_to_cartesian(state: np.ndarray) -> np.ndarray:
    """16-dim state -> 6-dim cartesian [xyz_m(3), euler_rad(3)].

    state layout: [xyz_mm(3), rot6d(6), joint_deg(6), gripper(1)].
    xyz: mm -> m. rotation: rot6d -> euler 'XYZ' radians.
    """
    xyz_m = state[..., :3] * 0.001
    rot6d = state[..., 3:9]
    single = rot6d.ndim == 1
    r = rot6d[None, :] if single else rot6d
    mats = np.stack([_rot6d_to_matrix(r[i]) for i in range(r.shape[0])])
    euler = Rotation.from_matrix(mats).as_euler("XYZ", degrees=False)
    if single:
        return np.concatenate([xyz_m, euler[0]])
    return np.concatenate([xyz_m, euler], axis=-1)


def _resize_image(image: np.ndarray) -> np.ndarray:
    if image.shape[0] == IMG_H and image.shape[1] == IMG_W:
        return image
    return np.array(Image.fromarray(image).resize((IMG_W, IMG_H), resample=Image.BICUBIC))


def _load_episode(episode_dir: Path):
    """Return (cartesian (T,6), gripper (T,1), images_table, images_hand, task, hz)."""
    d = np.load(episode_dir / "data.npz")
    state = np.asarray(d["state"], dtype=np.float32)
    grip = np.asarray(d["gripper_states"], dtype=np.float32).reshape(-1, 1)
    if np.allclose(grip.squeeze(), state[..., 15]):
        grip = state[..., 15:16].astype(np.float32)
    imgs_table = np.asarray(d["images_table"], dtype=np.uint8)
    imgs_hand = np.asarray(d["images_hand"], dtype=np.uint8)

    meta = json.loads((episode_dir / "meta.json").read_text())
    task = meta.get("task", "grasp motor shaft and insert into bushing")
    hz = float(meta.get("hz", 30.0))

    cartesian = _state_to_cartesian(state).astype(np.float32)
    return cartesian, grip, imgs_table, imgs_hand, task, int(round(hz))


def main(
    data_dir: str,
    *,
    repo_name: str,
    language_instruction: str | None = None,
    max_episodes: int | None = None,
    push_to_hub: bool = False,
):
    data_dir = Path(data_dir)
    episode_dirs = sorted(p for p in data_dir.glob("episode_*") if p.is_dir())
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]
    print(f"Found {len(episode_dirs)} episodes in {data_dir}")

    # Probe first episode for fps + task.
    cartesian0, _, _, _, task0, hz0 = _load_episode(episode_dirs[0])
    fps = hz0
    task = language_instruction or task0
    print(f"fps={fps}, task={task!r}, cartesian dim={cartesian0.shape[-1]}")

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="cr5af_gripper",
        fps=fps,
        features={
            "exterior_image_1_left": {"dtype": "image", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channel"]},
            "exterior_image_2_left": {"dtype": "image", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channel"]},
            "wrist_image_left": {"dtype": "image", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channel"]},
            "cartesian_position": {"dtype": "float32", "shape": (6,), "names": ["cartesian_position"]},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=4,
        image_writer_processes=0,
    )

    n_written = 0
    for ep in tqdm(episode_dirs, desc="Converting episodes"):
        try:
            cartesian, grip, imgs_table, imgs_hand, _, _ = _load_episode(ep)
        except Exception as e:
            print(f"SKIP {ep.name}: load failed ({e})")
            continue
        T = cartesian.shape[0]
        if T < 2:
            print(f"SKIP {ep.name}: too short ({T} frames)")
            continue

        # ABSOLUTE next-pose action: action[t] = [cartesian[t+1], gripper[t+1]]; pad last frame.
        next_cart = np.concatenate([cartesian[1:], cartesian[-1:]], axis=0)
        next_grip = np.concatenate([grip[1:], grip[-1:]], axis=0)
        actions = np.concatenate([next_cart, next_grip], axis=-1).astype(np.float32)

        for t in range(T):
            dataset.add_frame({
                "exterior_image_1_left": _resize_image(imgs_table[t]),
                "exterior_image_2_left": _resize_image(imgs_table[t]),
                "wrist_image_left": _resize_image(imgs_hand[t]),
                "cartesian_position": cartesian[t],
                "gripper_position": grip[t],  # (1,) float32; encode_unwrap handles the scalar mapping
                "actions": actions[t],
                "task": task,
            })
        dataset.save_episode()
        n_written += 1

    print(f"Wrote {n_written} episodes to {output_path}")
    if push_to_hub:
        dataset.push_to_hub(private=False, push_videos=True, license="apache-2.0")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
