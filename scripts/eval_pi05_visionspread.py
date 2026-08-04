#!/usr/bin/env python
"""Vision-spread test for PI0.5 SFT — the clean 'is vision alive?' check.

The near-identity collapse (absolute-action copy task) suppresses the policy's
dependence on the camera: the output tracks proprioception and ignores pixels.
This test isolates that dependence WITHOUT the OOD-feedback confound of the
closed-loop rollout.

The policy outputs an ABSOLUTE next pose reconstructed as `state + delta`, so
the raw action is trivially sensitive to the state (additive). We therefore
measure the spread of the model's actual prediction, the DELTA (= action - state):

  * image sensitivity: hold ONE state fixed, swap in K different frames' images
    -> spread of the predicted delta across images. Large  => vision drives motion.
  * state sensitivity: hold ONE image fixed, swap in K different frames' states
    -> spread of the predicted delta across states (its own state subtracted).

Verdict: image_spread >> state_spread  => vision-driven (alive).
         image_spread ~ 0              => collapse (pixels ignored).

Usage:
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python scripts/eval_pi05_visionspread.py \
        --checkpoint /data/.../pi05_shaft_insert_sft/19999 --num-anchors 16
"""
from __future__ import annotations

import logging
import pathlib

os = __import__("os")
os.environ.setdefault("HF_LEROBOT_HOME", "/datasets/lerobot")

import numpy as np  # noqa: E402

from openpi.training import config as _config  # noqa: E402
from openpi.policies.policy_config import create_trained_policy  # noqa: E402
from openpi.shared import normalize as _normalize  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_NAME = "expo_pi05_droid_lora_finetune_sft_cartesian_state"
REPO_ID = "cr5af/shaft_insert"
PROMPT = "grasp motor shaft and insert into bushing"

import tyro  # noqa: E402
from dataclasses import dataclass  # noqa: E402


@dataclass
class Args:
    checkpoint: str = f"checkpoints/{CONFIG_NAME}/pi05_shaft_insert_sft/19999"
    num_anchors: int = 16
    config: str = CONFIG_NAME


def to_uint8_hwc(image) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = (255.0 * image).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def _pos_spread_mm(deltas: np.ndarray) -> float:
    """L2 norm (mm) of per-axis std over samples, for the xyz delta channels."""
    return float(np.linalg.norm(np.std(deltas[:, :3], axis=0)) * 1000.0)


def _rot_spread(deltas: np.ndarray) -> float:
    """Mean per-dim std over samples for the 6 rot6d channels."""
    return float(np.mean(np.std(deltas[:, 3:9], axis=0)))


def main(args: Args) -> None:
    cfg = _config.get_config(args.config)
    ckpt = pathlib.Path(args.checkpoint)
    import glob

    ns_matches = glob.glob(str(ckpt / "assets" / "**" / "norm_stats.json"), recursive=True)
    if not ns_matches:
        raise FileNotFoundError(f"No norm_stats.json under {ckpt}/assets")
    norm_stats = _normalize.load(pathlib.Path(ns_matches[0]).parent)
    logger.info("Loaded norm_stats from %s", ns_matches[0])
    logger.info("Loading policy from %s", ckpt)
    policy = create_trained_policy(cfg, ckpt, default_prompt=PROMPT, norm_stats=norm_stats)

    ds = LeRobotDataset(REPO_ID)
    n = len(ds)
    idxs = np.linspace(0, n - 1, args.num_anchors).astype(int)
    logger.info("Sampling %d anchor frames across %d", len(idxs), n)

    anchors = []
    for idx in idxs:
        it = ds[int(idx)]
        cart = np.asarray(it["cartesian_position"], dtype=np.float32).flatten()
        grip = np.asarray(it["gripper_position"], dtype=np.float32).reshape(-1)
        anchors.append({
            "ext": to_uint8_hwc(it["exterior_image_1_left"]),
            "wrist": to_uint8_hwc(it["wrist_image_left"]),
            "state": np.concatenate([cart, grip]),  # 10D
        })

    def _infer(ext, wrist, state10):
        obs = {
            "observation/exterior_image_1_left": ext,
            "observation/wrist_image_left": wrist,
            "observation/cartesian_position": state10[:9].astype(np.float32),
            "observation/gripper_position": state10[9:10].astype(np.float32),
            "prompt": PROMPT,
        }
        return np.asarray(policy.infer(obs)["actions"])[0, :10]

    mid = len(anchors) // 2
    fixed_state = anchors[mid]["state"]
    fixed_img = anchors[mid]

    # image sensitivity: fixed state, vary image -> delta = action - fixed_state
    img_deltas = []
    for a in anchors:
        act = _infer(a["ext"], a["wrist"], fixed_state)
        img_deltas.append(act - fixed_state)
    img_deltas = np.stack(img_deltas)

    # state sensitivity: fixed image, vary state -> delta = action - that state
    st_deltas = []
    for a in anchors:
        act = _infer(fixed_img["ext"], fixed_img["wrist"], a["state"])
        st_deltas.append(act - a["state"])
    st_deltas = np.stack(st_deltas)

    img_pos, img_rot = _pos_spread_mm(img_deltas), _rot_spread(img_deltas)
    st_pos, st_rot = _pos_spread_mm(st_deltas), _rot_spread(st_deltas)

    print("\n=== PI0.5 SFT vision-spread test ===")
    print(f"anchors={len(anchors)}  (delta = predicted action - input state)\n")
    print(f"{'vary':<14} {'pos-delta spread (mm)':>22} {'rot6d-delta spread':>20}")
    print(f"{'image':<14} {img_pos:>22.3f} {img_rot:>20.5f}")
    print(f"{'state':<14} {st_pos:>22.3f} {st_rot:>20.5f}")
    ratio = img_pos / st_pos if st_pos > 1e-9 else float("inf")
    print(f"\nimage/state pos-spread ratio: {ratio:.2f}")
    verdict = ("VISION-DRIVEN (alive)" if ratio > 2.0
               else "MIXED" if ratio > 0.8
               else "PROPRIOCEPTION-DRIVEN (vision weak/dead)")
    print(f"verdict: {verdict}")
    # absolute magnitude sanity: how much does the predicted delta move per axis?
    print(f"\nmean |predicted xyz delta| (image-vary): "
          f"{float(np.mean(np.abs(img_deltas[:, :3])) * 1000):.3f} mm")
    print(f"mean |predicted gripper delta| (image-vary): {float(np.mean(np.abs(img_deltas[:, 9]))):.3f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
