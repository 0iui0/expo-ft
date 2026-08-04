#!/usr/bin/env python
"""Offline open-loop evaluation of PI0.5 SFT on the shaft_insert dataset.

Loads the trained checkpoint and compares the model's predicted next-pose
action against the ground-truth absolute action stored in the LeRobot
dataset. Reports per-dimension MAE/MSE (position in metres, rotation in
radians, gripper in [0,1]).

No robot or simulator is needed: this checks that the SFT policy converged to
a sensible action distribution (the prerequisite before online RL).

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_pi05_openloop.py \
        --checkpoint checkpoints/<config>/pi05_shaft_insert_sft/19999 \
        --num-frames 100
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
    num_frames: int = 100
    config: str = CONFIG_NAME


def to_uint8_hwc(image) -> np.ndarray:
    """Best-effort coercion to uint8 (H, W, 3) — matches DroidInputs._parse_image."""
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = (255.0 * image).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:  # (C,H,W) -> (H,W,C)
        image = np.transpose(image, (1, 2, 0))
    return image


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """rot6d (two 3-vectors) -> 3x3 rotation matrix (Gram-Schmidt)."""
    a = rot6d[:3] / max(np.linalg.norm(rot6d[:3]), 1e-8)
    b = rot6d[3:6] - np.dot(a, rot6d[3:6]) * a
    b = b / max(np.linalg.norm(b), 1e-8)
    c = np.cross(a, b)
    return np.column_stack([a, b, c])


def _rot6d_geodesic_deg(pred6d: np.ndarray, gt6d: np.ndarray) -> np.ndarray:
    """Per-sample geodesic angle (deg) between two rot6d rotations."""
    out = np.empty(pred6d.shape[0], dtype=np.float64)
    for i in range(pred6d.shape[0]):
        rp, rg = _rot6d_to_matrix(pred6d[i]), _rot6d_to_matrix(gt6d[i])
        cos = (np.trace(rp.T @ rg) - 1.0) / 2.0
        out[i] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return out


def main(args: Args) -> None:
    cfg = _config.get_config(args.config)
    ckpt = pathlib.Path(args.checkpoint)
    # create_trained_policy resolves asset_id from the config default, which does not
    # match the asset_id used during training. Load norm_stats explicitly.
    import glob

    ns_matches = glob.glob(str(ckpt / "assets" / "**" / "norm_stats.json"), recursive=True)
    if not ns_matches:
        raise FileNotFoundError(f"No norm_stats.json under {ckpt}/assets")
    norm_stats = _normalize.load(pathlib.Path(ns_matches[0]).parent)
    logger.info("Loaded norm_stats from %s", ns_matches[0])
    logger.info("Loading policy from %s", ckpt)
    policy = create_trained_policy(cfg, ckpt, default_prompt=PROMPT, norm_stats=norm_stats)

    logger.info("Loading dataset %s", REPO_ID)
    ds = LeRobotDataset(REPO_ID)
    n = len(ds)
    idxs = np.linspace(0, n - 1, args.num_frames).astype(int)
    logger.info("Dataset has %d frames; sampling %d", n, len(idxs))

    preds, gts, states, infer_ms = [], [], [], []
    for i, idx in enumerate(idxs):
        item = ds[int(idx)]
        cart = np.asarray(item["cartesian_position"], dtype=np.float32).flatten()  # 9D
        grip = np.asarray(item["gripper_position"], dtype=np.float32).reshape(-1)  # 1D
        obs = {
            "observation/exterior_image_1_left": to_uint8_hwc(item["exterior_image_1_left"]),
            "observation/wrist_image_left": to_uint8_hwc(item["wrist_image_left"]),
            "observation/cartesian_position": cart,
            "observation/gripper_position": grip,
            "prompt": PROMPT,
        }
        gt = np.asarray(item["actions"], dtype=np.float32).flatten()
        out = policy.infer(obs)
        pred_chunk = np.asarray(out["actions"])  # (action_horizon, 10)
        preds.append(pred_chunk[0, :10])  # next-pose only
        gts.append(gt)
        states.append(np.concatenate([cart, grip]))  # 10D current state = identity baseline
        infer_ms.append(out.get("policy_timing", {}).get("infer_ms", 0.0))
        if (i + 1) % 20 == 0 or i == 0:
            logger.info("  %d/%d  last infer=%.0fms", i + 1, len(idxs), infer_ms[-1])

    preds = np.stack(preds)
    gts = np.stack(gts)
    states = np.stack(states)
    err = preds - gts
    mae = np.mean(np.abs(err), axis=0)
    mse = np.mean(err**2, axis=0)

    print("\n=== PI0.5 SFT open-loop eval ===")
    labels = ["x_m", "y_m", "z_m", "r6_0", "r6_1", "r6_2", "r6_3", "r6_4", "r6_5", "gripper"]
    print(f"{'dim':<12} {'MAE':>12} {'MSE':>12}")
    for j, lab in enumerate(labels):
        print(f"{lab:<12} {mae[j]:>12.5f} {mse[j]:>12.6f}")

    def _report(tag: str, p: np.ndarray) -> None:
        e = p - gts
        pos_mm = float(np.mean(np.abs(e[:, :3])) * 1000.0)
        rot_deg = float(np.mean(_rot6d_geodesic_deg(p[:, 3:9], gts[:, 3:9])))
        grip = float(np.mean(np.abs(e[:, 9])))
        print(f"{tag:<10} pos {pos_mm:8.3f} mm | rot {rot_deg:7.3f} deg | grsp {grip:.4f}")

    print()
    _report("model", preds)
    _report("identity", states)  # naive baseline: predict current state
    print(f"\noverall MSE: {float(np.mean(mse)):.6f}")
    print(f"mean infer: {np.mean(infer_ms):.0f} ms  (excl. first-step JIT)")
    # Action range sanity (catch divergence/NaN).
    print(f"\npred range: [{preds.min():.3f}, {preds.max():.3f}]  gt range: [{gts.min():.3f}, {gts.max():.3f}]")
    print(f"pred gripper mean: {preds[:, 9].mean():.3f}  gt gripper mean: {gts[:, 9].mean():.3f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
