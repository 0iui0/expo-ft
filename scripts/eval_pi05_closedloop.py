#!/usr/bin/env python
"""Offline CLOSED-LOOP evaluation of PI0.5 SFT on shaft_insert (no sim / no robot).

Open-loop eval feeds the ground-truth state at every step, so it only measures
1-step prediction error — near-identity at ~26 Hz for every policy. This script
instead rolls out the policy autoregressively on proprioception:

    pred_state[0] = gt_state[0]
    for t: a = policy(image[t], pred_state[t]); pred_state[t+1] = a   # a is the
           absolute next pose (policy output transforms already reconstruct it)

The recorded demo image at frame t is replayed (we cannot render the diverged
state without a simulator). This makes the test a clean discriminator:

  * A vision-driven policy keeps predicting the demo's motion from the image, so
    the rolled-out trajectory tracks the ground truth -> small compounding drift.
  * A proprioceptive-copy policy (the collapse failure mode) ignores the image
    and echoes its drifted input state -> drift grows like the "frozen" baseline
    (state pinned at t=0), i.e. as far as the demo travels from its start.

Reports compounding position/rotation drift vs horizon, against the frozen
baseline for scale.

Usage:
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python scripts/eval_pi05_closedloop.py \
        --checkpoint /data/.../pi05_shaft_insert_sft/19999 \
        --num-episodes 3 --max-steps 300
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
    num_episodes: int = 3
    max_steps: int = 300  # cap per-episode rollout length to bound runtime
    config: str = CONFIG_NAME


def to_uint8_hwc(image) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = (255.0 * image).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """rot6d (two 3-vectors) -> 3x3 rotation matrix (Gram-Schmidt)."""
    a = rot6d[:3] / max(np.linalg.norm(rot6d[:3]), 1e-8)
    b = rot6d[3:6] - np.dot(a, rot6d[3:6]) * a
    b = b / max(np.linalg.norm(b), 1e-8)
    c = np.cross(a, b)
    return np.column_stack([a, b, c])


def _geodesic_deg(pred6d: np.ndarray, gt6d: np.ndarray) -> float:
    rp, rg = _rot6d_to_matrix(pred6d), _rot6d_to_matrix(gt6d)
    cos = (np.trace(rp.T @ rg) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _rollout(policy, ds, frm, to, max_steps):
    """Autoregressive proprioceptive rollout over one episode.

    Returns per-step (pos_drift_mm, rot_drift_deg, grip_abs_err) for the model
    rollout and for the frozen baseline (state pinned at t=0)."""
    T = min(to - frm, max_steps)
    # ground-truth trajectory (10D: xyz_m, rot6d, gripper)
    gt = np.stack([
        np.concatenate([
            np.asarray(ds[frm + t]["cartesian_position"], dtype=np.float32).flatten(),
            np.asarray(ds[frm + t]["gripper_position"], dtype=np.float32).reshape(-1),
        ])
        for t in range(T)
    ])

    pred_state = gt[0].copy()
    frozen = gt[0]
    model_m, frozen_m = [], []
    for t in range(T):
        item = ds[frm + t]
        obs = {
            "observation/exterior_image_1_left": to_uint8_hwc(item["exterior_image_1_left"]),
            "observation/wrist_image_left": to_uint8_hwc(item["wrist_image_left"]),
            "observation/cartesian_position": pred_state[:9].astype(np.float32),
            "observation/gripper_position": pred_state[9:10].astype(np.float32),
            "prompt": PROMPT,
        }
        # drift of CURRENT rolled-out state vs GT at this step
        model_m.append((
            float(np.linalg.norm(pred_state[:3] - gt[t][:3]) * 1000.0),
            _geodesic_deg(pred_state[3:9], gt[t][3:9]),
            float(abs(pred_state[9] - gt[t][9])),
        ))
        frozen_m.append((
            float(np.linalg.norm(frozen[:3] - gt[t][:3]) * 1000.0),
            _geodesic_deg(frozen[3:9], gt[t][3:9]),
            float(abs(frozen[9] - gt[t][9])),
        ))
        # step the policy; predicted absolute next pose becomes next state
        a = np.asarray(policy.infer(obs)["actions"])[0, :10]
        pred_state = a.astype(np.float32)
    return np.array(model_m), np.array(frozen_m)


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
    edi = ds.episode_data_index
    frm_all, to_all = edi["from"].tolist(), edi["to"].tolist()
    ep_idxs = np.linspace(0, len(frm_all) - 1, args.num_episodes).astype(int)
    logger.info("Rolling out %d episodes (cap %d steps): %s", len(ep_idxs), args.max_steps, ep_idxs.tolist())

    all_model, all_frozen = [], []
    for ei in ep_idxs:
        frm, to = frm_all[ei], to_all[ei]
        logger.info("episode %d: frames [%d,%d) len=%d", ei, frm, to, min(to - frm, args.max_steps))
        m, f = _rollout(policy, ds, frm, to, args.max_steps)
        all_model.append(m)
        all_frozen.append(f)

    # aggregate by step index across episodes that reached that horizon
    maxT = max(m.shape[0] for m in all_model)

    def _at(buckets, arrs, col):
        out = {}
        for h in buckets:
            vals = [a[h, col] for a in arrs if a.shape[0] > h]
            out[h] = (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)
        return out

    buckets = [b for b in (0, 5, 10, 25, 50, 100, 200, maxT - 1) if b < maxT]
    buckets = sorted(set(buckets))

    print("\n=== PI0.5 SFT closed-loop rollout (proprioceptive autoregression, image replay) ===")
    print(f"episodes={len(all_model)}  max horizon={maxT} steps (~{maxT/26:.1f}s @26Hz)\n")
    print(f"{'horizon':>8} | {'model pos mm':>13} {'frozen pos mm':>14} | "
          f"{'model rot°':>11} {'frozen rot°':>12} | {'n_eps':>5}")
    mp, fp = _at(buckets, all_model, 0), _at(buckets, all_frozen, 0)
    mr, fr = _at(buckets, all_model, 1), _at(buckets, all_frozen, 1)
    for h in buckets:
        print(f"{h:>8} | {mp[h][0]:>13.2f} {fp[h][0]:>14.2f} | "
              f"{mr[h][0]:>11.2f} {fr[h][0]:>12.2f} | {mp[h][1]:>5}")

    # whole-rollout means (per episode, then averaged)
    def _mean_over(arrs, col):
        return float(np.mean([a[:, col].mean() for a in arrs]))

    m_pos, f_pos = _mean_over(all_model, 0), _mean_over(all_frozen, 0)
    m_rot, f_rot = _mean_over(all_model, 1), _mean_over(all_frozen, 1)
    m_grp = _mean_over(all_model, 2)
    print("\nrollout-mean drift:")
    print(f"  model   pos {m_pos:8.2f} mm | rot {m_rot:7.2f}° | grip {m_grp:.3f}")
    print(f"  frozen  pos {f_pos:8.2f} mm | rot {f_rot:7.2f}°  (= how far the demo travels)")
    # Judge at the final horizon, where the demo has actually moved (frozen >> 0);
    # the ratio is meaningless on the static opening frames where frozen ~ 0.
    h = maxT - 1
    mp_f, fp_f = mp[h][0], fp[h][0]
    ratio = mp_f / fp_f if fp_f > 1.0 else float("nan")
    print(f"\n  at full horizon ({h} steps): model {mp_f:.1f} mm vs demo-travel {fp_f:.1f} mm")
    if np.isnan(ratio):
        print("  (demo barely moves over this horizon -> ratio uninformative; use longer/other episodes)")
    else:
        verdict = "TRACKS demo (vision-driven)" if ratio < 0.5 else "DRIFTS ~like frozen (proprioceptive copy)"
        print(f"  model/demo-travel ratio: {ratio:.3f}  ({verdict})")


if __name__ == "__main__":
    main(tyro.cli(Args))
