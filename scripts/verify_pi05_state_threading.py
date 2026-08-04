#!/usr/bin/env python
"""Deterministic verification of the delta->absolute state threading fix (Task #7).

The RL rollout path (`Pi05Agent.process_transformed_outputs`) used to feed
`dummy_state = zeros` into the output pipeline. With `AbsoluteActions` in
`data_transforms.outputs` (delta-xyz config), the absolute xyz is reconstructed
as `delta + Unnormalize(state)` -- so zeros makes it `delta + quantile_midpoint`,
i.e. wrong. The fix threads the normalized state the model consumed
(`transformed_inputs["state"]`), mirroring `Policy.infer` (policy.py:104).

This test needs NO model / GPU / flow-matching. It exercises the exact input and
output transform pipelines Pi05Agent builds, doing a round-trip on a REAL frame:

    raw absolute action (recorded next-frame pose)
        --INPUT pipeline (DeltaActions+Normalize)-->  normalized delta action
        --OUTPUT pipeline (Unnormalize+AbsoluteActions), state=S_norm --> absolute

With the correct state the reconstructed xyz must equal the recorded next-frame
xyz (round-trip identity, sub-mm). With state=zeros it lands far away (garbage).

Usage:
    JAXTYPING_DISABLE=1 CUDA_VISIBLE_DEVICES="" \
        python scripts/verify_pi05_state_threading.py \
        --checkpoint /data/openpi_checkpoints/.../pi05_shaft_insert_sft/19999
"""
from __future__ import annotations

import dataclasses
import glob
import logging
import pathlib

os = __import__("os")
os.environ.setdefault("HF_LEROBOT_HOME", "/datasets/lerobot")
os.environ.setdefault("JAXTYPING_DISABLE", "1")

import numpy as np  # noqa: E402

from openpi.training import config as _config  # noqa: E402
from openpi import transforms as _transforms  # noqa: E402
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
    checkpoint: str = f"/data/openpi_checkpoints/{CONFIG_NAME}/pi05_shaft_insert_sft/19999"
    config: str = CONFIG_NAME
    frame: int = 40  # a mid-episode frame where the demo is actually moving


def to_uint8_hwc(image) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = (255.0 * image).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def main(args: Args) -> None:
    cfg = _config.get_config(args.config)
    model = cfg.model
    data_config = cfg.data.create(cfg.assets_dirs, model)

    ckpt = pathlib.Path(args.checkpoint)
    ns_matches = glob.glob(str(ckpt / "assets" / "**" / "norm_stats.json"), recursive=True)
    if not ns_matches:
        raise FileNotFoundError(f"No norm_stats.json under {ckpt}/assets")
    norm_stats = _normalize.load(pathlib.Path(ns_matches[0]).parent)
    data_config = dataclasses.replace(data_config, norm_stats=norm_stats)
    logger.info("norm_stats from %s", ns_matches[0])

    uq = data_config.use_quantile_norm
    input_tf = _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=uq),
        *data_config.model_transforms.inputs,
    ])
    output_tf = _transforms.compose([
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(data_config.norm_stats, use_quantiles=uq),
        *data_config.data_transforms.outputs,
    ])
    action_dim = model.action_dim
    action_horizon = model.action_horizon

    ds = LeRobotDataset(REPO_ID)
    f = args.frame
    cur = np.concatenate([
        np.asarray(ds[f]["cartesian_position"], np.float32).flatten(),
        np.asarray(ds[f]["gripper_position"], np.float32).reshape(-1),
    ])  # 10D absolute
    nxt = np.concatenate([
        np.asarray(ds[f + 1]["cartesian_position"], np.float32).flatten(),
        np.asarray(ds[f + 1]["gripper_position"], np.float32).reshape(-1),
    ])  # 10D absolute (the recorded action target)

    # Run the INPUT pipeline: state is `cur`, action is the recorded next pose.
    # Keys are the flat dataset columns the RepackTransform maps from.
    ext = to_uint8_hwc(ds[f]["exterior_image_1_left"])
    raw = {
        "exterior_image_1_left": ext,
        "exterior_image_2_left": ext,  # dataset has no 2nd exterior; repack requires the key
        "wrist_image_left": to_uint8_hwc(ds[f]["wrist_image_left"]),
        "cartesian_position": cur[:9],
        "gripper_position": cur[9:10],
        "actions": np.tile(nxt.astype(np.float32), (action_horizon, 1)),  # (H,10) absolute; DeltaActions -> delta xyz
        "prompt": PROMPT,
    }
    tin = input_tf(raw)
    state_norm = np.asarray(tin["state"], np.float32).reshape(1, action_dim)  # (1, 32)
    acts_norm = np.asarray(tin["actions"], np.float32)  # (horizon, 32) normalized delta-xyz

    # Feed the SAME normalized action through the output pipeline with (a) the
    # threaded normalized state and (b) zeros -- replicating both branches of the
    # edited process_transformed_outputs.
    def decode(state_vec):
        out = output_tf({"state": state_vec.reshape(action_dim), "actions": acts_norm})
        return np.asarray(out["actions"])  # (horizon, env_action_dim=10)

    abs_with_state = decode(state_norm[0])
    abs_with_zeros = decode(np.zeros(action_dim, np.float32))

    # Step 0 of the chunk is the immediate next action; compare its xyz (meters).
    rec_xyz = nxt[:3]
    got_xyz = abs_with_state[0, :3]
    bad_xyz = abs_with_zeros[0, :3]

    err_state_mm = float(np.linalg.norm(got_xyz - rec_xyz) * 1000.0)
    err_zeros_mm = float(np.linalg.norm(bad_xyz - rec_xyz) * 1000.0)

    print("\n=== Task #7 state-threading verification (frame %d) ===" % f)
    print(f"recorded next-frame xyz (m):     {rec_xyz}")
    print(f"reconstructed xyz  w/ STATE (m): {got_xyz}   err = {err_state_mm:.3f} mm")
    print(f"reconstructed xyz  w/ ZEROS (m): {bad_xyz}   err = {err_zeros_mm:.1f} mm")
    print(f"\n  round-trip error with threaded state: {err_state_mm:.3f} mm")
    print(f"  round-trip error with zeros (old bug): {err_zeros_mm:.1f} mm")
    ok = err_state_mm < 1.0 and err_zeros_mm > err_state_mm * 10
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(state reconstructs the true pose; zeros is off by {err_zeros_mm/max(err_state_mm,1e-6):.0f}x)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
