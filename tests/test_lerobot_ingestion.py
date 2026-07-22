"""CPU test for the LeRobot v2 offline loader -> Gr00tReplayBuffer ingestion.

Validates ``iter_lerobot_v2_transitions`` / ``load_lerobot_v2_into_buffer``
(prefilling the RL replay buffer from a LeRobot v2 dataset -- the ingestion gap
``insert_dataset`` previously could not handle).  No GPU/checkpoint required.
Skips if ``/datasets/shaft_insert`` is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.not_gpu

DATASET = "/datasets/shaft_insert"


def _have() -> bool:
    return Path(DATASET, "meta", "info.json").exists()


def _expected_frames(n_episodes: int) -> int:
    eps = [json.loads(line) for line in Path(DATASET, "meta", "episodes.jsonl").read_text().splitlines()]
    return sum(e["length"] for e in eps[:n_episodes])


def _make_buffer():
    from expo_ft.data.gr00t_replay_buffer import Gr00tReplayBuffer

    return Gr00tReplayBuffer(
        camera_keys=["hand_view", "table_view"],
        video_horizon=1,
        image_size=(64, 64),
        env_state_dim=16,
        env_action_dim=16,
        env_action_horizon=8,
        capacity=2000,
        task_description="shaft insert",
        replan_steps=3,
        discount=0.99,
    )


def test_loader_inserts_all_frames():
    if not _have():
        pytest.skip(f"{DATASET} not present")
    from expo_ft.data.lerobot_loader import load_lerobot_v2_into_buffer

    rb = _make_buffer()
    rb.seed(0)
    load_lerobot_v2_into_buffer(rb, DATASET, max_episodes=2)
    assert len(rb) == _expected_frames(2), f"len={len(rb)} expected={_expected_frames(2)}"


def test_inserted_transitions_format_and_sampling():
    if not _have():
        pytest.skip(f"{DATASET} not present")
    from expo_ft.data.lerobot_loader import load_lerobot_v2_into_buffer

    rb = _make_buffer()
    rb.seed(1)
    load_lerobot_v2_into_buffer(rb, DATASET, max_episodes=2)

    # insert_dataset marks offline demos as HIL + success for the actor pool.
    assert rb.dataset_dict["is_hil"][: len(rb)].any()
    assert rb.dataset_dict["is_success"][: len(rb)].any()

    # Images are RGB, resized to the buffer size, and non-trivial.
    img0 = rb.dataset_dict["image_hand_view"][0]
    assert img0.shape == (64, 64, 3) and img0.dtype == np.uint8 and img0.std() > 1.0

    # State/action storage shapes.
    assert rb.dataset_dict["state"][0].shape == (16,)
    assert rb.dataset_dict["actions"][0].shape == (8, 16)  # 1-D target tiled to horizon

    # sample_jax reconstructs a finite, non-degenerate executed chunk (b8 #7).
    batch = rb.sample_jax(8)
    actions = np.asarray(batch["actions"])  # (8, replan_steps, 16)
    assert actions.shape == (8, 3, 16)
    assert np.isfinite(actions).all()

    # success_only sampling is populated (actor BC pool).
    assert rb.sample_jax(4, success_only=True) is not None
