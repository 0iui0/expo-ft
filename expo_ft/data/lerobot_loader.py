"""Load a LeRobot v2 dataset into GR00T replay-buffer transition dicts.

LeRobot v2 stores per-frame parquet (``observation.state``, ``action``) and
mp4 videos (``observation.images.<view>``).  This module yields one transition
dict per frame in the format ``Gr00tReplayBuffer.insert`` / ``insert_dataset``
expects (GR00T native: ``image`` + ``state`` + ``actions``), so offline demos
can prefill the RL replay buffer -- closing the ingestion gap noted in the
pipeline doc (``insert_dataset`` previously handled only GR00T-native and
OpenPI/Droid formats, not LeRobot v2).

Action representation (important): the dataset's ``action[t]`` is the absolute
target pose the demonstrator recorded, where the convert script set
``action[t] = state[t + lookahead]``.  The SFT policy is trained to output
exactly these lookahead targets, and online deployment executes them per step,
so the online ``real_action`` IS a lookahead target of the same form.  Hence we
insert ``action[t]`` directly -- the b8 chunk reconstruction then yields a
target chunk consistent with what the policy outputs.  No per-step recomputation
(e.g. ``state[t+1]``) is wanted: that would mismatch the trained policy.

Videos are decoded lazily per frame (cv2), so a 160-episode prefill streams
without holding all frames in memory.  ``BGR`` (cv2's native) is converted to
``RGB`` to match the GR00T/critic convention.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


def _resolve_paths(dataset_path: str) -> dict:
    """Read info.json templates for parquet/video path resolution."""
    info = json.loads(Path(dataset_path, "meta", "info.json").read_text())
    return {
        "data_tmpl": info["data_path"],
        "video_tmpl": info["video_path"],
        "chunks_size": info.get("chunks_size", 1000),
        "total_episodes": info["total_episodes"],
    }


def iter_lerobot_v2_transitions(
    dataset_path: str,
    camera_keys: Sequence[str],
    image_size: Tuple[int, int],
    max_episodes: int | None = None,
) -> Iterator[Dict]:
    """Yield per-frame transition dicts from a LeRobot v2 dataset.

    Args:
        dataset_path: root of the LeRobot v2 dataset (contains ``meta/``,
            ``data/``, ``videos/``).
        camera_keys: buffer camera views, e.g. ``["hand_view", "table_view"]``.
            Each maps to LeRobot video key ``observation.images.<view>``.
        image_size: buffer storage size ``(H, W)``; frames are resized to it.
        max_episodes: optional cap (for tests / partial prefills).

    Yields: dicts with ``image`` (``{view: (H, W, 3) uint8 RGB}``), ``state``
        ``(env_state_dim,) float32``, ``actions`` ``(env_action_dim,) float32``
        (the per-frame lookahead target), ``next_state``, ``rewards``,
        ``masks``, ``dones``.
    """
    meta = _resolve_paths(dataset_path)
    n = meta["total_episodes"]
    if max_episodes is not None:
        n = min(n, max_episodes)

    h, w = int(image_size[0]), int(image_size[1])

    for ei in range(n):
        ec = ei // meta["chunks_size"]
        pq_path = Path(
            dataset_path, meta["data_tmpl"].format(episode_chunk=ec, episode_index=ei)
        )
        df = pd.read_parquet(pq_path)
        states = np.stack(df["observation.state"].to_list()).astype(np.float32)
        actions = np.stack(df["action"].to_list()).astype(np.float32)

        caps = {}
        for view in camera_keys:
            vk = f"observation.images.{view}"
            vp = Path(
                dataset_path,
                meta["video_tmpl"].format(
                    episode_chunk=ec, episode_index=ei, video_key=vk
                ),
            )
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                raise FileNotFoundError(f"could not open video: {vp}")
            caps[view] = cap

        T = len(states)
        try:
            for t in range(T):
                images: Dict[str, np.ndarray] = {}
                for view in camera_keys:
                    ok, bgr = caps[view].read()
                    if not ok:
                        raise RuntimeError(
                            f"failed to read frame {t}/{T} of episode {ei} ({view})"
                        )
                    if bgr.shape[0] != h or bgr.shape[1] != w:
                        bgr = cv2.resize(bgr, (w, h))
                    images[view] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                is_last = t == T - 1
                yield {
                    "image": images,
                    "state": states[t],
                    "actions": actions[t],  # 1-D lookahead target; buffer tiles it
                    "next_state": states[t + 1] if not is_last else states[t],
                    "rewards": 0.0,
                    "masks": 0.0 if is_last else 1.0,
                    "dones": bool(is_last),
                }
        finally:
            for cap in caps.values():
                cap.release()


class _SizedStream:
    """A streaming iterable that also reports a precomputed length.

    ``Gr00tReplayBuffer.insert_dataset`` calls ``len(dataset)`` (for its log and
    ``tqdm`` total), which a raw generator cannot satisfy.  This wraps the lazy
    generator with a length derived from ``episodes.jsonl`` so the full 160-episode
    prefill streams frame-by-frame without materialising every frame in memory.
    """

    def __init__(self, gen: Iterator[Dict], length: int):
        self._gen = gen
        self._length = length

    def __iter__(self) -> Iterator[Dict]:
        return self._gen

    def __len__(self) -> int:
        return self._length


def _count_frames(dataset_path: str, max_episodes: int | None) -> int:
    eps_file = Path(dataset_path, "meta", "episodes.jsonl")
    lengths = [json.loads(line)["length"] for line in eps_file.read_text().splitlines()]
    if max_episodes is not None:
        lengths = lengths[:max_episodes]
    return int(sum(lengths))


def load_lerobot_v2_into_buffer(buffer, dataset_path: str, max_episodes: int | None = None):
    """Prefill a ``Gr00tReplayBuffer`` from a LeRobot v2 dataset (offline demos).

    Streams transitions into ``buffer.insert_dataset`` (which marks them
    ``is_hil``/``is_success`` for the actor success-only pool).  Memory-bounded:
    videos are decoded lazily, one frame at a time.
    """
    gen = iter_lerobot_v2_transitions(
        dataset_path, buffer._camera_keys, buffer._image_size, max_episodes
    )
    buffer.insert_dataset(_SizedStream(gen, _count_frames(dataset_path, max_episodes)))
    return buffer
