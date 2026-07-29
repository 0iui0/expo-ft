"""Replay buffer for GR00T VLA training within the EXPO-FT framework.

Stores raw (un-normalized) robot data with GR00T modality naming (view names
like ``hand_view``, ``table_view``) and outputs batches compatible with both
the GR00T actor (``Gr00tAgent.prepare_batch_for_actor``) and the EXPO critic.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from expo_ft.data.dataset import Dataset, DatasetDict

logger = logging.getLogger(__name__)


def prepare_gr00t_critic_batch(
    batch: DatasetDict,
    camera_keys: Sequence[str],
    padded_dim: int,
    action_dim: int,
    state_dim: int,
    action_horizon: int,
    replan_steps: int,
) -> DatasetDict:
    """Prepare a GR00T batch for critic training, analogous to ``prepare_critic_batch``.

    The key difference is ``camera_keys``, which specifies which views to
    concatenate into the critic's observation tensor.

    Supports both raw (env_dim) and padded (max_action_dim) action formats.

    Args:
        batch: Replay-buffer sample with ``image`` dict keyed by camera view.
        camera_keys: Ordered list of camera keys to concatenate for critic obs.
        padded_dim: Padded action/state dimension (``model_config.action_dim``).
        action_dim: True environment action dimension.
        state_dim: True environment state dimension.
        action_horizon: Action chunk horizon (env horizon).
        replan_steps: Number of replan steps for action truncation.

    Returns:
        Updated batch with critic-specific fields added.
    """
    batch_size = batch["state"].shape[0]

    # Concatenate camera views along the channel axis and normalize uint8 -> [0,1]
    # float32 for the JAX ResNet critic encoder. The critic is trained from
    # scratch; feeding raw uint8 (0-255) would produce huge activations, so we
    # normalize here (the OpenPI pipeline the critic was designed around fed
    # normalized float).
    def _as_float(img):
        return img.astype(jnp.float32) / 255.0

    obs_parts = [_as_float(batch["image"][k]) for k in camera_keys]
    batch["observations"] = jnp.concatenate(obs_parts, axis=-1)

    next_parts = [_as_float(batch["next_image"][k]) for k in camera_keys]
    batch["next_observations"] = jnp.concatenate(next_parts, axis=-1)

    batch["states"] = batch["state"].reshape(batch_size, -1)[..., :state_dim]
    batch["next_states"] = batch["next_state"].reshape(batch_size, -1)[..., :state_dim]
    batch["critic_states"] = batch["states"]
    batch["next_critic_states"] = batch["next_states"]

    # Actions arrive as the reconstructed executed chunk
    # (B, replan_steps, env_action_dim) from ``sample_jax`` (Fix #7). Trim to the
    # env action dim; the critic and residual actor consume the flattened C-step
    # chunk of shape (B, replan_steps * action_dim).  ``padded_dim`` /
    # ``action_horizon`` are retained in the signature for call-site compatibility
    # but no longer drive the reshape (the chunk length is already replan_steps).
    raw_actions = batch["actions"]
    actions_unpadded = raw_actions[..., :action_dim]  # (B, replan_steps, action_dim)
    batch["actions"] = actions_unpadded.reshape(batch_size, replan_steps * action_dim)
    # full_actions is unused on the GR00T path; keep shape-consistent for safety.
    batch["full_actions"] = batch["actions"]

    return batch


def create_gr00t_replay_buffer(config, example_action, capacity, task_description, replan_steps, seed):
    """Build a Gr00tReplayBuffer from agent config.

    Matches the ``create_replay_buffer`` signature so it can be used as a
    drop-in replacement in training scripts.
    """
    camera_keys = config.get("gr00t_camera_keys", ["hand_view", "table_view"])
    state_keys = config.get("gr00t_state_keys", ["eef_9d", "joint_pos", "gripper_pos"])
    action_keys = config.get("gr00t_action_keys", ["eef_9d", "joint_pos", "gripper_pos"])

    # Resolve actual dimensions from the checkpoint's processor
    from pathlib import Path
    import gr00t.model  # noqa: F401 — register GR00T processor/model with HF registry
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        Path(config.gr00t_model_path) / "processor"
        if not (Path(config.gr00t_model_path) / "processor_config.json").exists()
        else Path(config.gr00t_model_path)
    )
    stats = processor.state_action_processor.norm_params
    embodiment = config.gr00t_embodiment_tag

    state_dims = {k: int(stats[embodiment]["state"][k]["dim"].item()) for k in state_keys}
    action_dims = {k: int(stats[embodiment]["action"][k]["dim"].item()) for k in action_keys}
    env_state_dim = sum(state_dims.values())
    env_action_dim = sum(action_dims.values())

    # Env action horizon from modality config
    modality_cfg = processor.get_modality_configs()[embodiment]
    env_action_horizon = len(modality_cfg["action"].delta_indices)

    # Image shape from config (example_action is a flat ndarray, not a dict)
    image_h, image_w = getattr(config, "image_size", (256, 256))
    video_horizon = len(modality_cfg["video"].delta_indices)  # e.g. 2

    buf = Gr00tReplayBuffer(
        camera_keys=camera_keys,
        video_horizon=video_horizon,
        image_size=(image_h, image_w),
        env_state_dim=env_state_dim,
        env_action_dim=env_action_dim,
        env_action_horizon=env_action_horizon,
        capacity=capacity,
        task_description=task_description,
        replan_steps=replan_steps,
        discount=config.discount,
    )
    buf.seed(seed)
    return buf


class Gr00tReplayBuffer(Dataset):
    """Replay buffer that stores raw robot data for GR00T model training.

    Unlike ``PiReplayBuffer``, this buffer stores un-normalized data without
    applying any OpenPI transforms.  Normalization is handled at training time
    by ``Gr00tAgent.prepare_batch_for_actor``, which runs the GR00T processor.

    Batch structure (returned by ``sample_jax``):
    - ``image``: dict ``{view: ndarray (B, H, W, C) uint8}``
    - ``image_mask``: dict ``{view: ndarray (B,) bool}``
    - ``state``: ndarray ``(B, env_state_dim)`` float32 (raw)
    - ``actions``: ndarray ``(B, env_action_horizon, env_action_dim)`` float32 (raw)
    - ``prompt``: list of strings ``(B,)``
    - ``next_image``, ``next_state``: for temporal-difference learning
    - ``rewards``, ``masks``, ``dones``: RL training fields
    - ``is_hil``, ``is_success``: for HIL / success-only sampling
    """

    def __init__(
        self,
        *,
        camera_keys: Sequence[str],
        video_horizon: int,
        image_size: Tuple[int, int],
        env_state_dim: int,
        env_action_dim: int,
        env_action_horizon: int,
        capacity: int,
        task_description: str = "",
        replan_steps: int = 1,
        discount: float = 0.99,
    ):
        self._camera_keys = list(camera_keys)
        self._video_horizon = video_horizon
        self._image_size = image_size
        self._env_state_dim = env_state_dim
        self._env_action_dim = env_action_dim
        self._env_action_horizon = env_action_horizon
        self._capacity = capacity
        self._replan_steps = replan_steps
        self._discount = discount
        self._prompt = task_description

        h, w = image_size

        dataset_dict: DatasetDict = {}

        # Image storage: ndarray for each camera view for visual RL
        for view in camera_keys:
            dataset_dict[f"image_{view}"] = np.empty((capacity, h, w, 3), dtype=np.uint8)
            dataset_dict[f"image_mask_{view}"] = np.empty((capacity,), dtype=bool)
        # For next-frame variants
        for view in camera_keys:
            dataset_dict[f"next_image_{view}"] = np.empty((capacity, h, w, 3), dtype=np.uint8)
            dataset_dict[f"next_image_mask_{view}"] = np.empty((capacity,), dtype=bool)

        dataset_dict.update(
            state=np.empty((capacity, env_state_dim), dtype=np.float32),
            actions=np.empty((capacity, env_action_horizon, env_action_dim), dtype=np.float32),
            next_state=np.empty((capacity, env_state_dim), dtype=np.float32),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
            is_hil=np.empty((capacity,), dtype=bool),
            is_success=np.zeros((capacity,), dtype=bool),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._insert_index = 0
        self._buffer_keys = list(dataset_dict.keys())

    # -- size tracking -------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._size = 0
        self._insert_index = 0

    # -- snapshot persistence (survives training crashes) -----------------------

    def save_snapshot(self, path: str, *, online_start: int = 0) -> None:
        """Save the *online* buffer segment to a fast uncompressed .npz file.

        Only transitions from ``online_start`` onward are saved (the offline
        dataset, which is reloaded from disk on restart, is excluded).  No
        compression — images in a .npz compress poorly and deflate takes 30+
        seconds for thousands of frames; uncompressed writes finish in << 1 s.
        """
        import logging
        _log = logging.getLogger(__name__)
        if self._size <= online_start:
            _log.info("buffer snapshot: %d online transitions, skipped save",
                      self._size - online_start)
            return

        # Linearise the circular buffer [oldest, ..., newest)
        if self._size < self._capacity:
            indices = np.arange(online_start, self._size)
        else:
            start = self._insert_index
            all_indices = list(range(start, self._capacity)) + list(range(0, start))
            tail = min(online_start, len(all_indices))
            indices = np.array(all_indices[tail:])

        valid = {k: v[indices] for k, v in self.dataset_dict.items()}
        meta = {
            "_size": len(indices),
            "_insert_index": 0,  # linearised → starts at 0
            "_online_start": online_start,
        }
        np.savez(path, **meta, **valid)
        _log.info(
            "buffer snapshot saved: %d online transitions → %s (%.1f MB)",
            len(indices), path, __import__('os').path.getsize(path) / 1e6,
        )

    @classmethod
    def load_snapshot(cls, path: str, *, online_start: int = 0,
                      **buffer_kwargs) -> "Gr00tReplayBuffer":
        """Load a saved snapshot and reconstruct only the online portion.

        offline_start marks where offline data begins; the restored buffer
        is sized to hold capacity transitions with the online data appended.
        """
        import logging
        _log = logging.getLogger(__name__)
        data = np.load(path)
        saved_size = int(data["_size"])
        required_cap = online_start + saved_size
        cap = max(required_cap, buffer_kwargs.get("capacity", 5000))
        buf = cls(capacity=cap,
                  **{k: v for k, v in buffer_kwargs.items() if k != "capacity"})
        # Place online data after the (already-reserved) offline segment
        off = online_start
        for k in buf.dataset_dict:
            buf.dataset_dict[k][off:off + saved_size] = data[k][:saved_size]
        buf._size = off + saved_size
        buf._insert_index = (off + saved_size) % buf._capacity
        _log.info(
            "buffer snapshot loaded: %d online transitions + %d offline offset"
            " from %s", saved_size, online_start, path,
        )
        return buf

    def count_episodes_chronological(self) -> int:
        if self._size == 0:
            return 0
        dones = np.asarray(self.dataset_dict["dones"])
        max_len = min(self._size, self._capacity)
        if self._size < self._capacity:
            indices = list(range(max_len))
        else:
            start = self._insert_index
            indices = list(range(start, self._capacity)) + list(range(0, start))
        return int(np.sum(dones[indices]))

    def _find_episode_boundaries(self, max_len: int) -> Tuple[List[int], List[int]]:
        dones = np.asarray(self.dataset_dict["dones"])
        starts = [0]
        ends = []
        for i in range(max_len):
            if dones[i]:
                ends.append(i + 1)
                if i + 1 < max_len:
                    starts.append(i + 1)
        return starts, ends

    # -- marking success -----------------------------------------------------

    def mark_episode_success(self, start_idx: int, end_idx: int) -> None:
        if end_idx > start_idx:
            self.dataset_dict["is_success"][start_idx:end_idx] = True
        else:
            self.dataset_dict["is_success"][start_idx:self._capacity] = True
            self.dataset_dict["is_success"][:end_idx] = True

    def restore_success_marks(self, reward_threshold: float = 0.5) -> None:
        self.dataset_dict["is_success"][:] = False
        starts, ends = self._find_episode_boundaries(self._size)
        for s, e in zip(starts, ends):
            if self.dataset_dict["rewards"][e - 1] > reward_threshold:
                self.dataset_dict["is_success"][s:e] = True

    # -- convert to critic format for agent init -----------------------------

    def convert_to_critic_format(
        self, data_dict: DatasetDict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build example critic inputs for agent init.

        Accepts a dict either with flat ``image_<view>`` keys (from
        ``dataset_dict``) or a nested ``image`` dict (from sampling).
        """
        if "image" in data_dict:
            # Nested format from sample_jax
            obs_parts = [data_dict["image"][k] for k in self._camera_keys]
        else:
            # Flat format from dataset_dict
            obs_parts = [data_dict[f"image_{k}"] for k in self._camera_keys]
        critic_obs = np.concatenate(obs_parts, axis=-1)
        critic_state = np.asarray(data_dict["state"])
        if "actions" in data_dict:
            critic_action = np.asarray(data_dict["actions"])[..., :self._env_action_dim]
        else:
            critic_action = np.zeros((self._env_action_horizon, self._env_action_dim))
        return critic_obs, critic_state, critic_action

    # -- insert / load -------------------------------------------------------

    def insert(self, data_dict: DatasetDict):
        """Insert a single transition.

        Args:
            data_dict: Dict with keys:
                - ``image``: dict of ``{view: ndarray (H, W, C) uint8}``
                - ``state``: ndarray ``(env_state_dim,)`` float32
                - ``actions``: ndarray ``(action_horizon, env_action_dim)`` float32
                - ``next_image``: (optional) same format as above
                - ``next_state``: (optional) ndarray ``(env_state_dim,)``
                - ``rewards``, ``masks``, ``dones``: scalar
                - ``is_hil``, ``is_success``: (optional) bool
        """
        idx = self._insert_index

        for view in self._camera_keys:
            self.dataset_dict[f"image_{view}"][idx] = np.asarray(data_dict["image"][view])
            self.dataset_dict[f"image_mask_{view}"][idx] = True
            if "next_image" in data_dict and view in data_dict["next_image"]:
                self.dataset_dict[f"next_image_{view}"][idx] = np.asarray(data_dict["next_image"][view])
                self.dataset_dict[f"next_image_mask_{view}"][idx] = True
            else:
                # Default: copy current for identity prediction
                self.dataset_dict[f"next_image_{view}"][idx] = np.asarray(data_dict["image"][view])
                self.dataset_dict[f"next_image_mask_{view}"][idx] = False

        self.dataset_dict["state"][idx] = np.asarray(data_dict["state"], dtype=np.float32)
        self.dataset_dict["next_state"][idx] = np.asarray(
            data_dict.get("next_state", data_dict["state"]), dtype=np.float32
        )

        # Action chunk: pad/repeat to env_action_horizon if needed
        actions = np.asarray(data_dict["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = np.tile(actions[None, :], (self._env_action_horizon, 1))
        elif actions.shape[0] != self._env_action_horizon:
            actions = np.tile(actions[0:1], (self._env_action_horizon, 1))
        self.dataset_dict["actions"][idx] = actions

        self.dataset_dict["rewards"][idx] = np.asarray(data_dict["rewards"], dtype=np.float32)
        self.dataset_dict["masks"][idx] = np.asarray(data_dict.get("masks", 1.0), dtype=np.float32)
        self.dataset_dict["dones"][idx] = np.asarray(data_dict.get("dones", False), dtype=bool)
        self.dataset_dict["is_hil"][idx] = np.asarray(data_dict.get("is_hil", False), dtype=bool)
        self.dataset_dict["is_success"][idx] = np.asarray(data_dict.get("is_success", False), dtype=bool)

        # Advance
        self._insert_index = (idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def insert_dataset(self, dataset):
        """Load offline demos; mark as HIL/success so they enter actor sampling pools.

        Handles both GR00T native format (``image`` + ``state`` keys) and
        OpenPI/Droid format (``observations`` key with ``exterior_image_1_left``,
        ``wrist_image_left``, ``cartesian_position``, ``gripper_position``).
        """
        logger.info("Loading %s offline episodes into GR00T replay buffer ...", len(dataset))
        for transition in tqdm.tqdm(dataset, desc="Inserting demos"):
            transition_dict = self._adapt_offline_transition(dict(transition))
            transition_dict.setdefault("is_hil", True)
            transition_dict.setdefault("is_success", True)
            self.insert(transition_dict)

    def _adapt_offline_transition(self, data_dict: DatasetDict) -> DatasetDict:
        """Convert an offline dataset transition to GR00T replay buffer format.

        Detects whether the transition is already in GR00T format (has ``image``
        key) or OpenPI/Droid format (has ``observations`` key with camera/state
        flat keys), and converts the latter.
        """
        # Already GR00T format — pass through
        if "image" in data_dict and "state" in data_dict:
            return data_dict

        # OpenPI/Droid format: data has "observations" dict
        obs = data_dict.pop("observations", None)
        if obs is None:
            raise KeyError(
                "Offline transition must contain either 'image'+'state' (GR00T format) "
                "or 'observations' (OpenPI/Droid format). Got keys: " + str(list(data_dict.keys()))
            )

        # Map OpenPI camera keys to GR00T view keys.
        # Convention: wrist_image_left → first view, exterior_image_1_left → second view.
        # Camera images may be at obs["image"][key] (Droid HDF5) or obs[key] (flat).
        _img_source = obs.get("image", obs)
        _img_keys = [k for k in _img_source if isinstance(_img_source[k], np.ndarray) and _img_source[k].ndim >= 2]
        _openpi_cam_map = {
            "wrist_image_left": 0,
            "exterior_image_1_left": 1,
        }
        images: dict = {}
        for openpi_key, idx in _openpi_cam_map.items():
            if openpi_key in _img_source and idx < len(self._camera_keys):
                images[self._camera_keys[idx]] = np.asarray(_img_source[openpi_key])

        # If no explicit match, assign available image keys heuristically
        _unmatched = [k for k in _img_keys if k not in _openpi_cam_map]
        _assigned = len(images)
        for k in _unmatched:
            if _assigned < len(self._camera_keys):
                images[self._camera_keys[_assigned]] = np.asarray(_img_source[k])
                _assigned += 1

        # Map state: may be nested under obs["state"] (Droid HDF5) or flat obs keys
        _state_source = obs.get("state", obs)
        _openpi_state_order = ["cartesian_position", "cartesian_velocity",
                                "joint_position", "gripper_position", "gripper_velocity"]
        state_parts = []
        for k in _openpi_state_order:
            if k in _state_source:
                state_parts.append(np.asarray(_state_source[k], dtype=np.float32).reshape(-1))
        if not state_parts:
            state_parts = [np.zeros(self._env_state_dim, dtype=np.float32)]
        flat_state = np.concatenate(state_parts)
        if len(flat_state) != self._env_state_dim:
            padded = np.zeros(self._env_state_dim, dtype=np.float32)
            n = min(len(flat_state), self._env_state_dim)
            padded[:n] = flat_state[:n]
            flat_state = padded

        data_dict["image"] = images
        data_dict["state"] = flat_state.astype(np.float32)

        # Resize images to match GR00T processor expected size (stored in buffer)
        import cv2
        _img_sz = getattr(self, "_image_size", (256, 256))
        for v in data_dict["image"]:
            img = data_dict["image"][v]
            if img.shape[0] != _img_sz[0] or img.shape[1] != _img_sz[1]:
                data_dict["image"][v] = cv2.resize(img, (_img_sz[1], _img_sz[0]))
        return data_dict

    # -- sampling ------------------------------------------------------------

    def sample_jax(
        self,
        batch_size: int,
        keys: Optional[Sequence[str]] = None,
        data_sharding=None,
        hil_only: bool = False,
        success_only: bool = False,
    ) -> Optional[DatasetDict]:
        """Sample a batch of raw (un-normalized) transitions.

        Returns a ``DatasetDict`` with:
        - ``image``: ``{view: (B, H, W, C) uint8}``
        - ``image_mask``: ``{view: (B,) bool}``
        - ``state``: ``(B, env_state_dim) float32``
        - ``actions``: ``(B, env_action_horizon, env_action_dim) float32``
        - ``prompt``: list of strings ``(B,)``
        - ``next_image``, ``next_state``: for TD learning
        - ``rewards``, ``masks``, ``dones``, ``valids``, ``full_actions``
        - ``is_hil``, ``is_success``
        """
        assert len(self) >= self._replan_steps, "Replay buffer must have >= replan_steps"
        if not hasattr(self, "rng"):
            self.rng = jax.random.PRNGKey(self._seed or 42)

        if keys is None:
            keys = self._buffer_keys

        key, rng = jax.random.split(self.rng)
        max_start = len(self) - self._replan_steps

        if hil_only:
            eligible = np.flatnonzero(self.dataset_dict["is_hil"][:max_start])
            if len(eligible) == 0:
                raise ValueError("No HIL-annotated samples available.")
            sampled = jax.random.randint(key, (batch_size,), minval=0, maxval=len(eligible))
            indices = eligible[np.asarray(sampled)]
        elif success_only:
            eligible = np.flatnonzero(self.dataset_dict["is_success"][:max_start])
            if len(eligible) == 0:
                self.rng = rng
                return None
            sampled = jax.random.randint(key, (batch_size,), minval=0, maxval=len(eligible))
            indices = eligible[np.asarray(sampled)]
        else:
            indices = jax.random.randint(key, (batch_size,), minval=0, maxval=max_start)
        self.rng = rng

        # Assemble batch with image dicts
        batch: DatasetDict = {
            "image": {},
            "image_mask": {},
        }
        for k in keys:
            if k in self.dataset_dict:
                batch[k] = self.dataset_dict[k][indices]

        # Restructure flat image_{view} keys into dicts
        # Also extract prompt from stored string
        prompt = self._prompt if self._prompt else ""
        batch["prompt"] = [prompt] * batch_size if isinstance(prompt, str) else prompt

        for view in self._camera_keys:
            batch["image"][view] = self.dataset_dict[f"image_{view}"][indices]
            batch["image_mask"][view] = self.dataset_dict[f"image_mask_{view}"][indices]
            batch[f"image_{view}"]  # keep flat key in _buffer_keys listing

        # Reconstruct the TRUE executed C-step action chunk (Fix #7).
        # Every slot stores a single per-step action (online env steps and
        # per-step offline demos both insert one action), so reading the first
        # action of C consecutive slots yields the actually-executed chunk
        # a_{t:t+C} -- instead of the degenerate tiled [a_t] x H that the
        # generic gather above produced.  Symmetric with the multi-step reward
        # accumulation below.
        _np_idx = np.asarray(indices)
        _adim = self._env_action_dim
        _chunk_actions = np.stack([
            self.dataset_dict["actions"][(_np_idx + k) % self._capacity][:, 0, :_adim]
            for k in range(self._replan_steps)
        ], axis=1)  # (B, replan_steps, env_action_dim)

        # Hold the last in-episode action after a terminal so the reconstructed
        # chunk never concatenates the NEXT episode's actions (Fix #8, actor-BC
        # path).  The critic is already shielded -- `valids` zeros its loss and
        # `masks` zeroes the bootstrap when a terminal falls in the chunk -- so
        # this only protects success-only actor supervision near episode ends.
        _dones_chunk = np.stack([
            self.dataset_dict["dones"][(_np_idx + k) % self._capacity]
            for k in range(self._replan_steps)
        ], axis=1)  # (B, replan_steps)
        _has_done = _dones_chunk.any(axis=1)
        _first_done = np.argmax(_dones_chunk, axis=1)  # first True per row
        for b in np.flatnonzero(_has_done):
            d = int(_first_done[b])
            if d + 1 < self._replan_steps:
                _chunk_actions[b, d + 1:] = _chunk_actions[b, d]
        batch["actions"] = _chunk_actions

        next_idx = (indices + self._replan_steps) % self._capacity

        batch["next_image"] = {}
        batch["next_image_mask"] = {}
        for view in self._camera_keys:
            batch["next_image"][view] = self.dataset_dict[f"next_image_{view}"][next_idx]
            batch["next_image_mask"][view] = self.dataset_dict[f"next_image_mask_{view}"][next_idx]
        batch["next_state"] = self.dataset_dict["next_state"][next_idx]

        # Compute multi-step returns
        batch["valids"] = np.ones((batch_size,), dtype=np.float32)
        _prev_masks = batch["masks"].copy()
        for i in range(1, self._replan_steps):
            _nxt = (indices + i) % self._capacity
            batch["rewards"] += self.dataset_dict["rewards"][_nxt] * (self._discount ** i) * _prev_masks
            batch["valids"] = _prev_masks
            batch["masks"] = np.minimum(batch["masks"], self.dataset_dict["masks"][_nxt])
            batch["dones"] = np.logical_or(batch["dones"], self.dataset_dict["dones"][_nxt])
            _prev_masks = batch["masks"].copy()

        # Convert to JAX arrays
        def _to_jax(x):
            if isinstance(x, dict):
                return {k: _to_jax(v) for k, v in x.items()}
            elif isinstance(x, (np.ndarray, np.generic)):
                return jnp.asarray(x)
            try:
                return jnp.asarray(x)
            except (TypeError, ValueError):
                return x

        return _to_jax(batch)
