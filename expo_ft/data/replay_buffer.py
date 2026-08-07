import collections
import logging
import os
import pickle as std_pickle
from typing import Any, Dict, Optional, Tuple, Union

import cloudpickle
import etils.epath as epath
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from expo_ft.data.dataset import Dataset, DatasetDict
import openpi.training.config as _config
import openpi.transforms as _transforms


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


def create_replay_buffer(config, example_action, capacity, task_description, replan_steps, seed):
    """Build pi05 config and create a seeded PiReplayBuffer."""
    from expo_ft.utils.train_utils import build_pi05_config
    _, pi05_train_config, pi05_resize_size, _ = build_pi05_config(config)
    buf = PiReplayBuffer(
        example_action=example_action.squeeze(),
        capacity=capacity,
        pi_train_config=pi05_train_config,
        skip_norm_stats=False,
        resize_size=pi05_resize_size,
        task_description=task_description,
        replan_steps=replan_steps,
        discount=config.discount,
    )
    buf.seed(seed)
    return buf


class PiReplayBuffer(Dataset):
    """Replay buffer for Pi models with OpenPI-compatible transforms.
    
    Batch Structure:
        The batch structure matches OpenPI's DataLoader format.
        
        OpenPI DataLoader yields: (observation: Observation, actions: Array)
        
        observation: Observation (dataclass object) with attributes:
            - images: dict[str, Array] - Images in [-1, 1] float32
                Keys: "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", etc. (dataset-dependent)
                Shape: [b, h, w, c] where b=batch_size, h=height, w=width, c=channels
            - image_masks: dict[str, Array] - Boolean masks for each image view
                Same keys as images, Shape: [b] (bool) - True if image is valid
            - state: Array - Low-dimensional robot state (normalized)
                Shape: [b, s] where s=state_dim
            - tokenized_prompt: Array | None - Tokenized prompt (for pi0/pi05 models)
                Shape: [b, l] (int32) where l=max_token_len
            - tokenized_prompt_mask: Array | None - Mask for tokenized prompt
                Shape: [b, l] (bool)
            - token_ar_mask: Array | None - Auto-regressive mask (for FAST models)
                Shape: [b, l] (int32)
            - token_loss_mask: Array | None - Loss mask (for FAST models)
                Shape: [b, l] (bool)
        
        actions: Array - Action chunks (normalized)
            Shape: [b, action_horizon, action_dim]
            Type: float32  
    """
    
    def __init__(
        self,
        # example_observation,
        example_action,
        # example_state, 
        capacity: int,
        pi_train_config: _config.TrainConfig,
        skip_norm_stats: bool = False,
        resize_size: int = 224,
        task_description: str = None,
        replan_steps: int = None,
        discount: float = 0.99,
    ):
        data_config = pi_train_config.data.create(pi_train_config.assets_dirs, pi_train_config.model)
        model_config = pi_train_config.model
        padded_action_dim = model_config.action_dim
        action_horizon = model_config.action_horizon
        image_shape = wrist_image_shape = (resize_size, resize_size, 3)
        state_shape = (padded_action_dim,)
        action_chunk_shape = (action_horizon, padded_action_dim)
        max_token_len = model_config.max_token_len


        dataset_dict = dict(
            base_image=np.empty((capacity, *image_shape), dtype=np.uint8),
            left_wrist_image=np.empty((capacity, *wrist_image_shape), dtype=np.uint8),
            right_wrist_image=np.empty((capacity, *wrist_image_shape), dtype=np.uint8),
            base_image_mask=np.empty((capacity,), dtype=bool),
            left_wrist_image_mask=np.empty((capacity,), dtype=bool),
            right_wrist_image_mask=np.empty((capacity,), dtype=bool),
            state=np.empty((capacity, *state_shape), dtype=np.float32),
            actions=np.empty((capacity, *action_chunk_shape), dtype=np.float32),
            tokenized_prompt=np.empty((capacity, max_token_len), dtype=int),
            tokenized_prompt_mask=np.empty((capacity, max_token_len), dtype=bool),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
            is_hil=np.empty((capacity,), dtype=bool),
            hil_chunk=np.empty((capacity,), dtype=bool),
            is_success=np.zeros((capacity,), dtype=bool),
        )
        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        
        self._data_config = data_config
        self._action_horizon = action_horizon
        self._raw_action_dim = example_action.shape[-1]
        
        self._action_dim = padded_action_dim
        self._replan_steps = replan_steps
        self._discount = discount
        self._skip_norm_stats = skip_norm_stats
        self._prompt = task_description
        self._transform = self._build_transform_pipeline()
        self._buffer_keys = dataset_dict.keys()

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        """Clear the buffer (size 0, next insert at 0). Used when repopulating on resume."""
        self._size = 0
        self._insert_index = 0

    def count_episodes_chronological(self) -> int:
        """Number of complete episodes when scanning the buffer in chronological order (oldest to newest)."""
        if self._size == 0:
            return 0
        dones = np.asarray(self.dataset_dict["dones"])
        if self._size < self._capacity:
            indices = list(range(self._size))
        else:
            start = self._insert_index
            indices = list(range(start, self._capacity)) + list(range(0, start))
        return int(np.sum(dones[indices]))

    def mark_episode_success(self, start_idx: int, end_idx: int) -> None:
        """Mark all transitions in [start_idx, end_idx) as success."""
        if end_idx > start_idx:
            self.dataset_dict["is_success"][start_idx:end_idx] = True
        else:  # wrapped
            self.dataset_dict["is_success"][start_idx:self._capacity] = True
            self.dataset_dict["is_success"][:end_idx] = True

    def restore_success_marks(self, reward_threshold: float = 0.5) -> None:
        """Rebuild is_success flags from episode rewards after resume."""
        self.dataset_dict["is_success"][:] = False
        starts, ends = self._find_episode_boundaries_from_dones(
            self.dataset_dict["dones"], self._size
        )
        for s, e in zip(starts, ends):
            if self.dataset_dict["rewards"][e - 1] > reward_threshold:
                self.dataset_dict["is_success"][s:e] = True

    def convert_to_critic_format(
        self, data_dict: DatasetDict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build example critic inputs for agent init (not used for training batches)."""
        critic_obs = np.concatenate([data_dict["base_image"], data_dict["left_wrist_image"]], axis=-1)
        # keep state same as raw state (with padding) 
        critic_state = data_dict["state"] 
        if "actions" in data_dict:
            critic_action = data_dict["actions"][..., :self._raw_action_dim]
        else:
            critic_action = np.zeros((self._action_horizon, self._raw_action_dim))
        return critic_obs, critic_state, critic_action
    
    def _preprocess_single_transition(self, data_dict: DatasetDict, is_last: bool = False) -> DatasetDict:
        """Preprocess a single transition by applying transforms."""
        for key, value in data_dict.items():
            data_dict[key] = np.asarray(value)
            if "image" in key:
                data_dict[key] = data_dict[key].astype(np.uint8)
                assert np.max(data_dict[key]) > 1

        transformed = self._transform(data_dict)
        
        result = {
            "state": transformed["state"],
            "base_image": transformed["image"]["base_0_rgb"],
            "left_wrist_image": transformed["image"]["left_wrist_0_rgb"],
            "right_wrist_image": transformed["image"]["right_wrist_0_rgb"],
            "base_image_mask": transformed["image_mask"]["base_0_rgb"],
            "left_wrist_image_mask": transformed["image_mask"]["left_wrist_0_rgb"],
            "right_wrist_image_mask": transformed["image_mask"]["right_wrist_0_rgb"],
            "tokenized_prompt": transformed["tokenized_prompt"],
            "tokenized_prompt_mask": transformed["tokenized_prompt_mask"],
            "actions": transformed["actions"],
        }
        for k in ["rewards", "masks", "dones"]:
            result[k] = data_dict[k]
        result["is_hil"] = np.asarray(data_dict.get("is_hil", False), dtype=bool)
        result["hil_chunk"] = np.asarray(
            data_dict.get("hil_chunk", result["is_hil"]), dtype=bool
        )
        result["is_success"] = np.asarray(data_dict.get("is_success", False), dtype=bool)

        return result


    def insert(self, data_dict: DatasetDict):
        # Create action chunk for current transition by padding with itself
        action_chunk_raw = np.tile(data_dict["actions"][None, :], (self._action_horizon, 1))  # Shape: [action_horizon, action_dim]
    
        obs_data_dict = data_dict["observations"].copy()
        obs_data_dict["actions"] = action_chunk_raw
        obs_data_dict["rewards"] = np.asarray(data_dict["rewards"])
        obs_data_dict["masks"] = np.asarray(data_dict["masks"])
        obs_data_dict["dones"] = np.asarray(data_dict["dones"])
        is_hil = bool(data_dict.get("is_hil", False))
        obs_data_dict["is_hil"] = np.asarray(is_hil, dtype=bool)
        obs_data_dict["hil_chunk"] = np.asarray(is_hil, dtype=bool)
        obs_data_dict["is_success"] = np.asarray(data_dict.get("is_success", False), dtype=bool)
        preprocessed_dict = self._preprocess_single_transition(obs_data_dict)

        # Drop any keys not in buffer storage (e.g. stray fields) so _insert_recursively matches.
        row = {k: preprocessed_dict[k] for k in self.dataset_dict.keys()}
        _insert_recursively(self.dataset_dict, row, self._insert_index)
        
        preprocessed_action = row["actions"][0]
        for action_idx in range(1, self._action_horizon):
            if self._insert_index < action_idx and self._size < self._capacity:
                # Buffer hasn't wrapped yet and we don't have enough history
                break
            prev_idx = (self._insert_index - action_idx) % self._capacity
            if self.dataset_dict["dones"][prev_idx]:
                break
            self.dataset_dict["actions"][prev_idx, action_idx:] = np.tile(preprocessed_action[None, :], (self._action_horizon - action_idx, 1))
            if is_hil:
                self.dataset_dict["hil_chunk"][prev_idx] = True
        
        # Update insert index and size
        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
    
    def insert_dataset(self, dataset):
        """Load offline demos; mark as HIL/success so they enter actor sampling pools."""
        for transition in dataset:
            transition_dict = dict(transition)
            # Offline demos are treated as expert/HIL data for BC and success-only actor updates.
            transition_dict.setdefault("is_hil", True)
            transition_dict.setdefault("is_success", True)
            self.insert(transition_dict)

    def _build_transform_pipeline(self):
        """Build the transform pipeline similar to OpenPI's transform_iterable_dataset."""
        norm_stats = {}
        if not self._skip_norm_stats:
            if self._data_config.norm_stats is not None:
                norm_stats = self._data_config.norm_stats
            else:
                raise ValueError(
                    "Normalization stats not found. "
                    "Make sure your OpenPI config has AssetsConfig set up correctly "
                    "so that norm_stats are loaded into data_config.norm_stats. "
                    "The norm stats file should be located at: "
                    "assets_dir / asset_id / norm_stats.json"
                )
        
        transforms = [
            *self._data_config.repack_transforms.inputs,
            *self._data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats, use_quantiles=self._data_config.use_quantile_norm
            ),
            *self._data_config.model_transforms.inputs,
        ]
        
        return _transforms.compose(transforms)
    

    def _clear_batch(self, batch):
        """Recursively clear a batch dictionary to free memory."""
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, dict):
                    self._clear_batch(v)
            batch.clear()

    def apply_data_sharding(self, batch, data_sharding):
        """Apply data sharding to a batch.
        
        Uses jax.make_array_from_process_local_data for multi-device sharding,
        matching OpenPI's pattern. For single-device (data_sharding=None), uses
        default device_put.
        """
        if data_sharding is None:
            # Match DataLoader default behavior: create NamedSharding for JAX
            default_sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
            sharded_batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(default_sharding, x), batch)
        else:
            sharded_batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), batch)
        
        # Clear the original CPU batch to free memory immediately
        self._clear_batch(batch)
        
        return sharded_batch

    def _find_episode_boundaries_from_dones(self, dones: np.ndarray, dataset_len: int) -> tuple[list, list]:
        episode_starts = [0]
        episode_ends = []
        for i in range(dataset_len):
            if dones[i]:
                episode_ends.append(i + 1)
                if i + 1 < dataset_len:
                    episode_starts.append(i + 1)
        if len(episode_ends) < len(episode_starts):
            episode_ends.append(dataset_len)
        return episode_starts, episode_ends
    
    def sample_jax(self, batch_size: int, keys=None, data_sharding=None,
                       hil_only: bool = False, success_only: bool = False):
        assert len(self) >= self._replan_steps, "Replay buffer size must be greater than replan steps"
        if not hasattr(self, "rng"):
            self.rng = jax.random.PRNGKey(self._seed or 42)

        if keys is None:
            keys = self.dataset_dict.keys()

        key, rng = jax.random.split(self.rng)
        max_start = len(self) - self._replan_steps
        if hil_only:
            eligible_indices = np.flatnonzero(self.dataset_dict["hil_chunk"][:max_start])
            if len(eligible_indices) == 0:
                raise ValueError(
                    "No HIL-annotated action chunks available for hil_only sampling."
                )
            sampled_positions = jax.random.randint(
                key, (batch_size,), minval=0, maxval=len(eligible_indices)
            )
            indices = eligible_indices[np.asarray(sampled_positions)]
        elif success_only:
            eligible_indices = np.flatnonzero(self.dataset_dict["is_success"][:max_start])
            if len(eligible_indices) == 0:
                self.rng = rng
                return None
            sampled_positions = jax.random.randint(
                key, (batch_size,), minval=0, maxval=len(eligible_indices)
            )
            indices = eligible_indices[np.asarray(sampled_positions)]
        else:
            indices = jax.random.randint(
                key, (batch_size,), minval=0, maxval=max_start
            )
        self.rng = rng

        jax_dataset_dict = {k: self.dataset_dict[k][indices] for k in keys}

        next_indices = (indices + self._replan_steps) % self._capacity

        jax_dataset_dict.update(
            {
                "next_base_image": self.dataset_dict["base_image"][next_indices],
                "next_left_wrist_image": self.dataset_dict["left_wrist_image"][next_indices],
                "next_right_wrist_image": self.dataset_dict["right_wrist_image"][next_indices],
                "next_base_image_mask": self.dataset_dict["base_image_mask"][next_indices],
                "next_left_wrist_image_mask": self.dataset_dict["left_wrist_image_mask"][next_indices],
                "next_right_wrist_image_mask": self.dataset_dict["right_wrist_image_mask"][next_indices],
                "next_state": self.dataset_dict["state"][next_indices],

                # Per-step loss weight within the replan horizon (1 while episode continues).
                "valids": np.ones((batch_size,), dtype=np.float32),
            }
        )
        _prev_masks = jax_dataset_dict["masks"].copy()
        for i in range(1, self._replan_steps):
            _indices = (indices + i) % self._capacity
            jax_dataset_dict["rewards"] += self.dataset_dict["rewards"][_indices] * (self._discount ** i) * _prev_masks
            jax_dataset_dict["valids"] = _prev_masks
            # Continuation mask: product of per-step masks out to step i (0 after terminal).
            jax_dataset_dict["masks"] = np.minimum(jax_dataset_dict["masks"], self.dataset_dict["masks"][_indices])
            jax_dataset_dict["dones"] = np.logical_or(jax_dataset_dict["dones"], self.dataset_dict["dones"][_indices])

            _prev_masks = jax_dataset_dict["masks"].copy()

        def to_jax_array(x):
            """Keep the sampled batch as numpy (CPU) so apply_data_sharding can move it
            directly to the data_sharding device. The previous `jnp.asarray(x)` created
            every leaf on the DEFAULT device (GPU0 = sampling card), which materialized
            the ~735 MiB critic batch on GPU0 (already full of the 3.3B sampling cache)
            and OOM'd before apply_data_sharding could move it to GPU1. numpy leaves
            work because _convert_to_openpi_format only uses .dtype/.astype and
            apply_data_sharding uses make_array_from_process_local_data (CPU source)."""
            if isinstance(x, dict):
                return {k: to_jax_array(v) for k, v in x.items()}
            elif isinstance(x, (np.ndarray, np.generic)):
                return np.asarray(x)
            else:
                try:
                    return np.asarray(x)
                except (TypeError, ValueError):
                    return x

        jax_dataset_dict = to_jax_array(jax_dataset_dict)
        
        return jax_dataset_dict

    def _convert_to_openpi_format(self, batch):
        openpi_batch = batch.copy()
        openpi_batch["image"] = {
            "base_0_rgb": batch["base_image"],
            "left_wrist_0_rgb": batch["left_wrist_image"],
            "right_wrist_0_rgb": batch["right_wrist_image"],
        }
        openpi_batch["image_mask"] = {
            "base_0_rgb": batch["base_image_mask"],
            "left_wrist_0_rgb": batch["left_wrist_image_mask"],
            "right_wrist_0_rgb": batch["right_wrist_image_mask"],
        }
        openpi_batch["next_image"] = {
            "base_0_rgb": batch["next_base_image"],
            "left_wrist_0_rgb": batch["next_left_wrist_image"],
            "right_wrist_0_rgb": batch["next_right_wrist_image"],
        }
        openpi_batch["next_image_mask"] = {
            "base_0_rgb": batch["next_base_image_mask"],
            "left_wrist_0_rgb": batch["next_left_wrist_image_mask"],
            "right_wrist_0_rgb": batch["next_right_wrist_image_mask"],
        }   
        for img_key in ["image", "next_image"]:
            for key in openpi_batch[img_key]:
                if openpi_batch[img_key][key].dtype == np.uint8:
                    openpi_batch[img_key][key] = openpi_batch[img_key][key].astype(np.float32) / 255.0 * 2.0 - 1.0
        
        del openpi_batch["base_image"]
        del openpi_batch["left_wrist_image"]
        del openpi_batch["right_wrist_image"]
        del openpi_batch["base_image_mask"]
        del openpi_batch["left_wrist_image_mask"]
        del openpi_batch["right_wrist_image_mask"]
        del openpi_batch["next_base_image"]
        del openpi_batch["next_left_wrist_image"]
        del openpi_batch["next_right_wrist_image"]
        del openpi_batch["next_base_image_mask"]
        del openpi_batch["next_left_wrist_image_mask"]
        del openpi_batch["next_right_wrist_image_mask"]
        return openpi_batch

    def get_iterator(
        self, queue_size: int = 3, sample_args: dict = {}, data_sharding=None
    ):
        """Get iterator over batches from replay buffer with transforms.
        
        Uses prefetching for efficient data loading. See:
        https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        
        Args:
            queue_size: Number of batches to prefetch (default 3).
            sample_args: Arguments passed to sample_jax() method.
            data_sharding: Optional data sharding specification.
            
        Yields:
            OpenPI-format batch dict (after transforms and n-step packing).
        """
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                sample_kwargs = dict(sample_args)
                if 'data_sharding' not in sample_kwargs:
                    sample_kwargs['data_sharding'] = data_sharding
                batch = self.sample_jax(**sample_kwargs)
                openpi_batch = self._convert_to_openpi_format(batch)
                queue.append(openpi_batch)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)


def save_replay_buffer_transition(
    checkpoint_dir: epath.Path | str,
    transition: dict,
    *,
    step: int,
) -> None:
    """Save one replay-buffer transition to `<checkpoint_dir>/buffers/{step:012d}.pkl`."""
    if jax.process_index() != 0:
        return

    buffer_dir = epath.Path(checkpoint_dir) / "buffers"
    buffer_dir.mkdir(parents=True, exist_ok=True)

    fname = buffer_dir / f"{step:012d}.pkl"
    tmp = epath.Path(str(fname) + ".tmp")
    with tmp.open("wb") as f:
        cloudpickle.dump(transition, f, protocol=std_pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, fname)


def restore_replay_buffer(
    checkpoint_dir: epath.Path | str,
    replay_buffer: Any,
    *,
    up_to_step: int | None = None,
    max_transitions: int | None = None,
) -> Any:
    """Restore replay buffer by re-inserting saved transitions from disk.
    If max_transitions is set, stop after inserting that many (e.g. for faster loading)."""
    if jax.process_index() != 0:
        return replay_buffer

    buffer_dir = epath.Path(checkpoint_dir) / "buffers"
    if not buffer_dir.exists():
        return replay_buffer

    if up_to_step is not None:
        cutoff = int(up_to_step)
        files = sorted(
            [p for p in buffer_dir.glob("*.pkl") if p.stem.isdigit() and int(p.stem) <= cutoff],
            key=lambda p: int(p.stem),
        )
    else:
        files = sorted(
            [p for p in buffer_dir.glob("*.pkl") if p.stem.isdigit()],
            key=lambda p: int(p.stem),
        )

    inserted = 0
    skipped = 0
    for p in tqdm.tqdm(files, desc="Loading replay buffer", unit="trans"):
        if max_transitions is not None and inserted >= max_transitions:
            break
        with p.open("rb") as f:
            transition = cloudpickle.load(f)
        if "actions" in transition and np.allclose(transition["actions"], -1):
            skipped += 1
            continue
        replay_buffer.insert(transition)
        inserted += 1
    if skipped > 0:
        logging.info("Skipped %d dummy (action=-1) transitions during replay buffer restore.", skipped)

    return replay_buffer
