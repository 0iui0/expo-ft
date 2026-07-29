"""Gr00tAgent: VLA agent wrapper for GR00T N1.7 model within the EXPO-FT framework.

Converts between EXPO's JAX-native batch format and GR00T's PyTorch-native
processing, implementing the ``vla_base.Model`` interface so Gr00tN1d7 can
serve as the VLA actor in EXPOLearnerGR00T.
"""
from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import jax
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from expo_ft.agents.vla.vla_base import Model
from expo_ft.agents.vla.gr00t_train_state import PyTorchTrainState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # gr00t is imported lazily at runtime (initialize / prepare_batch_for_actor)
    # so this module stays importable without the gr00t package installed.
    from gr00t.data.embodiment_tags import EmbodimentTag


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert all floating point tensors to a target dtype."""
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x


class Gr00tAgent(Model):
    """EXPO-FT Model wrapper around the GR00T N1.7 VLA model.

    Key differences from Pi05Agent:
    - ``train_step`` is a plain Python function (not JAX jitted) that runs
      PyTorch forward / backward / optimizer step.
    - ``prepare_batch_for_actor`` converts raw (or replay-buffer) data into the
      collated input dict that ``Gr00tN1d7.forward`` expects.
    - ``sample_actions`` runs ``model.get_action`` (flow-matching diffusion).
    - Model parameters are stored in a ``PyTorchTrainState`` opaque pytree leaf.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        processor: Any,
        optimizer: torch.optim.Optimizer,
        embodiment_tag: EmbodimentTag,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        lr_scheduler: Any = None,
    ):
        self.model = model
        self.processor = processor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.embodiment_tag = embodiment_tag
        self.device = device
        self.dtype = dtype

        modality_cfg = self._get_modality_cfg()

        # Language key from modality config (e.g. "annotation.human.task_description")
        self._lang_key = modality_cfg["language"].modality_keys[0]

        # View keys from modality config
        self._video_keys = modality_cfg["video"].modality_keys
        self._state_keys = modality_cfg["state"].modality_keys
        self._action_keys = modality_cfg["action"].modality_keys

        # Per-modality dimension info (unpadded)
        stats = self._get_stats()
        self._state_dims = {k: stats["state"][k]["dim"].item() for k in self._state_keys}
        self._action_dims = {k: stats["action"][k]["dim"].item() for k in self._action_keys}

        # Total environment action / state dims (sum of unpadded modalities)
        self.action_dim = sum(self._action_dims.values())
        self.state_dim = sum(self._state_dims.values())

        # Padded model config -- EXPOLearner uses model_config.action_dim
        # as the padded dimension for batch preparation.
        self.model_config = SimpleNamespace(
            action_dim=processor.max_action_dim,
            action_horizon=processor.max_action_horizon,
        )

        # JAX sharding fields (unused for PyTorch, required by interface)
        self.mesh = None
        self.infer_sharding = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @classmethod
    def initialize(
        cls,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        device: Union[str, torch.device] = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        lr: float = 1e-5,
        weight_decay: float = 1e-5,
        ema_decay: float = 0.999,
        freeze_backbone: bool = False,
        **kwargs,
    ) -> Tuple[Gr00tAgent, PyTorchTrainState, dict]:
        """Load GR00T model + processor and create a Gr00tAgent.

        Args:
            model_path: Path to the finetuned GR00T checkpoint directory.
            embodiment_tag: Embodiment tag (string or enum).
            device: Torch device for the model.
            dtype: Torch dtype (default: bfloat16).
            lr: Learning rate for the AdamW optimizer.
            weight_decay: Weight decay.
            ema_decay: EMA decay rate for target params.
            **kwargs: Ignored (for API compatibility).

        Returns:
            (agent, actor_train_state, sharding_info)
            ``sharding_info`` is an empty dict (PyTorch does not use JAX sharding).
        """
        # Import to register model/processor
        import gr00t.model  # noqa: F401
        from gr00t.data.embodiment_tags import EmbodimentTag as ETag

        if isinstance(embodiment_tag, str):
            embodiment_tag = ETag.resolve(embodiment_tag)

        # Resolve processor directory (checkpoint may save under "processor/")
        model_dir = model_path
        if not (model_dir / "processor_config.json").exists():
            processor_dir = model_dir / "processor"
        else:
            processor_dir = model_dir

        # Load model
        model = AutoModel.from_pretrained(model_dir)
        model.to(device=device, dtype=dtype)
        model.train()

        # Optionally freeze the VLM backbone (vision encoder + LLM) and train
        # only the DiT action head. This matches the EXPO-FT recipe (base image
        # encoder frozen, §C.1) and is REQUIRED to fit Adam optimizer states
        # on a single 32 GiB GPU: the full model (~3B params) needs ~12 GiB of
        # Adam state, which OOMs together with the model + critic + activations.
        if freeze_backbone:
            backbone = getattr(model, "backbone", None)
            if backbone is not None:
                for p in backbone.parameters():
                    p.requires_grad_(False)
                n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
                n_total = sum(p.numel() for p in model.parameters())
                logger.info(
                    "freeze_gr00t_backbone=True: backbone frozen, "
                    "training %d / %d params (%.1f%%)",
                    n_train, n_total, 100.0 * n_train / max(n_total, 1),
                )

        # Load processor
        processor = AutoProcessor.from_pretrained(processor_dir)
        processor.train()

        # Create optimizer over trainable params only
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        # Create opaque PyTorchTrainState
        train_state = PyTorchTrainState.create(model, opt, ema_decay=ema_decay)

        # Build agent
        agent = cls(
            model=model,
            processor=processor,
            optimizer=opt,
            embodiment_tag=embodiment_tag,
            device=torch.device(device) if isinstance(device, str) else device,
            dtype=dtype,
        )

        return agent, train_state, {}

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def get_params(self, train_state: PyTorchTrainState) -> Dict[str, np.ndarray]:
        """Return best available params (EMA if present, else current)."""
        return train_state.get_best_params()

    def init_target_params(
        self, rng: jax.random.PRNGKey, *, resume: bool = False
    ) -> None:
        """Base-VLA target EMA is unused on the GR00T EXPO path.

        EXPO's OTF/next-action sampling uses the *online* actor directly
        (see ``EXPOLearnerGR00T``); only the critic has a Polyak target. We
        therefore return None so ``EXPOLearner.create`` stores a no-op target
        and ``update_actor`` does not maintain it.
        """
        return None

    # ------------------------------------------------------------------
    # Modality / dimension helpers
    # ------------------------------------------------------------------

    def _get_modality_cfg(self) -> dict:
        return self.processor.get_modality_configs()[self.embodiment_tag.value]

    def _get_stats(self) -> dict:
        return self.processor.state_action_processor.norm_params[self.embodiment_tag.value]

    def _split_state(self, flat_state: np.ndarray) -> Dict[str, np.ndarray]:
        """Split a flat concatenated state array into per-modality keys.

        ``flat_state`` shape: ``(D_total,)`` or ``(T, D_total)``.
        """
        result = {}
        start = 0
        for key in self._state_keys:
            dim = self._state_dims[key]
            result[key] = flat_state[..., start : start + dim]
            start += dim
        return result

    def _split_action(self, flat_action: np.ndarray) -> Dict[str, np.ndarray]:
        """Split a flat action array into per-modality keys.

        ``flat_action`` shape: ``(horizon, total_env_dim)``.
        """
        result = {}
        start = 0
        for key in self._action_keys:
            dim = self._action_dims[key]
            result[key] = flat_action[..., start : start + dim]
            start += dim
        return result

    # ------------------------------------------------------------------
    # Batch preparation for actor training
    # ------------------------------------------------------------------

    def prepare_batch_for_actor(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert a training batch into the collated dict that ``model.forward`` expects.

        The incoming ``batch`` is assumed to contain raw data (uint8 images,
        raw state, raw actions) -- as produced by the GR00T replay buffer
        (``gr00t_replay_buffer.py``, Task #4).

        Each sample is funneled through ``processor.__call__`` (normalisation +
        image transforms + VLM tokenisation) and the results are collated via
        the data collator.

        Returns a dict of torch tensors on ``self.device`` ready for
        ``model.forward(collated)``.
        """
        modality_cfg = self._get_modality_cfg()
        from gr00t.data.types import MessageType, VLAStepData

        # Infer batch size
        if "actions" in batch:
            B = batch["actions"].shape[0]
        elif "action" in batch:
            B = batch["action"].shape[0]
        elif "state" in batch:
            B = batch["state"].shape[0]
        else:
            raise KeyError("batch must contain one of 'actions', 'action', or 'state'")

        # Language prompt
        if "prompt" in batch:
            prompts = (
                [str(batch["prompt"])] * B
                if isinstance(batch["prompt"], str)
                else [str(p) for p in batch["prompt"]]
            )
        else:
            prompts = [""] * B

        processed_samples = []
        for i in range(B):
            # --- Images: each view as a single-frame list ---
            images = {}
            for view in modality_cfg["video"].modality_keys:
                img = batch["image"][view][i]
                images[view] = [np.asarray(img)]

            # --- State: split flat array into per-key ---
            # Add the temporal axis if missing: the GR00T processor's __call__
            # expects per-modality state of shape (T, D), but a replay-buffer
            # sample's state arrives as a flat (D,) vector.
            state_array = np.asarray(batch["state"][i])
            if state_array.ndim == 1:
                state_array = state_array[np.newaxis, :]  # (1, D)
            states = self._split_state(state_array)

            # --- Actions: split flat array into per-key with temporal dim ---
            if "actions" in batch:
                action_array = np.asarray(batch["actions"][i])
            elif "action" in batch:
                action_array = np.asarray(batch["action"][i])
            else:
                action_array = np.zeros(
                    (self.model_config.action_horizon, self.action_dim), dtype=np.float32
                )
            if action_array.ndim == 1:
                # Flattened (C*dim,) -> (C, dim): preserve the temporal axis so
                # _split_action yields (C, modality_dim) per key and GR00T's
                # action_mask supervises only the C executed steps (Fix #7).
                action_array = action_array.reshape(-1, self.action_dim)
            actions = self._split_action(action_array)

            # Build VLAStepData and process
            vla_step = VLAStepData(
                images=images,
                states=states,
                actions=actions,
                text=prompts[i],
                embodiment=self.embodiment_tag,
            )
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]
            processed = self.processor(messages)
            processed_samples.append(processed)

        # Collate into a single batch dict
        collated = self.processor.collator(processed_samples)["inputs"]

        # Move to device and convert dtype
        def _to_device(x):
            if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
                return x.to(device=self.device, dtype=self.dtype)
            elif isinstance(x, torch.Tensor):
                return x.to(device=self.device)
            return x

        return {k: _to_device(v) for k, v in collated.items()}

    def process_raw_inputs_batch(
        self,
        obs_list,
        action_dim: int,
        resize_size,
        normalize: bool = True,
    ):
        """Batched ``process_raw_inputs`` for a list of raw observations (Fix #9).

        ``process_observation`` natively supports batch ``B > 1`` (it tokenizes
        all B prompts with shared padding), so a chunk of next-states is
        processed in one call instead of B sequential single-obs calls.  Each
        obs has ``video.<view>`` (H, W, C), ``state.<key>`` (dim,), and
        ``prompt``.
        """
        modality_cfg = self._get_modality_cfg()
        obs: Dict[str, Any] = {}
        for view in modality_cfg["video"].modality_keys:
            obs[f"video.{view}"] = np.stack(
                [np.asarray(o[f"video.{view}"]) for o in obs_list]
            )[:, np.newaxis, ...]  # (B, 1, H, W, C)
        for key in modality_cfg["state"].modality_keys:
            obs[f"state.{key}"] = np.stack(
                [np.asarray(o[f"state.{key}"]).reshape(-1) for o in obs_list]
            )[:, np.newaxis, :]  # (B, 1, dim)
        lang_key = modality_cfg["language"].modality_keys[0]
        obs[lang_key] = [str(o.get("prompt", "")) for o in obs_list]  # B flat strings

        processed = self.processor.process_observation(obs, self.embodiment_tag)

        def _to_device(x):
            if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
                return x.to(device=self.device, dtype=self.dtype)
            elif isinstance(x, torch.Tensor):
                return x.to(device=self.device)
            return x

        return {k: _to_device(v) for k, v in processed.items()}

    # ------------------------------------------------------------------
    # Training step (PyTorch eager)
    # ------------------------------------------------------------------

    def train_step(
        self,
        rng_key: jax.random.PRNGKey,
        train_state: PyTorchTrainState,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[PyTorchTrainState, Dict[str, float]]:
        """Run one training step: forward, backward, optimizer step.

        Args:
            rng_key: JAX RNG key (unused -- GR00T uses its own torch RNG).
            train_state: Current PyTorchTrainState with params.
            batch: Collated batch dict (from ``prepare_batch_for_actor``).

        Returns:
            (new_train_state, info_dict)
        """
        # The live model + optimizer are held inside train_state; update them
        # in place. No full state_dict numpy round-trip (bf16-safe, fast).
        model = train_state.model
        model.train()
        train_state.optimizer.zero_grad()

        if getattr(self, "_use_gradient_checkpointing", False):
            # Wrap forward in checkpoint: activations are recomputed during
            # backward, trading ~15% wall time for ~25-30% VRAM savings.
            def _ckpt_forward(*args, **kwargs):
                return model.forward(*args, **kwargs)

            outputs = torch.utils.checkpoint.checkpoint(
                _ckpt_forward, batch, use_reentrant=False
            )
        else:
            outputs = model.forward(batch)

        loss = outputs["loss"]
        loss.backward()

        # Acquire model lock during optimizer step to prevent concurrent
        # inference reads of partially-updated parameters (async mode).
        lock = getattr(self, "_model_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            train_state.optimizer.step()
        finally:
            if lock is not None:
                lock.release()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        new_step = train_state.step + 1
        new_train_state = train_state.update_ema().replace(step=new_step)

        info: Dict[str, float] = {
            "actor_loss": float(loss.detach()),
            "actor_state_step": float(new_step),
        }
        return new_train_state, info

    def enable_gradient_checkpointing(self, enabled: bool = True):
        """Toggle gradient checkpointing for the training forward pass.

        Reduces peak VRAM ~25-30% at the cost of ~15% slower backward.
        Call before training starts (not thread-safe).
        """
        self._use_gradient_checkpointing = enabled

    def enable_model_lock(self, enabled: bool = True):
        """Create a threading lock for safe concurrent model access.

        When enabled, ``train_step`` and ``sample_actions`` acquire this lock
        around the critical sections (optimizer step and model forward). This
        prevents parameter corruption when the actor and learner thread share
        the same live model on the same GPU.

        Only needed for async learner/actor mode; sync training has no
        concurrent access. Call before threading starts.
        """
        import threading
        self._model_lock = threading.Lock() if enabled else None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process_raw_inputs(
        self,
        raw_observations: Union[Dict, np.ndarray],
        action_dim: int,
        resize_size: int,
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """Convert a raw env observation into the model inference format.

        Expects env observations with flat ``video.<view>``, ``state.<key>``
        and ``prompt`` keys.  Uses ``processor.process_observation`` to
        normalise state, process images, and tokenise the language instruction.

        Returns a ``BatchFeature`` dict (tensors on ``self.device``) that can
        be passed directly to ``sample_actions``.
        """
        modality_cfg = self._get_modality_cfg()

        obs: Dict[str, Any] = {}

        for key, value in raw_observations.items():
            if key.startswith("video."):
                obs[key] = np.asarray(value)[np.newaxis, np.newaxis]
            elif key.startswith("state."):
                obs[key] = np.asarray(value)[np.newaxis, np.newaxis]

        # Language key: process_observation iterates this as B string prompts
        # (formalize_language lower-cases each via re.sub), so a flat list of
        # strings is required -- not a nested [[prompt]].
        lang_key = modality_cfg["language"].modality_keys[0]
        prompt = str(raw_observations.get("prompt", ""))
        obs[lang_key] = [prompt]

        processed = self.processor.process_observation(obs, self.embodiment_tag)

        # Move tensors to device
        def _to_device(x):
            if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
                return x.to(device=self.device, dtype=self.dtype)
            elif isinstance(x, torch.Tensor):
                return x.to(device=self.device)
            return x

        return {k: _to_device(v) for k, v in processed.items()}

    def sample_actions(
        self,
        transformed_inputs: Dict,
        train_state: PyTorchTrainState,
        rng: jax.random.PRNGKey,
        train: bool = False,
        num_samples: int = 1,
    ) -> Tuple[np.ndarray, float]:
        """Run GR00T flow-matching inference to produce action predictions.

        Args:
            transformed_inputs: BatchFeature from ``process_raw_inputs``.
            train_state: Current or cached train state for weight loading.
            rng: Ignored (PyTorch model uses its own RNG).
            train: Unused (model always in eval mode for inference).
            num_samples: Number of action samples (currently only 1 supported).

        Returns:
            (actions, infer_ms)
            ``actions`` shape: ``(num_samples, action_horizon, env_action_dim)``
        """
        # GR00T path uses no actor EMA (ema_decay=None, see build_gr00t): the
        # live model holds the online params, so inference runs on it as-is.
        self.model.train(False)

        # N-sample via a SINGLE batched forward: GR00T's flow-matching head draws
        # independent per-batch-element initial noise (gr00t_n1d7 torch.randn), so
        # tiling the input num_samples x along the batch axis yields num_samples
        # diverse chunks in one call.
        inputs = (
            self._repeat_inputs(transformed_inputs, num_samples)
            if num_samples > 1
            else transformed_inputs
        )

        t0 = time.perf_counter()
        lock = getattr(self, "_model_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            with torch.inference_mode():
                model_pred = self.model.get_action(inputs)
        finally:
            if lock is not None:
                lock.release()
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # (num_samples, max_horizon, max_dim) -> (num_samples, env_horizon, env_dim)
        pred = model_pred["action_pred"].float()
        env_cfg = self._get_modality_cfg()
        env_horizon = len(env_cfg["action"].delta_indices)
        pred = pred[:, :env_horizon, : self.action_dim]
        return pred.cpu().numpy(), infer_ms

    def sample_training_actions(
        self,
        transformed_inputs: Dict,
        train_state: PyTorchTrainState,
        rng: jax.random.PRNGKey,
        train: bool = True,
        num_samples: int = 1,
    ) -> Tuple[np.ndarray, float]:
        """Same as ``sample_actions`` for training-time action selection."""
        return self.sample_actions(transformed_inputs, train_state, rng, train, num_samples)

    def process_transformed_outputs(
        self,
        transformed_actions: Union[np.ndarray, torch.Tensor],
        unnormalize: bool = True,
        state: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Decode and unnormalise model actions back to environment action space.

        Uses the GR00T processor's ``decode_action`` to convert per-modality
        normalised actions into physical (absolute) actions.

        Args:
            transformed_actions: Model output actions, shape
                ``(N, action_horizon, padded_dim)`` or ``(N, action_horizon, env_dim)``.
            unnormalize: If False, skip decode and just trim padding.
            state: Optional per-modality reference state dict
                ``{key: (N, T_state, D)}`` (raw, unnormalised) required when the
                embodiment uses RELATIVE actions (e.g. CR5AF eef_9d/joint_pos) so
                ``decode_action`` can convert relative deltas to absolute actions.
                ``N`` must match ``transformed_actions.shape[0]``. Ignored when
                ``unnormalize`` is False or the processor has ``use_relative_action``
                disabled.

        Returns:
            Unnormalised flat actions, shape ``(N, action_horizon, env_action_dim)``.
        """
        if isinstance(transformed_actions, torch.Tensor):
            arr = transformed_actions.cpu().numpy()
        else:
            arr = np.asarray(transformed_actions)

        if not unnormalize:
            env_horizon = len(self._get_modality_cfg()["action"].delta_indices)
            return arr[:, :env_horizon, : self.action_dim]

        decoded = self.processor.decode_action(arr, self.embodiment_tag, state=state)
        flat = self._concat_action(decoded)
        return flat.astype(np.float32)

    def _concat_action(self, per_key: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate per-modality actions in the canonical order."""
        return np.concatenate([per_key[k] for k in self._action_keys], axis=-1)

    # ------------------------------------------------------------------
    # Critic-facing helpers (obs/state built from raw env data, NOT from GR00T)
    # ------------------------------------------------------------------

    def _repeat_inputs(self, inputs: Dict[str, Any], n: int) -> Dict[str, Any]:
        """Tile every tensor in a BatchFeature dict n x along the batch axis."""
        out: Dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                out[k] = v.repeat(n, *([1] * (v.dim() - 1)))
            elif isinstance(v, dict):
                out[k] = self._repeat_inputs(v, n)
            else:
                out[k] = v
        return out

    @staticmethod
    def _resize_image(img: np.ndarray, size) -> np.ndarray:
        """Resize an (H, W, 3) uint8 image to size=(h, w)."""
        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=tuple(size), mode="bilinear", align_corners=False)
        return t[0].permute(1, 2, 0).to(torch.uint8).numpy()

    def convert_env_obs_to_gr00t(
        self,
        env_obs: Dict[str, Any],
        side_camera_id: str = "",
        wrist_camera_id: str = "",
    ) -> Dict[str, Any]:
        """Convert an env observation to GR00T format (``video.<view>``, ``state.<key>``).

        If the observation already has ``video.*`` keys, returns as-is.
        Otherwise converts from OpenPI/Droid-style keys using the camera ID
        mapping.  Missing state modalities are zero-padded to the correct
        dimension from the processor's modality config.

        Args:
            env_obs: Raw env observation dict (OpenPI or GR00T format).
            side_camera_id: OpenPI camera key for the static/scene camera
                (e.g. ``"table_view"``, ``"exterior_image_1_left"``).
            wrist_camera_id: OpenPI camera key for the wrist camera
                (e.g. ``"hand_view"``, ``"wrist_image_left"``).

        Returns:
            Dict with ``video.<view>`` (uint8), ``state.<key>`` (float32),
            and ``prompt`` keys.
        """
        # Already GR00T format — pass through
        if any(k.startswith("video.") for k in env_obs):
            return dict(env_obs)

        gr00t_obs: Dict[str, Any] = {}

        # --- Images: map env camera keys to GR00T video keys ---
        # Build a mapping from whatever image-like keys are present.
        # Priority: explicit camera_id args, then heuristic matching.
        _env_img_keys = [
            k for k in env_obs
            if isinstance(env_obs[k], np.ndarray)
            and env_obs[k].ndim >= 2
            and k not in ("prompt",)
            and not k.startswith("state.")
        ]

        if side_camera_id or wrist_camera_id:
            # Explicit mapping via task-config camera IDs
            _cam_map = {}
            if wrist_camera_id and len(self._video_keys) >= 1:
                _cam_map[wrist_camera_id] = self._video_keys[0]
            if side_camera_id and len(self._video_keys) >= 2:
                _cam_map[side_camera_id] = self._video_keys[1]
            # Also try common OpenPI key names as fallback
            _cam_map.setdefault("wrist_image_left", self._video_keys[0] if len(self._video_keys) >= 1 else "")
            _cam_map.setdefault("exterior_image_1_left", self._video_keys[1] if len(self._video_keys) >= 2 else "")
        else:
            # Heuristic: assign env image keys in sorted order to GR00T video keys
            _cam_map = {}
            for i, k in enumerate(sorted(_env_img_keys)[: len(self._video_keys)]):
                _cam_map[k] = self._video_keys[i]

        for env_key, gr00t_view in _cam_map.items():
            if env_key in env_obs and gr00t_view:
                gr00t_obs[f"video.{gr00t_view}"] = np.asarray(env_obs[env_key])

        # --- State: map known OpenPI keys + zero-pad missing ---
        # Common OpenPI → GR00T state-key heuristics.
        _openpi_state_map = {
            "cartesian_position": self._state_keys[0] if len(self._state_keys) > 0 else None,
            "joint_position": self._state_keys[1] if len(self._state_keys) > 1 else None,
            "gripper_position": self._state_keys[-1] if len(self._state_keys) > 2 else None,
        }

        provided: set = set()
        for openpi_key, gr00t_key in _openpi_state_map.items():
            if gr00t_key and openpi_key in env_obs:
                val = np.asarray(env_obs[openpi_key], dtype=np.float32).reshape(-1)
                target_dim = self._state_dims.get(gr00t_key, len(val))
                if len(val) != target_dim:
                    # Truncate or zero-pad to match expected dimension
                    padded = np.zeros(target_dim, dtype=np.float32)
                    n = min(len(val), target_dim)
                    padded[:n] = val[:n]
                    val = padded
                gr00t_obs[f"state.{gr00t_key}"] = val
                provided.add(gr00t_key)

        # Zero-pad any remaining state keys
        for key in self._state_keys:
            if key not in provided:
                gr00t_obs[f"state.{key}"] = np.zeros(
                    self._state_dims.get(key, 1), dtype=np.float32
                )

        # --- Prompt ---
        gr00t_obs["prompt"] = str(env_obs.get("prompt", ""))

        return gr00t_obs

    def build_obs_dict(self, images: Dict[str, np.ndarray], flat_state: np.ndarray, prompt: str) -> Dict[str, Any]:
        """Repack flat env state + per-view images + prompt into raw-obs keys
        (``video.<view>`` / ``state.<key>`` / ``prompt``) for process_raw_inputs.
        """
        obs: Dict[str, Any] = {}
        for k, v in self._split_state(np.asarray(flat_state)).items():
            obs[f"state.{k}"] = np.asarray(v)
        for view in self._video_keys:
            obs[f"video.{view}"] = np.asarray(images[view])
        obs["prompt"] = prompt
        return obs

    def critic_inputs_from_observation(self, obs: Dict[str, Any], image_size=None) -> Tuple[np.ndarray, np.ndarray]:
        """Build the critic observation and state from a raw env observation.

        Returns (critic_obs, critic_state) with a leading batch dim of 1:
        - critic_obs: ``(1, H, W, 3*n_views)`` uint8 (caller normalizes to float);
        - critic_state: ``(1, state_dim)`` float32.
        """
        views = []
        for view in self._video_keys:
            img = np.asarray(obs[f"video.{view}"])
            if image_size is not None and tuple(img.shape[:2]) != tuple(image_size):
                img = self._resize_image(img, image_size)
            views.append(img)
        critic_obs = np.concatenate(views, axis=-1)[np.newaxis].astype(np.uint8)
        flat = np.concatenate(
            [np.asarray(obs[f"state.{k}"]).reshape(-1) for k in self._state_keys], axis=0
        )
        return critic_obs, flat[np.newaxis].astype(np.float32)


def build_gr00t(config, seed, mesh, data_sharding, replicated_sharding, resume, default_prompt):
    """Build Gr00Tactor, train state, and target params from agent config.

    Matches the ``build_pi05`` signature so it can be used interchangeably in
    training scripts.

    Returns (actor, actor_train_state, target_actor_params, agent_kwargs, metadata)
    where metadata is a dict with action_horizon, resize_size, freeze_encoder
    ready to pass into ``EXPOLearner.create``.
    """
    from pathlib import Path

    model_path = Path(config.gr00t_model_path)
    embodiment_tag = config.gr00t_embodiment_tag

    actor, actor_train_state, _ = Gr00tAgent.initialize(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        device="cuda:0",
        dtype=torch.bfloat16,
        lr=config.get("actor_lr", 1e-5),
        weight_decay=config.get("weight_decay", 1e-5),
        # Target base-VLA EMA for TD next-action sampling (paper Sec C.2,
        # tau_pi).  None disables it (the online actor is used, current
        # default); 0.999 ~= tau_pi=1e-3 and enables a target forward via
        # ``PyTorchTrainState.run_with_ema``.
        ema_decay=config.get("gr00t_actor_ema_decay", None),
        # Freeze the VLM backbone, train only the DiT action head (paper §C.1,
        # and required to fit Adam states in 32 GiB).
        freeze_backbone=config.get("freeze_gr00t_backbone", False),
    )

    # Base-VLA target EMA is unused on the GR00T EXPO path (OTF/next-action
    # sampling uses the online actor; only the critic has a Polyak target).
    target_actor_params = None

    # Gradient checkpointing (VRAM/throughput tradeoff)
    if config.get("use_gradient_checkpointing", False):
        actor.enable_gradient_checkpointing(True)

    # Model access lock for async learner/actor safety (Gap #5)
    if config.get("use_model_lock", False):
        actor.enable_model_lock(True)

    # Config knobs consumed by EXPOLearnerGR00T through the actor instance
    # (the agent is a flax struct, so plain attributes on the non-struct
    # Gr00tAgent are the simplest channel).
    actor.gr00t_critic_aug_full = bool(config.get("use_full_augmentation", True))
    actor.gr00t_target_sample_chunk = int(config.get("gr00t_target_sample_chunk", 8))

    metadata = dict(
        action_horizon=len(
            actor._get_modality_cfg()["action"].delta_indices
        ),
        resize_size=None,  # GR00T handles image sizing internally
        freeze_encoder=False,
    )

    # Plumb the EXPO hyperparameters from the model config into the learner.
    # (Returning {} here previously left every config knob dead: the agent was
    # always built with EXPOLearner.create's defaults -- N=32, num_qs=2,
    # n_edit_samples=0, latent_dim_image=50, actor_success_only=False -- ignoring
    # the paper/config-tuned values below.)
    agent_kwargs = dict(
        N=config.get("N", 32),
        n_edit_samples=config.get("n_edit_samples", 0),
        num_qs=config.get("num_qs", 2),
        num_min_qs=config.get("num_min_qs", None),
        critic_layer_norm=config.get("critic_layer_norm", False),
        critic_dropout_rate=config.get("critic_dropout_rate", None),
        critic_weight_decay=config.get("critic_weight_decay", None),
        use_critic_resnet=config.get("use_critic_resnet", False),
        use_pnorm=config.get("use_pnorm", False),
        hidden_dims=tuple(config.get("hidden_dims", (256, 256))),
        latent_dim_image=config.get("latent_dim_image", 50),
        latent_dim_state=config.get("latent_dim_state", 50),
        include_state=config.get("include_state", True),
        encoder_stage_sizes=tuple(config.get("encoder_stage_sizes", (2, 2, 2, 2))),
        encoder_num_filters=config.get("encoder_num_filters", 64),
        actor_lr=config.get("actor_lr", 3e-4),
        critic_lr=config.get("critic_lr", 3e-4),
        temp_lr=config.get("temp_lr", 3e-4),
        edit_scale=config.get("edit_scale", 1.0),
        actor_drop=config.get("actor_drop", None),
        actor_tau=config.get("actor_tau", 1e-3),
        adjust_target_entropy=config.get("adjust_target_entropy", False),
        entropy_scale=config.get("entropy_scale", 1.0),
        init_temperature=config.get("init_temperature", 1.0),
        batch_split=config.get("batch_split", 1),
        encode_batch_split=config.get("encode_batch_split", 1),
        discount=config.get("discount", 0.99),
        tau=config.get("tau", 0.005),
        freeze_critic_encoder=config.get("freeze_critic_encoder", False),
        actor_success_only=config.get("actor_success_only", False),
        use_full_augmentation=config.get("use_full_augmentation", True),
    )

    return actor, actor_train_state, target_actor_params, agent_kwargs, metadata
