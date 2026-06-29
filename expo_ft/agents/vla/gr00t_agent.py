"""Gr00tAgent: VLA agent wrapper for GR00T N1.7 model within the EXPO-FT framework.

Converts between EXPO's JAX-native batch format and GR00T's PyTorch-native
processing, implementing the ``vla_base.Model`` interface so Gr00tN1d7 can
serve as the VLA actor in EXPOLearnerGR00T.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import jax
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from expo_ft.agents.vla.vla_base import Model
from expo_ft.agents.vla.gr00t_train_state import PyTorchTrainState

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

        # Load processor
        processor = AutoProcessor.from_pretrained(processor_dir)
        processor.train()

        # Create optimizer
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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

    def _load_params(self, params_dict: Dict[str, np.ndarray]) -> None:
        """Load numpy-array parameters into the PyTorch model."""
        state_dict: Dict[str, torch.Tensor] = {}
        existing_tensors = {n: p for n, p in self.model.named_parameters()}
        existing_tensors.update({n: b for n, b in self.model.named_buffers()})
        for name, arr in params_dict.items():
            if name in existing_tensors:
                ref = existing_tensors[name]
                state_dict[name] = torch.tensor(arr, dtype=ref.dtype, device=ref.device)
            else:
                state_dict[name] = torch.tensor(arr, device=self.device)
        self.model.load_state_dict(state_dict, strict=False)

    def _extract_params(self) -> Dict[str, np.ndarray]:
        """Extract current model parameters as a numpy dict."""
        return {
            name: tensor.detach().cpu().numpy()
            for name, tensor in self.model.state_dict().items()
        }

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
            state_array = np.asarray(batch["state"][i])
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

        outputs = model.forward(batch)
        loss = outputs["loss"]
        loss.backward()

        train_state.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        new_step = train_state.step + 1
        new_train_state = train_state.update_ema().replace(step=new_step)

        info: Dict[str, float] = {
            "actor_loss": float(loss.detach()),
            "actor_state_step": float(new_step),
        }
        return new_train_state, info

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

        # Language key
        lang_key = modality_cfg["language"].modality_keys[0]
        prompt = str(raw_observations.get("prompt", ""))
        obs[lang_key] = [[prompt]]

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
        # Load best available params (EMA if present)
        best_params = train_state.get_best_params()
        if best_params:
            self._load_params(best_params)
        self.model.eval()

        t0 = time.perf_counter()
        with torch.inference_mode():
            model_pred = self.model.get_action(transformed_inputs)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # Model output: (1, max_action_horizon, max_action_dim)
        pred = model_pred["action_pred"].float()
        # Slice to actual horizon and env action dim
        env_cfg = self._get_modality_cfg()
        env_horizon = len(env_cfg["action"].delta_indices)
        pred = pred[:, :env_horizon, : self.action_dim]
        pred_np = pred.cpu().numpy()
        return pred_np, infer_ms

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
    ) -> np.ndarray:
        """Decode and unnormalise model actions back to environment action space.

        Uses the GR00T processor's ``decode_action`` to convert per-modality
        normalised actions into physical (absolute) actions.

        Args:
            transformed_actions: Model output actions, shape
                ``(N, action_horizon, padded_dim)`` or ``(N, action_horizon, env_dim)``.
            unnormalize: If False, skip decode and just trim padding.

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

        decoded = self.processor.decode_action(arr, self.embodiment_tag, state=None)
        flat = self._concat_action(decoded)
        return flat.astype(np.float32)

    def _concat_action(self, per_key: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate per-modality actions in the canonical order."""
        return np.concatenate([per_key[k] for k in self._action_keys], axis=-1)


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
        ema_decay=config.get("ema_decay", 0.999),
    )

    # Base-VLA target EMA is unused on the GR00T EXPO path (OTF/next-action
    # sampling uses the online actor; only the critic has a Polyak target).
    target_actor_params = None

    metadata = dict(
        action_horizon=len(
            actor._get_modality_cfg()["action"].delta_indices
        ),
        resize_size=None,  # GR00T handles image sizing internally
        freeze_encoder=False,
    )

    return actor, actor_train_state, target_actor_params, {}, metadata
