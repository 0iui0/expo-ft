"""PyTorch-compatible TrainState for managing GR00T model weights within the EXPO-FT JAX framework.

This module provides a PyTorch-native alternative to JAX's TrainState that can be used
as the `actor_train_state` field in EXPOLearner. It manages:
- Model parameters (as a flat Python dict of numpy arrays)
- Optimizer state
- EMA parameters (polyak averaging)
- Step counter

Key design: registered as an opaque JAX pytree node so EXPOLearner can hold it
without JAX trying to trace its contents. The state_dict is stored as numpy arrays
for efficient serialization.
"""

import copy
from typing import Any, Dict, Optional

import jax
import numpy as np
import torch


class PyTorchTrainState:
    """Manage PyTorch model + optimizer state, compatible with EXPOLearner interface.

    Unlike JAX TrainState which holds jax arrays, this stores model params as numpy arrays
    (extracted from state_dict) and reconstructs torch tensors on demand. This allows
    EXPOLearner to treat `actor_train_state` as an opaque pytree leaf.

    Attributes:
        step: Training step counter.
        params: Model parameters as {name: numpy_array} dict.
        ema_params: EMA parameters (optional, same format as params).
        optimizer_state: Serialized PyTorch optimizer state dict (picklable).
        model_def: Reference to the model class for re-merging params.
            Stored as (class_module, class_name) tuple.
        tx_config: Optimizer config for reconstruction.
    """

    def __init__(
        self,
        step: int = 0,
        params: Optional[Dict[str, np.ndarray]] = None,
        ema_params: Optional[Dict[str, np.ndarray]] = None,
        optimizer_state: Optional[Dict] = None,
        model_def: Optional[tuple] = None,
        tx_config: Optional[Dict] = None,
    ):
        self.step = step
        self.params = params or {}
        self.ema_params = ema_params
        self.optimizer_state = optimizer_state
        self.model_def = model_def  # (module_path, class_name) for lazy import
        self.tx_config = tx_config or {}

    @classmethod
    def create(
        cls,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ema_decay: Optional[float] = None,
    ) -> "PyTorchTrainState":
        """Create a PyTorchTrainState from a live model and optimizer.

        Args:
            model: PyTorch model whose parameters will be tracked.
            optimizer: PyTorch optimizer for this model.
            ema_decay: If provided, enable EMA tracking with this decay rate.

        Returns:
            A new PyTorchTrainState with extracted params.
        """
        params = cls._extract_params(model)
        optimizer_state = optimizer.state_dict()
        # Store class info as tuple for lazy reconstruction
        model_def = (model.__class__.__module__, model.__class__.__qualname__)
        tx_config = {
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0),
        }

        ema_params = None
        if ema_decay is not None:
            ema_params = copy.deepcopy(params)

        return cls(
            step=0,
            params=params,
            ema_params=ema_params,
            optimizer_state=optimizer_state,
            model_def=model_def,
            tx_config=tx_config,
        )

    @staticmethod
    def _extract_params(model: torch.nn.Module) -> Dict[str, np.ndarray]:
        """Extract model state_dict as numpy arrays (independent copies)."""
        result = {}
        for name, tensor in model.state_dict().items():
            result[name] = tensor.detach().cpu().numpy().copy()
        return result

    @staticmethod
    def _load_params_into_model(
        model: torch.nn.Module, params: Dict[str, np.ndarray]
    ) -> None:
        """Load numpy params dict back into a PyTorch model."""
        state_dict = {}
        existing = model.state_dict()
        for name, arr in params.items():
            if name in existing:
                dtype = existing[name].dtype
                device = existing[name].device
            else:
                dtype = torch.float32
                device = torch.device("cpu")
            state_dict[name] = torch.tensor(arr, dtype=dtype, device=device)
        model.load_state_dict(state_dict, strict=False)

    def update_ema(self, ema_decay: float) -> "PyTorchTrainState":
        """Apply polyak averaging to EMA params.

        Args:
            ema_decay: EMA decay rate (e.g., 0.999).

        Returns:
            New PyTorchTrainState with updated ema_params.
        """
        if self.ema_params is None:
            return self

        new_ema = {}
        for name, param in self.params.items():
            old_ema = self.ema_params.get(name, param)
            new_ema[name] = ema_decay * old_ema + (1.0 - ema_decay) * param
        return PyTorchTrainState(
            step=self.step,
            params=self.params,
            ema_params=new_ema,
            optimizer_state=self.optimizer_state,
            model_def=self.model_def,
            tx_config=self.tx_config,
        )

    def get_best_params(self) -> Dict[str, np.ndarray]:
        """Return best available params (EMA if available, else current)."""
        if self.ema_params is not None:
            return self.ema_params
        return self.params

    def incremental_update_target(
        self, target_params: Dict[str, np.ndarray], tau: float
    ) -> Dict[str, np.ndarray]:
        """Polyak average update for target network params (used by EXPOLearner).

        Args:
            target_params: Current target params.
            tau: Soft update rate (e.g., 0.001).

        Returns:
            Updated target params.
        """
        best_params = self.get_best_params()
        new_target = {}
        for name, target_val in target_params.items():
            new_val = target_val if name not in best_params else (
                tau * best_params[name] + (1.0 - tau) * target_val
            )
            new_target[name] = new_val
        return new_target

    def replace(self, **kwargs) -> "PyTorchTrainState":
        """Create a new state with some fields replaced (mirrors JAX TrainState API)."""
        return PyTorchTrainState(
            step=kwargs.get("step", self.step),
            params=kwargs.get("params", self.params),
            ema_params=kwargs.get("ema_params", self.ema_params),
            optimizer_state=kwargs.get("optimizer_state", self.optimizer_state),
            model_def=kwargs.get("model_def", self.model_def),
            tx_config=kwargs.get("tx_config", self.tx_config),
        )


# Register PyTorchTrainState as an opaque JAX pytree node.
# This prevents JAX from trying to trace its internals (numpy arrays, dicts, etc.)
# and allows EXPOLearner to hold it as a regular pytree leaf.
def _flatten_state(state):
    """Flatten: treat the entire state as one opaque leaf."""
    # We store the whole object as one auxiliary value
    return (), state


def _unflatten_state(aux_data, children):
    """Unflatten: reconstruct from the opaque leaf."""
    # children is (), aux_data is the original state object
    return aux_data


jax.tree_util.register_pytree_node(
    PyTorchTrainState,
    _flatten_state,
    _unflatten_state,
)
