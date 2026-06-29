"""PyTorch TrainState for GR00T within EXPO-FT's JAX learner.

GPU-resident design: the live ``nn.Module`` and optimizer are held *by reference*
inside this state. There is no per-step numpy round-trip of the full state_dict
(the previous design both crashed on bfloat16 tensors — numpy has no bf16 — and
was a throughput/memory killer for a multi-B VLA in an online RL loop).

Only the trainable (selective-unfreeze) subset of parameters is mirrored into a
torch-native EMA dict for inference / target use. The state is registered as an
*opaque* JAX pytree leaf so ``EXPOLearner`` (a flax ``struct.PyTreeNode``) can
hold it without JAX trying to trace its internals.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import jax
import torch


@dataclass
class PyTorchTrainState:
    """Live PyTorch model + optimizer + torch EMA of trainable params.

    Attributes:
        model: live ``nn.Module`` on device; params are mutated in place by the
            optimizer during ``Gr00tAgent.train_step``.
        optimizer: optimizer constructed over ``requires_grad`` params only.
        ema_decay: Polyak averaging factor for the EMA, or ``None`` to disable.
        step: training-step counter.
        ema_params: ``{name: gpu_tensor}`` mirror of the trainable params, or
            ``None`` when EMA is disabled. Used for inference weights.
        trainable_names: cached tuple of trainable parameter names.
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    ema_decay: Optional[float]
    step: int = 0
    ema_params: Optional[Dict[str, torch.Tensor]] = None
    trainable_names: Tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ema_decay: Optional[float] = None,
    ) -> "PyTorchTrainState":
        """Build a state from a live model + optimizer.

        Only parameters with ``requires_grad=True`` (the selective-unfreeze
        subset) are tracked for EMA. The optimizer is assumed to already be
        constructed over exactly that subset.
        """
        model.train()
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        ema_params = (
            {n: p.detach().clone() for n, p in trainable.items()}
            if ema_decay is not None
            else None
        )
        return cls(
            model=model,
            optimizer=optimizer,
            ema_decay=ema_decay,
            step=0,
            ema_params=ema_params,
            trainable_names=tuple(trainable.keys()),
        )

    # ------------------------------------------------------------------
    # Param access
    # ------------------------------------------------------------------
    def trainable_params(self) -> Dict[str, torch.Tensor]:
        """Current trainable params (live references, not copies)."""
        return {n: p for n, p in self.model.named_parameters() if p.requires_grad}

    def get_best_params(self) -> Dict[str, torch.Tensor]:
        """EMA params if available, else current trainable params."""
        return self.ema_params if self.ema_params is not None else self.trainable_params()

    def load_params_into_model(self, params: Dict[str, torch.Tensor]) -> None:
        """Copy a param dict (e.g. EMA) into the live model in place."""
        own = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, tensor in params.items():
                if name in own:
                    own[name].copy_(tensor)

    # ------------------------------------------------------------------
    # EMA (torch-native; works on bf16/fp32 GPU tensors)
    # ------------------------------------------------------------------
    def update_ema(self) -> "PyTorchTrainState":
        """Polyak-average current trainable params into the EMA dict."""
        if self.ema_params is None or self.ema_decay is None:
            return self
        decay = self.ema_decay
        new_ema = {
            name: decay * self.ema_params[name] + (1.0 - decay) * p.detach()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }
        return dataclasses.replace(self, ema_params=new_ema)

    # ------------------------------------------------------------------
    # Functional update helper (mirrors JAX/flax replace API)
    # ------------------------------------------------------------------
    def replace(self, **kwargs) -> "PyTorchTrainState":
        return dataclasses.replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Register as an opaque JAX pytree leaf.
#
# Flatten returns no children and the whole object as auxiliary data, so JAX
# treats the live PyTorch state as a single opaque leaf (it never tries to
# trace/transfer the torch tensors or the optimizer). This lets EXPOLearner
# hold it as a regular field while update() runs eagerly.
# ---------------------------------------------------------------------------
def _flatten_state(state):
    # Treat the whole object as one opaque leaf (no children).
    return (), state


def _unflatten_state(aux_data, children):
    return aux_data


jax.tree_util.register_pytree_node(
    PyTorchTrainState,
    _flatten_state,
    _unflatten_state,
)
