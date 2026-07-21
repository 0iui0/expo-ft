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
import numpy as np
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

    def run_with_ema(self, fn):
        """Run ``fn`` with the EMA (target) trainable params in the live model.

        Used by ``sample_batch_actions`` to draw TD next-actions from the target
        base-VLA (paper Sec C.2, tau_pi).  No-op when EMA is disabled.  Saves the
        online trainable params, swaps the EMA in, runs ``fn``, and restores the
        online params in a ``finally``.  In async mode the caller must hold the
        model lock so the swap is not observed mid-forward by the actor thread.
        """
        if self.ema_params is None or self.ema_decay is None:
            return fn()
        online = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        self.load_params_into_model(self.ema_params)
        try:
            return fn()
        finally:
            self.load_params_into_model(online)

    # ------------------------------------------------------------------
    # Checkpoint serialization (CPU offload for orbax compatibility)
    # ------------------------------------------------------------------
    def state_dict(self):
        """Return model + optimizer state dicts on CPU for orbax checkpointing.

        Returns a dict with keys ``model`` and ``optimizer``, each mapping
        parameter names to CPU ``np.ndarray`` (fp32-safe for msgpack/orbax).
        """
        import numpy as np  # noqa: F811

        model_sd = {}
        for name, p in self.model.state_dict().items():
            model_sd[name] = p.detach().cpu().to(torch.float).numpy()

        opt_sd = {}
        for group_key, group_vals in self.optimizer.state_dict().items():
            if group_key == "param_groups":
                opt_sd[group_key] = group_vals
            elif group_key == "state":
                state_ser = {}
                for param_id, state_vals in group_vals.items():
                    state_ser[str(param_id)] = {
                        k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                        for k, v in state_vals.items()
                    }
                opt_sd[group_key] = state_ser
            else:
                opt_sd[group_key] = group_vals

        return {"model": model_sd, "optimizer": opt_sd}

    def load_state_dict(self, ckpt) -> "PyTorchTrainState":
        """Restore model + optimizer state from a CPU checkpoint dict.

        Loads fp32 CPU arrays back into the live model/optimizer, casting to
        the model's native dtype. Returns a new ``PyTorchTrainState`` with the
        restored step counter (optimizer state is mutated in place).
        """
        import numpy as np  # noqa: F811

        native_dtype = next(self.model.parameters()).dtype
        device = next(self.model.parameters()).device

        # Restore model
        model_sd = {}
        for name, arr in ckpt["model"].items():
            t = torch.from_numpy(np.asarray(arr)).to(device=device, dtype=native_dtype)
            if name in self.model.state_dict():
                target_shape = self.model.state_dict()[name].shape
                if t.shape != target_shape:
                    t = t.reshape(target_shape)
            model_sd[name] = t
        self.model.load_state_dict(model_sd, strict=False)

        # Restore optimizer
        opt_sd = ckpt["optimizer"]
        if "state" in opt_sd:
            restored_state = {}
            for param_id_str, state_vals in opt_sd["state"].items():
                restored_state[int(param_id_str)] = {
                    k: torch.from_numpy(np.asarray(v)).to(device=device, dtype=torch.float)
                    if isinstance(v, np.ndarray)
                    else v
                    for k, v in state_vals.items()
                }
            opt_sd["state"] = restored_state
        self.optimizer.load_state_dict(opt_sd)

        # Rebuild EMA from restored model
        if self.ema_decay is not None:
            new_ema = {
                n: p.detach().clone()
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }
            return dataclasses.replace(self, ema_params=new_ema)

        return self

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
