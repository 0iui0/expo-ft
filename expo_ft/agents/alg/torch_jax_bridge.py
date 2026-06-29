"""Centralized conversions at the GR00T (PyTorch actor) / EXPO (JAX critic) boundary.

The actor is PyTorch; the critic, residual actor, temperature and image encoder
are JAX. The only arrays that actually cross this boundary are *actions*: the
actor returns numpy via ``.cpu().numpy()`` and the critic consumes jax arrays.
Critic images do NOT cross the boundary (they come straight from the replay
buffer as jax arrays).

Keeping the few conversions in one tested place avoids the dtype / device /
layout bugs that multiply when torch and jax arrays are mixed ad-hoc.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import torch


def torch_to_np(t: Any) -> np.ndarray:
    """torch tensor (any device/dtype) -> contiguous float32 numpy on host."""
    if isinstance(t, torch.Tensor):
        return t.detach().to(torch.float32).cpu().numpy()
    return np.asarray(t, dtype=np.float32)


def np_to_jax(a: Any, dtype=None) -> jnp.ndarray:
    arr = np.asarray(a)
    if dtype is not None:
        arr = arr.astype(dtype)
    else:
        arr = arr.astype(np.float32)
    return jnp.asarray(arr)


def torch_to_jax(t: Any, dtype=None) -> jnp.ndarray:
    """torch tensor -> jax array (host round-trip via numpy, float32)."""
    return np_to_jax(torch_to_np(t), dtype=dtype)
