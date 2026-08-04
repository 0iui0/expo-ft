#!/usr/bin/env python3
"""Launch openpi train.py with flax monkeypatch for jax 0.10.x compatibility.

jax 0.10.0 removed the ``concrete`` kwarg from ``jax.checkpoint``, but the
installed flax still passes it from ``nn.remat``. This wrapper patches flax
before importing openpi, so the SFT config ``expo_pi05_droid_lora_finetune_sft_cartesian_state``
can train on GPU1 (RTX 5090, sm_120) with jax 0.10.1.dev.
"""

# ── monkeypatch flax remat (strip concrete kwarg) ──────────────────────
import flax.linen.transforms as _ft
import flax.linen.partitioning as _fp

_orig_remat = _ft.remat


def _patched_remat(
    target,
    variables=True,
    rngs=True,
    concrete=False,  # noqa: ARG001 — swallowed for jax 0.10 compat
    prevent_cse=True,
    static_argnums=(),
    policy=None,
    methods=None,
):
    return _orig_remat(
        target,
        variables=variables,
        rngs=rngs,
        prevent_cse=prevent_cse,
        static_argnums=static_argnums,
        policy=policy,
        methods=methods,
    )


_ft.remat = _patched_remat
if hasattr(_fp, "remat"):
    _fp.remat = _patched_remat

# ── launch training ────────────────────────────────────────────────────
import importlib.util
import pathlib
import sys

import openpi  # noqa: E402
from openpi.training import config as _config  # noqa: E402

# openpi.scripts is NOT an importable package (train.py is a standalone script
# under the repo's scripts/, sibling to src/openpi). Load it by file path.
# Guarded under __main__: the data loader spawns worker processes that re-import
# this module; only the parent should load train.py and launch training.
if __name__ == "__main__":
    _openpi_root = pathlib.Path(openpi.__file__).parents[2]  # .../openpi (repo root)
    _train_path = _openpi_root / "scripts" / "train.py"
    _spec = importlib.util.spec_from_file_location("openpi_train", _train_path)
    _train = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_train)

    print(f"[patched] Launching PI0.5 SFT with args: {sys.argv[1:]}")
    _train.main(_config.cli())  # tyro parses sys.argv[1:]
