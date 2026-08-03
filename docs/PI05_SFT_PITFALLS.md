# PI0.5 shaft_insert SFT — Pitfalls & Lessons Learned

Hard-won gotchas from fine-tuning PI0.5 (openpi, JAX flow-matching VLA) on the
`cr5af/shaft_insert` task with the CR5AF robot + DH PGE gripper. None of these
are obvious from the code; each cost real debugging time. Read this before
launching a training or eval run.

Shared paths referenced below: `/cache` (wheels, base model params),
`/datasets` (LeRobot datasets), `/data` (large scratch, 11 TB free). The repo
root is written as `<repo>`.

---

## 1. Action representation — the near-identity collapse (the big one)

**Symptom.** SFT converges cleanly (flow-matching loss ~0.26) but open-loop eval
collapses: position MAE ~83 mm, rotation garbage, gripper output constant. The
model scored *worse than doing nothing* — a trivial identity map (predict the
current pose) scored ~1.1 mm.

**Root cause.** Actions were stored as the **absolute next-frame pose** at
~26 Hz. `|action − state|` median ≈ 0.20 mm → `corr(action, current_state) ≈
0.99`. SFT degenerates into a copy task (`output := input`); vision receives no
gradient pressure. GR00T hit the identical collapse on this task.

**Fixes (applied together, one retrain):**
- **Delta-on-xyz.** Translation channels become `action_xyz −= state_xyz`, which
  drives `corr → 0` and forces the model to predict motion from vision. This is
  the primary lever. Implemented with openpi's `DeltaActions` / `AbsoluteActions`
  transforms using `make_bool_mask(3, -7)` = `[T,T,T,F,F,F,F,F,F,F]`. The dataset
  stores absolute poses; the transform converts at load/inference time.
- **Native rot6d, no euler round-trip.** Rotation was being derived
  `rot6d → matrix → euler("XYZ")`. True frame-to-frame rotation is ~0°/frame, but
  euler decomposition injects 12–18°/frame of representational noise the model
  then fits. The raw recordings are already rot6d — keep them native. Cartesian
  state is 9D `[xyz_m(3), rot6d(6)]`; actions are 10D `[xyz_m(3), rot6d(6),
  gripper(1)]`.
- **Gripper stays absolute.** It is binary and changes in <0.5% of steps; delta
  would be a degenerate near-zero distribution.

**Lesson.** Flow-matching loss is a *misleading* health metric for imitation
learning. Always evaluate against an **identity/naive baseline** (predict current
state). "Vision is alive" — same state + different real images changes the
prediction — is the decisive check, not loss.

---

## 2. Rebuilt unified venv (self-built jaxlib cu13 + torch cu13 + flax 0.12.7)

The RTX 5090 (sm_120, 32 GB) needs a self-built jaxlib/XLA and current flax.
Four adjustments are required after any jax/flax rebuild — none obvious from code:

1. **Run `scripts/train.py` directly.** Do *not* use the `run_pi05_sft_patched.py`
   wrapper. On flax 0.12.7 the `remat` `concrete`-kwarg monkeypatch is no longer
   needed, and the wrapper breaks: it re-executes under torch's spawn dataloader
   workers (`runpy`/`exec_module` bypasses the `if __name__` guard), and
   `openpi.scripts` is not an importable package. `train.py`'s own `__main__`
   guard handles spawn correctly.

2. **Set `JAXTYPING_DISABLE=1`.** `openpi.training.utils.TrainState` is
   `@typecheck`'d (jaxtyped + beartype); current beartype rejects optax's loose
   `OptState = ArrayTree` union as a false positive on `opt_state`. The env var
   is honored globally.

3. **Use `--batch-size 32`, not the config's 64.** The self-built jaxlib/XLA does
   not fuse the LM-logits cross-entropy, so batch 64 attempts a single 19.52 GiB
   `jit__mean` allocation → `RESOURCE_EXHAUSTED` on 32 GB. Batch 32 halves it.
   Pair with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`. Rate ~2.8 s/it → ~15.5 h for
   20k steps. (An older jax build fit batch 64; the regression is XLA-fusion, not
   the model.)

4. **flax 0.12.7 `flat_state()` returns `(path, Param)` tuples**, and `Param` is
   unhashable. In `openpi/src/openpi/shared/nnx_utils.py` (`state_map`), build the
   key set as `{path for path, _ in state.filter(filter).flat_state()}` — not
   `set(state.filter(filter).flat_state())`, which raises `TypeError: unhashable`.

---

## 3. Checkpoints will fill the OS disk — write them to `/data`

**Symptom (predicted, avoided).** Training crashes on disk-full around step
~16000.

**Root cause.** The SFT config does not override `save_interval`, so openpi's
default **1000** applies → a 20k-step run writes **~20 checkpoints**. Each PI0.5
checkpoint is **~10.7 GB** (full-precision inference `params` ≈ 12 G base + LoRA
opt state) → ~215 GB total. The OS nvme was ~95% full (~200 GB free).

**Fix.** `--checkpoint-base-dir /data/openpi_checkpoints` (large scratch volume).
`checkpoint_dir` resolves to `<base>/<config-name>/<exp-name>`. Norm stats
(`assets/`) are saved into each checkpoint dir automatically, so eval on a
`/data` checkpoint resolves them without extra flags.

**Resume gotcha.** After a crash, relaunch with `--resume`, **never**
`--overwrite` — `--overwrite` calls `rmtree` on the checkpoint dir and destroys
all progress. Orbax commits checkpoints atomically (`*-tmp` → final dir on
completion), so a power loss mid-write leaves the last *complete* checkpoint
intact and resumable.

---

## 4. Offline / unattended runs (surviving a network cut + session end)

- **`WANDB_MODE=offline`** (+ `WANDB_DIR=<rundir>`) so wandb never touches the
  network. Training is otherwise fully local: base params from `/cache`, dataset
  from `/datasets`. `unset https_proxy http_proxy ALL_PROXY all_proxy` before
  launch — the proxy dies with the network and stale vars break local calls.
- **Detach properly.** `setsid nohup bash orchestrator.sh </dev/null &` survives
  both the controlling session ending and a network cut. (It does **not** survive
  a machine reboot.)
- **Open-loop eval is ~5.6 s/frame** (~10 min for 100 frames), excluding
  first-step JIT. Schedule accordingly. If the training GPU is still busy, run
  eval on the other GPU with `XLA_PYTHON_CLIENT_PREALLOCATE=false` to co-exist.
- **Norm stats must be recomputed** after any change to `data_transforms.inputs`
  (e.g. adding `DeltaActions`) or to feature shapes — they are computed *after*
  input transforms, so stale stats silently mis-scale the delta distribution.

## 5. Deploy — image color order

RealSense delivers **BGR8**. PaliGemma expects **RGB**. Sending raw BGR to the
visual encoder blinds it and produces fixed/constant output. Convert BGR→RGB at
the dataset-conversion boundary and in the deploy adapter.

---

## Tooling gotchas (host, not model)

- A 120 s shell-timeout wrapper will kill long `sleep`s — run long waits as
  detached background processes and poll, don't block.
- `pkill -f <pattern>` matches its own command line and can kill the wrong thing.
  Kill by PID, or filter on `comm == python` with `ps`/`awk`.
