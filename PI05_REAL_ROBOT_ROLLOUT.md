# PI0.5 shaft_insert — Real-Robot Rollout Runbook (Route 2)

Serve the PI0.5 `shaft_insert` SFT checkpoint over openpi's websocket policy
server (GPU host) and drive the CR5AF arm + DH PGE gripper from a thin deploy
client (robot host). The server reconstructs absolute cartesian actions from
delta-xyz internally (`Policy.infer` threads the observation state), so the
client only forwards observations, executes the returned chunk, and enforces a
workspace clamp + NaN guard.

**Topology**
- GPU / policy host `192.168.16.155` — holds the checkpoint and the unified `.venv`.
- Robot host `thor` (`ssh thor`) — reaches the arm `192.168.5.1` + RealSense; runs
  the `hil-serl` venv at `~/workspaces/hil-serl/.venv`.

Checkpoint: `/data/openpi_checkpoints/expo_pi05_droid_lora_finetune_sft_cartesian_state/pi05_shaft_insert_sft/19999`

---

## Step 0 — Prerequisites on the robot host (one-time)

Sync `thor:~/workspaces/expo-ft` to the PI0.5 branch so these files are present:
`client/deploy_pi05_cr5af.py`, `client/envs/cr5af_gripper_droid_env.py`,
`client/envs/cr5af_gripper_env.py`, `configs/task/cr5af_gripper.py`.

Ensure the client venv resolves all deps (`tyro`, `openpi_client`, `websockets`,
`msgpack_numpy`, `pyrealsense2`, `numpy`, `scipy`):

```bash
ssh thor 'cd ~/workspaces/expo-ft && ~/workspaces/hil-serl/.venv/bin/python -c \
  "import tyro, openpi_client, websockets, msgpack_numpy, pyrealsense2, numpy, scipy; print(\"deps OK\")"'
```

If `openpi_client` is missing and the venv has no `pip`, add it via a `.pth`
(editable-equivalent; all its deps are already present):

```bash
ssh thor 'cd ~/workspaces/expo-ft && \
  SP=$(~/workspaces/hil-serl/.venv/bin/python -c "import site;print(site.getsitepackages()[0])") && \
  SRC=$(cd expo_ft/agents/vla/openpi/packages/openpi-client/src && pwd) && \
  echo "$SRC" > "$SP/openpi_client.pth"'
```

---

## Step 1 — Start the policy server (GPU host `192.168.16.155`)

```bash
cd <expo-ft-repo>
XLA_PYTHON_CLIENT_PREALLOCATE=false JAXTYPING_DISABLE=1 CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python expo_ft/agents/vla/openpi/scripts/serve_policy.py \
  --port 8000 --default-prompt "grasp motor shaft and insert into bushing" \
  policy:checkpoint \
    --policy.config expo_pi05_droid_lora_finetune_sft_cartesian_state \
    --policy.dir /data/openpi_checkpoints/expo_pi05_droid_lora_finetune_sft_cartesian_state/pi05_shaft_insert_sft/19999
```

Wait for the server to log metadata and `listening on 0.0.0.0:8000`.
From the robot host confirm reachability: `nc -zv 192.168.16.155 8000`.

---

## Step 2 — Dry run (NO robot motion)

```bash
ssh thor 'cd ~/workspaces/expo-ft && ~/workspaces/hil-serl/.venv/bin/python \
  client/deploy_pi05_cr5af.py \
  --server-ip 192.168.16.155 --port 8000 --robot-ip 192.168.5.1 \
  --task "grasp motor shaft and insert into bushing" --dry-run'
```

Expect: `chunk (16, 10), finite=True, first-step |Δxyz| = <cm-scale> mm`.
Decisive "policy alive on real images" check:
- first-step delta ≈ 0 mm → vision collapse (policy ignoring camera);
- huge delta → normalization/state problem.
Resolve before moving hardware.

---

## Step 3 — First episode: translation-only, low speed, e-stop in hand

```bash
ssh thor 'cd ~/workspaces/expo-ft && ~/workspaces/hil-serl/.venv/bin/python \
  client/deploy_pi05_cr5af.py \
  --server-ip 192.168.16.155 --port 8000 --robot-ip 192.168.5.1 \
  --task "grasp motor shaft and insert into bushing" \
  --workspace-min 369 -245 110 --workspace-max 820 299 442 \
  --speed 20 --translation-only --max-steps 200'
```

Arm should move smoothly toward the shaft and actuate the gripper. `Ctrl-C`
triggers a zero-velocity stop + gripper open.

---

## Step 4 — Full 6-DOF

Drop `--translation-only` (optionally raise `--speed`). Run a few episodes.
Watch for: origin drift (bad absolute reconstruction), freeze/NaN (check server
log), or constant pose (vision regression).

---

## Safety layers

1. **Client workspace clamp** (`deploy_pi05_cr5af.py`): xyz target clipped into
   `--workspace-min/--workspace-max` (mm).
2. **Env velocity clip** (`client/envs/cr5af_gripper_env.py`): ServoP capped at
   ±50 mm/s and ±30°/s per tick; `--translation-only` zeroes rotation velocity.
3. **NaN/Inf guard**: non-finite actions hold the previous pose.

Always dry-run first; always start translation-only + low speed with the e-stop
in hand.

## Notes

- The client action chunk is **absolute** cartesian `[xyz_m(3), rot6d(6),
  grip(1)]` — delta→absolute is reconstructed server-side, no client math.
- `--replan-steps` (default 8) actions are executed per inference at
  `--control-hz` (default 30).
- Online EXPO-FT RL (native `run_client.py` + `train_pi_robo.py`) is the
  follow-on after this SFT hardware validation passes.
