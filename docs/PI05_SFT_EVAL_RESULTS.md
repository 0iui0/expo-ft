# PI0.5 shaft_insert SFT — Offline Evaluation Results

Checkpoint: `pi05_shaft_insert_sft` step 19999 (20k-step run, delta-xyz + native
rot6d action rep). Config `expo_pi05_droid_lora_finetune_sft_cartesian_state`.
Three complementary offline evals (no robot / no simulator). Scripts under
`scripts/eval_pi05_*.py`.

## Background

The previous SFT (absolute next-frame pose) collapsed into a copy task: pos MAE
~83 mm open-loop, rotation garbage, gripper constant — worse than a do-nothing
identity baseline (~1 mm). The rework switched translation to **delta-on-xyz**
(state subtracted at load) and rotation to **native rot6d** (no euler
round-trip), keeping gripper absolute. These evals verify the fix.

## 1. Open-loop (single-step prediction, 100 frames)

Feeds ground-truth state each step; measures 1-step action error vs the dataset.

| metric | model | identity baseline |
|---|---|---|
| position MAE | **0.348 mm** | 0.496 mm |
| rotation (rot6d geodesic) | **2.161°** | 3.048° |
| gripper (mean) | 0.431 | gt 0.420 |

Position MAE dropped from 83 mm → 0.35 mm and **beats identity** on all three
axes; rotation is no longer garbage; gripper tracks (non-constant). All three
collapse symptoms resolved. Caveat: at ~26 Hz single-step motion is tiny, so
absolute numbers are near-identity for any reasonable policy — a weak
discriminator on its own, hence the two tests below.

## 2. Closed-loop rollout (proprioceptive autoregression, image replay)

No sim, so we roll out on proprioception: `pred_state[t+1] = policy(image[t],
pred_state[t])` (the policy output already reconstructs the absolute next pose).
The recorded demo image at frame t is replayed. Baseline "frozen" = state pinned
at t=0, i.e. how far the demo travels from its start. 3 episodes × ≤400 steps.

| horizon (steps) | model drift (mm) | demo travel (mm) | model rot (°) |
|---|---|---|---|
| 25 | 14 | 1 | 6 |
| 50 | 34 | 52 | 13 |
| 100 | 86 | 168 | 19 |
| 200 | 154 | 198 | 20 |
| 399 | 399 | 176 | 46 |

- **Mid-horizon (50–200 steps): model drift < demo travel** — the policy is
  predicting real motion, not echoing state (a copy policy would drift like the
  frozen baseline).
- **Long-horizon (>200): diverges** — classic behavior-cloning compounding error.

**Important caveat.** The image is replayed on the demo trajectory while the
state diverges, so the policy receives `(drifted state, demo image)` pairs that
never occurred in training (state and image were always consistent). This OOD
mismatch inflates the long-horizon divergence — it is partly a surrogate
artifact, not purely model error. A real closed loop (camera tracks the actual
robot, providing corrective visual feedback) is expected to do better. This
offline surrogate cannot settle robustness; the real-robot rollout is the gate.

## 3. Vision-spread test (is vision alive?)

Isolates the policy's dependence on pixels without the OOD confound. The output
is `state + delta`; we measure the spread of the predicted **delta** (motion)
when varying only the image (fixed state) vs varying only the state (fixed
image). 16 anchor frames.

| vary | pos-delta spread (mm) | rot6d-delta spread |
|---|---|---|
| **image** (fixed state) | 1.111 | 0.10372 |
| **state** (fixed image) | 0.720 | 0.00328 |
| **image / state ratio** | 1.54× | **31.6×** |

- **Rotation is strongly vision-driven (31×):** swapping the image swings the
  rot6d prediction; swapping the state barely moves it.
- **Position is vision-led but moderate (1.54×):** the per-step translation is
  ~0.47 mm, so magnitudes are small, but image still drives it more than state.
- Under collapse the image spread would be ≈0; it is clearly non-zero →
  **vision is alive.**

## Conclusion

The delta-xyz + native-rot6d rework **fixed the near-identity collapse**. Vision
demonstrably drives the prediction (rotation strongly, position moderately), and
single-step accuracy beats the identity baseline. Long-horizon open-loop drift
remains — expected for behavior cloning and amplified by the image-replay
surrogate — so the decisive robustness check is a **real-robot closed-loop
rollout** with live visual feedback (next step).
