# CR5AF → GR00T N1.7 → EXPO-FT 端到端指南

本指南一步一步带你：**采集 CR5AF 数据 → BC 微调 GR00T N1.7 → rollout 测试 → 接入 EXPO-FT 在线 RL**。
所有命令均来自当前仓库的真实脚本，已在 RTX 5090 (GPU1) + 统一 venv 上验证。

> 背景与架构详见 [`GR00T_EXPO_RL_PIPELINE.md`](./GR00T_EXPO_RL_PIPELINE.md)。本文件是"按顺序执行"的操作手册。

---

## 0. 前置条件

| 项 | 要求 |
|---|---|
| 两个仓库 | `~/workspace/3rd/Isaac-GR00T`（SFT + 数据工具）、`~/workspace/3rd/expo-ft`（RL） |
| 统一 venv | `expo-ft/.venv`（py3.12, jax cuda13 + torch cu130 + gr00t + transformers 4.57.3）。**所有 RL 步骤用它** |
| SFT venv | `Isaac-GR00T/.venv`（py3.10, torch cu130 + gr00t 全量 deps：deepspeed/flash-attn/torchcodec）。**SFT 用它** |
| 基座模型 | `~/.cache/modelscope/nv-community/GR00T-N1.7-3B`（3.14B，bf16） |
| GPU | RTX 5090，**用 GPU1**：`CUDA_VISIBLE_DEVICES=1` |
| Embodiment | `NEW_EMBODIMENT`：2 相机（hand_view/table_view），state/action = `eef_9d(9)+joint_pos(6)+gripper_pos(1)=16`，action horizon 10，video 2 帧 `[-20,0]`，**RELATIVE 动作**（eef_9d/joint_pos 相对、gripper 绝对） |

**RL 运行必带两个环境变量**（jax+torch 共存必需，否则 jax 预占 75% 显存饿死 torch）：
```bash
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

---

## Step 1：采集 CR5AF 数据

采集在 **thor（机器人控制机）** 上进行（需要 RealSense D405/D455 + SpaceMouse + ServoP），不是训练机。

```bash
# 在 thor 上：
cd ~/workspaces/hil-serl
.venv/bin/python record_demo.py \
  --robot-ip 192.168.5.1 \
  --task "pick motor housing and place on fixture" \
  --grasp-pose grasp_housing \
  --output-dir recordings/cr5af_demos \
  --translation-only --preview
```
产出 `.npz` 原始录制（每 episode 含 CR5AF RT state、D405/D455 图像、TopHand 夹爪状态）。

**关键经验（已踩坑，见 Isaac-GR00T `examples/CR5AF/README.md`）：**
- **rot6d 必须用 `tool_vector[3:6]`（TCP axis-angle）算**，绝不能用 RT 自带 quaternion（offset 1384 的 quat 不代表 TCP 姿态，误差 120°）。
- **手动曝光**：录制和推理用相同曝光值，否则 D405 图像分布漂移。

### 1.1 转成 LeRobot v2

```bash
# 训练机上（数据量大）：
cd ~/workspace/3rd/Isaac-GR00T
.venv/bin/python examples/CR5AF/convert_to_lerobot.py \
  --input-dir recordings/cr5af_demos \
  --output-dir /datasets/cr5af_grasp_housing \
  --fps 30 \
  --lookahead 50        # ⚠️ 必须 50：action[t]=state[t+50]，把 0.05mm/步的 delta 放大到 4.4mm/步，模型才能跟踪
```
产出 LeRobot v2（parquet + mp4）。现有数据集见 `/datasets/cr5af_grasp_housing_l50`（成功 episodes，用于 BC）、`/datasets/cr5af_grasp_housing_l50_rl`（含失败，用于 critic）。

---

## Step 2：BC 微调 GR00T N1.7（SFT）

在训练机、**Isaac venv**、GPU1 上跑。生产脚本 `examples/CR5AF/finetune_l50.sh`（`--tune-visual --no-tune-llm --tune-top-llm-layers 2`，lookahead 50）。下面是最小可复现版（已在 RTX 5090 验证可产出可加载 checkpoint）：

```bash
cd ~/workspace/3rd/Isaac-GR00T
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="$HOME/.cache/modelscope/nv-community/GR00T-N1.7-3B"

.venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE" \
  --dataset-path /datasets/cr5af_grasp_housing_l50 \
  --modality-config-path examples/CR5AF/cr5af_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /tmp/cr5af_finetune \
  --experiment-name cr5af-grasp \
  --max-steps 20000 \
  --save-steps 2000 \
  --save-only-model \
  --global-batch-size 4 \
  --learning-rate 1e-5 \
  --tune-visual --no-tune-llm --tune-top-llm-layers 2
```

**checkpoint 产出位置**：`/tmp/cr5af_finetune/cr5af-grasp/checkpoint-<step>/`
内含 `config.json` + `processor_config.json`（根目录，**standalone 可加载**）+ `experiment_cfg/` + safetensors 分片。

> **调参红线（来自团队训练史 v1–v6）**：frozen VLM 不够（v1/v2 固定输出）；必须 `--tune-visual`（v3 起有轨迹方向）；RTX 5090 32GB 全调 LLM 装不下（需 80GB+），所以只调 top-2 LLM 层；想要更高精度需 500+ episodes 或 A100/H100。lookahead 50 是精度拐点（v5 能跟踪目标，~5mm）。

### 2.1 验证 checkpoint 可加载（统一 venv）

```bash
cd ~/workspace/3rd/expo-ft
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -c "
from pathlib import Path; import torch
from expo_ft.agents.vla.gr00t_agent import Gr00tAgent
actor, ts, _ = Gr00tAgent.initialize(
    model_path=Path('/tmp/cr5af_finetune/cr5af-grasp/checkpoint-20000'),
    embodiment_tag='NEW_EMBODIMENT', device='cuda:0', dtype=torch.bfloat16)
print('OK', sum(p.numel() for p in actor.model.parameters())/1e9, 'B params')
"
```
打印 `OK 3.144 B params` 即 SFT 成功、可进入 RL。若报 `Qwen3VL... ImportError` → venv 的 transformers 不是 4.57.3；若报 SVD 不收敛 → checkpoint 损坏或 embodiment 配置错。

---

## Step 3：Rollout 测试（部署验证）

GR00T 原生部署用 `deploy_cr5af.py`（ZMQ server/client 架构）：

```bash
# 训练机：起 policy server（GPU1）
cd ~/workspace/3rd/Isaac-GR00T
CUDA_VISIBLE_DEVICES=1 .venv/bin/python examples/CR5AF/deploy_cr5af.py \
  --server --model-path /tmp/cr5af_finetune/cr5af-grasp/checkpoint-20000 --port 5555

# thor：起 client，连机械臂
cd ~/workspaces/hil-serl
.venv/bin/python deploy_cr5af.py \
  --client --server-ip <训练机IP> --port 5555 \
  --robot-ip 192.168.5.1 --task "pick motor housing and place on fixture" \
  --grasp-pose grasp_housing --tophand-hand left \
  --translation-only --speed 100 \
  --hand-exposure <值> --hand-gain <值> --table-exposure <值> --table-gain <值>   # 与录制一致
```

**部署侧必检（踩坑清单）：**
- **相机 fps 必须 30**（与转换一致）。`delta_indices=[-20,0]` 在 15fps 下时间窗口翻倍 → 时序混乱。
- **手动曝光**与录制完全一致。
- ServoP 阻抗：用 `FCSetStiffness/FCSetDamping` + `ServoP(..., gain=300)`；`set_impedance()` 是无效 API。
- 延迟：TRT full pipeline (2 samples) ≈ 350ms 是当前最优；纯 PyTorch ≈ 600ms 太慢。

**离线 smoke**（不连机械臂，验证 checkpoint 推理）：用 `examples/CR5AF/preview_episode.py` 或本仓库的 `/tmp/smoke_learner.py`（已验证 forward 产 finite actions）。

---

## Step 4：接入 EXPO-FT 在线 RL

### 4.1 当前代码状态（重要）

本次会话已修复阻塞性 bug，**统一 venv 现在可以端到端跑一个 `agent.update()`**（jax critic + torch 3B actor 共存于 GPU1，已验证 finite losses）。修复清单：

| # | 文件 | 修复 |
|---|---|---|
| 1 | venv | transformers 固定 4.57.3（4.53.2 无 Qwen3VL；5.x 破坏 gr00t） |
| 2 | `train_gr00t_robo.py` | 单卡用 `SingleDeviceSharding`（原 `mesh=None` 崩 `fsdp_sharding`） |
| 3 | `gr00t_agent.py` + `expo_ft_gr00t.py` | `process_transformed_outputs` 传 state 给 `decode_action`（CR5AF RELATIVE 动作需要；原 `state=None` 崩） |
| 4 | `openpi/.../sharding.py` | `fsdp_sharding(mesh=None)` → `SingleDeviceSharding` fast path |
| 5 | `gr00t_agent.py` | `prepare_batch_for_actor` 给 1-D state 补时间维（原 IndexError） |
| + | `gr00t_agent.py build_gr00t` | 把 config 的 EXPO 超参（N/num_qs/n_edit/latent_dim_*/actor_success_only…）真正传给 learner（原返回 `{}`，全用默认值） |

> ⚠️ 修复 #4 在 **vendored openpi 克隆**（`expo_ft/agents/vla/openpi/`，git 不跟踪）里。换机器/重装 openpi 需重打该 patch（在 `fsdp_sharding` 开头加 `if mesh is None: return SingleDeviceSharding`）。

### 4.2 learner smoke（RL 更新步验证，不连机械臂）

```bash
cd ~/workspace/3rd/expo-ft
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python /tmp/smoke_learner.py
```
应打印 `✓ SMOKE TEST PASSED` + `actor_loss/critic_loss/q` 等有限值。这验证 jax+torch+gr00t 在一次 `update()` 里共存。

### 4.3 在线 RL 训练（需要 env server）

`train_gr00t_robo.py` 通过 `EnvClientWrapper(host=localhost, port=8102)` 连 **env server**（在线 rollout 用）。env server 是机器人侧的 RL 环境服务（step/reset/get_observation），与 `deploy_cr5af.py` 的 policy server 不同——它实现 EXPO-FT 的 RL step 协议。先确认 env server 已起：

```bash
# 参见 scripts/pick/run_server.sh（thor 侧）/ run_policy.sh（训练机侧），端口 8102
# SSH 反向隧道把 thor:8102 转发到训练机 localhost:8102
```

确认 `localhost:8102` 可达后，启动 RL：

```bash
cd ~/workspace/3rd/expo-ft
GR00T_CKPT=/tmp/cr5af_finetune/cr5af-grasp/checkpoint-20000 \
DATASET_PATH=/datasets/cr5af_grasp_housing_l50 \
bash scripts/train_gr00t_cr5af.sh
```
（脚本默认：batch 4、UTD 20、offline_ratio 0.5、replan 8、actor_lr 3e-5、critic_lr 3e-4、20000 步、wandb `expo-ft-gr00t`。改环境变量调参。）

### 4.4 还差的集成项（非本次范围）

- **CR5AF reward/success classifier**（论文 §4.3 rule-based 二分类）：env server 侧。已有 `Isaac-GR00T/examples/CR5AF/train_success_classifier.py` 可训。
- **HIL per-step-in-chunk 纠正**（论文 §4.2）：当前 `train_gr00t_robo.py` 只支持整段 override。
- **CR5AF env server**：需与 `EnvClientWrapper` 协议对齐（step/reset/get_observation/get_info_for_step）。这是 rollout 能跑的前提。

---

## 速查：各阶段用哪个 venv / GPU

| 阶段 | 仓库 | venv | GPU | 关键命令 |
|---|---|---|---|---|
| 采集 | Isaac-GR00T | thor: hil-serl | — | `record_demo.py` |
| 转 LeRobot | Isaac-GR00T | Isaac `.venv` | CPU | `convert_to_lerobot.py --lookahead 50` |
| SFT (BC) | Isaac-GR00T | Isaac `.venv` | GPU1 | `launch_finetune.py --tune-visual …` |
| checkpoint 验证 / learner smoke | expo-ft | expo-ft `.venv` | GPU1 | `Gr00tAgent.initialize(...)` / `smoke_learner.py` |
| rollout | Isaac-GR00T | Isaac `.venv` (server) + thor (client) | GPU1 | `deploy_cr5af.py --server/--client` |
| 在线 RL | expo-ft | expo-ft `.venv` | GPU1 | `train_gr00t_cr5af.sh` |

**两个 venv 都跑在 GPU1，都要 `CUDA_VISIBLE_DEVICES=1`；expo-ft venv 还要 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。**
