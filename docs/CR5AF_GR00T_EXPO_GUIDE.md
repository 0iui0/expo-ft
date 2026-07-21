# CR5AF → GR00T N1.7 → EXPO-FT 端到端指南

本指南一步一步带你：**采集 CR5AF 数据 → BC 微调 GR00T N1.7 → rollout 测试 → 接入 EXPO-FT 在线 RL**。
所有命令均来自当前仓库的真实脚本，已在 RTX 5090 (GPU1) + 统一 venv 上验证。

> 背景与架构详见 [`GR00T_EXPO_RL_PIPELINE.md`](./GR00T_EXPO_RL_PIPELINE.md)。本文件是"按顺序执行"的操作手册。

> ⚠️ **2026-07 末端执行器变更**：TopHand 灵巧手已无库存，CR5AF 现改用 **DH PGE 1-DOF 夹爪**（wrist-aviation 485 总线，DobotStudio DHGrip 插件模式控制开合）。当前任务为 **"grasp motor shaft and insert into bushing"**（轴套插装），数据集 `/datasets/shaft_insert`（LeRobot v2，5 episodes）。采集代码在 `thor:~/workspaces/cr5af_gripper/`（`record_demo_gripper.py`、`dh_gripper.py`、`convert_to_lerobot.py`）。
>
> **embodiment 维度未变**——仍为 16-dim（`eef_9d(9)+joint_pos(6)+gripper_pos(1)`，2 相机 hand_view/table_view），故 `examples/CR5AF/cr5af_config.py` 的 `NEW_EMBODIMENT` 配置、expo-ft 的 `configs/task/cr5af.py`、以及全部 b8 修复（RELATIVE decode 等）**原样沿用，无需改任何代码**。已验证：SFT 产出 checkpoint → 统一 venv 加载 → `tests/test_gr00t_learner_smoke.py` 通过（jax critic + torch 3B actor 共存于 GPU1）。下文 Step 1–2 为夹爪 + shaft_insert 路径；只有采集脚本 / 任务 / 数据集 / reward 判据变了。

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

## Step 1：采集 CR5AF 数据（DH PGE 夹爪）

采集在 **thor（机器人控制机）** 上进行，代码在 `thor:~/workspaces/cr5af_gripper/`（需 RealSense D405/D455 + SpaceMouse + ServoP + DobotStudio DHGrip 插件已启用、`grip_open`/`grip_close` 项目已建）。**末端执行器现为 DH PGE 1-DOF 夹爪**（非 TopHand）。

```bash
# 在 thor 上：
cd ~/workspaces/cr5af_gripper
~/workspaces/hil-serl/.venv/bin/python record_demo_gripper.py \
  --robot-ip 192.168.5.1 \
  --task "grasp motor shaft and insert into bushing" \
  --grasp-pose grasp_shaft \
  --output-dir recordings/shaft_insert \
  --translation-only --preview
```
产出 `recordings/shaft_insert/episode_*/data.npz`（每 episode 含 CR5AF RT state、D405/D455 图像、DH PGE 夹爪状态）。

**关键经验（已踩坑，见 Isaac-GR00T `examples/CR5AF/README.md`）：**
- **rot6d 必须用 `tool_vector[3:6]`（TCP axis-angle）算**，绝不能用 RT 自带 quaternion（offset 1384 的 quat 不代表 TCP 姿态，误差 120°）。
- **手动曝光**：录制和推理用相同曝光值，否则 D405 图像分布漂移。

### 1.1 转成 LeRobot v2

```bash
# 在 thor 上（原始录制所在机）：
cd ~/workspaces/cr5af_gripper
~/workspaces/hil-serl/.venv/bin/python convert_to_lerobot.py \
  --input-dir recordings/shaft_insert \
  --output-dir /datasets/shaft_insert \
  --fps 30
  # --lookahead N   # action[t]=state[t+N]（绝对目标）；N 越大单步 delta 越大。grasp_housing 曾用 50；
                    # shaft_insert 用本脚本默认。转换后把 /datasets/shaft_insert 同步到训练机。
```
产出 LeRobot v2（parquet + mp4，state/action=16）。`/datasets/shaft_insert`（5 episodes）已同步到训练机。

> ⚠️ **rot6d / SVD 踩坑**：若 SFT 报 `Rotation.from_matrix` → `SVD did not converge`，是转换出的 rot6d 有坏帧（`eef_9d[3:9]` 必须是有效旋转）。**重新用 `convert_to_lerobot.py` 转换即可**（2026-07 在 shaft_insert 上遇到过，重转后 SFT 正常）。rot6d 必须由 `tool_vector[3:6]`（TCP axis-angle）算，不能用 RT 自带 quaternion。

---

## Step 2：BC 微调 GR00T N1.7（SFT）

在训练机、**Isaac venv**、GPU1 上跑。**当前 shaft_insert 流水线测试脚本**：`examples/CR5AF/finetune_shaft_insert_test.sh`（`--tune-visual --no-tune-llm --tune-top-llm-layers 2`，500 步 → 产出可加载 `checkpoint-500`，已验证可进入 RL）。生产级长 run 把 `--max-steps/--save-steps` 调大、episode 补到 500+。下面是最小可复现版：

```bash
cd ~/workspace/3rd/Isaac-GR00T
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="$HOME/.cache/modelscope/nv-community/GR00T-N1.7-3B"

.venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE" \
  --dataset-path /datasets/shaft_insert \
  --modality-config-path examples/CR5AF/cr5af_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /tmp/cr5af_shaft_insert_test \
  --experiment-name shaft-insert-pipeline-test \
  --max-steps 20000 \
  --save-steps 2000 \
  --save-only-model \
  --global-batch-size 4 \
  --learning-rate 1e-5 \
  --tune-visual --no-tune-llm --tune-top-llm-layers 2
```

**checkpoint 产出位置**：`/tmp/cr5af_shaft_insert_test/shaft-insert-pipeline-test/checkpoint-<step>/`（测试脚本产 `checkpoint-500`）
内含 `config.json` + `model-*.safetensors` + `model.safetensors.index.json` + `processor_config.json` + `statistics.json` + `embodiment_id.json`（根目录，**standalone 可加载**）；`experiment_cfg/`、`processor/` 在实验根目录。

> **调参红线（来自团队训练史 v1–v6）**：frozen VLM 不够（v1/v2 固定输出）；必须 `--tune-visual`（v3 起有轨迹方向）；RTX 5090 32GB 全调 LLM 装不下（需 80GB+），所以只调 top-2 LLM 层；想要更高精度需 500+ episodes 或 A100/H100。lookahead 50 是精度拐点（v5 能跟踪目标，~5mm）。

### 2.1 验证 checkpoint 可加载（统一 venv）

```bash
cd ~/workspace/3rd/expo-ft
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -c "
from pathlib import Path; import torch
from expo_ft.agents.vla.gr00t_agent import Gr00tAgent
actor, ts, _ = Gr00tAgent.initialize(
    model_path=Path('/tmp/cr5af_shaft_insert_test/shaft-insert-pipeline-test/checkpoint-500'),
    embodiment_tag='NEW_EMBODIMENT', device='cuda:0', dtype=torch.bfloat16)
print('OK', sum(p.numel() for p in actor.model.parameters())/1e9, 'B params')
"
```
打印 `OK 3.144 B params` 即 SFT 成功、可进入 RL。若报 `Qwen3VL... ImportError` → venv 的 transformers 不是 4.57.3；若报 SVD 不收敛 → checkpoint 损坏或 embodiment 配置错。

---

## Step 3：Rollout 测试（部署验证）

> ⚠️ **夹爪 rollout 暂未就绪**：`deploy_cr5af.py` 是 TopHand 时代的部署脚本（含 `--tophand-hand`）。DH PGE 夹爪的在线 rollout / env server（C1）尚未实现——这是接 EXPO-FT 在线 RL 的硬阻塞，需要机器人侧开发（见 §4.4）。下面的命令保留作历史参考；夹爪版需新建 env server，协议见 `expo_ft/env/env_client.py` 的 `EnvClientWrapper`（step/reset/get_observation/get_info_for_step，端口 8102，observation 须为 GR00T flat-key 格式）。

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

**离线 smoke**（不连机械臂，验证 checkpoint 推理 + RL 更新步）：用 `examples/CR5AF/preview_episode.py`（纯推理）或本仓库 `tests/test_gr00t_learner_smoke.py`（见 §4.2，jax+torch+gr00t 共存验证，需 `GR00T_CKPT`）。

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

已沉淀为正式 GPU 集成测试 `tests/test_gr00t_learner_smoke.py`（jax critic + torch 3B actor 共存于一次 `agent.update()`）。需 `GR00T_CKPT` 指向一个可加载 checkpoint：

```bash
cd ~/workspace/3rd/expo-ft
GR00T_CKPT=/tmp/cr5af_shaft_insert_test/shaft-insert-pipeline-test/checkpoint-500 \
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -m pytest tests/test_gr00t_learner_smoke.py -s -v
```
应 `1 passed`（约 75s），内部断言 finite losses + dead-config 超参到位。已在 shaft_insert checkpoint-500 上验证通过。无 `GR00T_CKPT` 或无 CUDA 时自动 skip。

### 4.3 在线 RL 训练（需要 env server）

`train_gr00t_robo.py` 通过 `EnvClientWrapper(host=localhost, port=8102)` 连 **env server**（在线 rollout 用）。env server 是机器人侧的 RL 环境服务（step/reset/get_observation），与 `deploy_cr5af.py` 的 policy server 不同——它实现 EXPO-FT 的 RL step 协议。先确认 env server 已起：

```bash
# 参见 scripts/pick/run_server.sh（thor 侧）/ run_policy.sh（训练机侧），端口 8102
# SSH 反向隧道把 thor:8102 转发到训练机 localhost:8102
```

确认 `localhost:8102` 可达后，启动 RL：

```bash
cd ~/workspace/3rd/expo-ft
GR00T_CKPT=/tmp/cr5af_shaft_insert_test/shaft-insert-pipeline-test/checkpoint-500 \
DATASET_PATH=/datasets/shaft_insert \
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
| 采集 | cr5af_gripper | thor: hil-serl `.venv` | — | `record_demo_gripper.py`（DH PGE 夹爪） |
| 转 LeRobot | cr5af_gripper | thor: hil-serl `.venv` | CPU | `convert_to_lerobot.py`（thor 上跑）→ `/datasets/shaft_insert` |
| SFT (BC) | Isaac-GR00T | Isaac `.venv` | GPU1 | `examples/CR5AF/finetune_shaft_insert_test.sh` |
| checkpoint 验证 / learner smoke | expo-ft | expo-ft `.venv` | GPU1 | `Gr00tAgent.initialize(...)` / `tests/test_gr00t_learner_smoke.py` |
| rollout | Isaac-GR00T | Isaac `.venv` (server) + thor (client) | GPU1 | `deploy_cr5af.py --server/--client` |
| 在线 RL | expo-ft | expo-ft `.venv` | GPU1 | `train_gr00t_cr5af.sh` |

**两个 venv 都跑在 GPU1，都要 `CUDA_VISIBLE_DEVICES=1`；expo-ft venv 还要 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。**
