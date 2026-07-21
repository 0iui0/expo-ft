# GR00T N1.7 + EXPO-FT Online RL Training Pipeline

> 本文档描述将 GR00T N1.7 VLA 模型接入 EXPO-FT 框架进行在线强化学习真机训练的完整流程。

---

## 目录

1. [架构概览](#1-架构概览)
2. [环境与依赖](#2-环境与依赖)
3. [Phase 1: GR00T SFT / BC 预训练](#3-phase-1-gr00t-sft--bc-预训练)
4. [Phase 2: 环境服务器搭建](#4-phase-2-环境服务器搭建)
5. [Phase 3: 离线数据集准备](#5-phase-3-离线数据集准备)
6. [Phase 4: EXPO-FT RL 在线训练](#6-phase-4-expo-ft-rl-在线训练)
7. [配置参考](#7-配置参考)
8. [监控与调试](#8-监控与调试)
9. [常见问题](#9-常见问题)

---

## 1. 架构概览

### 1.1 算法原理

EXPO-FT 是一个 **Sample-Efficient RL Finetuning** 方法，专为 VLA (Vision-Language-Action) 模型设计：

```
EXPO = Base VLA (π_θ) + Edit Policy (residual Δ) + OTF argmax-Q selection + RedQ critic ensemble
```

核心思想：
- **Base VLA** (GR00T N1.7): 已通过 BC/SFT 学到合理的初始策略，在 RL 阶段继续用 success-only BC 更新以维持 output distribution 稳定性。
- **Edit Policy**: 一个轻量级残差网络，输出对 base action 的修正量 Δ。
- **OTF (Online Test-time Filtering)**: 推理时生成 N 个 base action chunks（通过 flow-matching 的 per-sample noise），加上 M 个 edited variants，用 RedQ critic ensemble 做 argmax-Q 选择最优 action。
- **RedQ Critic**: 10 个 Q 网络（ensemble），min-2 做保守估计，含 image encoder + state encoder。

### 1.2 设备布局 (Dual-GPU)

```
┌─────────────────────────────────────────────────┐
│ GPU0 (RTX 5090)                                 │
│ ┌───────────────────────────────────────────┐   │
│ │  Actor (PyTorch)                          │   │
│ │  - GR00T N1.7-3B model (~3.1B params, bf16)           │   │
│ │  - Flow-matching inference (action gen)   │   │
│ │  - Actor BC training (success-only)       │   │
│ │  - Isaac-GR00T venv (torch 2.11+cu130)    │   │
│ └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ GPU1 (RTX 5090)                                 │
│ ┌───────────────────────────────────────────┐   │
│ │  Learner (JAX)                            │   │
│ │  - RedQ Critic ensemble (10 Q-nets)       │   │
│ │  - Image Encoder (ResNet, shared)          │   │
│ │  - Residual Actor (lightweight MLP)       │   │
│ │  - Temperature parameter                  │   │
│ │  - expo-ft venv (JAX 0.10 selfbuilt)      │   │
│ └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

- **Async 模式** (`train_gr00t_robo_async.py`): Actor 主线程在 GPU0 采样，Learner 后台线程在 GPU1 更新。适合在线真机。
- **Sync 模式** (`train_gr00t_robo.py`): 单线程交替采样和更新。适合离线数据调试。

### 1.3 数据流

```
Environment Server (WebSocket :8102)
    │
    │  obs: {video.hand_view, video.table_view, state.eef_9d, state.joint_pos, ...}
    ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Actor Thread │───▶│ Gr00tReplayBuffer│───▶│ Gr00tBatchProc   │
│ (GPU0)       │    │ (raw uint8/float) │    │ (UTD=20x mixer)  │
└──────────────┘    └──────────────────┘    └────────┬─────────┘
                                                     │
                                              ┌──────▼──────────┐
                                              │ Learner Thread   │
                                              │ (GPU1)           │
                                              │ 1. Critic update │
                                              │ 2. Actor BC      │
                                              │ 3. Residual act  │
                                              └──────────────────┘
```

---

## 2. 环境与依赖

### 2.1 硬件要求

| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | 2× RTX 4090 (24GB) | 2× RTX 5090 (32GB) |
| RAM | 64 GB | 128 GB |
| Disk | 200 GB | 500 GB (SSD) |

### 2.2 软件环境

本项目使用**双 venv 架构** — Actor 和 Learner 各用自己的 Python 环境：

#### expo-ft venv (Learner: JAX, GPU1)

```bash
cd ~/workspace/3rd/expo-ft

# 创建 Python 3.12 venv
uv venv --python 3.12

# 安装项目依赖
uv sync

# 手动安装自编译 JAX (RTX 5090 sm_120 需要自编译)
# JAX 0.10.1 自编译路径
source .venv/bin/activate
pip install ~/.cache/bazel_old/jax/jaxlib/tools/dist/jaxlib-0.10.1.dev0+selfbuilt-cp312-cp312-linux_x86_64.whl
pip install ~/.cache/bazel_old/jax/jax_cuda13_plugin-0.10.1.dev0+selfbuilt-cp312-cp312-linux_x86_64.whl
pip install ~/.cache/bazel_old/jax/jax_cuda13_pjrt-0.10.1.dev0+selfbuilt-cp312-cp312-linux_x86_64.whl
```

验证:
```bash
.venv/bin/python -c "import jax; print('JAX', jax.__version__); print('Devices:', jax.devices())"
# Expected output: JAX 0.10.1.dev...  Devices: [CudaDevice(id=0), CudaDevice(id=1)]
```

#### Isaac-GR00T venv (Actor: PyTorch, GPU0)

Actor 推理/训练使用 Isaac-GR00T 的 Python 环境:

```bash
# Isaac-GR00T 已有 venv:
~/workspace/3rd/Isaac-GR00T/.venv/bin/python -c "import torch; print('PyTorch', torch.__version__)"
# Expected output: PyTorch 2.11.0+cu130
```

### 2.3 GR00T 模型下载

```bash
# ModelScope: GR00T N1.7-3B (版本 N1.7，3B 参数变体；非 7B)
# 模型路径: ~/.cache/modelscope/nv-community/GR00T-N1.7-3B/
#   (软链接 GR00T-N1.7-3B -> GR00T-N1___7-3B)

# 目录结构:
# GR00T-N1.7-3B/   (~3.14B params, SFT 时 47.67% trainable)
#   config.json
#   model-00001-of-00002.safetensors   (~3GB)
#   model-00002-of-00002.safetensors   (~3GB)
#   processor_config.json
#   statistics.json
#   embodiment_id.json
```

---

## 3. Phase 1: GR00T SFT / BC 预训练

> **目的**: 在目标 embodiment 上对 GR00T N1.7 base model 做 behavior cloning，获得初始策略。
> **前提**: 需要收集若干条人工遥控示教数据（推荐 50-100 条成功 episode）。

### 3.1 注册 Embodiment

CR5AF 的 modality 配置定义在 `examples/CR5AF/cr5af_config.py`，通过 `register_modality_config(..., embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)` 注册为 `NEW_EMBODIMENT`：

- 2 路相机: `hand_view` (D405 腕部) + `table_view` (D455 固定第三视角)
- state: `eef_9d`(9) + `joint_pos`(6) + `gripper_pos`(1) = 16 维
- action: 同 16 维，10 步预测 horizon，delta_indices `range(0,10)`
- video: 2 帧 history，delta_indices `[-20, 0]`

`launch_finetune.py` 会自动 `import` 该文件完成注册（见 `--modality-config-path`）。

### 3.2 准备示教数据

CR5AF 实际使用 **LeRobot v2 格式**（非 Droid HDF5）。用 `examples/CR5AF/convert_to_lerobot.py` 把遥控原始数据转换为：

```
/datasets/cr5af_grasp_housing_l50/
  meta/        # info.json, stats.json, modality.json, episodes.jsonl, tasks.jsonl
  data/chunk-000/episode_XXXXXX.parquet   # state(16) + action(16) per frame, fps=30
  videos/chunk-000/<view>/episode_XXXXXX.mp4
```

### 3.3 运行 SFT Finetuning

入口是 `gr00t/experiment/launch_finetune.py`（**不是** `scripts/finetune.py`，该文件不存在）。参考封装脚本 `examples/CR5AF/finetune_l50.sh`：

```bash
cd ~/workspace/3rd/Isaac-GR00T

.venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path "$HOME/.cache/modelscope/nv-community/GR00T-N1.7-3B" \
  --dataset-path /datasets/cr5af_grasp_housing_l50 \
  --modality-config-path examples/CR5AF/cr5af_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /tmp/cr5af_finetune_v5 \
  --experiment-name grasp-housing-v5-l50 \
  --max-steps 20000 --save-steps 5000 \
  --global-batch-size 4 --dataloader-num-workers 0 \
  --learning-rate 1e-5 --episode-sampling-rate 0.1 \
  --tune-visual --no-tune-llm --tune-top-llm-layers 2
```

> 仅用于跑通 RL 代码路径时，可用短 run（`--max-steps 600 --save-steps 300 --save-only-model`），见 `examples/CR5AF/finetune_test_ckpt.sh`。生产级策略照上面的长 run。
> ⚠️ 不要把 `WANDB_API_KEY` 写进脚本（`finetune_l50.sh` 历史版本含硬编码 key，勿复制传播）；默认 `use_wandb=False`，需要时用环境变量传入。

### 3.4 验证 SFT Checkpoint

可加载的 checkpoint 落在 **`<output_dir>/<experiment_name>/checkpoint-<step>/`**（每个 `save_steps` 保存一次；`CheckpointFormatCallback.on_save` 会把 `processor/` 复制进该目录，使其自包含）：

```
/tmp/cr5af_finetune_v5/grasp-housing-v5-l50/checkpoint-5000/
  config.json
  model-00001-of-00002.safetensors   # GR00T-N1.7-3B 权重 (~3GB)
  model-00002-of-00002.safetensors
  model.safetensors.index.json
  processor_config.json              # 复制自 processor/
  statistics.json
  embodiment_id.json
  experiment_cfg/                    # conf.yaml, config.yaml 等
```

`Gr00tAgent.initialize` 通过 `AutoModel.from_pretrained(checkpoint_dir)` + `AutoProcessor.from_pretrained(checkpoint_dir)` 加载，因此 RL 配置里的 `gr00t_model_path` 应指向这个 `checkpoint-<step>/` 目录。注意：`processor/` 与 `experiment_cfg/` 在实验根目录也存在，但**没有 `checkpoint-<step>/` 子目录就意味着没有保存权重**（v5/v6 历史run 被中断在此状态，不可直接用于 RL）。

---

## 4. Phase 2: 环境服务器搭建

### 4.1 环境服务器

EXPO-FT 通过 WebSocket (端口 8102) 连接环境服务器。环境服务器负责：
- 管理机器人硬件
- 接收 action 指令
- 返回 observation (GR00T 格式)
- 处理 episode 重置

### 4.2 Observation 格式

环境服务器必须返回 GR00T flat-key 格式的 observation:

```python
observation = {
    "video.hand_view":    np.ndarray (H, W, 3) uint8,     # 手部相机
    "video.table_view":   np.ndarray (H, W, 3) uint8,     # 桌面相机
    "state.eef_9d":       np.ndarray (9,) float32,         # 末端位姿
    "state.joint_pos":    np.ndarray (6,) float32,         # 关节位置
    "state.gripper_pos":  np.ndarray (1,) float32,         # 夹爪位置
    "prompt":             "pick up the housing",            # 任务指令
}
```

### 4.3 启动环境服务器

```bash
# 在机器人控制机上启动环境服务器
# (具体命令取决于你的环境服务器实现)
python env_server.py --port 8102
```

### 4.4 验证连接

```bash
# 在训练机上验证可以连接到环境服务器
cd ~/workspace/3rd/expo-ft
source .venv/bin/activate

python -c "
import time
from expo_ft.env.env_client import EnvClientWrapper

env = EnvClientWrapper(
    env_creation_request={'example_action': np.zeros((1,16)), 'env_usage': 'train'},
    host='localhost',  # 或机器人控制机 IP
    port=8102,
)
obs = env.get_observation()
print('Observation keys:', list(obs.keys()))
print('Done')
"
```

---

## 5. Phase 3: 离线数据集准备

### 5.1 数据集格式

CR5AF 使用 **LeRobot v2 格式**（parquet + mp4，**非 Droid HDF5**）。实际数据集：

```
/datasets/cr5af_grasp_housing_l50/     # 103 episodes, fps=30, state/action dim=16, ~509MB
  meta/{info.json,stats.json,modality.json,episodes.jsonl,tasks.jsonl}
  data/chunk-000/episode_XXXXXX.parquet
  videos/chunk-000/{hand_view,table_view}/episode_XXXXXX.mp4
```

> ⚠️ **数据接入缺口**：`Gr00tReplayBuffer.insert_dataset` → `_adapt_offline_transition` 只识别 GR00T 原生格式（`image`+`state`）或 OpenPI/Droid 格式（`observations`），**不能直接读 LeRobot parquet**。在线 RL 时 replay buffer 由 env 循环逐 transition 填充（见 `train_gr00t_robo.py`），离线示教灌入需要先转成上述 transition 格式（参考 `examples/CR5AF/convert_to_lerobot.py` 的逆方向）。`process_droid_dataset` 仅适用于 Droid HDF5，对 LeRobot 数据无效。

### 5.2 任务配置文件

参考 `configs/task/cr5af.py`，按需修改:

```python
# configs/task/my_task.py
import ml_collections
import numpy as np

def get_config():
    config = ml_collections.ConfigDict()
    config.env_type = "droid"
    config.action_space = "cartesian_velocity"
    config.gripper_action_space = "velocity"
    config.side_camera_id = "table_view"
    config.wrist_camera_id = "hand_view"
    config.image_size = (256, 256)
    config.control_hz = 8
    # Action dim = eef_9d(9) + joint_pos(6) + gripper_pos(1) = 16
    config.example_action = np.zeros((1, 16), dtype=np.float32)
    config.residual_action_xyzg = False
    return config
```

### 5.3 验证数据集

```bash
# LeRobot 数据集元信息与维度核对（用 Isaac-GR00T venv，可读 parquet）
cd ~/workspace/3rd/Isaac-GR00T
.venv/bin/python -c "
import json, pandas as pd
meta = json.load(open('/datasets/cr5af_grasp_housing_l50/meta/info.json'))
print('episodes:', meta['total_episodes'], 'fps:', meta['fps'])
print('state shape:', meta['features']['observation.state']['shape'])
print('action shape:', meta['features']['action']['shape'])
df = pd.read_parquet('/datasets/cr5af_grasp_housing_l50/data/chunk-000/episode_000000.parquet')
print('columns:', list(df.columns))
"
```

`state`/`action` shape 应为 `[16]`。这只是确认数据可读；要灌入 `Gr00tReplayBuffer` 仍需 transition 格式转换（见 5.1 缺口说明）。

---

## 6. Phase 4: EXPO-FT RL 在线训练

### 6.1 配置文件准备

编辑 `configs/model/expo_ft_gr00t_config.py`，确认关键字段:

```python
# --- 必须修改的字段 ---
config.gr00t_model_path = "/tmp/cr5af_finetune_v5/grasp-housing-v5-l50/checkpoint-5000"  # 指向含 safetensors 的 checkpoint-<step>/
config.gr00t_embodiment_tag = "NEW_EMBODIMENT"                 # 与 cr5af_config.py register 一致
config.gr00t_camera_keys = ["hand_view", "table_view"]         # 相机视角
config.gr00t_state_keys = ["eef_9d", "joint_pos", "gripper_pos"]  # 状态 key
config.gr00t_action_keys = ["eef_9d", "joint_pos", "gripper_pos"] # 动作 key

# --- 训练参数 ---
config.actor_lr = 3e-5              # Actor (GR00T) 学习率
config.critic_lr = 3e-4             # Critic 学习率
config.edit_scale = 0.2             # Edit policy 缩放
config.N = 8                        # OTF base samples
config.n_edit_samples = 8           # Edit policy samples
config.num_qs = 10                  # RedQ ensemble size
config.num_min_qs = 2               # RedQ min Q 数

# --- VRAM 优化 ---
config.use_gradient_checkpointing = False   # VRAM 不够时设为 True
config.freeze_gr00t_backbone = False       # 冻结 backbone 节省 VRAM

# --- Async 安全 ---
config.use_model_lock = False       # 遇到 NaN 时设为 True
```

### 6.2 启动 Sync 训练（调试用）

Single-process, single-GPU variant。适合离线调试和概念验证。

```bash
cd ~/workspace/3rd/expo-ft
source .venv/bin/activate

# 设置 GPU1 给 JAX learner
export CUDA_VISIBLE_DEVICES=0,1

python train_gr00t_robo.py \
    --config configs/model/expo_ft_gr00t_config.py \
    --config_task configs/task/cr5af.py \
    --dataset_path /datasets/cr5af_grasp_housing \
    --gr00t_model_path /path/to/your/sft/checkpoint \
    --max_steps 100000 \
    --batch_size 64 \
    --utd_ratio 20 \
    --update_type episode \
    --output_dir ./logs/gr00t_rl_debug \
    --tqdm
```

### 6.3 启动 Async 训练（正式用）

Dual-GPU async mode。Actor (主线程, GPU0) 采样，Learner (后台线程, GPU1) 更新。适用于在线真机。

> **重要**: 当前版本使用单进程多线程架构。Actor 推理在 GPU0 上使用 Isaac-GR00T 的 PyTorch 模型。如果需要真正的双进程分离（Actor 进程用 Isaac-GR00T venv，Learner 进程用 expo-ft venv），见 [6.4](#64-双进程分离架构实验性)。

```bash
cd ~/workspace/3rd/expo-ft
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

python train_gr00t_robo_async.py \
    --config configs/model/expo_ft_gr00t_config.py \
    --config_task configs/task/cr5af.py \
    --dataset_path /datasets/cr5af_grasp_housing \
    --gr00t_model_path /path/to/your/sft/checkpoint \
    --client_host localhost \
    --client_port 8102 \
    --max_steps 100000 \
    --batch_size 64 \
    --utd_ratio 20 \
    --offline_ratio 0.0 \
    --ep_timeout_secs 120 \
    --checkpoint_model \
    --checkpoint_interval 5000 \
    --output_dir ./logs/gr00t_rl_async \
    --run_name gr00t_cr5af_run1 \
    --tqdm
```

关键参数说明:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_steps` | 100000 | 总训练步数 |
| `--batch_size` | 64 | Critic 批大小 |
| `--utd_ratio` | 20 | Update-To-Data ratio (每次 update 做 20 轮 critic 更新) |
| `--offline_ratio` | 0.0 | 0 = 纯在线; 0.5 = 一半在线一半离线 |
| `--ep_timeout_secs` | 120 | Episode 超时 (超过此时间无新 episode 则暂停 update) |
| `--checkpoint_interval` | 0 | 每隔 N 步保存 checkpoint (0 = 只在最后保存) |
| `--replan_steps` | 8 | 每次推理执行的动作步数 (8 步 @ 8Hz = 1 秒) |

### 6.4 双进程分离架构 (实验性)

如果需要 Actor 进程完全在 Isaac-GR00T venv 运行（独立 PyTorch 依赖），需要:

1. 在 Isaac-GR00T venv 中安装 expo-ft (不装 JAX):
```bash
cd ~/workspace/3rd/expo-ft
source ~/workspace/3rd/Isaac-GR00T/.venv/bin/activate
# 只安装 expo-ft 中 PyTorch 需要的部分
pip install -e . --no-deps  # 然后手动装需要的
```

2. 进程间通过 multiprocessing + shared memory 或 Redis 传递参数

> 注: 当前 `train_gr00t_robo_async.py` 的 threading 架构在单进程内工作，两个 GPU 各司其职，已验证可运行。

### 6.5 恢复训练

```bash
python train_gr00t_robo_async.py \
    ... (同上参数) \
    --resume \
    --output_dir ./logs/gr00t_rl_async
```

### 6.6 训练过程预期

1. **启动阶段**: 加载 GR00T 模型 (~6GB, 可能需要 30-60s)，加载离线数据集。
2. **预热阶段**: 前 10 个 episode 只采样不更新（warm-up）。JAX 首次编译需要 ~1-2 分钟。
3. **训练循环**: 
   - 每个 env step: Actor 采样 action chunk，环境执行 `replan_steps` 步
   - 每个 episode 结束后: Learner 做 UTD=20 轮更新
   - wandb 实时记录: actor_loss, critic_loss, q_values, sample_time
4. **Checkpoint**: 保存路径 `./logs/<run_name>/checkpoints/step_<N>/`

---

## 7. 配置参考

### 7.1 核心超参数来自 EXPO-FT 论文

| 参数 | 论文推荐值 | 本项目默认 | 说明 |
|------|-----------|-----------|------|
| `N` (OTF samples) | 4-8 | 8 | 更多样本 → 更好的 Q 选择，但推理更慢 |
| `n_edit_samples` | 4-8 | 8 | Edit policy 候选数 |
| `num_qs` | 10 | 10 | RedQ ensemble 大小 |
| `num_min_qs` | 2 | 2 | In-sample minimum Q 数 |
| `edit_scale` | 0.2 | 0.2 | Edit policy 的 action scaling |
| `actor_lr` | 1e-5 ~ 3e-5 | 3e-5 | GR00T actor BC 学习率 |
| `critic_lr` | 1e-4 ~ 3e-4 | 3e-4 | Critic ensemble 学习率 |
| `utd_ratio` | 10-20 | 20 | 每 episode 的 critic update 轮数 |
| `discount` | 0.99 | 0.99 | RL discount factor |
| `replan_steps` | 8 | 8 | 动作执行步数 (@8Hz = 1s re-plan) |

### 7.2 VRAM 优化策略

GR00T N1.7-3B 模型 (~3.1B params ≈ 6GB bf16)，建议:

1. **默认配置**: `freeze_gr00t_backbone=False`, `use_gradient_checkpointing=False`
   - 需要 ~24-28GB GPU 显存 (RTX 4090 边缘)
   
2. **低 VRAM 配置** (24GB GPU):
   ```python
   config.freeze_gr00t_backbone = True          # 只训练 action head
   config.use_gradient_checkpointing = True     # 减少 25-30% VRAM
   ```

3. **Critic 的 image encoding**: Critic 使用自己的 ResNet encoder (不 share GR00T backbone)，所以 critic 侧不需加载 3B 模型。

---

## 8. 监控与调试

### 8.1 wandb Dashboard

训练会自动上传到 wandb。重点关注:

| Metric | 正常范围 | 异常信号 |
|--------|---------|---------|
| `training/actor_loss` | 0.01 ~ 0.1 | >1.0 → 数据/配置问题 |
| `training/critic_loss` | 从高下降 | 不下降 → 学习率问题; 突然 NaN → 网络不稳定 |
| `training/q` | 0.0 ~ 1.0 | 持续 < 0 → reward 设计问题 |
| `training/q_min/q_max` | - | max/min 差距持续增大 → overestimation |
| `training/update_time_avg_ms` | 500-2000ms | >5s → VRAM 不足 |
| `episode/success_rate` | 逐渐上升 | 不上升 → RL 不收敛 |

### 8.2 TensorBoard (备选)

Checkpoint 目录兼容 TensorBoard 读取。

### 8.3 常见问题诊断

```
Q: JAX XLA OOM on GPU1 ?
A: 减小 batch_size 或 utd_ratio. XLA 编译需要额外 2-3GB.

Q: "No kernel image available" on GPU0 ?
A: PyTorch 版本不支持 sm_120. 使用 Isaac-GR00T 的 PyTorch 2.11+cu130.

Q: actor_loss becomes NaN ?
A: 1) 启用 use_model_lock=True
   2) 降低 actor_lr 到 1e-5
   3) 检查 SFT checkpoint 是否完整

Q: 环境连接失败 ?
A: 1) 确认 env server 在 8102 端口运行
   2) 检查防火墙: ssh -R 8102:localhost:8102 <训练机>
   3) 用 python -c "from expo_ft.env.env_client import EnvClient; EnvClient(host='...').reset()" 测试
```

---

## 9. 完整运行检查清单

启动前依次确认:

- [ ] **硬件**: 2× GPU 可用 (`nvidia-smi`), GPU0 ≥ 24GB
- [ ] **expo-ft venv**: JAX 0.10+ 自编译, 2 GPUs detected
- [ ] **Isaac-GR00T venv**: PyTorch 2.11+cu130, GPU0 可用
- [ ] **GR00T 模型**: SFT checkpoint 目录包含 `config.json` + safetensors + processor
- [ ] **环境服务器**: WebSocket `:8102` 可达
- [ ] **离线数据集**: `/datasets/...` 存在且可 load
- [ ] **任务配置**: `configs/task/*.py` 的 `example_action` 维度匹配
- [ ] **模型配置**: `gr00t_model_path`, `gr00t_embodiment_tag`, camera/state/action keys 正确
- [ ] **输出目录**: `./logs/` 可写

---

## 附录 A: 文件速查

| 文件 | 用途 |
|------|------|
| `train_gr00t_robo_async.py` | Async 在线 RL 训练主脚本 (双 GPU, 多线程) |
| `train_gr00t_robo.py` | Sync 训练脚本 (单 GPU, 简单调试) |
| `expo_ft/agents/vla/gr00t_agent.py` | GR00T actor: 模型加载, 推理, BC 训练 |
| `expo_ft/agents/vla/gr00t_train_state.py` | PyTorchTrainState: 模型+优化器状态管理 + checkpoint 序列化 |
| `expo_ft/agents/alg/expo_ft_gr00t.py` | EXPOLearnerGR00T: OTF 采样, critic 更新, update loop |
| `expo_ft/data/gr00t_replay_buffer.py` | GR00T 格式 replay buffer (存储+采样) |
| `expo_ft/data/gr00t_batch_processor.py` | 在线/离线 batch 混合器 |
| `configs/model/expo_ft_gr00t_config.py` | 模型超参数配置 |
| `configs/task/cr5af.py` | CR5AF 任务配置 |
| `expo_ft/env/env_client.py` | 环境服务器 WebSocket 客户端 |

## 附录 B: GR00T 模型路径说明

GR00T N1.7 基础模型目录结构:
```
~/.cache/modelscope/nv-community/GR00T-N1.7-3B/
├── config.json                     # HF model config
├── model-00001-of-00002.safetensors   # ~3B params, shard 1
├── model-00002-of-00002.safetensors   # ~3B params, shard 2
├── model.safetensors.index.json       # weight map
├── processor_config.json              # Processor modality + transform config
├── statistics.json                    # State/action normalization stats
├── embodiment_id.json                 # Embodiment registry
└── scheduler.pt                       # (SFT only) LR scheduler state
```

SFT finetuned checkpoint 可能有不同结构（单文件 safetensors 或分片），但至少需要 `config.json` + `*.safetensors` + `processor_config.json`。

---

## 附录 C: 与实际仓库的对齐 / 已知审查结论

### C.1 `examples/CR5AF/` 工作区（Isaac-GR00T 侧）

CR5AF 的 SFT/部署/数据脚本集中在 `Isaac-GR00T/examples/CR5AF/`，本文档此前的版本未提及：

| 文件 | 用途 |
|------|------|
| `cr5af_config.py` | 注册 `NEW_EMBODIMENT` modality（2 cam + eef_9d/joint_pos/gripper_pos，10 步 action horizon） |
| `convert_to_lerobot.py` | 原始遥控数据 → LeRobot v2 格式 |
| `finetune_l50.sh` | 生产级 SFT 启动脚本（lookahead=50，tune-visual + top-2 LLM） |
| `finetune_test_ckpt.sh` | 短 run，仅为产出可加载 checkpoint 跑通 RL 代码 |
| `train_iql_critic.py` | 状态-only IQL Critic+Value（obs/act dim=16，τ=0.7 expectile） |
| `train_success_classifier.py` | 成功检测分类器（reward 来源） |
| `deploy_cr5af.py` | 真机部署推理 |
| `README.md` | CR5AF 完整 finetune/部署历史与经验 |

### C.2 EXPO-FT vs IQL+QGF

团队 CR5AF **当前在用的 RL 路线是 IQL Critic + QGF guidance**（状态-only，`train_iql_critic.py`），与本文档描述的 EXPO-FT（图像 critic + 残差 edit + OTF argmax-Q）是两条独立代码路径。`expo_ft/agents/alg/expo_ft_gr00t.py` 是 EXPO-FT 路径，与 IQL/QGF 未集成。选型时需明确走哪条线。

### C.3 已审查的 "剩余 Issue" 结论（对照代码与论文）

- **#8 `% capacity` 换行**：**诊断错误**。`sample_jax` 中 `max_start = len(self) - replan_steps`（第 436 行）已保证 `indices + replan_steps < capacity`，取模**永不触发**。跨 episode 顾虑对 critic 已被屏蔽：TD 目标第 309 行 `* masks` 屏蔽 bootstrap，critic 损失第 330 行 `* valids` 在 chunk 内 terminal 时零化样本。n-step return（第 499–506 行）是标准做法。仅 actor BC 路径动作块在 episode 末尾有低优先级缺口。
- **奖励分类器**：expo-ft 内确实没有，但架构上 reward 由 env 服务端提供（`train_gr00t_robo.py` 第 264 行 `env.get_info_for_step()`），分类器在 `examples/CR5AF/train_success_classifier.py`。是 env 侧接线任务，非 expo-ft 缺组件。
- **HIL**：**完整且正确**。检测（`action_type=="human"`）、标记（`is_hil`）、清除（`action_plan.clear()`）均实现；接管期间 `env.step` 返回的 `real_action`（人类实际动作）被逐 step 存入缓冲区（第 314 行），chunk 在采样时重构（Fix #7），无需"存储块替换"。

> **真正的阻塞点是缺失 SFT checkpoint 权重**（v5/v6 历史run 仅存 `processor/`+`experiment_cfg/`，无 `checkpoint-<step>/`），而非 #8/HIL/奖励。
