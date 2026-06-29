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
│ │  - GR00T N1.7 model (~7B, bf16)           │   │
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
cd /home/zpa/workspace/3rd/expo-ft

# 创建 Python 3.12 venv
uv venv --python 3.12

# 安装项目依赖
uv sync

# 手动安装自编译 JAX (RTX 5090 sm_120 需要自编译)
# JAX 0.10.1 自编译路径
source .venv/bin/activate
pip install /home/zpa/.cache/bazel_old/jax/jaxlib/tools/dist/jaxlib-0.10.1.dev0+selfbuilt-cp312-cp312-linux_x86_64.whl
pip install /home/zpa/.cache/bazel_old/jax/jax_cuda13_plugin-0.10.1.dev0+selfbuilt-cp312-cp312-linux_x86_64.whl
pip install /home/zpa/.cache/bazel_old/jax/jax_cuda13_pjrt-0.10.1.dev0+selfbuilt-cp312-cp312-linux_x86_64.whl
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
/home/zpa/workspace/3rd/Isaac-GR00T/.venv/bin/python -c "import torch; print('PyTorch', torch.__version__)"
# Expected output: PyTorch 2.11.0+cu130
```

### 2.3 GR00T 模型下载

```bash
# 从 ModelScope 下载 GR00T N1.7 基础模型
# (如果已下载则跳过)
mkdir -p ~/.cache/modelscope_old/nv-community/
# 模型路径: ~/.cache/modelscope_old/nv-community/GR00T-N1___7-3B/

# 目录结构:
# GR00T-N1___7-3B/
#   config.json
#   model-00001-of-00002.safetensors   (~3.5GB)
#   model-00002-of-00002.safetensors   (~3.5GB)
#   processor_config.json
#   statistics.json
#   embodiment_id.json
```

---

## 3. Phase 1: GR00T SFT / BC 预训练

> **目的**: 在目标 embodiment 上对 GR00T N1.7 base model 做 behavior cloning，获得初始策略。
> **前提**: 需要收集若干条人工遥控示教数据（推荐 50-100 条成功 episode）。

### 3.1 注册 Embodiment

在 Isaac-GR00T 中注册你的新 embodiment（例如 `CR5AF_TOP_HAND`）。

参考: Isaac-GR00T 仓库的 embodiment 注册流程。

### 3.2 准备示教数据

将遥控数据转为 GR00T 格式存储：

- 图片: 每个视角保存为 `video.<view>` key，格式 uint8 (H, W, 3)
- 状态: 每个 modality 保存为 `state.<key>` key，格式 float32
- 动作: 每个 modality 保存为 `action.<key>` key，格式 float32
- 语言指令: 自然语言 task description

### 3.3 运行 SFT Finetuning

```bash
cd /home/zpa/workspace/3rd/Isaac-GR00T
source .venv/bin/activate

# 使用 Isaac-GR00T 的 finetune 脚本
# 示例 (以实际脚本为准):
python scripts/finetune.py \
    --model_path ~/.cache/modelscope_old/nv-community/GR00T-N1___7-3B/ \
    --embodiment CR5AF_TOP_HAND \
    --dataset_path /datasets/cr5af_demos/ \
    --output_dir ./checkpoints/cr5af_sft/ \
    --num_epochs 10 \
    --batch_size 2 \
    --learning_rate 3e-5
```

### 3.4 验证 SFT Checkpoint

确保 SFT 输出目录包含以下文件:

```
checkpoints/cr5af_sft/
  config.json
  model.safetensors           # 或 model-00001-of-00002.safetensors + model-00002-of-00002.safetensors
  processor_config.json
  statistics.json
  scheduler.pt                 # (可选)
  trainer_state.json            # (可选)
```

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
cd /home/zpa/workspace/3rd/expo-ft
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

即使做纯在线 RL，也需要一个离线数据集（用于初始化 replay buffer 和 agent 结构推断）。

数据集目录结构:
```
/datasets/cr5af_grasp_housing/
  train/
    data_0.hdf5   # 或 .npz 文件
    data_1.hdf5
    ...
```

每个 sample 包含:
```python
{
    "image": {
        "hand_view":  np.ndarray (H, W, 3) uint8,
        "table_view": np.ndarray (H, W, 3) uint8,
    },
    "state":      np.ndarray (state_dim,) float32,
    "actions":    np.ndarray (action_horizon, action_dim) float32,
    "prompt":     "task description string",
}
```

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

### 5.3 验证数据集加载

```bash
cd /home/zpa/workspace/3rd/expo-ft
source .venv/bin/activate

python -c "
from expo_ft.env.droid_utils import process_droid_dataset
from configs.task import cr5af

config = cr5af.get_config()
dataset = process_droid_dataset('/datasets/cr5af_grasp_housing', config, num_data=5)
print(f'Loaded {len(dataset)} episodes')
print(f'Sample keys: {list(dataset[0].keys())}')
"
```

---

## 6. Phase 4: EXPO-FT RL 在线训练

### 6.1 配置文件准备

编辑 `configs/model/expo_ft_gr00t_config.py`，确认关键字段:

```python
# --- 必须修改的字段 ---
config.gr00t_model_path = "/path/to/your/sft/checkpoint"       # SFT 输出目录
config.gr00t_embodiment_tag = "CR5AF_TOP_HAND"                 # Embodiment tag
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
cd /home/zpa/workspace/3rd/expo-ft
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
cd /home/zpa/workspace/3rd/expo-ft
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
cd /home/zpa/workspace/3rd/expo-ft
source /home/zpa/workspace/3rd/Isaac-GR00T/.venv/bin/activate
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

1. **启动阶段**: 加载 GR00T 模型 (~7GB, 可能需要 30-60s)，加载离线数据集。
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

GR00T N1.7 模型很大 (~7B params ≈ 14GB bf16)，建议:

1. **默认配置**: `freeze_gr00t_backbone=False`, `use_gradient_checkpointing=False`
   - 需要 ~24-28GB GPU 显存 (RTX 4090 边缘)
   
2. **低 VRAM 配置** (24GB GPU):
   ```python
   config.freeze_gr00t_backbone = True          # 只训练 action head
   config.use_gradient_checkpointing = True     # 减少 25-30% VRAM
   ```

3. **Critic 的 image encoding**: Critic 使用自己的 ResNet encoder (不 share GR00T backbone)，所以 critic 侧不需加载 7B 模型。

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
~/.cache/modelscope_old/nv-community/GR00T-N1___7-3B/
├── config.json                     # HF model config
├── model-00001-of-00002.safetensors   # 7B params, shard 1
├── model-00002-of-00002.safetensors   # 7B params, shard 2
├── model.safetensors.index.json       # weight map
├── processor_config.json              # Processor modality + transform config
├── statistics.json                    # State/action normalization stats
├── embodiment_id.json                 # Embodiment registry
└── scheduler.pt                       # (SFT only) LR scheduler state
```

SFT finetuned checkpoint 可能有不同结构（单文件 safetensors 或分片），但至少需要 `config.json` + `*.safetensors` + `processor_config.json`。
