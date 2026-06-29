# EXPO-FT × GR00T N1.7 适配方案

## 核心策略: PyTorch-JAX Bridge

GR00T N1.7 是 PyTorch 模型, EXPO-FT 全部基于 JAX/Flax + OpenPI。
采用 **桥接方案**: GR00T actor 保持 PyTorch, critic/residual actor/batch_encoder 保持 JAX。
通过 numpy 中间层交换 tensor。

```
                    ┌──────────────────────┐
                    │  EXPOLearner (JAX)   │
                    │  ┌────────────────┐  │
                    │  │ critic (JAX)   │  │
                    │  │ residual (JAX) │  │
                    │  │ encoder (JAX)  │  │
                    │  └────────────────┘  │
                    │         │             │
                    │  actions: jax→np→torch│
                    │  obs: torch→np→jax    │
                    │         │             │
                    │  ┌────────────────┐  │
                    │  │ Gr00tAgent     │  │
                    │  │ (PyTorch)      │  │
                    │  └────────────────┘  │
                    └──────────────────────┘
```

## Phase 1: Gr00tAgent VLA Wrapper (核心)

实现 `Gr00tAgent(Model)` 接口, 桥接 GR00T PyTorch 到 EXPO-FT JAX 框架。

### 文件: `expo_ft/agents/vla/gr00t.py`

必须实现 `Model` ABC 的 7 个方法:

| 方法 | 功能 | 桥接要点 |
|------|------|---------|
| `initialize` | 加载模型, 创建 train_state | PyTorch 模型加载, 需自定义 TrainState 包装 |
| `get_params` | 提取最佳参数 | 从 PyTorch state_dict 提取 |
| `init_target_params` | 创建 EMA target 参数 | 复制 PyTorch state_dict |
| `process_raw_inputs` | 观测预处理 | GR00T processor 处理, 输出转 numpy |
| `process_transformed_outputs` | 动作反归一化 | GR00T processor decode, numpy→jax |
| `sample_actions` | 推理采样动作 | torch inference→numpy→jax |
| `sample_training_actions` | 训练采样动作 | 同上, train=True |
| `prepare_batch_for_actor` | 为 actor loss 格式化 batch | GR00T collator 格式 |
| `train_step` | 训练一步 | torch autograd, 返回 (new_state, info) |

### 关键设计决策:

1. **TrainState 包装**: PyTorch 模型不能放进 JAX TrainState。
   - `actor_train_state` 存为纯 Python dict (PyTorch state_dict)
   - 提供 `PyTorchTrainState` 类, 管理 optimizer state + model params
   - EMA 用 `optax.incremental_update` 等效的 torch polyak averaging

2. **Inference 流程**:
   ```python
   def sample_actions(self, transformed_inputs, train_state, rng, train, num_samples):
       # 1. numpy/jax → torch (observation)
       # 2. torch inference (GR00T flow matching denoising)
       # 3. torch → numpy → jax (actions)
   ```

3. **Sharding**: GR00tAgent.infer_sharding 用 `SingleDeviceSharding`
   (PyTorch 推理在单一 GPU), EXPOLearner 的编码/采样在 JAX 设备。

4. **随机数**: JAX rng → numpy seed → torch manual_seed for reproducibility

### 文件: `expo_ft/agents/vla/gr00t_train_state.py`

```python
class PyTorchTrainState:
    """Manage PyTorch model + optimizer state, compatible with EXPOLearner interface."""
    params: dict  # PyTorch state_dict
    ema_params: dict | None
    optimizer: torch.optim.Optimizer
    step: int
    model_def: torch.nn.Module  # kept alive for re-merge
```

## Phase 2: GR00T Config & Build 函数

### 文件: `configs/model/expo_ft_gr00t_config.py`

```python
def get_config():
    config = sac_config.get_config()
    config.model_cls = "EXPOLearner"
    config.use_gr00t = True
    config.gr00t_model_path = ""  # SFT checkpoint path
    config.gr00t_embodiment_tag = ""  # e.g. "new_robot"
    config.gr00t_resize_size = 224
    config.freeze_gr00t_encoder = True  # freeze VLM backbone
    # ... critic/residual/encoder params same as pi05 config
```

### 文件: `expo_ft/utils/train_utils.py` (扩展)

增加 `build_gr00t_config()` 函数, 类似 `build_pi05_config()`。

### 文件: `expo_ft/agents/vla/gr00t_builder.py`

```python
def build_gr00t(config, seed, mesh, data_sharding, replicated_sharding,
                resume, default_prompt):
    """Build GR00T actor, train_state, target_params, metadata."""
    actor = Gr00tAgent.initialize(config, ...)
    return actor, actor_train_state, target_actor_params, agent_kwargs, metadata
```

## Phase 3: Replay Buffer 适配

### 文件: `expo_ft/data/gr00t_replay_buffer.py`

GR00T 的观测格式不同于 pi05 (DROID 格式):
- pi05: `base_image, left_wrist_image, right_wrist_image, state`
- GR00T: 动态 `video.{cam_name}`, `state.{state_key}` (由 embodiment tag 决定)

需要适配:
- 根据 embodiment tag 的 `ModalityConfig` 确定图像/状态键
- 动态构建 `dataset_dict` shape
- `convert_to_critic_format()` 使用 GR00T 的 normalization
- `_preprocess_single_transition()` 使用 GR00T processor 而非 OpenPI transforms

## Phase 4: 批处理 & Critic 编码适配

### 文件: `expo_ft/data/batch_processor.py` (修改)

- `prepare_critic_batch()` 需适配 GR00T 的 action 格式
- `prepare_actor_sampling_batch()` 需适配 GR00T 的观测格式
- `extract_critic_fields()` 需适配 GR00T 的 state/action 维度

### 文件: `expo_ft/agents/alg/batch_utils.py` (修改)

- critic batch 提取需适配 GR00T 的 multi-view 图像格式

## Phase 5: 环境适配

### 文件: `configs/task/gr00t_task_config.py`

CR5AF 机器人 + TopHand 手的任务配置:
- 6DOF arm + dexterous hand action space
- D405/D455 camera 观测
- Jetson Thor 部署

### 文件: `client/envs/gr00t_env.py`

GR00T 环境封装, 类似 `droid_env.py` 但适配 CR5AF 硬件接口。

## Phase 6: 训练脚本

### 文件: `scripts/gr00t/` 目录

- `collect_data.sh` - 数据采集
- `convert_data.sh` - 数据转换到 GR00T LeRobot 格式
- `calculate_norm.sh` - 计算归一化统计
- `finetune_gr00t.sh` - SFT 预训练 (用 GR00T 原生 finetune)
- `run_server.sh` - EXPO-FT RL 微调
- `run_server_async.sh` - 异步 RL 微调
- `eval_policy.sh` - 评估

### 文件: `train_pi_robo.py` (修改)

增加 GR00T 分支:
```python
if model_cls == "EXPOLearner":
    if config.use_gr00t:
        from expo_ft.agents.vla.gr00t_builder import build_gr00t
        actor, ... = build_gr00t(FLAGS.config, ...)
    else:
        from expo_ft.agents.vla.pi05 import build_pi05
        actor, ... = build_pi05(FLAGS.config, ...)
```

## 实施顺序

```
Phase 1  ──→  Phase 2  ──→  Phase 3  ──→  Phase 4  ──→  Phase 5  ──→  Phase 6
(核心wrapper)   (config)      (buffer)       (batch)        (env)         (scripts)
   2-3 days       1 day        1-2 days       1 day          2 days        1 day
```

## 关键风险 & 缓解

1. **PyTorch-JAX 内存冲突**: 两个框架同时占用 GPU。
   - 缓解: critic/batch_encoder 较小, GR00T backbone frozen (encoder detached)
   - 可选: critic 在 CPU 上运行, GR00T 在 GPU 上

2. **JAX jit 与 PyTorch 交互**: JAX jit 不能调用 PyTorch。
   - 缓解: `sample_actions` 不被 jit, 作为普通 Python 函数
   - EXPOLearner 的 `_update_jit` 中 `update_actor` 部分不能 jit → 拆分 jit 区域

3. **观测编码器不一致**: GR00T 用自己的 VLM backbone 编码, EXPO-FT critic 用 ResNetV2 编码。
   - 缓解: EXPO-FT 原本就是这样! critic 有自己独立的 encoder (ResNetV2)
   - VLA actor 用 VLM backbone, critic 用 ResNetV2, 两者独立编码

4. **RTC (Receding Temporal Control)**: GR00T 支持 action chunk inpainting。
   - 缓解: EXPO-FT 的 `replan_steps` 机制类似, 可以对接
