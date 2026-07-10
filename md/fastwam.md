# FastWAM 中 RoboTwin 从训练到推理的完整链路

本文档基于对本地仓库 `/nav-oss/yangganlin/tzz_workspace/FastWAM` 的代码阅读整理，重点说明 RoboTwin 数据在 FastWAM 中从离线训练到 RoboTwin 环境评测的完整工程路径。文档中的代码路径均相对于仓库根目录 `FastWAM/`。

需要先说明几个当前工作区状态：

- 当前 `data/robotwin2.0/` 下只看到 `dataset_stats.json` 和 `robotwin2.0.tar.gz.part-00`，没有看到已解压后的 `data/robotwin2.0/robotwin2.0/{data,meta,videos}`。
- 当前未找到 `third_party/RoboTwin/task_config/`，但 RoboTwin 评测代码依赖该目录下的 `_eval_step_limit.yml`、`demo_clean.yml`、`demo_randomized.yml` 以及 embodiment/camera 配置。
- FastWAM 训练默认使用预计算 T5 文本 embedding；训练时 `load_text_encoder=false`，推理/评测时 `load_text_encoder=true`。
- 本文档中“未找到/需验证”的地方均来自当前本地代码状态，不代表论文或官方仓库一定没有。

---

## 1. 仓库整体结构

### 1.1 顶层目录

| 路径 | 作用 |
| --- | --- |
| `README.md` | 官方使用说明，包含 RoboTwin 数据下载、模型准备、训练和评测命令。 |
| `configs/` | Hydra 配置入口，覆盖数据、模型、训练、任务和仿真评测配置。 |
| `scripts/` | 训练、分布式启动、文本 embedding 预计算、Action DiT 初始化等脚本。 |
| `src/fastwam/` | FastWAM 主代码，包括数据集、processor、模型、MoT、scheduler、trainer、runtime。 |
| `experiments/robotwin/` | RoboTwin 评测入口、并行任务管理器、FastWAM policy adapter。 |
| `third_party/RoboTwin/` | 内置 RoboTwin 环境和官方评测脚本。当前本地缺少 `task_config/`。 |
| `data/` | 默认数据目录。RoboTwin 配置默认读取 `data/robotwin2.0/robotwin2.0`。 |
| `checkpoints/` | 默认模型权重目录。README 中要求放置 Wan2.2 和 Action DiT 初始化权重。 |
| `runs/` | 默认训练输出目录，保存 checkpoint、日志、config、eval 可视化结果。 |
| `evaluate_results/` | RoboTwin rollout/评测默认输出目录。 |

### 1.2 配置文件

| 路径 | 作用 |
| --- | --- |
| `configs/train.yaml` | 基础训练配置，包括 `output_dir`、`seed`、`mixed_precision`、wandb 开关等。 |
| `configs/data/robotwin.yaml` | RoboTwin 数据集配置。定义 dataset dir、三相机、action/state 维度、processor、归一化统计等。 |
| `configs/model/fastwam.yaml` | FastWAM 主模型配置，创建 `fastwam.runtime.create_fastwam`。定义 Wan2.2 Video DiT、Action DiT、scheduler、loss 权重等。 |
| `configs/model/fastwam_joint.yaml` | Fast-WAM-Joint 模型配置，创建 `create_fastwam_joint`，推理时联合生成未来视频和动作。 |
| `configs/model/fastwam_idm.yaml` | Fast-WAM-IDM 模型配置，创建 `create_fastwam_idm`，推理时先生成视频再由 inverse dynamics 预测动作。 |
| `configs/task/robotwin_uncond_3cam_384_1e-4.yaml` | RoboTwin Fast-WAM 训练任务配置，默认 `data=robotwin`、`model=fastwam`。 |
| `configs/task/robotwin_joint_3cam_384_1e-4.yaml` | RoboTwin Fast-WAM-Joint 训练配置。 |
| `configs/task/robotwin_idm_3cam_384_1e-4.yaml` | RoboTwin Fast-WAM-IDM 训练配置。 |
| `configs/sim_robotwin.yaml` | RoboTwin 在线评测配置，包含 checkpoint、环境路径、rollout 参数、并行评测参数。 |

### 1.3 数据处理相关代码

| 路径 | 作用 |
| --- | --- |
| `src/fastwam/datasets/lerobot/robot_video_dataset.py` | RoboTwin 训练数据集入口 `RobotVideoDataset`。负责从 LeRobot 数据集中取样，选择视频帧、拼接三相机、读取文本缓存、输出训练 batch 字段。 |
| `src/fastwam/datasets/lerobot/base_lerobot_dataset.py` | `BaseLerobotDataset`，封装 LeRobot/MultiLeRobotDataset，构造 delta timestamps、episode train/val split、统计 action/state 归一化参数。 |
| `src/fastwam/datasets/lerobot/processors/fastwam_processor.py` | `FastWAMProcessor`，处理图像变换、instruction、action/state 归一化、action/state 合并。 |
| `src/fastwam/datasets/lerobot/transforms/action_state_merger.py` | `ConcatLeftAlign`，把多个 action/state key 按顺序拼成一个扁平向量。RoboTwin 中最终是 14 维。 |
| `src/fastwam/datasets/lerobot/utils/normalizer.py` | `LinearNormalizer`，负责 action/state 的 z-score 或 min-max 归一化/反归一化。 |
| `src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py` | vendored LeRobot 数据集实现，读取 `meta/*.jsonl`、parquet、mp4 视频，并支持多数据集拼接。 |
| `scripts/precompute_text_embeds.py` | 训练前将 `meta/tasks.jsonl` 中的任务文本编码成 T5 embedding 缓存。 |

### 1.4 模型定义与训练相关代码

| 路径 | 作用 |
| --- | --- |
| `src/fastwam/runtime.py` | Hydra 配置实例化入口。构建模型、数据集和 trainer。 |
| `src/fastwam/trainer.py` | `Wan22Trainer`，实现训练循环、eval、checkpoint、resume、分布式 accelerator。 |
| `src/fastwam/models/wan22/fastwam.py` | Fast-WAM 主模型，包含训练 loss、联合视频/动作 flow matching、Fast-WAM 低延迟 `infer_action`。 |
| `src/fastwam/models/wan22/fastwam_joint.py` | Fast-WAM-Joint，实现推理时视频和动作联合去噪。 |
| `src/fastwam/models/wan22/fastwam_idm.py` | Fast-WAM-IDM，实现先视频再动作的串行推理。 |
| `src/fastwam/models/wan22/mot.py` | Mixture-of-Transformers 核心，实现视频专家和动作专家 Q/K/V 拼接 attention，以及推理时 video K/V cache。 |
| `src/fastwam/models/wan22/action_dit.py` | Action DiT 定义，把动作序列作为 diffusion token 建模。 |
| `src/fastwam/models/wan22/wan_video_dit.py` | Wan2.2 Video DiT 包装，包含视频 patchify、视频 attention mask、上下文 cross-attention 等。 |
| `src/fastwam/models/wan22/scheduler.py` | Flow matching scheduler，用于训练加噪和推理去噪。 |
| `scripts/train.py` | Hydra 训练入口，调用 `fastwam.runtime.run_training`。 |
| `scripts/train_zero1.sh` | DeepSpeed ZeRO-1 分布式训练启动脚本。 |
| `scripts/train_zero2.sh` | DeepSpeed ZeRO-2 分布式训练启动脚本。 |
| `scripts/preprocess_action_dit_backbone.py` | 从 Wan2.2 Video DiT 初始化较小的 Action DiT backbone 权重。 |

### 1.5 推理 / 评测 / rollout 相关代码

| 路径 | 作用 |
| --- | --- |
| `experiments/robotwin/run_robotwin_manager.py` | RoboTwin 多任务、多 GPU 评测 manager。为 clean/random 两阶段启动子进程，汇总 success rate。 |
| `experiments/robotwin/eval_robotwin_single.py` | 单任务评测 wrapper。把 FastWAM config 转换成 RoboTwin 官方 `script/eval_policy.py` 参数。 |
| `experiments/robotwin/fastwam_policy/deploy_policy.py` | FastWAM 在 RoboTwin 中的 policy adapter。负责 observation 转 tensor、调用 `model.infer_action`、反归一化动作、执行 action chunk。 |
| `experiments/robotwin/fastwam_policy/deploy_policy.yml` | RoboTwin policy 默认参数。部分字段会被 `sim_robotwin.yaml` 覆盖。 |
| `third_party/RoboTwin/script/eval_policy.py` | RoboTwin 官方评测脚本。初始化任务环境、选择 seen/unseen instruction、执行 rollout、统计成功率。 |
| `third_party/RoboTwin/envs/_base_task.py` | RoboTwin task base class。提供 `get_obs`、`take_action`、camera 渲染、qpos action 执行、成功判断入口。 |

---

## 2. RoboTwin 数据格式与放置位置

### 2.1 默认数据位置

RoboTwin 配置默认读取：

```yaml
# configs/data/robotwin.yaml
train:
  dataset_dirs:
    - ./data/robotwin2.0/robotwin2.0
val:
  dataset_dirs:
    - ./data/robotwin2.0/robotwin2.0/
```

README 期望数据解压后的结构是：

```text
data/robotwin2.0/
├── dataset_stats.json
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

当前本地工作区未看到完整解压后的 `robotwin2.0/{data,meta,videos}`，因此如果直接启动训练，数据集初始化大概率会失败。

### 2.2 LeRobot 数据集内部结构

数据读取由 `src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py` 完成，遵循 LeRobot 格式：

```text
robotwin2.0/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   ├── episodes.jsonl
│   ├── stats.json
│   └── episodes_stats.jsonl
└── videos/
    └── chunk-000/
        ├── observation.images.cam_high/
        ├── observation.images.cam_left_wrist/
        └── observation.images.cam_right_wrist/
```

关键 metadata：

- `meta/tasks.jsonl`：保存 `task_index -> task string` 映射。`scripts/precompute_text_embeds.py` 读取这里生成文本 embedding。
- `meta/episodes.jsonl`：保存每个 episode 的 index、长度、关联 tasks 等信息。`BaseLerobotDataset` 按 episode 做 train/val split。
- `meta/info.json`：保存 fps、features、video key、总 episode 数等。
- parquet 文件：保存 action、state、timestamp、task_index 等结构化数据。
- mp4 文件：保存每个 camera 的视频。

### 2.3 RoboTwin 默认 observation/action schema

`configs/data/robotwin.yaml` 中定义的 `shape_meta` 是训练链路的 schema 来源：

```yaml
shape_meta:
  images:
    cam_high:
      raw_shape: [3, 480, 640]
      shape: [3, 240, 320]
    cam_left_wrist:
      raw_shape: [3, 480, 640]
      shape: [3, 240, 320]
    cam_right_wrist:
      raw_shape: [3, 480, 640]
      shape: [3, 240, 320]
  actions:
    default:
      raw_shape: [14]
      shape: [14]
  states:
    default:
      raw_shape: [14]
      shape: [14]
```

训练侧最终使用：

- 三个相机：`cam_high`、`cam_left_wrist`、`cam_right_wrist`
- action：14 维 qpos action
- proprio/state：14 维 qpos state
- language instruction：来自 LeRobot 的 `task` 字段

---

## 3. RoboTwin 训练链路

### 3.1 训练前准备

#### 3.1.1 解压 RoboTwin 数据

README 给出的数据解压命令是：

```bash
cd data/robotwin2.0
cat robotwin2.0.tar.gz.part-* | tar -xzf -
```

解压后需要确认：

```text
data/robotwin2.0/robotwin2.0/data
data/robotwin2.0/robotwin2.0/meta
data/robotwin2.0/robotwin2.0/videos
data/robotwin2.0/dataset_stats.json
```

#### 3.1.2 准备 Action DiT 初始化权重

`configs/model/fastwam.yaml` 默认读取：

```yaml
action_dit_pretrained_path: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

生成命令：

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

对应代码：`scripts/preprocess_action_dit_backbone.py`。

该脚本加载 Wan2.2 Video DiT，并把可迁移的 transformer backbone 权重插值/缩放到 1024 hidden dim 的 Action DiT。`action_encoder.` 和 `head.` 不从视频模型加载，会随机初始化。

#### 3.1.3 预计算文本 embedding

FastWAM 训练时默认不加载 T5 文本编码器：

```yaml
# configs/model/fastwam.yaml
load_text_encoder: false
```

训练数据集会从缓存读取 `context` 和 `context_mask`，所以训练前要运行：

```bash
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
```

多 GPU 可使用：

```bash
torchrun --nproc_per_node=8 scripts/precompute_text_embeds.py \
  task=robotwin_uncond_3cam_384_1e-4
```

关键代码：`scripts/precompute_text_embeds.py`。

脚本行为：

1. 从 Hydra 配置中找到 `dataset_dirs`、`text_embedding_cache_dir` 和 `context_len`。
2. 读取每个数据集目录下的 `meta/tasks.jsonl`。
3. 对每个 task 构造 prompt：

```python
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)
```

4. 使用 Wan T5/text encoder 得到 `context` 和 `mask`。
5. 保存到：

```text
data/text_embeds_cache/robotwin/
```

缓存文件名由 prompt 的 sha256、context length、encoder id 组成，例如：

```text
<sha256>.t5_len128.wan22ti2v5b.pt
```

### 3.2 启动训练

Fast-WAM 基础模型：

```bash
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4
```

Fast-WAM-Joint：

```bash
bash scripts/train_zero1.sh 8 task=robotwin_joint_3cam_384_1e-4
```

Fast-WAM-IDM：

```bash
bash scripts/train_zero1.sh 8 task=robotwin_idm_3cam_384_1e-4
```

`scripts/train_zero1.sh` 的作用：

- 第一个参数是进程数，例如 `8`。
- 使用 `scripts/accelerate_configs/accelerate_zero1_ds.yaml` 启动 accelerate/deepspeed。
- 自动构造输出目录：

```text
runs/<task_basename>/<run_id>/
```

训练入口：

```text
scripts/train.py
  -> fastwam.runtime.run_training(cfg)
  -> instantiate model
  -> build_datasets(cfg.data)
  -> Wan22Trainer.train()
```

### 3.3 关键训练配置

RoboTwin Fast-WAM 默认任务配置：`configs/task/robotwin_uncond_3cam_384_1e-4.yaml`

```yaml
defaults:
  - override /data: robotwin
  - override /model: fastwam

batch_size: 16
num_workers: 8
learning_rate: 1e-4
num_epochs: 5
gradient_accumulation_steps: 1
weight_decay: 1e-2
lr_scheduler_type: cosine
save_every: 2500
eval_every: 500
log_every: 10
```

基础训练配置：`configs/train.yaml`

```yaml
mixed_precision: bf16
seed: 42
max_grad_norm: 1.0
wandb:
  enabled: false
```

RoboTwin 数据配置：`configs/data/robotwin.yaml`

```yaml
num_frames: 33
action_video_freq_ratio: 4
video_size: [384, 320]
concat_multi_camera: robotwin
text_embedding_cache_dir: ./data/text_embeds_cache/robotwin
context_len: 128
pretrained_norm_stats: ./data/robotwin2.0/dataset_stats.json
val_set_proportion: 0.01
```

两个关键数字：

- `num_frames=33`：一次样本取 33 个时刻。
- `action_video_freq_ratio=4`：视频取 `0,4,8,...,32`，得到 9 帧视频；action 使用 `0..31`，得到 32 个动作。

代码中实际 action horizon 是 `num_frames - 1 = 32`。

### 3.4 数据索引与采样

数据集实例化路径：

```text
fastwam.runtime.build_datasets
  -> hydra.utils.instantiate(cfg.data.train)
  -> RobotVideoDataset
  -> BaseLerobotDataset
  -> MultiLeRobotDataset / LeRobotDataset
```

关键代码：

- `src/fastwam/runtime.py`
- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
- `src/fastwam/datasets/lerobot/base_lerobot_dataset.py`
- `src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py`

`BaseLerobotDataset` 构造 LeRobot delta timestamps：

- image/state：`t = 0..32`
- action：`t = 0..31`
- 时间间隔由 `global_sample_stride / fps` 决定，RoboTwin 默认 `global_sample_stride=1`。

train/val split 逻辑在 `BaseLerobotDataset` 中：

- 读取每个数据集的全部 episode index。
- 使用固定 seed `42` shuffle。
- 根据 `val_set_proportion=0.01` 切分。
- `is_training_set=true` 使用训练 episode；`is_training_set=false` 使用验证 episode。

训练时 dataloader 使用 `Wan22Trainer` 中的 `ResumableEpochSampler`：

- 每个 epoch 重新按 `seed + epoch` 生成随机 permutation。
- 支持从 checkpoint 恢复到 batch offset。
- 每个分布式 rank 取自己的 shard。

### 3.5 单个样本如何构造

核心代码：`RobotVideoDataset._get`，路径：

```text
src/fastwam/datasets/lerobot/robot_video_dataset.py
```

流程：

1. 从 `BaseLerobotDataset` 取出原始 LeRobot 样本。
2. 经 `FastWAMProcessor.preprocess` 处理图像、动作、状态、instruction。
3. 从 33 帧中按 `action_video_freq_ratio=4` 选视频帧：

```python
video_sample_indices = range(0, num_frames, action_video_freq_ratio)
# 0, 4, 8, ..., 32
```

4. 三相机拼接为一张 RoboTwin 画面：

```text
cam_high        -> resize 到 256 x 320，放上方
cam_left_wrist  -> resize 到 128 x 160，放下方左侧
cam_right_wrist -> resize 到 128 x 160，放下方右侧

最终图像尺寸：384 x 320
```

5. 图像归一化到 `[-1, 1]`，并调整为：

```text
video: [C, T_video, H, W] = [3, 9, 384, 320]
```

6. action：

```text
action: [32, 14]
```

7. proprio：

```text
proprio: [32, 14]
```

processor 先取 33 个 state，`RobotVideoDataset` 中使用 `sample["proprio"][:-1, :]` 与 32 个 action 对齐。

8. instruction：

```python
task = sample["instruction"]
prompt = DEFAULT_PROMPT.format(task=task)
```

9. 读取预计算 T5 embedding：

```text
context: [128, 4096]
context_mask: [128]
```

当前代码在读取缓存时先用 mask 把 padding context 置零，然后把 `context_mask` 重新设为全 1，以匹配 Wan 模型中的上下文处理方式。

最终单样本字段大致为：

```python
{
    "video": Tensor[3, 9, 384, 320],
    "action": Tensor[32, 14],
    "proprio": Tensor[32, 14],
    "prompt": str,
    "context": Tensor[128, 4096],
    "context_mask": Tensor[128],
    "image_is_pad": Tensor[...],
    "action_is_pad": Tensor[32],
    "proprio_is_pad": Tensor[32],
}
```

batch 后主要维度为：

```text
video:   [B, 3, 9, 384, 320]
action:  [B, 32, 14]
proprio: [B, 32, 14]
context: [B, 128, 4096]
```

### 3.6 Processor、归一化和 action/state 组织

核心代码：

- `src/fastwam/datasets/lerobot/processors/fastwam_processor.py`
- `src/fastwam/datasets/lerobot/utils/normalizer.py`
- `src/fastwam/datasets/lerobot/transforms/action_state_merger.py`

`FastWAMProcessor` 负责：

- image transform：默认 `ToTensor + Resize([240, 320])`，之后 `RobotVideoDataset` 还会做 RoboTwin 三相机拼接和 resize。
- instruction 处理：默认使用低层 task 文本。
- action/state 归一化：RoboTwin 使用 `norm_default_mode="z-score"`。
- action/state 合并：`ConcatLeftAlign` 把多个 key 拼成向量。RoboTwin 只有 `default`，因此就是 14 维。

归一化统计来自：

```yaml
pretrained_norm_stats: ./data/robotwin2.0/dataset_stats.json
```

如果训练集配置中没有提供 `pretrained_norm_stats`，`RobotVideoDataset` 会在 main process 上调用 `BaseLerobotDataset.get_dataset_stats()` 计算并保存到工作目录下的 `dataset_stats.json`。验证集必须能拿到统计文件，否则会报错。

### 3.7 模型输入输出

训练模型入口：

```text
FastWAM.training_loss(batch)
```

路径：

```text
src/fastwam/models/wan22/fastwam.py
```

`FastWAM.build_inputs` 中的主要输入：

```text
video:   [B, 3, 9, 384, 320]
action:  [B, 32, 14]
proprio: [B, 32, 14]
context: [B, 128, 4096]
```

视频会进入 Wan VAE，得到 latent。由于配置中：

```yaml
fuse_vae_embedding_in_latents: true
```

代码会保留当前帧 latent，并在训练加噪后把第一帧 latent 替换回干净当前帧。这对应论文笔记中的 `F0=干净当前帧 token`。

proprio 只取第一个状态：

```python
proprio_token = proprio_encoder(proprio[:, 0, :])
```

然后作为额外 context token 拼到文本 context 里。

模型输出：

- `pred_video`：预测视频 latent 的 flow target。
- `pred_action`：预测 action flow target。

### 3.8 Loss 计算

核心代码：`FastWAM.training_loss`。

训练步骤：

1. VAE 编码视频：`video -> video_latents`。
2. 对视频 latent 加 flow noise：`video_scheduler.add_noise(...)`。
3. 对 action 加 flow noise：`action_scheduler.add_noise(...)`。
4. 第一帧视频 latent 保持 clean：`noisy_video[:, :, 0:1] = first_frame_latents`。
5. 经过 Video DiT pre-processing 和 Action DiT pre-processing 后，送入 MoT。
6. 分别计算视频和动作 MSE：

```text
loss_video  = MSE(pred_video, video_target)
loss_action = MSE(pred_action, action_target)
loss        = lambda_video * loss_video + lambda_action * loss_action
```

`configs/model/fastwam.yaml` 中显式配置：

```yaml
loss:
  lambda_action: 1.0
```

`lambda_video` 在 runtime 创建模型时使用默认值 `1.0`。

padding mask：

- `image_is_pad` 用于 mask 视频 loss。
- `action_is_pad` 用于 mask action loss。

### 3.9 训练时 MoT attention 结构

实现位置：

- `src/fastwam/models/wan22/fastwam.py`
- `src/fastwam/models/wan22/mot.py`
- `src/fastwam/models/wan22/wan_video_dit.py`

Fast-WAM 训练时的 attention mask 由 `FastWAM._build_mot_attention_mask` 构造：

- 视频 token 对视频 token 的 mask 来自 `video_expert.build_video_to_video_mask`。
- `video_attention_mask_mode="first_frame_causal"` 时，第一帧 query 不能看未来视频 token。
- action token 可以看 action token。
- action token 只允许看 first-frame video token，不能看未来视频 token。

这与论文笔记中的约束一致：

```text
F0 -> F0
F+ -> F0, F+
A  -> F0, A
A  不读取 F+
```

MoT 的实现方式在 `src/fastwam/models/wan22/mot.py`：

1. 每一层分别用 video expert 和 action expert 生成 Q/K/V。
2. 将两类 token 沿 token 维拼接。
3. 使用统一 Flash Attention。
4. 输出再按 token 区间切回 video/action。
5. 两个 expert 各自走自己的 output projection、cross-attention、FFN。

### 3.10 Checkpoint、日志与评估

训练器：`src/fastwam/trainer.py`。

权重 checkpoint：

```text
runs/<task>/<run_id>/checkpoints/weights/step_XXXXXX.pt
```

由 `model.save_checkpoint(...)` 保存，主要包括：

- `mot`
- `proprio_encoder`
- `step`
- `dtype`
- 可选 optimizer state

完整 accelerate state：

```text
runs/<task>/<run_id>/checkpoints/state/step_XXXXXX/
```

其中还会保存：

```text
trainer_state.json
```

resume 配置字段：

```yaml
resume: null
```

支持两种形式：

- `resume=/path/to/step_XXXXXX.pt`：只加载模型权重。
- `resume=/path/to/checkpoints/state/step_XXXXXX`：恢复 accelerate 完整训练状态。

训练日志由 `Wan22Trainer._log_train` 输出，包含：

- `loss`
- `loss_video`
- `loss_action`
- learning rate
- step speed
- ETA

wandb 默认关闭。如需打开：

```bash
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4 \
  wandb.enabled=true \
  wandb.project=fastwam
```

`eval_every=500` 时，trainer 会从 val set 采样样本，调用：

```python
model.training_loss(sample)
model.infer(...)
```

并保存可视化视频和指标。注意基础 `FastWAM.infer()` 当前调用的是 `infer_joint()`，因此 trainer 内部 eval 会生成视频和动作；而 RoboTwin 在线部署使用的是低延迟的 `infer_action()`，不会生成未来视频。

### 3.11 多 GPU / 分布式训练

支持 accelerate + DeepSpeed。

相关文件：

- `scripts/train_zero1.sh`
- `scripts/train_zero2.sh`
- `scripts/accelerate_configs/accelerate_zero1_ds.yaml`
- `scripts/accelerate_configs/accelerate_zero2_ds.yaml`
- `scripts/accelerate_configs/ds_zero1_config.json`
- `scripts/accelerate_configs/ds_zero2_config.json`

常用启动：

```bash
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4
```

多节点脚本支持环境变量：

```bash
NNODES
NODE_RANK
MASTER_ADDR
MASTER_PORT
```

README 提到官方 RoboTwin 训练为了速度使用了 64 GPU；代码本身可以用较少 GPU 跑，只是训练时间会变长。

---

## 4. RoboTwin 推理与评测链路

### 4.1 评测启动命令

README 推荐命令：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=path/to/checkpoint.pt \
  EVALUATION.dataset_stats_path=./data/robotwin2.0/dataset_stats.json \
  MULTIRUN.num_gpus=8
```

如果使用 self-trained checkpoint，`ckpt` 通常类似：

```text
runs/robotwin_uncond_3cam_384_1e-4/<run_id>/checkpoints/weights/step_XXXXXX.pt
```

### 4.2 评测 manager

入口：`experiments/robotwin/run_robotwin_manager.py`。

主要逻辑：

1. 读取 `configs/sim_robotwin.yaml`。
2. 检查 `ckpt` 是否存在。
3. 如果 `EVALUATION.task_name=null`，从下列文件读取全部任务列表：

```text
third_party/RoboTwin/task_config/_eval_step_limit.yml
```

当前本地工作区未找到 `third_party/RoboTwin/task_config/`，所以这里需要补齐 RoboTwin 官方配置后才能跑全任务评测。

4. 对每个 task 跑两个阶段：

```text
clean  -> task_config=demo_clean
random -> task_config=demo_randomized
```

5. 使用多 GPU 子进程队列调度：

```yaml
MULTIRUN:
  num_gpus: 8
  max_tasks_per_gpu: 2
```

6. 汇总每个任务的 clean/random success rate，输出：

```text
evaluate_results/robotwin/<ckpt_tag>/<run_ts>/
├── manager.log
├── summary.csv
├── summary.json
└── failed_tasks.txt
```

### 4.3 单任务评测 wrapper

文件：`experiments/robotwin/eval_robotwin_single.py`。

这个脚本不直接跑环境，而是把 FastWAM 参数转换为 RoboTwin 官方 `script/eval_policy.py` 的参数。

关键行为：

1. 确保 policy 软链接存在：

```text
third_party/RoboTwin/policy/fastwam_policy
  -> experiments/robotwin/fastwam_policy
```

2. 解析 dataset stats：

优先使用：

```yaml
EVALUATION.dataset_stats_path
```

如果未设置，会从 checkpoint 父目录向上搜索 `dataset_stats.json`。

3. 在 `third_party/RoboTwin` 下启动：

```bash
python -u script/eval_policy.py \
  --config policy/fastwam_policy/deploy_policy.yml \
  --overrides \
  --task_name <task> \
  --task_config <demo_clean_or_demo_randomized> \
  --ckpt_setting <ckpt> \
  --policy_name fastwam_policy \
  ...
```

4. 设置：

```text
CUDA_VISIBLE_DEVICES=<gpu_id>
```

### 4.4 RoboTwin 官方 eval_policy

文件：`third_party/RoboTwin/script/eval_policy.py`。

核心流程：

1. 根据 `task_name` 动态导入任务环境：

```python
task = importlib.import_module(f"envs.{args['task_name']}")
TASK_ENV = getattr(task, args["task_name"])()
```

2. 读取任务配置：

```text
task_config/<demo_clean_or_demo_randomized>.yml
```

以及 embodiment、camera 配置。

3. 对每个 episode，先用 expert policy 检查 seed 是否可解。
4. 从任务描述中选择 instruction：

```python
instruction = results[0][instruction_type]
```

其中 `instruction_type` 默认来自 `configs/sim_robotwin.yaml`：

```yaml
EVALUATION:
  instruction_type: unseen
```

可以改成：

```bash
EVALUATION.instruction_type=seen
```

5. rollout 循环：

```python
while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
    observation = TASK_ENV.get_obs()
    eval(TASK_ENV, model, observation)
    if TASK_ENV.eval_success:
        break
```

如果开启：

```yaml
EVALUATION.skip_get_obs_within_replan: true
```

并且 policy 实现了 `should_request_observation()`，则在 action chunk 尚未执行完时跳过 RGB observation 渲染，以加速评测。

6. 成功率统计：

```text
success_rate = success_episodes / eval_num_episodes
```

结果写入：

```text
<task_name>_result_clean.txt
<task_name>_result_random.txt
```

### 4.5 FastWAM RoboTwin policy adapter

核心文件：

```text
experiments/robotwin/fastwam_policy/deploy_policy.py
```

RoboTwin 调用入口：

```python
def get_model(usr_args):
    return WorldActionRobotWinPolicy(...)
```

#### 4.5.1 checkpoint 加载

`WorldActionRobotWinPolicy.__init__` 中会：

1. compose `sim_robotwin.yaml` 和 task config。
2. 实例化模型。
3. 加载 checkpoint：`self.model.load_checkpoint(...)`。
4. 设置 `self.model.eval()`。

`configs/sim_robotwin.yaml` 中推理侧覆盖：

```yaml
model:
  load_text_encoder: true
  skip_dit_load_from_pretrain: true
  action_dit_pretrained_path: null
```

这意味着推理时不会重新加载 Action DiT 初始化权重，而是直接从训练 checkpoint 恢复 MoT/Action 部分，同时加载 text encoder 用于在线 instruction 编码。

#### 4.5.2 processor 和归一化

policy adapter 会重新实例化训练配置中的 processor：

```python
processor = hydra.utils.instantiate(cfg.data.train.processor)
processor.set_normalizer_from_stats(stats)
```

stats 来自：

```yaml
EVALUATION.dataset_stats_path
```

该 stats 必须与训练时 action/state 的归一化方式一致，否则动作反归一化会错误。

#### 4.5.3 Observation 到模型输入

RoboTwin `BaseTask.get_obs()` 返回的 observation 包含：

```python
observation["observation"]["head_camera"]["rgb"]
observation["observation"]["left_camera"]["rgb"]
observation["observation"]["right_camera"]["rgb"]
observation["joint_action"]["vector"]
```

代码位置：

- `third_party/RoboTwin/envs/_base_task.py`
- `experiments/robotwin/fastwam_policy/deploy_policy.py`

policy adapter 中 `_build_robotwin_image_tensor` 与训练时保持相同拼接方式：

```text
head_camera  -> 320 x 256，上方
left_camera  -> 160 x 128，下方左侧
right_camera -> 160 x 128，下方右侧
最终尺寸     -> 384 x 320
```

然后转为：

```text
input_image: [1, 3, 384, 320], range [-1, 1]
```

proprio/state：

```python
state = observation["joint_action"]["vector"]
```

经 processor 的 state normalizer 得到：

```text
proprio: [1, 14]
```

language instruction：

```python
instruction = task_env.get_instruction()
prompt = DEFAULT_PROMPT.format(task=instruction)
```

#### 4.5.4 模型输出 action

policy 调用：

```python
action = self.model.infer_action(
    prompt=prompt,
    input_image=image,
    action_horizon=32,
    proprio=proprio,
    num_inference_steps=10,
    ...
)
```

默认参数来自 `configs/sim_robotwin.yaml`：

```yaml
EVALUATION:
  action_horizon: null
  replan_steps: 24
  num_inference_steps: 10
```

如果 `action_horizon=null`，代码使用：

```python
action_horizon = cfg.data.train.num_frames - 1
# 32
```

对于 `FastWAMJoint` 和 `FastWAMIDM`，`infer_action` 签名需要 `num_video_frames`，policy adapter 会根据函数签名自动传入：

```python
num_video_frames = (num_frames - 1) // action_video_freq_ratio + 1
# 9
```

#### 4.5.5 action 反归一化与执行

模型输出是归一化 action：

```text
action: [32, 14]
```

policy adapter 使用训练时的 action normalizer 反归一化：

```python
normalizer.backward(action)
```

然后只把前 `replan_steps` 个动作放入队列：

```python
pending_actions = action[:replan_steps]
```

每次 RoboTwin 调用 policy，执行一个动作：

```python
task_env.take_action(action, action_type="qpos")
```

`BaseTask.take_action` 会把 14 维 qpos 拆成：

```text
left_arm:       6
left_gripper:  1
right_arm:      6
right_gripper: 1
```

然后通过 sim controller 执行，并调用任务自身的 `check_success()` 更新 `eval_success`。

### 4.6 episode 终止与 success rate

episode 终止条件来自 `eval_policy.py`：

- 达到 `TASK_ENV.step_lim`
- 或 `TASK_ENV.eval_success=True`

每个 task 的 step limit 通常来自：

```text
third_party/RoboTwin/task_config/_eval_step_limit.yml
```

当前本地该文件未找到。

success rate 写入结果文件，manager 再解析最后的数字行，生成 `summary.csv/json`。

---

## 5. FastWAM 核心机制在代码中的实现位置

### 5.1 world/action model 主体

基础 Fast-WAM：

```text
src/fastwam/models/wan22/fastwam.py
```

核心类：

```python
class FastWAM(nn.Module)
```

组成：

- `video_expert`：Wan2.2 Video DiT。
- `action_expert`：Action DiT。
- `mot`：Mixture-of-Transformers，将 video/action expert 组织为统一模型。
- `vae`：Wan VAE，用于视频 latent 编码/解码。
- `text_encoder/tokenizer`：训练时通常关闭，推理时开启。
- `proprio_encoder`：`Linear(proprio_dim, 4096)`，把当前 proprio 编码成 context token。

### 5.2 representation encoder

视觉 representation：

- VAE 编码：`FastWAM.build_inputs`
- VAE 实现来自 `src/fastwam/models/wan22/wan_vae.py`

语言 representation：

- 训练：预计算缓存，`RobotVideoDataset._get_cached_text_context`
- 推理：在线 T5 encode，`FastWAM._encode_prompt`

proprio representation：

```python
self.proprio_encoder = nn.Linear(proprio_dim, 4096)
```

训练和推理都只使用当前 proprio，作为额外 context token。

### 5.3 dynamics / transition / latent state

代码中没有一个显式命名为 `dynamics` 或 `transition` 的模块。FastWAM 的 dynamics 建模体现在：

- 视频 latent flow matching：`FastWAM.training_loss`
- action flow matching：`FastWAM.training_loss`
- MoT 中 video/action token 的受限交互：`FastWAM._build_mot_attention_mask`
- 推理时基于当前帧 latent 的 action diffusion：`FastWAM.infer_action`

也就是说，world model 的动态预测主要通过 Video DiT 对未来视频 latent 的 flow loss 学到，而不是一个单独的 transition network。

### 5.4 actor / policy / action head

Action model：

```text
src/fastwam/models/wan22/action_dit.py
```

关键结构：

- `action_encoder: Linear(action_dim, hidden_dim)`
- DiT blocks
- `head: Linear(hidden_dim, action_dim)`

FastWAM 并不是传统 BC 的单步确定性 action head，而是对整段 action chunk 做 diffusion/flow denoising。

在线 policy adapter：

```text
experiments/robotwin/fastwam_policy/deploy_policy.py
```

它负责把 RoboTwin observation 转成 `infer_action` 所需的输入，并把归一化 action chunk 反归一化成 qpos action。

### 5.5 MoT 双专家实现

核心文件：

```text
src/fastwam/models/wan22/mot.py
```

对应论文笔记中的“双专家 MoT”：

- video expert hidden dim：3072
- action expert hidden dim：1024
- 两者都使用 24 heads、head dim 128
- 因此 Q/K/V 形状可以统一成 `[B, S, 24, 128]`

训练 `forward`：

- 每层计算 video/action 的 Q/K/V。
- token 维拼接。
- 统一 attention。
- split 回 video/action。
- 各自经过各自的 output projection、cross attention、FFN。

推理 `prefill_video_cache`：

- 只对当前帧 video token 运行 Video DiT。
- 每层保存 video K/V cache。

推理 `forward_action_with_video_cache`：

- 每个 action denoising step 只重算 action expert。
- action query attend 到 `[cached_video_kv; current_action_kv]`。

这就是 Fast-WAM 低延迟推理的核心工程实现。

### 5.6 Fast-WAM 低延迟推理

核心函数：

```python
FastWAM.infer_action(...)
```

路径：

```text
src/fastwam/models/wan22/fastwam.py
```

流程：

1. 输入当前图像：`input_image: [1, 3, H, W]`。
2. VAE 编码当前图像为 first-frame latent。
3. Video DiT 对当前帧做一次 prefill：`self.mot.prefill_video_cache(...)`。
4. 初始化 action noise：`action_latents: [1, action_horizon, action_dim]`。
5. 迭代 `num_inference_steps=10` 次预测 action noise 并 scheduler step。
6. 返回 CPU action：`[action_horizon, action_dim]`。

推理过程中未来视频 token 不存在，因此基础 Fast-WAM 不生成未来视频。

### 5.7 Fast-WAM-Joint

文件：

```text
src/fastwam/models/wan22/fastwam_joint.py
```

主要差异：

- `_build_mot_attention_mask` 允许 action attend 到完整视频序列。
- `infer_action` 会同时初始化未来 video latent 和 action latent。
- 每个 denoising step 同时更新视频和动作。

这对应论文笔记中的 “Fast-WAM-Joint：推理时视频与动作联合去噪”。

### 5.8 Fast-WAM-IDM

文件：

```text
src/fastwam/models/wan22/fastwam_idm.py
```

主要差异：

- 训练时有视频 flow loss。
- inverse dynamics action branch 使用条件视频。
- 训练中以 `video_cond_noise_prob=0.5` 对真实未来视频 latent 加噪，减轻 train/test mismatch。
- 推理分两阶段：

```text
stage 1: 当前帧 + instruction -> 生成未来视频 latent
stage 2: 当前帧 + 生成视频 -> 预测 action
```

这对应论文笔记中的 Fast-WAM-IDM。

### 5.9 RoboTwin-specific adaptation

RoboTwin 适配主要不在模型里，而在数据和 policy adapter 里：

- 三相机拼接：
  - 训练：`RobotVideoDataset._concat_robotwin_cameras`
  - 推理：`WorldActionRobotWinPolicy._build_robotwin_image_tensor`
- action/state 14 维 qpos：
  - 配置：`configs/data/robotwin.yaml`
  - 执行：`third_party/RoboTwin/envs/_base_task.py`
- action chunk：
  - 训练：`num_frames - 1 = 32`
  - 推理：`action_horizon=32`
  - rollout 执行：每次只执行 `replan_steps` 个动作，默认 24。
- language instruction：
  - 训练：LeRobot `task`
  - 推理：RoboTwin task environment `get_instruction()`

### 5.10 与普通 BC / VLA policy 的差异

从代码看，FastWAM 与普通行为克隆/VLA 的主要差异是：

1. 训练目标不只是 action imitation，还包含视频 latent flow matching。
2. action 不是单步确定性输出，而是 action chunk diffusion/flow denoising。
3. 当前帧通过 VAE latent 进入 Video DiT，不是简单 CNN/ViT encoder。
4. MoT 训练时让视频和动作共享 attention 结构，但通过 mask 防止 action 读取未来视频。
5. 基础 Fast-WAM 推理时不生成未来视频，只缓存当前帧 video K/V，再迭代动作去噪。

### 5.11 未找到或需要进一步验证的点

- 代码中没有单独命名为 `world model`、`dynamics model`、`transition model` 的类；这些能力分散在 Video DiT、VAE、MoT 和 flow loss 中。
- `configs/sim_robotwin.yaml` 中有 `text_cfg_scale` 和 `negative_prompt` 字段，policy adapter 也会传参，但基础 `FastWAM.infer_action` 中没有看到显式 classifier-free guidance 逻辑；这些字段对基础 Fast-WAM 可能没有实际效果。
- 当前工作区未找到 `third_party/RoboTwin/task_config/`，因此 RoboTwin 全量评测配置需要补齐后才能验证。
- 当前本地未看到完整解压的 RoboTwin 数据目录，因此训练链路尚未在当前状态下实际跑通。

---

## 6. 只训练 RoboTwin 的部分任务

### 6.1 当前代码是否支持按 task name 过滤

当前代码中，RoboTwin 数据配置只有：

```yaml
dataset_dirs:
  - ./data/robotwin2.0/robotwin2.0
```

没有看到类似：

```yaml
task_names:
  - ...
```

的配置字段。

`BaseLerobotDataset` 当前按 episode 做 train/val split，但没有按 task name 过滤 episode。因此当前代码原生不支持通过 config 直接指定部分任务训练。

### 6.2 不改代码的可行方式

不改代码时，可以准备一个物理过滤后的 LeRobot 数据集目录：

```text
data/robotwin_subset/
├── data/
├── meta/
└── videos/
```

要求：

- 只保留目标 task 对应 episodes。
- 重写或同步 `meta/episodes.jsonl`。
- 保留或重写 `meta/tasks.jsonl`。
- 保持 parquet、mp4 路径与 LeRobot metadata 一致。
- 重新计算或准备对应 `dataset_stats.json`。

然后训练时覆盖：

```bash
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4 \
  data.train.dataset_dirs='[./data/robotwin_subset]' \
  data.val.dataset_dirs='[./data/robotwin_subset]' \
  data.train.pretrained_norm_stats=./data/robotwin_subset/dataset_stats.json \
  data.val.pretrained_norm_stats=./data/robotwin_subset/dataset_stats.json
```

这种方式最不影响代码，但需要额外写数据过滤脚本。

### 6.3 推荐的最小侵入式代码改法

如果希望通过 Hydra config 指定任务名，建议只改数据集层，不改模型和 trainer。

第一步，在 `configs/data/robotwin.yaml` 给 train/val 都加：

```yaml
task_names: null
```

第二步，给 `RobotVideoDataset.__init__` 增加参数并传下去：

```python
class RobotVideoDataset(Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        ...,
        task_names=None,
        ...
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=shape_meta,
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            task_names=task_names,
        )
```

第三步，在 `BaseLerobotDataset` 里按 episode task 过滤：

```python
def _episode_matches_task(meta, episode_index: int, task_names: set[str]) -> bool:
    if not task_names:
        return True

    episode = meta.episodes[episode_index]
    episode_tasks = episode.get("tasks", [])
    if isinstance(episode_tasks, str):
        episode_tasks = [episode_tasks]

    return any(task in task_names for task in episode_tasks)
```

在构造 episode indices 时，把原来的全部 episode：

```python
episode_indices = list(range(meta.total_episodes))
```

替换为：

```python
if task_names is None:
    candidate_indices = list(range(meta.total_episodes))
else:
    names = set(task_names)
    candidate_indices = [
        episode_index
        for episode_index in range(meta.total_episodes)
        if _episode_matches_task(meta, episode_index, names)
    ]

if len(candidate_indices) == 0:
    raise ValueError(f"No episodes matched task_names={task_names}")
```

然后对 `candidate_indices` 做原有的 shuffle 和 train/val split。

说明：

- LeRobot 的 `episodes.jsonl` 通常包含 `tasks` 字段；如果某些数据版本不包含，需要根据 `task_index` 从 parquet 或 metadata 中再查一次。
- 这一路径只过滤 episode，不需要改 `RobotVideoDataset._get`、processor、trainer 或模型。

### 6.4 文本 embedding 缓存

最简单做法是不改 `scripts/precompute_text_embeds.py`，继续对完整 `meta/tasks.jsonl` 预计算文本 embedding。这样会多算一些未训练任务的文本，但不影响训练正确性。

如果任务数很多、想节省预计算时间，可以给 `scripts/precompute_text_embeds.py` 也加 `task_names` 过滤，让 `_read_unique_prompts` 只保留目标 task。

### 6.5 按部分任务训练的示例命令

完成上述最小代码改动后，可以这样训练：

```bash
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
```

```bash
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4 \
  'data.train.task_names=[click_alarmclock,turn_switch]' \
  'data.val.task_names=[click_alarmclock,turn_switch]' \
  output_dir=./runs/robotwin_subset_click_turn
```

如果目标任务很少，验证集可能被 `val_set_proportion=0.01` 切得太小。可以临时调大：

```bash
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4 \
  'data.train.task_names=[click_alarmclock,turn_switch]' \
  'data.val.task_names=[click_alarmclock,turn_switch]' \
  data.train.val_set_proportion=0.1 \
  data.val.val_set_proportion=0.1
```

---

## 7. 添加自己的新任务进行训练

这里假设你的自定义任务数据格式和 RoboTwin 一样，也就是 LeRobot 格式、三相机、14 维 action、14 维 state。

### 7.1 数据目录命名

推荐每个新任务或新数据集使用独立目录：

```text
data/my_tasks/
└── my_task_lerobot/
    ├── data/
    ├── meta/
    └── videos/
```

如果多个自定义任务放在一个 LeRobot 数据集里，也可以：

```text
data/my_tasks/
└── custom_robotwin_like/
    ├── data/
    ├── meta/
    └── videos/
```

要求 metadata 中的 task string 写入：

```text
meta/tasks.jsonl
```

episode 与 task 对应关系写入：

```text
meta/episodes.jsonl
```

### 7.2 schema 必须与配置一致

如果完全复用 RoboTwin 配置，数据中需要有以下 feature key：

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
observation.state
action
task_index
```

并且：

```text
action dim = 14
state dim  = 14
camera count = 3
```

如果 camera 名称或 action/state 维度不同，需要同步修改：

```text
configs/data/robotwin.yaml
```

尤其是：

- `shape_meta.images`
- `shape_meta.actions`
- `shape_meta.states`
- `processor.action_output_dim`
- `processor.proprio_output_dim`
- `concat_multi_camera`

如果仍然希望使用当前 RoboTwin 三相机拼接逻辑，camera 数必须是 3，并且语义上要能对应 high / left wrist / right wrist。

### 7.3 是否需要添加 task config

只训练离线 policy 时，不需要添加 RoboTwin `task_config`。

最少需要：

- LeRobot 数据目录。
- `meta/tasks.jsonl` 中有 task 文本。
- action/state/camera schema 与配置一致。
- 文本 embedding 预计算。
- dataset stats。

只有当你想在 RoboTwin/SAPIEN 在线环境中评测这个新任务时，才需要添加：

- `third_party/RoboTwin/envs/<task_name>.py`
- `third_party/RoboTwin/task_config/*.yml`
- 对应 assets、camera config、embodiment config
- instruction 描述生成逻辑
- task 的 `check_success()`

当前本地仓库缺少 `third_party/RoboTwin/task_config/`，因此在线新任务评测需要先补齐官方 RoboTwin 配置体系。

### 7.4 最少配置改法：把自定义数据加入 dataset_dirs

如果你的数据与 RoboTwin 格式完全一致，可以直接通过 Hydra 覆盖：

```bash
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4 \
  'data.train.dataset_dirs=[./data/robotwin2.0/robotwin2.0,./data/my_tasks/my_task_lerobot]' \
  'data.val.dataset_dirs=[./data/my_tasks/my_task_lerobot]'
```

训练：

```bash
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4 \
  'data.train.dataset_dirs=[./data/robotwin2.0/robotwin2.0,./data/my_tasks/my_task_lerobot]' \
  'data.val.dataset_dirs=[./data/my_tasks/my_task_lerobot]' \
  output_dir=./runs/robotwin_plus_my_task
```

注意：`MultiLeRobotDataset` 要求多个数据集有兼容的 common features 和 fps。如果 fps、feature 名称、action 维度不一致，需要先转换数据或单独写新的 data config。

### 7.5 dataset_stats 怎么处理

当前 RoboTwin 配置默认使用：

```yaml
pretrained_norm_stats: ./data/robotwin2.0/dataset_stats.json
```

如果加入自己的数据，推荐重新计算或准备覆盖所有训练数据的 stats。最小做法：

1. 暂时把训练配置中的 `pretrained_norm_stats` 设为 `null`。
2. 让 `RobotVideoDataset` 在训练集上计算 stats。
3. 保存到本次 `output_dir/dataset_stats.json`。
4. 评测时显式指定该 stats。

评测示例：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./runs/robotwin_plus_my_task/<run_id>/checkpoints/weights/step_XXXXXX.pt \
  EVALUATION.dataset_stats_path=./runs/robotwin_plus_my_task/<run_id>/dataset_stats.json
```

如果只训练自定义数据，也可以手动准备：

```text
data/my_tasks/dataset_stats.json
```

并覆盖：

```bash
data.train.pretrained_norm_stats=./data/my_tasks/dataset_stats.json \
data.val.pretrained_norm_stats=./data/my_tasks/dataset_stats.json
```

### 7.6 只训练离线 policy 的最小步骤

在“自定义任务数据和 RoboTwin 格式一样”的前提下，最少不需要改代码，只需要：

1. 放置 LeRobot 格式数据。
2. 覆盖 `dataset_dirs`。
3. 准备或重新计算 `dataset_stats.json`。
4. 运行文本 embedding 预计算。
5. 启动训练。

示例：

```bash
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4 \
  'data.train.dataset_dirs=[./data/my_tasks/my_task_lerobot]' \
  'data.val.dataset_dirs=[./data/my_tasks/my_task_lerobot]'
```

```bash
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4 \
  'data.train.dataset_dirs=[./data/my_tasks/my_task_lerobot]' \
  'data.val.dataset_dirs=[./data/my_tasks/my_task_lerobot]' \
  data.train.pretrained_norm_stats=null \
  data.val.pretrained_norm_stats=null \
  output_dir=./runs/my_task_fastwam
```

需要注意：验证集在 `pretrained_norm_stats=null` 时可能无法自动获得 stats，具体取决于 `runtime.build_datasets` 如何把训练集 stats 传给 val。更稳妥的流程是先用训练集计算并保存 stats，再在后续训练/评测中显式指定 stats 文件。

---

## 8. 常用复现命令

### 8.1 预处理 Action DiT

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

### 8.2 预计算 RoboTwin 文本 embedding

```bash
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
```

### 8.3 训练 Fast-WAM

```bash
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4
```

### 8.4 训练 Joint / IDM 对照

```bash
bash scripts/train_zero1.sh 8 task=robotwin_joint_3cam_384_1e-4
```

```bash
bash scripts/train_zero1.sh 8 task=robotwin_idm_3cam_384_1e-4
```

### 8.5 RoboTwin 在线评测

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./runs/robotwin_uncond_3cam_384_1e-4/<run_id>/checkpoints/weights/step_XXXXXX.pt \
  EVALUATION.dataset_stats_path=./data/robotwin2.0/dataset_stats.json \
  MULTIRUN.num_gpus=8
```

只评测单个任务：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./runs/robotwin_uncond_3cam_384_1e-4/<run_id>/checkpoints/weights/step_XXXXXX.pt \
  EVALUATION.dataset_stats_path=./data/robotwin2.0/dataset_stats.json \
  EVALUATION.task_name=<robotwin_task_name> \
  MULTIRUN.num_gpus=1
```

---

## 9. 复现检查清单

训练前确认：

- `data/robotwin2.0/robotwin2.0/{data,meta,videos}` 已存在。
- `data/robotwin2.0/dataset_stats.json` 已存在，或准备让训练集重新计算。
- `data/text_embeds_cache/robotwin/` 中已生成 T5 embedding 缓存。
- `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` 已存在。
- `DIFFSYNTH_MODEL_BASE_PATH` 指向 Wan2.2 权重所在目录。

评测前确认：

- checkpoint 路径存在。
- `EVALUATION.dataset_stats_path` 与训练使用的归一化统计一致。
- `third_party/RoboTwin/task_config/` 已补齐。
- `third_party/RoboTwin/policy/fastwam_policy` 软链接可由 `eval_robotwin_single.py` 自动创建。
- 当前环境能正常启动 RoboTwin/SAPIEN 仿真。

---

## 10. 总结

FastWAM 在 RoboTwin 上的代码链路可以概括为：

```text
LeRobot RoboTwin 数据
  -> BaseLerobotDataset episode/window 采样
  -> FastWAMProcessor 图像/action/state/instruction 处理
  -> RobotVideoDataset 三相机拼接 + 文本 embedding 缓存
  -> FastWAM.training_loss
      - 视频 latent flow loss
      - action flow loss
      - MoT 受限混合 attention
  -> Wan22Trainer 分布式训练与 checkpoint
  -> RoboTwin policy adapter
      - 当前三相机 observation 拼接
      - 当前 proprio 编码
      - 在线 instruction 编码
      - FastWAM.infer_action 使用 video K/V cache 生成 action chunk
  -> RoboTwin env take_action(qpos)
  -> success rate 汇总
```

从代码看，Fast-WAM 的核心工程取舍是：训练时保留视频预测作为辅助世界建模任务；推理时基础 Fast-WAM 只编码当前帧并缓存 video K/V，然后仅对 action token 做多步去噪，从而避免测试时生成未来视频。
