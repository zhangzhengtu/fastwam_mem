# FastWAM Release 权重在 RoboTwin 中的推理 / 评测接口说明

本文档记录当前仓库在 **RoboTwin assets 已准备、FastWAM release 权重已下载** 之后，如何继续完成 RoboTwin 推理、rollout 和 success rate 评测。

仓库根目录：

```text
/nav-oss/yangganlin/tzz_workspace/FastWAM
```

当前已确认存在的关键文件：

```text
checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
checkpoints/Wan-AI/Wan2.2-TI2V-5B/
checkpoints/Wan-AI/Wan2.1-T2V-1.3B/
checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/
third_party/RoboTwin/assets/
third_party/RoboTwin/task_config/_eval_step_limit.yml
third_party/RoboTwin/task_config/demo_clean.yml
third_party/RoboTwin/task_config/demo_randomized.yml
third_party/RoboTwin/task_config/_camera_config.yml
third_party/RoboTwin/task_config/_embodiment_config.yml
```

当前 `third_party/RoboTwin/assets/` 和 `third_party/RoboTwin/task_config/` 都已经在代码期望的位置。RoboTwin 的评测脚本除了 `assets/` 之外，还会读取 `task_config/` 下的任务列表、clean/random 配置、相机配置和 embodiment 配置。

---

## 1. 推理链路总览

FastWAM release 权重在 RoboTwin 中的在线评测链路是：

```text
run_robotwin_manager.py
  -> eval_robotwin_single.py
    -> third_party/RoboTwin/script/eval_policy.py
      -> policy/fastwam_policy/deploy_policy.py
        -> WorldActionRobotWinPolicy
          -> FastWAM.infer_action(...)
          -> task_env.take_action(action, action_type="qpos")
```

主要代码位置：

| 路径 | 作用 |
| --- | --- |
| `experiments/robotwin/run_robotwin_manager.py` | 多任务、多 GPU 评测 manager。全量评测推荐入口。 |
| `experiments/robotwin/eval_robotwin_single.py` | 单任务评测 wrapper。把 FastWAM policy 目录复制到 RoboTwin policy 路径，转发 checkpoint、stats 和评测参数到 RoboTwin 官方脚本。 |
| `experiments/robotwin/fastwam_policy/deploy_policy.py` | FastWAM policy adapter。把 RoboTwin observation 转成 FastWAM 输入，调用 `infer_action`，反归一化 action 并执行。 |
| `experiments/robotwin/fastwam_policy/deploy_policy.yml` | RoboTwin policy 默认参数。实际多数参数会被 `sim_robotwin.yaml` 和命令行 override 覆盖。 |
| `third_party/RoboTwin/script/eval_policy.py` | RoboTwin 官方 rollout loop。初始化环境、生成 instruction、循环执行 policy、统计成功率。 |
| `configs/sim_robotwin.yaml` | FastWAM RoboTwin 评测的 Hydra 配置入口。 |

release 权重路径：

```text
./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
```

release 权重对应的归一化统计：

```text
./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
```

推理时不需要离线 RoboTwin 训练数据目录，但必须提供与 checkpoint 匹配的 `dataset_stats.json`，因为 policy adapter 会用它反归一化 14 维 qpos action。

---

## 2. 推理前检查

建议所有命令都从仓库根目录执行：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
```

### 2.1 检查 release 权重和 stats

```bash
test -f ./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
test -f ./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
```

不要把 release 权重和 `./data/robotwin2.0/dataset_stats.json` 随意混用。推荐 release 权重始终配套使用：

```text
checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
```

### 2.2 检查 Wan / VAE / T5 基础权重

虽然评测加载的是 FastWAM release checkpoint，但模型实例化仍然需要 Wan 相关组件，例如 VAE、T5 tokenizer/text encoder 等。当前本地已看到：

```text
checkpoints/Wan-AI/Wan2.2-TI2V-5B/
checkpoints/Wan-AI/Wan2.1-T2V-1.3B/
checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/
```

建议设置：

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
```

如果不设置，底层加载逻辑可能会去默认路径或尝试按模型名解析，导致找不到本地权重。

### 2.3 检查 RoboTwin assets

你当前的 assets 路径：

```text
third_party/RoboTwin/assets
```

当前本地检查显示该目录非空，包含 objects、embodiments、background_texture 等资源。

快速检查：

```bash
test -d ./third_party/RoboTwin/assets
find ./third_party/RoboTwin/assets -maxdepth 2 -type d | head
```

### 2.4 检查 RoboTwin task_config

RoboTwin 在线评测还需要：

```text
third_party/RoboTwin/task_config/
├── _eval_step_limit.yml
├── _camera_config.yml
├── _embodiment_config.yml
├── demo_clean.yml
└── demo_randomized.yml
```

代码依赖位置：

- `experiments/robotwin/run_robotwin_manager.py` 读取 `third_party/RoboTwin/task_config/_eval_step_limit.yml` 作为全任务列表。
- `third_party/RoboTwin/script/eval_policy.py` 读取 `task_config/demo_clean.yml` 或 `task_config/demo_randomized.yml`。
- `third_party/RoboTwin/script/eval_policy.py` 还会读取 `_camera_config.yml` 和 `_embodiment_config.yml`。

检查命令：

```bash
test -f ./third_party/RoboTwin/task_config/_eval_step_limit.yml
test -f ./third_party/RoboTwin/task_config/demo_clean.yml
test -f ./third_party/RoboTwin/task_config/demo_randomized.yml
test -f ./third_party/RoboTwin/task_config/_camera_config.yml
test -f ./third_party/RoboTwin/task_config/_embodiment_config.yml
```

当前本地已确认这些关键文件存在。如果之后移动目录或切换分支导致文件不存在，即使 `assets/` 已经下载好了，评测也会失败。常见报错包括：

```text
Task list file not found: .../third_party/RoboTwin/task_config/_eval_step_limit.yml
task config file is missing
FileNotFoundError: ./task_config/demo_randomized.yml
```

### 2.5 检查 policy 目录复制

`eval_robotwin_single.py` 会自动把 FastWAM policy adapter 复制到 RoboTwin 的 policy 搜索路径：

```text
third_party/RoboTwin/policy/fastwam_policy
```

你的对象存储挂载不支持软链接，因此当前代码已经改成直接复制目录，不再执行 `ln -sfn`。手动复制命令是：

```bash
cp -r "$PWD/experiments/robotwin/fastwam_policy" \
  "$PWD/third_party/RoboTwin/policy/"
```

### 2.6 Python / Conda 环境应该怎么选

结论：**SAPIEN/RoboTwin 不一定必须装在 `/shared/smartbot/yangganlin/anaconda3/envs/fastwam`，但启动评测的那个 Python 环境必须同时能 import FastWAM 和 RoboTwin/SAPIEN。**

原因是当前代码不是两个进程分别使用两个环境的结构。实际启动链路里：

```text
run_robotwin_manager.py
  -> 使用 sys.executable 启动 eval_robotwin_single.py
    -> 再使用同一个 sys.executable 启动 third_party/RoboTwin/script/eval_policy.py
      -> eval_policy.py 在同一个 Python 进程里 import fastwam_policy
      -> fastwam_policy/deploy_policy.py 再 import fastwam、torch、hydra 等 FastWAM 依赖
      -> RoboTwin env 同时 import sapien、RoboTwin env/task 依赖
```

也就是说，**最终执行 `eval_policy.py` 的 Python 环境必须同时具备两边依赖**：

```text
FastWAM 侧：torch, hydra, omegaconf, transformers, safetensors, fastwam package ...
RoboTwin 侧：sapien, RoboTwin 仿真/渲染相关依赖 ...
```

你现在有两个环境：

```text
/shared/smartbot/yangganlin/anaconda3/envs/fastwam
/shared/smartbot/yangganlin/anaconda3/envs/RoboTwin
```

推荐优先使用下面两种方案之一。

#### 方案 A：在 RoboTwin 环境里补 FastWAM 依赖，使用 RoboTwin 环境启动评测

如果 `/shared/smartbot/yangganlin/anaconda3/envs/RoboTwin` 已经能正常跑 SAPIEN 渲染，这是更稳的方案，因为 SAPIEN/渲染依赖通常更挑环境。

```bash
source /shared/smartbot/yangganlin/anaconda3/bin/activate \
  /shared/smartbot/yangganlin/anaconda3/envs/RoboTwin

cd /nav-oss/yangganlin/tzz_workspace/FastWAM
python -m pip install -e .
```

然后检查这个环境是否同时能导入两侧依赖：

```bash
python - <<'PY'
import sys
print("python:", sys.executable)

import torch
import hydra
import omegaconf
import fastwam
print("fastwam side ok")

import sapien
print("sapien side ok")
PY
```

通过后，后续所有评测命令都在这个 `RoboTwin` 环境下执行。

#### 方案 B：在 fastwam 环境里补 RoboTwin/SAPIEN 依赖，使用 fastwam 环境启动评测

如果 `/shared/smartbot/yangganlin/anaconda3/envs/fastwam` 已经能稳定加载 FastWAM/Wan 权重，也可以把 RoboTwin/SAPIEN 依赖安装到这个环境里。

```bash
source /shared/smartbot/yangganlin/anaconda3/bin/activate \
  /shared/smartbot/yangganlin/anaconda3/envs/fastwam

cd /nav-oss/yangganlin/tzz_workspace/FastWAM
python -m pip install -e .
```

然后安装或确认 RoboTwin/SAPIEN 依赖，并检查：

```bash
python - <<'PY'
import sys
print("python:", sys.executable)

import fastwam
import sapien
print("FastWAM + SAPIEN ok")
PY
```

#### 不推荐：manager 用 fastwam 环境，RoboTwin 子进程用另一个环境

当前 `run_robotwin_manager.py` 和 `eval_robotwin_single.py` 都用 `sys.executable` 启动子进程，所以子进程会继承你当前激活的 Python。除非改代码显式指定另一个 Python，否则不会自动切到 `/shared/smartbot/yangganlin/anaconda3/envs/RoboTwin/bin/python`。

即使改成让 RoboTwin 子进程用 `RoboTwin` 环境，`eval_policy.py` 仍然会在该环境里 import `fastwam_policy/deploy_policy.py` 和 `fastwam` 包，因此 `RoboTwin` 环境还是必须装 FastWAM 依赖。

### 2.7 检查 SAPIEN / RoboTwin 渲染

```python
from test_render import Sapien_TEST
Sapien_TEST()
```

因此评测前需要保证 RoboTwin/SAPIEN 环境可用。若这里失败，通常不是 FastWAM 权重问题，而是仿真环境、渲染、显卡或依赖没有配好。

---

## 3. 推荐运行方式

### 3.1 最小 smoke test：单任务、少量 episode

建议先跑一个单任务、少量 episode，确认 checkpoint、stats、RoboTwin 环境和 policy adapter 都能串起来。

示例任务使用当前仓库中存在的 env 文件 `click_alarmclock.py`：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.eval_num_episodes=3 \
  MULTIRUN.num_gpus=1 \
  MULTIRUN.max_tasks_per_gpu=1
```

这个命令会通过 manager 依次跑：

```text
click_alarmclock + demo_clean
click_alarmclock + demo_randomized
```

如果只是想更快验证一条链路，也可以直接调用单任务 wrapper，只跑一个 phase。

只跑 randomized：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"

python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  gpu_id=0 \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.task_config=demo_randomized \
  EVALUATION.eval_num_episodes=3 \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
```

只跑 clean：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"

python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  gpu_id=0 \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.task_config=demo_clean \
  EVALUATION.eval_num_episodes=3 \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
```

### 3.2 正式全任务评测

全任务评测不传 `EVALUATION.task_name`，manager 会从：

```text
third_party/RoboTwin/task_config/_eval_step_limit.yml
```

读取全部 task name。

8 GPU 示例：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=2
```

每个任务会先跑 clean phase，再跑 randomized phase：

```text
demo_clean -> demo_randomized
```

如果显存或仿真资源紧张，可以降低并发：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=4 \
  MULTIRUN.max_tasks_per_gpu=1
```

### 3.3 使用 seen instruction

默认配置：

```yaml
EVALUATION:
  instruction_type: unseen
```

代码中会在 `third_party/RoboTwin/script/eval_policy.py` 里执行：

```python
instruction = np.random.choice(results[0][instruction_type])
TASK_ENV.set_instruction(instruction=instruction)
```

如果要评测 seen instruction：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.instruction_type=seen \
  MULTIRUN.num_gpus=8
```

README 中也说明官方默认按 unseen instruction 评测；seen instruction 往往会略高一些。

### 3.4 保存更连贯的评测视频

默认配置：

```yaml
EVALUATION:
  skip_get_obs_within_replan: true
```

含义：在一个 action chunk 的连续执行窗口内，只有需要重新规划时才重新渲染 RGB observation。这样更快，但保存出来的视频会像低帧率。

如果你想保存更完整的视频：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.skip_get_obs_within_replan=false \
  MULTIRUN.num_gpus=1 \
  MULTIRUN.max_tasks_per_gpu=1
```

注意：是否真正保存视频还取决于 `task_config/demo_clean.yml` 或 `demo_randomized.yml` 中的 `eval_video_log` 设置。

---

## 4. 关键配置项解释

配置入口：`configs/sim_robotwin.yaml`

```yaml
ckpt: null
gpu_id: 0

model:
  load_text_encoder: true
  skip_dit_load_from_pretrain: true
  action_dit_pretrained_path: null

EVALUATION:
  robotwin_root: third_party/RoboTwin
  policy_name: fastwam_policy
  task_name: null
  task_config: demo_randomized
  instruction_type: unseen
  eval_num_episodes: 100
  action_horizon: null
  replan_steps: 24
  num_inference_steps: ${eval_num_inference_steps}
  sigma_shift: null
  text_cfg_scale: 1.0
  negative_prompt: ""
  rand_device: cpu
  tiled: false
  timing_enabled: false
  skip_get_obs_within_replan: true
  dataset_stats_path: null
  device: cuda

MULTIRUN:
  enabled: false
  num_gpus: 8
  max_tasks_per_gpu: 2
```

常用字段：

| 字段 | 作用 |
| --- | --- |
| `ckpt` | FastWAM checkpoint 路径。release 使用 `./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt`。 |
| `EVALUATION.dataset_stats_path` | action/state 归一化统计。release 权重必须使用对应 release stats。 |
| `EVALUATION.task_name` | 单任务评测时指定；为 `null` 时 manager 从 `_eval_step_limit.yml` 读取全任务。 |
| `EVALUATION.task_config` | 单任务 wrapper 使用；manager 会自动对每个任务跑 `demo_clean` 和 `demo_randomized`。 |
| `EVALUATION.instruction_type` | `seen` 或 `unseen`。默认 `unseen`。 |
| `EVALUATION.eval_num_episodes` | 每个 phase 的 episode 数。默认 100，smoke test 可设为 1-3。 |
| `EVALUATION.action_horizon` | 模型一次生成的 action chunk 长度。默认 `null`，代码使用训练配置 `num_frames - 1 = 32`。 |
| `EVALUATION.replan_steps` | 每次重新规划后实际执行多少个动作。默认 24，会被 clamp 到 `[1, action_horizon]`。 |
| `EVALUATION.num_inference_steps` | action diffusion/flow denoising 步数。默认 10。 |
| `EVALUATION.skip_get_obs_within_replan` | 是否在 action queue 未空时跳过 RGB observation 渲染。默认 true，加速但视频低帧率。 |
| `MULTIRUN.num_gpus` | manager 使用的 GPU 数。 |
| `MULTIRUN.max_tasks_per_gpu` | 每张 GPU 上最多同时跑几个任务子进程。 |

推理侧模型覆盖：

```yaml
model:
  load_text_encoder: true
  skip_dit_load_from_pretrain: true
  action_dit_pretrained_path: null
```

含义：

- `load_text_encoder=true`：在线 RoboTwin instruction 需要即时编码文本。
- `skip_dit_load_from_pretrain=true`：不再用 Wan 预训练权重初始化 DiT，而是从 release checkpoint 加载训练后的 MoT/action 参数。
- `action_dit_pretrained_path=null`：推理时不再需要 Action DiT 初始化权重参与加载；不过本地基础组件仍需要可访问。

---

## 5. 推理时 observation / action 接口

FastWAM 的 RoboTwin adapter 在：

```text
experiments/robotwin/fastwam_policy/deploy_policy.py
```

### 5.1 RoboTwin observation 输入

RoboTwin 环境在 `third_party/RoboTwin/envs/_base_task.py` 中提供 `get_obs()`，policy adapter 使用这些字段：

```python
observation["observation"]["head_camera"]["rgb"]
observation["observation"]["left_camera"]["rgb"]
observation["observation"]["right_camera"]["rgb"]
observation["joint_action"]["vector"]
```

其中：

- `head_camera.rgb`：头部相机 RGB。
- `left_camera.rgb`：左腕相机 RGB。
- `right_camera.rgb`：右腕相机 RGB。
- `joint_action.vector`：14 维当前 qpos/state。

### 5.2 三相机拼接

policy adapter 的 `_build_robotwin_image_tensor` 与训练时保持一致：

```text
head_camera  -> resize 到 320 x 256，放上方
left_camera  -> resize 到 160 x 128，放下方左侧
right_camera -> resize 到 160 x 128，放下方右侧

最终图像：384 x 320 x 3
```

然后转为：

```text
input_image: [1, 3, 384, 320]
range: [-1, 1]
dtype: bf16 by default
device: cuda by default
```

### 5.3 proprio/state 归一化

adapter 从：

```python
observation["joint_action"]["vector"]
```

取 14 维 state，经训练时同一套 processor 和 normalizer 归一化：

```python
processor.set_normalizer_from_stats(dataset_stats)
```

这里的 `dataset_stats` 就是命令中的：

```text
EVALUATION.dataset_stats_path
```

### 5.4 language instruction

RoboTwin eval loop 会生成并设置 instruction：

```python
TASK_ENV.set_instruction(instruction=instruction)
```

policy adapter 每次重新规划时读取：

```python
instruction = task_env.get_instruction()
prompt = DEFAULT_PROMPT.format(task=instruction)
```

prompt 模板来自训练数据集代码：

```python
"A video recorded from a robot's point of view executing the following instruction: {task}"
```

### 5.5 FastWAM action 推理

adapter 调用：

```python
pred = self.model.infer_action(
    prompt=prompt,
    input_image=image_tensor,
    action_horizon=self.action_horizon,
    proprio=proprio,
    num_inference_steps=self.num_inference_steps,
    sigma_shift=self.sigma_shift,
    seed=self.seed,
    rand_device=self.rand_device,
    tiled=self.tiled,
)
```

基础 Fast-WAM 的 `infer_action` 在：

```text
src/fastwam/models/wan22/fastwam.py
```

其推理逻辑是：

```text
当前图像
  -> VAE encode 当前帧 latent
  -> Video DiT 单次 prefill
  -> 每层保存 video K/V cache
  -> Action DiT 迭代去噪 10 steps
  -> 输出 action chunk
```

这里不会生成未来视频。未来视频 token 在基础 Fast-WAM 推理时不存在。

默认 action 输出：

```text
action: [32, 14]
```

### 5.6 action 反归一化与执行

模型输出是归一化 action。adapter 使用 release stats 反归一化：

```python
action_chunk = self._denormalize_action(action_tensor)[0]
```

然后只将前 `replan_steps` 个动作放入队列：

```python
n_exec = min(self.replan_steps, action_chunk.shape[0])
pending_actions.append(action_chunk[i])
```

每个 simulator step 执行一个动作：

```python
task_env.take_action(action, action_type="qpos")
```

RoboTwin base task 会把 14 维 qpos 拆为：

```text
left_arm:       6
left_gripper:  1
right_arm:      6
right_gripper: 1
```

---

## 6. episode、终止条件和 success rate

rollout loop 在：

```text
third_party/RoboTwin/script/eval_policy.py
```

每个 episode 的基本过程：

1. `TASK_ENV.setup_demo(...)` 初始化环境。
2. expert 先执行一次可行性检查，过滤不稳定 seed。
3. `generate_episode_descriptions(...)` 生成 seen/unseen instruction。
4. `TASK_ENV.set_instruction(...)` 设置语言指令。
5. 调用 `reset_model(model)` 清空 FastWAM action queue。
6. 在 `TASK_ENV.take_action_cnt < TASK_ENV.step_lim` 内循环：
   - 如果需要 observation，调用 `TASK_ENV.get_obs()`。
   - 调用 `fastwam_policy.eval(TASK_ENV, model, observation)`。
   - policy 内部执行 `task_env.take_action(...)`。
   - 如果 `TASK_ENV.eval_success=True`，episode 成功并提前结束。
7. 记录 success/fail。

终止条件：

```text
TASK_ENV.take_action_cnt >= TASK_ENV.step_lim
或
TASK_ENV.eval_success == True
```

`step_lim` 通常来自：

```text
third_party/RoboTwin/task_config/_eval_step_limit.yml
```

成功率：

```text
success_rate = TASK_ENV.suc / TASK_ENV.test_num
```

结果文件：

```text
_result_clean.txt
_result_random.txt
```

manager 会解析这些文件中的最后一个浮点数作为 success rate。

---

## 7. 输出目录

### 7.1 manager 输出

使用 release checkpoint 时，`ckpt_tag` 是 checkpoint 文件名 stem：

```text
robotwin_uncond_3cam_384
```

manager 输出目录：

```text
evaluate_results/robotwin/robotwin_uncond_3cam_384/<run_timestamp>/
```

典型文件：

```text
evaluate_results/robotwin/robotwin_uncond_3cam_384/<run_timestamp>/
├── manager.log
├── summary.csv
├── summary.json
├── failed_tasks.txt
└── <task_name>/
    ├── _result_clean.txt
    ├── _result_random.txt
    └── ...
```

`summary.csv` 里会有：

```text
task,clean,random,mean
```

`summary.json` 记录同样的聚合结果。

### 7.2 单任务 wrapper 输出

`eval_robotwin_single.py` 最终也会把输出整理到：

```text
evaluate_results/robotwin/<ckpt_tag>/<run_timestamp>/<task_name>/
```

并保存单任务 log：

```text
eval_<task_name>_<timestamp>.log
```

---

## 8. 常见问题和排查

### 8.1 `Task list file not found`

报错形态：

```text
Task list file not found:
.../third_party/RoboTwin/task_config/_eval_step_limit.yml
```

原因：全任务 manager 需要从 `_eval_step_limit.yml` 读取任务列表，但当前 `task_config/` 不存在或路径不对。

处理：

```bash
ls ./third_party/RoboTwin/task_config
```

确认至少存在：

```text
_eval_step_limit.yml
demo_clean.yml
demo_randomized.yml
_camera_config.yml
_embodiment_config.yml
```

### 8.2 `task config file is missing`

报错来自：

```text
third_party/RoboTwin/script/eval_policy.py
```

通常是缺：

```text
third_party/RoboTwin/task_config/_camera_config.yml
```

或当前工作目录不在 `third_party/RoboTwin`。正常通过 `eval_robotwin_single.py` 启动时，代码会把 cwd 设置为 RoboTwin root；如果手动直接进 `third_party/RoboTwin` 跑，要注意相对路径。

### 8.3 `Dataset stats path not found`

release 权重请使用：

```text
./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
```

不要省略：

```text
EVALUATION.dataset_stats_path=...
```

`deploy_policy.py` 中 `_resolve_dataset_stats_path` 明确要求这个字段存在。

### 8.4 `Checkpoint not found`

检查：

```bash
ls -lh ./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
```

命令中可以用相对路径，`eval_robotwin_single.py` 会以项目根目录解析；也可以直接传绝对路径。

### 8.5 `No Task`

报错来自：

```python
importlib.import_module(f"envs.{task_name}")
getattr(envs_module, task_name)
```

说明 `EVALUATION.task_name` 和 `third_party/RoboTwin/envs/<task_name>.py` 对不上。

当前仓库可见的一些 env 名称包括：

```text
click_alarmclock
turn_switch
place_a2b_left
place_a2b_right
open_laptop
open_microwave
stack_blocks_two
stack_blocks_three
```

全任务评测应以 `_eval_step_limit.yml` 的 key 为准。

### 8.6 显存不足或并发过高

降低并发：

```bash
MULTIRUN.num_gpus=1
MULTIRUN.max_tasks_per_gpu=1
```

也可以先降低 episode 数做 smoke test：

```bash
EVALUATION.eval_num_episodes=1
```

### 8.7 视频保存低帧率

默认：

```yaml
EVALUATION.skip_get_obs_within_replan: true
```

这会加速评测，但 action chunk 内部不每步重新渲染 observation，因此保存视频低帧率。需要完整视频时设置：

```bash
EVALUATION.skip_get_obs_within_replan=false
```

如果日志里出现：

```text
Error writing trailer: Invalid argument
Error closing file: Invalid argument
```

并且生成的 `.mp4` 无法打开，通常不是 rollout 失败，而是 ffmpeg 直接向 `/nav-oss` 这类对象存储挂载写 MP4 时，收尾写 moov/trailer 所需的文件操作不被完整支持。当前代码已改为先把 ffmpeg 输出写到本机临时目录：

```text
/tmp/fastwam_robotwin_eval_video/
```

ffmpeg 正常关闭后，再把完整 mp4 文件复制到 `evaluate_results/`。已经损坏的旧 mp4 一般无法补救，需要重新跑对应 episode 才会生成可打开的视频。

### 8.8 SAPIEN / rendering 失败

`eval_policy.py` 启动时会先跑 `Sapien_TEST()`。如果这里失败，优先检查：

- CUDA/driver 是否可用。
- SAPIEN 环境变量是否正确。
- headless rendering 是否配置好。
- RoboTwin 官方依赖是否装完整。
- assets 和 task_config 路径是否完整。

### 8.9 Python 环境只装了一半依赖

如果使用 `/shared/smartbot/yangganlin/anaconda3/envs/RoboTwin` 启动，但没有补 FastWAM 依赖，常见报错是：

```text
ModuleNotFoundError: No module named 'hydra'
ModuleNotFoundError: No module named 'omegaconf'
ModuleNotFoundError: No module named 'fastwam'
ModuleNotFoundError: No module named 'transformers'
```

处理方式是在当前激活的 `RoboTwin` 环境里安装 FastWAM：

```bash
source /shared/smartbot/yangganlin/anaconda3/bin/activate \
  /shared/smartbot/yangganlin/anaconda3/envs/RoboTwin

cd /nav-oss/yangganlin/tzz_workspace/FastWAM
python -m pip install -e .
```

如果使用 `/shared/smartbot/yangganlin/anaconda3/envs/fastwam` 启动，但没有补 RoboTwin/SAPIEN 依赖，常见报错是：

```text
ModuleNotFoundError: No module named 'sapien'
Sapien_TEST failed
```

处理方式是在 `fastwam` 环境里补齐 RoboTwin 官方依赖，或者直接改用已经能跑 SAPIEN 的 `RoboTwin` 环境，并在其中安装 FastWAM 依赖。

---

## 9. 一套推荐执行顺序

从你当前状态继续，建议按下面顺序跑：

```bash
source /shared/smartbot/yangganlin/anaconda3/bin/activate \
  /shared/smartbot/yangganlin/anaconda3/envs/RoboTwin

cd /nav-oss/yangganlin/tzz_workspace/FastWAM
python -m pip install -e .
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
```

这里默认选择 `RoboTwin` 环境作为统一运行环境，因为你已经有单独的 RoboTwin 环境。如果你决定使用 `fastwam` 环境，则把第一行换成：

```bash
source /shared/smartbot/yangganlin/anaconda3/bin/activate \
  /shared/smartbot/yangganlin/anaconda3/envs/fastwam
```

但无论选择哪个环境，都要确认它同时能 import FastWAM 和 SAPIEN：

```bash
python - <<'PY'
import sys
print("python:", sys.executable)
import fastwam
import hydra
import torch
import sapien
print("FastWAM + RoboTwin/SAPIEN environment ok")
PY
```

检查文件：

```bash
test -f ./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
test -f ./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
test -d ./third_party/RoboTwin/assets
test -f ./third_party/RoboTwin/task_config/_eval_step_limit.yml
test -f ./third_party/RoboTwin/task_config/demo_clean.yml
test -f ./third_party/RoboTwin/task_config/demo_randomized.yml
test -f ./third_party/RoboTwin/task_config/_camera_config.yml
test -f ./third_party/RoboTwin/task_config/_embodiment_config.yml
```

复制 policy 目录：

```bash
cp -r "$PWD/experiments/robotwin/fastwam_policy" \
  "$PWD/third_party/RoboTwin/policy/"
```

先跑单任务 smoke test：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.eval_num_episodes=3 \
  MULTIRUN.num_gpus=1 \
  MULTIRUN.max_tasks_per_gpu=1
```

smoke test 通过后跑全任务：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=2
```

如果要看完整视频，把最后一个命令加上：

```bash
EVALUATION.skip_get_obs_within_replan=false
```

如果要评测 seen instruction，再加：

```bash
EVALUATION.instruction_type=seen
```

---

## 10. 最小接口总结

release RoboTwin 推理最核心的入口命令是。执行前先激活一个同时包含 FastWAM 和 RoboTwin/SAPIEN 依赖的环境：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

FastWAM policy 实际接收的 observation 是：

```text
head_camera.rgb
left_camera.rgb
right_camera.rgb
joint_action.vector
instruction
```

模型一次输出：

```text
32 x 14 action chunk
```

评测默认执行：

```text
每次重规划执行 24 步
每次 action denoising 10 steps
instruction_type = unseen
clean + randomized 两阶段
```

最终结果看：

```text
evaluate_results/robotwin/robotwin_uncond_3cam_384/<run_timestamp>/summary.csv
evaluate_results/robotwin/robotwin_uncond_3cam_384/<run_timestamp>/summary.json
```
