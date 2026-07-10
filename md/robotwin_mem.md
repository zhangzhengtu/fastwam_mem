# RoboTwin-Mem cover_blocks_hard 推理实现方案

本文档记录 `RoboTwin-Mem` 推理实现方案和当前落地状态。已在 FastWAM 里新增：

```text
experiments/robotwin_mem/
```

并让下面这套已训练权重可以在 `third_party/RoboTwin-Mem` 的 `cover_blocks_hard` 任务上完整跑推理：

```text
/nav-oss/yangganlin/tzz_workspace/FastWAM/runs/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4/checkpoints/weights/step_003000.pt
/nav-oss/yangganlin/tzz_workspace/FastWAM/runs/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4/dataset_stats.json
```

## 1. 已确认的现状

### 1.1 训练权重对应的 FastWAM 配置

该 run 的 `config.yaml` 显示：

- `task=robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4`
- `model._target_=fastwam.runtime.create_fastwam_idm`
- `data.task_name=cover_blocks_hard`
- 三相机输入：
  - `cam_high`
  - `cam_left_wrist`
  - `cam_right_wrist`
- 输入拼接模式：`concat_multi_camera: robotwin`
- 单帧拼接后视频尺寸：`384 x 320`
- `num_frames: 33`
- `action_video_freq_ratio: 4`
- 推理时 `num_video_frames=(33 - 1) / 4 + 1 = 9`
- `action_horizon` 默认应为 `num_frames - 1 = 32`
- `state/action` 都是 14 维，对应 aloha-agilex 双臂 qpos
- 训练时重新计算了该任务自己的 stats，所以推理必须使用同一目录下的 `dataset_stats.json`

注意：当前 run 目录下实际找到的权重是：

```text
runs/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4/checkpoints/weights/step_003000.pt
```

虽然 run config 里有 `max_steps: 6000`、`save_every: 3000`，目前可用权重以 `step_003000.pt` 为准。

### 1.2 现有 `experiments/robotwin` 推理链路

现有链路是：

```text
experiments/robotwin/eval_robotwin_single.py
  -> copy experiments/robotwin/fastwam_policy 到 third_party/RoboTwin/policy/fastwam_policy
  -> 在 third_party/RoboTwin 下执行 script/eval_policy.py
  -> script/eval_policy.py import policy/fastwam_policy/deploy_policy.py
  -> deploy_policy.py 加载 FastWAM checkpoint + dataset_stats
  -> 每次 replan 从 RoboTwin observation 拼接三相机图像、归一化 qpos、infer_action、反归一化 action
  -> TASK_ENV.take_action(action, action_type="qpos")
```

现有 `fastwam_policy/deploy_policy.py` 已经具备这次需要的核心能力：

- 从 `configs/sim_robotwin.yaml` + `task=<hydra task>` 组合出模型和 processor 配置
- `model.load_checkpoint(checkpoint_path)`
- `processor.set_normalizer_from_stats(dataset_stats)`
- 将 `observation["observation"]["head_camera"]["rgb"]` resize 到 `320x256`
- 将左右腕相机 resize 到 `160x128` 后横向拼接，再接到 head 下方，得到 `[384, 320, 3]`
- 将像素转成 `[-1, 1]` 的 `[1, 3, 384, 320]`
- 从 `observation["joint_action"]["vector"]` 取 14 维 proprio
- `DEFAULT_PROMPT.format(task=instruction)`
- 调 `model.infer_action(..., action_horizon, proprio, num_video_frames=9, ...)`
- action 反归一化后按 `replan_steps` 入队执行

因此 `robotwin_mem` 的 policy adapter 可以先复制该实现，只做命名和少量健壮性改动。

### 1.3 `RoboTwin-Mem` 与原 `RoboTwin` 的关键差异

`third_party/RoboTwin-Mem/README.md` 写明：

```text
In RoboTwin-Mem, we always use demo_clean setting.
```

当前 `third_party/RoboTwin-Mem/task_config/` 只有：

```text
demo_clean.yml
_eval_step_limit.yml
_camera_config.yml
_embodiment_config.yml
```

所以不能照搬原 `run_robotwin_manager.py` 的 clean/random 两阶段评测。`robotwin_mem` 首版应只跑 `demo_clean`。

`RoboTwin-Mem` 的 `script/eval_policy.py` 当前也还没同步原 `RoboTwin/script/eval_policy.py` 里已有的 FastWAM 适配：

- `eval_num_episodes` 固定为 100，不能从 wrapper 覆盖
- 结果固定写到 bench 内部 `eval_result/.../_result.txt`
- 不支持 `eval_output_dir`
- 不支持 `_result_clean.txt/_result_random.txt` 这种 manager 可解析命名
- 不支持 `skip_get_obs_within_replan`
- 结果文件格式是 `Success Rate: <value>` 和 `Reward: <value>`，不是纯浮点行

同时，`RoboTwin-Mem` 的环境接口和原 RoboTwin 基本兼容：

- `TASK_ENV.get_obs()` 返回 `observation` 和 `joint_action["vector"]`
- 双臂时 `joint_action["vector"] = left_jointstate + right_jointstate`
- `TASK_ENV.set_instruction()` / `get_instruction()` 可用
- `TASK_ENV.take_action(action, action_type="qpos")` 可用
- `cover_blocks_hard` 的 step limit 是 `1700`

## 2. 推荐实现范围

新增目录：

```text
experiments/robotwin_mem/
├── eval_robotwin_mem_single.py
├── run_robotwin_mem_manager.py
└── fastwam_policy/
    ├── __init__.py
    ├── deploy_policy.py
    └── deploy_policy.yml
```

已对 `third_party/RoboTwin-Mem/script/eval_policy.py` 做小型兼容补丁。理由是：只在 wrapper 里解析原生 `eval_result` 可以凑合跑，但无法干净控制 episode 数、输出目录、视频目录和 manager 汇总；而原 `third_party/RoboTwin/script/eval_policy.py` 已经有这套补丁，直接把同等能力移植到 Mem 版本最稳。

如果后面希望严格不改 `third_party/RoboTwin-Mem`，备选方案是让 `eval_robotwin_mem_single.py` 从 stdout 里解析 `Data has been saved to .../_result.txt`，再复制和转换结果到 `evaluate_results/robotwin_mem/...`。这个方案侵入更小，但无法覆盖 `eval_num_episodes`，不建议作为最终版。

## 3. 文件级方案

### 3.1 `experiments/robotwin_mem/eval_robotwin_mem_single.py`

以 `experiments/robotwin/eval_robotwin_single.py` 为蓝本，主要改动：

1. 常量改为：

```python
POLICY_NAME = "fastwam_policy"
DEFAULT_BENCH_ROOT = PROJECT_ROOT / "third_party" / "RoboTwin-Mem"
```

2. policy source 改为：

```text
PROJECT_ROOT / "experiments" / "robotwin_mem" / "fastwam_policy"
```

3. 默认 bench root 使用 `third_party/RoboTwin-Mem`。仍允许用户用 `EVALUATION.robotwin_root=...` 覆盖。

4. 输出目录改为：

```text
evaluate_results/robotwin_mem/<ckpt_tag>/<run_ts>/<task_name>/
```

5. `ckpt_tag` 需要比原版更稳。当前权重路径是：

```text
runs/<run_name>/checkpoints/weights/step_003000.pt
```

建议 tag 生成规则：

- 如果路径位于 `runs/<run_name>/checkpoints/weights/<step>.pt`，tag 用 `<run_name>_<step>`
- 否则沿用 `ckpt_path.stem`

这样当前 tag 会是：

```text
robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4_step_003000
```

6. `dataset_stats_path` 解析仍按：

- 优先 `EVALUATION.dataset_stats_path`
- 然后从 checkpoint 的父级向上找 `dataset_stats.json`

当前 checkpoint 在 `checkpoints/weights` 下，向上 3-4 级能找到 run 根目录的 `dataset_stats.json`。

7. 传给 `RoboTwin-Mem/script/eval_policy.py` 的 overrides 至少包括：

```text
task_name=cover_blocks_hard
task_config=demo_clean
ckpt_setting=<step_003000.pt>
policy_name=fastwam_policy
instruction_type=unseen
eval_num_episodes=<N>
sim_cfg_path=<FastWAM/configs/sim_robotwin.yaml>
sim_task=robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4
eval_output_dir=<evaluate_results/.../cover_blocks_hard>
mixed_precision=bf16
device=cuda
dataset_stats_path=<run/dataset_stats.json>
action_horizon=null  # deploy_policy 内部回落到 32
replan_steps=24
num_inference_steps=10
sigma_shift=null
text_cfg_scale=1.0
negative_prompt=""
rand_device=cpu
tiled=false
timing_enabled=false
skip_get_obs_within_replan=false
```

8. 启动命令保持 bench 兼容：

```python
cmd = [
    sys.executable,
    "-u",
    "script/eval_policy.py",
    "--config",
    "policy/fastwam_policy/deploy_policy.yml",
    "--overrides",
    *overrides,
]
cwd = third_party/RoboTwin-Mem
```

### 3.2 `experiments/robotwin_mem/fastwam_policy/deploy_policy.py`

首版可以复制 `experiments/robotwin/fastwam_policy/deploy_policy.py`，建议做这些小调整：

1. 类名改为 `WorldActionRobotWinMemPolicy`，日志中区分 Mem。

2. `_build_robotwin_image_tensor` 保持训练一致的三相机拼接：

```text
head_camera rgb -> 320x256
left_camera rgb -> 160x128
right_camera rgb -> 160x128
bottom = concat(left, right) -> 320x128
image = concat(head, bottom) -> 320x384
```

3. 对 Mem 的 observation 做更明确的错误提示：

- 缺 `head_camera` / `left_camera` / `right_camera` 时直接报错
- `joint_action["vector"]` 不是 14 维时报错，提示该权重只支持 aloha-agilex 双臂 14 维

4. `get_model()` 继续通过 `sim_cfg_path + sim_task` 组合 FastWAM 配置。当前必须传：

```text
sim_task=robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4
```

否则 `configs/sim_robotwin.yaml` 默认会落到 `robotwin_uncond_3cam_384_1e-4`，模型类型会不对。

5. `action_horizon` 默认逻辑保留：

```python
action_horizon = cfg.data.train.num_frames - 1  # 32
```

6. `num_video_frames` 保留：

```python
(cfg.data.train.num_frames - 1) // cfg.data.train.action_video_freq_ratio + 1  # 9
```

7. `eval()` / `reset_model()` 保持 RoboTwin policy 协议不变：

```python
def eval(TASK_ENV, model, observation):
    model.step(TASK_ENV, observation)

def reset_model(model):
    model.reset()
```

### 3.3 `experiments/robotwin_mem/fastwam_policy/deploy_policy.yml`

可以复制原 `deploy_policy.yml`，但默认值改成 Mem 当前目标：

```yaml
policy_name: fastwam_policy
task_name: cover_blocks_hard
task_config: demo_clean
ckpt_setting: null
seed: 0
instruction_type: unseen
eval_num_episodes: 100

sim_cfg_path: null
sim_task: robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4
sim_cfg_name: sim_robotwin.yaml
mixed_precision: bf16
device: cuda
dataset_stats_path: null
action_horizon: null
replan_steps: 24
num_inference_steps: 10
sigma_shift: null
text_cfg_scale: 1.0
negative_prompt: ""
rand_device: cpu
tiled: false
timing_enabled: false
skip_get_obs_within_replan: false
```

### 3.4 `experiments/robotwin_mem/run_robotwin_mem_manager.py`

以原 `run_robotwin_manager.py` 为蓝本，但首版只需要单阶段：

- `SINGLE_ENTRY = experiments/robotwin_mem/eval_robotwin_mem_single.py`
- `EVAL_STEP_LIMIT_FILE = third_party/RoboTwin-Mem/task_config/_eval_step_limit.yml`
- 默认任务：
  - 如果 `EVALUATION.task_name` 非空，只跑该任务
  - 如果为空，可以从 `_eval_step_limit.yml` 读取所有 Mem 任务，但本次建议默认命令显式传 `cover_blocks_hard`
- `phase_to_task_config` 不再需要 clean/random map，固定 `task_config=demo_clean`
- summary 字段改为：

```text
task_name,demo_clean_success_rate,reward
```

当前 `cover_blocks_hard` 先只需要 success rate；reward 可以后续从 `_result.txt` 解析，也可以先留空。

结果解析支持两种格式：

1. 推荐补丁后的纯浮点行：

```text
0.42
```

2. Mem 原生格式：

```text
Success Rate: 0.42
Reward: 0.73
```

这样即使 `third_party/RoboTwin-Mem/script/eval_policy.py` 没完全补齐，manager 也能更耐用。

### 3.5 `third_party/RoboTwin-Mem/script/eval_policy.py` 兼容补丁

建议把当前 `third_party/RoboTwin/script/eval_policy.py` 里已有的 FastWAM 适配同步到 Mem 版本，但保留 Mem 的 reward 统计：

1. 增加参数解析：

```python
skip_get_obs_within_replan = parse_bool(usr_args.get("skip_get_obs_within_replan", False))
eval_num_episodes = int(usr_args.get("eval_num_episodes", 100))
eval_output_dir = usr_args.get("eval_output_dir")
```

2. `test_num = eval_num_episodes`，不再固定 100。

3. 如果传了 `eval_output_dir`，结果和视频都写到该目录。

4. result 文件建议命名：

```text
_result_demo_clean.txt
```

同时为了兼容简单脚本，可以额外写一份：

```text
_result.txt
```

5. result 内容首行写纯 success rate，后面再写 reward：

```text
0.42
Reward: 0.73
```

这样 manager 可以直接按最后/第一条浮点解析。

6. 推理循环支持 action queue 时跳过不必要的 `get_obs()`：

```python
need_obs = True
if skip_get_obs_within_replan and hasattr(model, "should_request_observation"):
    need_obs = bool(model.should_request_observation())
observation = TASK_ENV.get_obs() if need_obs else None
eval_func(TASK_ENV, model, observation)
```

7. 视频尺寸逻辑建议同步原 RoboTwin 当前补丁的 `get_eval_video_size()`，避免 head + wrist 合成视频时尺寸不匹配。

## 4. 最终推理命令

单任务直接评测：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"

python experiments/robotwin_mem/eval_robotwin_mem_single.py \
  task=robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4 \
  ckpt=./runs/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4/checkpoints/weights/step_003000.pt \
  EVALUATION.robotwin_root=third_party/RoboTwin-Mem \
  EVALUATION.task_name=cover_blocks_hard \
  EVALUATION.task_config=demo_clean \
  EVALUATION.dataset_stats_path=./runs/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4/dataset_stats.json \
  EVALUATION.eval_num_episodes=100 \
  EVALUATION.num_inference_steps=10 \
  gpu_id=0
```

如果要快速 smoke test，补丁完成后可先跑 1 个 episode：

```bash
python experiments/robotwin_mem/eval_robotwin_mem_single.py \
  task=robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4 \
  ckpt=./runs/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4/checkpoints/weights/step_003000.pt \
  EVALUATION.robotwin_root=third_party/RoboTwin-Mem \
  EVALUATION.task_name=cover_blocks_hard \
  EVALUATION.task_config=demo_clean \
  EVALUATION.dataset_stats_path=./runs/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4/dataset_stats.json \
  EVALUATION.eval_num_episodes=1 \
  gpu_id=0
```

manager 版本命令：

```bash
python experiments/robotwin_mem/run_robotwin_mem_manager.py \
  task=robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4 \
  ckpt=./runs/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4/checkpoints/weights/step_003000.pt \
  EVALUATION.robotwin_root=third_party/RoboTwin-Mem \
  EVALUATION.task_name=cover_blocks_hard \
  EVALUATION.task_config=demo_clean \
  EVALUATION.dataset_stats_path=./runs/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4/dataset_stats.json \
  MULTIRUN.num_gpus=1 \
  MULTIRUN.max_tasks_per_gpu=1
```

## 5. 预期输出

建议输出目录：

```text
evaluate_results/robotwin_mem/
└── robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4_step_003000/
    └── <run_ts>/
        ├── manager.log                      # manager 模式
        ├── summary.csv                      # manager 模式
        ├── summary.json                     # manager 模式
        ├── eval_config_cover_blocks_hard.yaml
        ├── eval_cover_blocks_hard_<ts>.log
        └── cover_blocks_hard/
            ├── _result_demo_clean.txt
            ├── _result.txt
            ├── eval_log.txt
            └── episode*_success-*.mp4
```

`_result_demo_clean.txt` 首行应是 success rate，例如：

```text
0.37
Reward: 0.62
```

## 6. 验证步骤

1. 静态检查：

```bash
python -m py_compile \
  experiments/robotwin_mem/eval_robotwin_mem_single.py \
  experiments/robotwin_mem/run_robotwin_mem_manager.py \
  experiments/robotwin_mem/fastwam_policy/deploy_policy.py
```

2. 检查 Hydra 任务配置能解析：

```bash
python experiments/robotwin_mem/eval_robotwin_mem_single.py \
  task=robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4 \
  ckpt=./runs/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4/checkpoints/weights/step_003000.pt \
  EVALUATION.dataset_stats_path=./runs/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4/dataset_stats.json \
  EVALUATION.task_name=cover_blocks_hard \
  --cfg job --resolve
```

3. smoke test：`EVALUATION.eval_num_episodes=1`

4. full eval：`EVALUATION.eval_num_episodes=100`

5. 检查日志中至少出现：

```text
Initialized WorldActionRobotWinMemPolicy
ckpt=.../step_003000.pt
stats=.../dataset_stats.json
horizon=32
replan=24
Task Name: cover_blocks_hard
Policy Name: fastwam_policy
```

## 7. 风险与注意点

1. 这套权重是 14 维双臂 aloha-agilex qpos，不能直接用于 Mem 的单臂任务。首版只保证 `cover_blocks_hard`。

2. `RoboTwin-Mem` 只提供 `demo_clean.yml`，不要传 `demo_randomized`。

3. 必须传 run 自己的 `dataset_stats.json`，不要混用 release 的 `robotwin_uncond_3cam_384_dataset_stats.json`。

4. `sim_task` 必须是：

```text
robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4
```

否则会按默认 `robotwin_uncond_3cam_384_1e-4` 组模型，和 checkpoint 不匹配。

5. `cover_blocks_hard` step limit 是 1700，100 episodes 会比较慢。建议先 1 episode smoke test。

6. `RoboTwin-Mem` 依赖 SAPIEN/CuRobo/ffmpeg/GPU 环境；如果 smoke test 在 `Sapien_TEST()` 或 assets 初始化阶段失败，优先检查 bench 环境和资产路径，不先怀疑 FastWAM adapter。

7. 当前仓库里 `third_party/RoboTwin/script/eval_policy.py` 已经有 FastWAM 适配补丁；给 Mem 做同样补丁时要保留 Mem 版 reward 逻辑和 hard task instruction 生成逻辑。

## 8. 实施顺序

1. 复制 `experiments/robotwin/fastwam_policy` 到 `experiments/robotwin_mem/fastwam_policy`，做 Mem 命名和 14 维检查。

2. 新建 `eval_robotwin_mem_single.py`，复用单任务 wrapper，默认 bench root 改成 `third_party/RoboTwin-Mem`，输出根改成 `evaluate_results/robotwin_mem`。

3. 给 `third_party/RoboTwin-Mem/script/eval_policy.py` 同步原 RoboTwin 已有的 `eval_output_dir`、`eval_num_episodes`、`skip_get_obs_within_replan`、结果命名补丁。

4. 新建 `run_robotwin_mem_manager.py`，先只支持 `demo_clean` 单阶段汇总。

5. 跑 `py_compile`。

6. 跑 1 episode smoke test。

7. 跑 100 episode full eval。
