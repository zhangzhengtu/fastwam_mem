
# RoboTwin 单任务权重推理 / 评测命令

本文档记录当前已经训练好的 `adjust_bottle` 单任务 Fast-WAM 权重如何在 RoboTwin 中跑推理 / rollout / success rate 评测。

当前权重目录：

```text
/nav-oss/yangganlin/tzz_workspace/FastWAM/runs/robotwin_adjust_bottle_3cam_384_1e-4/checkpoints/weights
```

目前已有：

```text
step_002500.pt
step_003000.pt
```

建议默认使用最终权重：

```text
/nav-oss/yangganlin/tzz_workspace/FastWAM/runs/robotwin_adjust_bottle_3cam_384_1e-4/checkpoints/weights/step_003000.pt
```

评测时必须使用同一次训练生成的 stats：

```text
/nav-oss/yangganlin/tzz_workspace/FastWAM/runs/robotwin_adjust_bottle_3cam_384_1e-4/dataset_stats.json
```

不要使用 release stats，否则 action/state 归一化会和单任务训练不一致。

---

## 1. 基础检查

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"

ls runs/robotwin_adjust_bottle_3cam_384_1e-4/checkpoints/weights/step_003000.pt
ls runs/robotwin_adjust_bottle_3cam_384_1e-4/dataset_stats.json
ls third_party/RoboTwin/assets
```

---

## 2. 推荐：用 manager 跑完整 clean + randomized 评测

`run_robotwin_manager.py` 会对指定任务自动跑两轮：

```text
demo_clean
demo_randomized
```

先小规模跑 3 个 episode 验证链路：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  ckpt=./runs/robotwin_adjust_bottle_3cam_384_1e-4/checkpoints/weights/step_003000.pt \
  EVALUATION.dataset_stats_path=./runs/robotwin_adjust_bottle_3cam_384_1e-4/dataset_stats.json \
  EVALUATION.task_name=adjust_bottle \
  EVALUATION.eval_num_episodes=3 \
  MULTIRUN.num_gpus=1 \
  MULTIRUN.max_tasks_per_gpu=1
```

确认能跑通后，把 episode 数改大，例如 100：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  ckpt=./runs/robotwin_adjust_bottle_3cam_384_1e-4/checkpoints/weights/step_003000.pt \
  EVALUATION.dataset_stats_path=./runs/robotwin_adjust_bottle_3cam_384_1e-4/dataset_stats.json \
  EVALUATION.task_name=adjust_bottle \
  EVALUATION.eval_num_episodes=100 \
  MULTIRUN.num_gpus=1 \
  MULTIRUN.max_tasks_per_gpu=1
```

输出目录在：

```text
evaluate_results/robotwin/
```

重点看：

```text
summary.csv
summary.json
adjust_bottle/_result_clean.txt
adjust_bottle/_result_random.txt
manager.log
```

---

## 3. 只跑 clean 或 randomized 单阶段

如果只想跑一个 phase，用 `eval_robotwin_single.py`。

只跑 clean：

```bash
python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  ckpt=./runs/robotwin_adjust_bottle_3cam_384_1e-4/checkpoints/weights/step_003000.pt \
  EVALUATION.dataset_stats_path=./runs/robotwin_adjust_bottle_3cam_384_1e-4/dataset_stats.json \
  EVALUATION.task_name=adjust_bottle \
  EVALUATION.task_config=demo_clean \
  EVALUATION.eval_num_episodes=3 \
  gpu_id=0
```

只跑 randomized：

```bash
python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  ckpt=./runs/robotwin_adjust_bottle_3cam_384_1e-4/checkpoints/weights/step_003000.pt \
  EVALUATION.dataset_stats_path=./runs/robotwin_adjust_bottle_3cam_384_1e-4/dataset_stats.json \
  EVALUATION.task_name=adjust_bottle \
  EVALUATION.task_config=demo_randomized \
  EVALUATION.eval_num_episodes=3 \
  gpu_id=0
```

---

## 4. 换 checkpoint

如果要评测 `step_002500.pt`，只改 `ckpt`：

```bash
ckpt=./runs/robotwin_adjust_bottle_3cam_384_1e-4/checkpoints/weights/step_002500.pt
```

`dataset_stats_path` 仍然使用同一个训练目录下的：

```bash
EVALUATION.dataset_stats_path=./runs/robotwin_adjust_bottle_3cam_384_1e-4/dataset_stats.json
```

---

## 5. 多 GPU 并行说明

单任务评测通常 1 张 GPU 就够，因为只有 `adjust_bottle` 一个任务。

如果后续一次评多个任务，才需要调大：

```bash
MULTIRUN.num_gpus=<N>
```

单任务下即使申请多张卡，manager 也只有一个任务要跑，收益不明显。

---

## 6. 常用参数

```text
EVALUATION.task_name=adjust_bottle
```

指定 RoboTwin 任务名。

```text
EVALUATION.eval_num_episodes=3
```

每个 phase 跑多少个 episode。manager 会 clean 跑 3 个、randomized 再跑 3 个。

```text
EVALUATION.dataset_stats_path=...
```

必须指向单任务训练生成的 `dataset_stats.json`。

```text
EVALUATION.replan_steps=24
```

默认每次模型输出 action chunk 后，环境执行 24 步再重新规划。

```text
EVALUATION.num_inference_steps=10
```

动作 denoising steps，默认来自训练配置的 `eval_num_inference_steps`。

---

## 7. 结果判断

manager 正常结束时会打印类似：

```text
done task=adjust_bottle phase=clean gpu=0 success_rate=...
done task=adjust_bottle phase=random gpu=0 success_rate=...
summary saved: .../summary.csv and .../summary.json
manager finished successfully
```

最终成功率以 `summary.csv` / `summary.json` 为准。

如果视频无法打开，但 `_result_clean.txt` / `_result_random.txt` 和 `summary.json` 正常生成，通常不影响 success rate 统计。此前对象存储上 MP4 写 trailer 可能失败，当前代码已改成先写 `/tmp` 再复制到结果目录。
