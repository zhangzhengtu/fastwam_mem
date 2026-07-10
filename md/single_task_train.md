# RoboTwin 单任务训练配置与启动流程

本文档只记录当前要落地的版本：**指定一个 RoboTwin 任务名，训练时 train/val 都使用该任务下的 clean + randomized 两个 LeRobot 数据目录，并重新计算该单任务自己的 action/state stats**。

当前已新增两个配置文件：

```text
configs/data/robotwin_single_task.yaml
configs/task/robotwin_adjust_bottle_fastwam_3cam_384_1e-4.yaml
```

默认任务是：

```text
adjust_bottle
```

默认数据根目录是：

```text
/mnt/inspurfs/efm_t/yangganlin/data/RoboTwin-LeRobot-v3.0
```

如果你的真实数据在另一个路径，例如：

```text
/mnt/inspurfs/efm_t/yangyuqiang/dataset/RoboTwin-LeRobot-v3.0
```

训练和预计算命令里覆盖：

```bash
data.robotwin_v3_root=/mnt/inspurfs/efm_t/yangyuqiang/dataset/RoboTwin-LeRobot-v3.0
```

---

## 1. 数据目录要求

以 `adjust_bottle` 为例，配置会读取两个 LeRobot 数据目录：

```text
${data.robotwin_v3_root}/adjust_bottle/aloha-agilex_randomized_500
${data.robotwin_v3_root}/adjust_bottle/aloha-agilex_clean_50
```

每个目录必须是 LeRobot 格式：

```text
aloha-agilex_randomized_500/
├── data/
├── meta/
└── videos/

aloha-agilex_clean_50/
├── data/
├── meta/
└── videos/
```

训练前建议先检查：

```bash
ls /mnt/inspurfs/efm_t/yangganlin/data/RoboTwin-LeRobot-v3.0/adjust_bottle/aloha-agilex_randomized_500/meta/tasks.jsonl
ls /mnt/inspurfs/efm_t/yangganlin/data/RoboTwin-LeRobot-v3.0/adjust_bottle/aloha-agilex_clean_50/meta/tasks.jsonl
```

如果实际根目录不同，把命令里的根目录替换掉，或在 Hydra 命令中覆盖 `data.robotwin_v3_root`。

---

## 2. 新增数据配置

文件：

```text
configs/data/robotwin_single_task.yaml
```

内容：

```yaml
defaults:
  - robotwin
  - _self_

# Single-task RoboTwin LeRobot v3.0 data config.
# Override `task_name` to train another task with the same directory layout.
task_name: adjust_bottle
robotwin_v3_root: /mnt/inspurfs/efm_t/yangganlin/data/RoboTwin-LeRobot-v3.0
embodiment: aloha-agilex
clean_dir_name: ${data.embodiment}_clean_50
randomized_dir_name: ${data.embodiment}_randomized_500

train:
  dataset_dirs:
    - ${data.robotwin_v3_root}/${data.task_name}/${data.randomized_dir_name}
    - ${data.robotwin_v3_root}/${data.task_name}/${data.clean_dir_name}
  val_set_proportion: 0.01
  is_training_set: true
  pretrained_norm_stats: null
  text_embedding_cache_dir: ./data/text_embeds_cache/robotwin_single/${data.task_name}

val:
  dataset_dirs:
    - ${data.robotwin_v3_root}/${data.task_name}/${data.randomized_dir_name}
    - ${data.robotwin_v3_root}/${data.task_name}/${data.clean_dir_name}
  val_set_proportion: 0.01
  is_training_set: false
  pretrained_norm_stats: null
  text_embedding_cache_dir: ./data/text_embeds_cache/robotwin_single/${data.task_name}
```

说明：

- `train.dataset_dirs` 包含 randomized + clean。
- `val.dataset_dirs` 也包含 randomized + clean。
- `val_set_proportion: 0.01` 表示每个 dataset 内按 episode 切出 1% 做 val。
- `pretrained_norm_stats: null` 表示不使用全局 stats，训练开始时重新计算该单任务 stats。
- 训练生成的 stats 会保存到 `<output_dir>/dataset_stats.json`。

---

## 3. 新增任务配置

文件：

```text
configs/task/robotwin_adjust_bottle_fastwam_3cam_384_1e-4.yaml
```

内容：

```yaml
# @package _global_

defaults:
  - override /data: robotwin_single_task
  - override /model: fastwam
  - _self_

batch_size: 16
num_workers: 8

data:
  task_name: adjust_bottle

model:
  mot_checkpoint_mixed_attn: false

lr_scheduler_type: "cosine"
learning_rate: 1e-4
num_epochs: 5
max_steps: 3000
log_every: 10
save_every: 2500
eval_every: 500

gradient_accumulation_steps: 1
weight_decay: 1e-2
resume: null
```

命名规则：

```text
configs/task/robotwin_<task_name>_3cam_384_1e-4.yaml
```

例如后续要给 `click_alarmclock` 建固定配置，就复制一份并改名：

```text
configs/task/robotwin_click_alarmclock_3cam_384_1e-4.yaml
```

然后把其中：

```yaml
data:
  task_name: click_alarmclock
```

---

## 4. 配置解析检查

可以先不训练，只检查 Hydra 最终解析出的 dataset 路径：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM

/shared/smartbot/yangganlin/anaconda3/envs/fastwam/bin/python \
  scripts/train.py \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  --cfg job --resolve
```

你应该看到：

```yaml
data:
  train:
    dataset_dirs:
    - /mnt/inspurfs/efm_t/yangganlin/data/RoboTwin-LeRobot-v3.0/adjust_bottle/aloha-agilex_randomized_500
    - /mnt/inspurfs/efm_t/yangganlin/data/RoboTwin-LeRobot-v3.0/adjust_bottle/aloha-agilex_clean_50
    pretrained_norm_stats: null
  val:
    dataset_dirs:
    - /mnt/inspurfs/efm_t/yangganlin/data/RoboTwin-LeRobot-v3.0/adjust_bottle/aloha-agilex_randomized_500
    - /mnt/inspurfs/efm_t/yangganlin/data/RoboTwin-LeRobot-v3.0/adjust_bottle/aloha-agilex_clean_50
    pretrained_norm_stats: null
```

如果根目录不同：

```bash
/shared/smartbot/yangganlin/anaconda3/envs/fastwam/bin/python \
  scripts/train.py \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  data.robotwin_v3_root=/mnt/inspurfs/efm_t/yangyuqiang/dataset/RoboTwin-LeRobot-v3.0 \
  --cfg job --resolve
```

---

## 5. 预计算文本 embedding

训练前先预计算文本 embedding：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
mkdir -p slurm

srun -p wam_agent \
  --gres=gpu:1 \
  --ntasks-per-node=1 \
  -o slurm/precompute_text_embeds.out \
  -e slurm/precompute_text_embeds.err \
  python scripts/precompute_text_embeds.py \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4
```

如果真实根目录不是配置默认值：

```bash
srun -p wam_agent \
  --gres=gpu:1 \
  --ntasks-per-node=1 \
  -o slurm/precompute_text_embeds.out \
  -e slurm/precompute_text_embeds.err \
  python scripts/precompute_text_embeds.py \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  data.robotwin_v3_root=/mnt/inspurfs/efm_t/yangyuqiang/dataset/RoboTwin-LeRobot-v3.0
```

生成位置：

```text
data/text_embeds_cache/robotwin_single/adjust_bottle/
```

---

## 6. 启动训练

正式训练：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
mkdir -p slurm

srun -p wam_agent \
  --gres=gpu:16 \
  --ntasks-per-node=1 \
  -o slurm/train_adjust_bottle_16gpu.out \
  -e slurm/train_adjust_bottle_16gpu.err \
  bash scripts/train_zero1.sh 16 \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  output_dir=./runs/robotwin_adjust_bottle_fastwam_3cam_384_1e-4
```

这条命令的实际默认配置是：

```text
模型: Fast-WAM / uncond, 即 configs/model/fastwam.yaml
数据: adjust_bottle 单任务，不是 50 个 task
训练集: adjust_bottle/aloha-agilex_randomized_500 + adjust_bottle/aloha-agilex_clean_50
验证集: 同样来自 randomized + clean，每个 LeRobot dataset 按 episode 切 1%
stats: 不使用 release stats，训练启动时重新计算该单任务自己的 dataset_stats.json
batch_size: 16 / GPU
16 卡 global batch: 16 * 16 = 256
num_epochs: 5
max_steps: 3000
learning_rate: 1e-4
lr_scheduler_type: cosine
gradient_accumulation_steps: 1
mixed_precision: bf16
save_every: 每 2500 个 optimizer step 保存一次
eval_every: 每 500 个 optimizer step 做一次 val/infer 评估
```

注意：`scripts/train_zero1.sh` 本来会自动加时间戳输出目录：

```text
./runs/<task_name>/<RUN_ID>
```

但如果命令里显式传了：

```bash
output_dir=./runs/robotwin_adjust_bottle_fastwam_3cam_384_1e-4
```

就会覆盖脚本默认输出目录，所有日志、stats、checkpoint 都直接写到这个目录下。为了避免多次实验混在一起，建议正式训练时要么不传 `output_dir`，要么自己加一个 run 子目录。

如果真实根目录不是配置默认值：

```bash
srun -p wam_agent \
  --gres=gpu:16 \
  --ntasks-per-node=1 \
  -o slurm/train_adjust_bottle_16gpu.out \
  -e slurm/train_adjust_bottle_16gpu.err \
  bash scripts/train_zero1.sh 16 \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  data.robotwin_v3_root=/mnt/inspurfs/efm_t/yangyuqiang/dataset/RoboTwin-LeRobot-v3.0 \
  output_dir=./runs/robotwin_adjust_bottle_fastwam_3cam_384_1e-4
```

调试小步跑：

```bash
srun -p wam_agent \
  --gres=gpu:1 \
  --ntasks-per-node=1 \
  -o slurm/debug_adjust_bottle.out \
  -e slurm/debug_adjust_bottle.err \
  bash scripts/train_zero1.sh 1 \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  max_steps=20 \
  eval_every=10 \
  save_every=20 \
  batch_size=1 \
  num_workers=0 \
  output_dir=./runs/debug_robotwin_adjust_bottle
```

一天内训练完的推荐命令：

```bash
srun -p wam_agent \
  --gres=gpu:16 \
  --ntasks-per-node=1 \
  -o slurm/train_adjust_bottle_16gpu_day1.out \
  -e slurm/train_adjust_bottle_16gpu_day1.err \
  bash scripts/train_zero1.sh 16 \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  output_dir=./runs/robotwin_adjust_bottle_fastwam_3cam_384_1e-4/day1_3000steps
```

说明：

- 当前任务配置已经固定 `max_steps: 3000`，总训练长度由 3000 个 optimizer step 控制。
- 单任务数据量远小于 50-task 全量数据，用固定 `max_steps` 控制预算，比只改 `num_epochs` 更稳。
- 其他训练超参保持不变：`batch_size=16`、`learning_rate=1e-4`、`save_every=2500`、`eval_every=500`。
- 如果显存不够，把 `batch_size=16` 改成 `batch_size=8` 或 `batch_size=4`。
- 如果只是确认链路是否能跑通，用上面的 `max_steps=20` 调试命令即可。

如果要改卡数，需要同时改两个位置：

```bash
--gres=gpu:<N>
bash scripts/train_zero1.sh <N>
```

例如改成 8 卡：

```bash
srun -p wam_agent \
  --gres=gpu:8 \
  --ntasks-per-node=1 \
  -o slurm/train_adjust_bottle_8gpu.out \
  -e slurm/train_adjust_bottle_8gpu.err \
  bash scripts/train_zero1.sh 8 \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  output_dir=./runs/robotwin_adjust_bottle_fastwam_3cam_384_1e-4/day1_3000steps_8gpu
```

`--gres=gpu:<N>` 是 Slurm 申请的 GPU 数，`train_zero1.sh <N>` 是 accelerate 在当前节点启动的进程数。单机训练时这两个数字要一致。改卡数后，global batch 会跟着变成：

```text
global batch = batch_size * GPU 数 * gradient_accumulation_steps
```

当前配置是 `batch_size=16`、`gradient_accumulation_steps=1`，所以 16 卡是 `256`，8 卡是 `128`。如果只是从 16 卡改 8 卡，训练仍然能跑，但每个 optimizer step 看到的样本数会变小；若希望保持同样 global batch，可以额外把 `gradient_accumulation_steps=2`，不过这就不再是“其他超参不变”。

训练时会重新计算单任务 stats，并保存：

```text
./runs/robotwin_adjust_bottle_fastwam_3cam_384_1e-4/dataset_stats.json
```

实际路径取决于 `output_dir`。

---

## 7. 权重保存逻辑

训练 checkpoint 保存在：

```text
<output_dir>/checkpoints/
├── weights/
│   └── step_XXXXXX.pt
└── state/
    └── step_XXXXXX/
```

两类文件用途不同：

```text
weights/step_XXXXXX.pt
```

用于推理、评测、rollout。后续 `ckpt=` 应该指向这个 `.pt` 文件。

```text
state/step_XXXXXX/
```

用于继续训练，里面是 accelerate/deepspeed 的 optimizer、scheduler、随机状态和 trainer_state。继续训练时 `resume=` 应该指向这个目录。

保存触发规则：

- 每当 `global_step % save_every == 0` 时保存一次。
- 训练达到 `max_steps` 时一定会再保存一次最终 checkpoint。
- `save_every` 按 optimizer step 计数，不按 epoch 计数。

例如一天内训练命令使用：

```bash
max_steps=3000 save_every=2500
```

会保存：

```text
step_002500.pt
step_003000.pt
```

训练生成的归一化统计保存在：

```text
<output_dir>/dataset_stats.json
```

这个文件和 checkpoint 是一套，评测时必须一起使用。

---

## 8. 训练后评测

用自己训练出的 checkpoint 评测时，必须使用同一次训练生成的 stats：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  ckpt=./runs/robotwin_adjust_bottle_fastwam_3cam_384_1e-4/checkpoints/weights/step_XXXXXX.pt \
  EVALUATION.dataset_stats_path=./runs/robotwin_adjust_bottle_fastwam_3cam_384_1e-4/dataset_stats.json \
  EVALUATION.task_name=adjust_bottle \
  EVALUATION.eval_num_episodes=100 \
  MULTIRUN.num_gpus=1 \
  MULTIRUN.max_tasks_per_gpu=1
```

不要用 release stats：

```text
checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
```

因为单任务训练时 action/state 归一化重新计算过，评测必须保持一致。

---

## 9. 切换到其他任务

临时切换任务：

```bash
srun -p wam_agent \
  --gres=gpu:1 \
  --ntasks-per-node=1 \
  -o slurm/precompute_text_embeds_click_alarmclock.out \
  -e slurm/precompute_text_embeds_click_alarmclock.err \
  python scripts/precompute_text_embeds.py \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  data.task_name=click_alarmclock
```

```bash
srun -p wam_agent \
  --gres=gpu:16 \
  --ntasks-per-node=1 \
  -o slurm/train_click_alarmclock_16gpu.out \
  -e slurm/train_click_alarmclock_16gpu.err \
  bash scripts/train_zero1.sh 16 \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  data.task_name=click_alarmclock \
  output_dir=./runs/robotwin_click_alarmclock_3cam_384_1e-4
```

长期使用建议新增任务专属配置：

```text
configs/task/robotwin_click_alarmclock_3cam_384_1e-4.yaml
```

内容只需要复制 `robotwin_adjust_bottle_fastwam_3cam_384_1e-4.yaml`，然后改：

```yaml
data:
  task_name: click_alarmclock
```

---

## 10. 本次新增配置

单任务配置文件统一命名为：

```text
configs/task/robotwin_<task_name>_<model_config>_3cam_384_1e-4.yaml
```

其中当前使用：

```text
fastwam
fastwamidm
```

### 10.1 move_can_pot

已新增任务配置：

```text
configs/task/robotwin_move_can_pot_fastwam_3cam_384_1e-4.yaml
```

该配置复用：

```text
configs/data/robotwin_single_task.yaml
```

解析后的训练 / 验证数据目录是：

```text
/mnt/inspurfs/efm_t/yangganlin/data/RoboTwin-LeRobot-v3.0/move_can_pot/aloha-agilex_randomized_500
/mnt/inspurfs/efm_t/yangganlin/data/RoboTwin-LeRobot-v3.0/move_can_pot/aloha-agilex_clean_50
```

使用方式：

```bash
task=robotwin_move_can_pot_fastwam_3cam_384_1e-4
```

### 10.2 cover_blocks_hard

已新增数据配置：

```text
configs/data/robotwin_cover_blocks_hard_extra.yaml
```

已新增任务配置：

```text
configs/task/robotwin_cover_blocks_hard_fastwam_3cam_384_1e-4.yaml
```

解析后的训练 / 验证数据目录是：

```text
/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/data/extra_data/cover_blocks_hard
```

使用方式：

```bash
task=robotwin_cover_blocks_hard_fastwam_3cam_384_1e-4
```

如果要训练 IDM 版本，使用：

```text
configs/task/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4.yaml
```

对应命令参数：

```bash
task=robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4
```

注意：当前按 `cover_blocks_hard` 本身就是一个 LeRobot 数据目录处理，也就是该目录下应直接包含 `data/`、`meta/`、`videos/`。如果实际是多级子目录，需要把 `configs/data/robotwin_cover_blocks_hard_extra.yaml` 里的 `train.dataset_dirs` 和 `val.dataset_dirs` 改成具体子目录列表。

---

## 11. 四个单任务完整训练流程

下面四个流程彼此独立，直接复制对应任务的小节即可。每个流程都包含：

```text
检查数据 -> 检查 Hydra 解析 -> 预计算文本 embedding -> 16 GPU 正式训练
```

### 11.1 adjust_bottle / fastwam

配置文件：

```text
configs/task/robotwin_adjust_bottle_fastwam_3cam_384_1e-4.yaml
```

检查数据：

```bash
ls /mnt/inspurfs/efm_t/yangganlin/data/RoboTwin-LeRobot-v3.0/adjust_bottle/aloha-agilex_randomized_500/meta/tasks.jsonl
ls /mnt/inspurfs/efm_t/yangganlin/data/RoboTwin-LeRobot-v3.0/adjust_bottle/aloha-agilex_clean_50/meta/tasks.jsonl
```

检查配置解析：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM

/shared/smartbot/yangganlin/anaconda3/envs/fastwam/bin/python \
  scripts/train.py \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  --cfg job --resolve
```

预计算文本 embedding：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
mkdir -p slurm

srun -p wam_agent \
  --gres=gpu:1 \
  --ntasks-per-node=1 \
  -o slurm/precompute_adjust_bottle_fastwam.out \
  -e slurm/precompute_adjust_bottle_fastwam.err \
  python scripts/precompute_text_embeds.py \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4
```

正式训练：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
mkdir -p slurm

srun -p wam_agent \
  --gres=gpu:16 \
  --ntasks-per-node=1 \
  -o slurm/train_adjust_bottle_fastwam_16gpu.out \
  -e slurm/train_adjust_bottle_fastwam_16gpu.err \
  bash scripts/train_zero1.sh 16 \
  task=robotwin_adjust_bottle_fastwam_3cam_384_1e-4 \
  output_dir=./runs/robotwin_adjust_bottle_fastwam_3cam_384_1e-4
```

训练产物：

```text
./runs/robotwin_adjust_bottle_fastwam_3cam_384_1e-4/dataset_stats.json
./runs/robotwin_adjust_bottle_fastwam_3cam_384_1e-4/checkpoints/weights/step_XXXXXX.pt
```

### 11.2 move_can_pot / fastwam

配置文件：

```text
configs/task/robotwin_move_can_pot_fastwam_3cam_384_1e-4.yaml
```

检查数据：

```bash
ls /mnt/inspurfs/efm_t/yangganlin/data/RoboTwin-LeRobot-v3.0/move_can_pot/aloha-agilex_randomized_500/meta/tasks.jsonl
ls /mnt/inspurfs/efm_t/yangganlin/data/RoboTwin-LeRobot-v3.0/move_can_pot/aloha-agilex_clean_50/meta/tasks.jsonl
```

检查配置解析：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM

/shared/smartbot/yangganlin/anaconda3/envs/fastwam/bin/python \
  scripts/train.py \
  task=robotwin_move_can_pot_fastwam_3cam_384_1e-4 \
  --cfg job --resolve
```

预计算文本 embedding：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
mkdir -p slurm

srun -p wam_agent \
  --gres=gpu:1 \
  --ntasks-per-node=1 \
  -o slurm/precompute_move_can_pot_fastwam.out \
  -e slurm/precompute_move_can_pot_fastwam.err \
  python scripts/precompute_text_embeds.py \
  task=robotwin_move_can_pot_fastwam_3cam_384_1e-4
```

正式训练：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
mkdir -p slurm

srun -p wam_agent \
  --gres=gpu:8 \
  --ntasks-per-node=1 \
  -o slurm/train_move_can_pot_fastwam_8gpu.out \
  -e slurm/train_move_can_pot_fastwam_8gpu.err \
  bash scripts/train_zero1.sh 8 \
  task=robotwin_move_can_pot_fastwam_3cam_384_1e-4 \
  output_dir=./runs/robotwin_move_can_pot_fastwam_3cam_384_1e-4
```

训练产物：

```text
./runs/robotwin_move_can_pot_fastwam_3cam_384_1e-4/dataset_stats.json
./runs/robotwin_move_can_pot_fastwam_3cam_384_1e-4/checkpoints/weights/step_XXXXXX.pt
```

### 11.3 cover_blocks_hard / fastwam

配置文件：

```text
configs/task/robotwin_cover_blocks_hard_fastwam_3cam_384_1e-4.yaml
```

检查数据：

```bash
ls /mnt/inspurfs/efm_t/yangganlin/workspace_tzz/data/extra_data/cover_blocks_hard/meta/tasks.jsonl
ls /mnt/inspurfs/efm_t/yangganlin/workspace_tzz/data/extra_data/cover_blocks_hard/data
ls /mnt/inspurfs/efm_t/yangganlin/workspace_tzz/data/extra_data/cover_blocks_hard/videos
```

检查配置解析：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM

/shared/smartbot/yangganlin/anaconda3/envs/fastwam/bin/python \
  scripts/train.py \
  task=robotwin_cover_blocks_hard_fastwam_3cam_384_1e-4 \
  --cfg job --resolve
```

预计算文本 embedding：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
mkdir -p slurm

srun -p wam_agent \
  --gres=gpu:1 \
  --ntasks-per-node=1 \
  -o slurm/precompute_cover_blocks_hard_fastwam.out \
  -e slurm/precompute_cover_blocks_hard_fastwam.err \
  python scripts/precompute_text_embeds.py \
  task=robotwin_cover_blocks_hard_fastwam_3cam_384_1e-4
```

正式训练：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
mkdir -p slurm

srun -p wam_agent \
  --gres=gpu:8 \
  --ntasks-per-node=1 \
  -o slurm/train_cover_blocks_hard_fastwam_8gpu.out \
  -e slurm/train_cover_blocks_hard_fastwam_8gpu.err \
  bash scripts/train_zero1.sh 8 \
  task=robotwin_cover_blocks_hard_fastwam_3cam_384_1e-4 \
  output_dir=./runs/robotwin_cover_blocks_hard_fastwam_3cam_384_1e-4
```

训练产物：

```text
./runs/robotwin_cover_blocks_hard_fastwam_3cam_384_1e-4/dataset_stats.json
./runs/robotwin_cover_blocks_hard_fastwam_3cam_384_1e-4/checkpoints/weights/step_XXXXXX.pt
```

### 11.4 cover_blocks_hard / fastwamidm

配置文件：

```text
configs/task/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4.yaml
```

检查数据：

```bash
ls /mnt/inspurfs/efm_t/yangganlin/workspace_tzz/data/extra_data/cover_blocks_hard/meta/tasks.jsonl
ls /mnt/inspurfs/efm_t/yangganlin/workspace_tzz/data/extra_data/cover_blocks_hard/data
ls /mnt/inspurfs/efm_t/yangganlin/workspace_tzz/data/extra_data/cover_blocks_hard/videos
```

检查配置解析：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM

/shared/smartbot/yangganlin/anaconda3/envs/fastwam/bin/python \
  scripts/train.py \
  task=robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4 \
  --cfg job --resolve
```

预计算文本 embedding：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
mkdir -p slurm

srun -p wam_agent \
  --gres=gpu:1 \
  --ntasks-per-node=1 \
  -o slurm/precompute_cover_blocks_hard_fastwamidm.out \
  -e slurm/precompute_cover_blocks_hard_fastwamidm.err \
  python scripts/precompute_text_embeds.py \
  task=robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4
```

正式训练：

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
mkdir -p slurm

srun -p wam_agent \
  --gres=gpu:8 \
  --ntasks-per-node=1 \
  -o slurm/train_cover_blocks_hard_fastwamidm_8gpu.out \
  -e slurm/train_cover_blocks_hard_fastwamidm_8gpu.err \
  bash scripts/train_zero1.sh 8 \
  task=robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4 \
  output_dir=./runs/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4
```

训练产物：

```text
./runs/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4/dataset_stats.json
./runs/robotwin_cover_blocks_hard_fastwamidm_3cam_384_1e-4/checkpoints/weights/step_XXXXXX.pt
```

---

## 12. 注意事项

1. clean/randomized 两个 dataset 的 fps 必须一致。

2. clean/randomized 两个 dataset 的 schema 必须一致。

当前 RoboTwin 配置期望：

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
observation.state
action
```

3. 不要混用 `_v3.0`、`_v3.0_v3.0` 后缀目录。

当前配置默认使用：

```text
aloha-agilex_clean_50
aloha-agilex_randomized_500
```

4. 训练和评测必须使用同一个 stats。

单任务训练会生成自己的：

```text
dataset_stats.json
```

评测时要传这个文件。
