# FastWAM 引入 EventVLA 顺序采样与 KEM 预测的可行方案

## 目标

在不改变 FastWAM 主体训练范式的前提下，参考 EventVLA 的两部分机制：

1. `sequence_sampler.py` 的 episode 顺序采样：让训练 batch 按轨迹内时间顺序产生 anchor，并携带 episode 边界、上一个 anchor、近似关键帧等元信息。
2. `EventVLA.py` 的 KEM 预测：在 action chunk 内预测关键事件概率，训练时用 episode 级 `keyframe_steps` 生成 `chunk_keyframe_target`，推理时可用 threshold、NMS、cooldown 选择未来 event commit。

FastWAM 当前训练输入已经是 `video/action/proprio/context`，模型在 `FastWAM.training_loss()` 中通过 `video_expert.pre_dit()`、`action_expert.pre_dit()`、`MoT` 和两个 post head 同时训练视频与动作。最自然的移植方式是：

- 数据侧先补齐 episode/step/keyframe 元信息、顺序 sampler，以及按关键帧标签取出的历史关键帧 memory。
- 训练时就把 teacher 历史关键帧塞进模型作为条件输入，避免只学 KEM 头、模型主体却从未见过 memory 的训练/推理分布错位。
- 模型侧在 `tokens_out["action"]` 上增加一个轻量 KEM head，输出 `[B, action_horizon]` 的 chunk 内关键帧概率。
- 第一阶段使用标注关键帧构造 teacher memory；第二阶段再参考 EventVLA 做 teacher-to-predict schedule，让训练逐步改用 KEM 预测维护的 runtime memory。

## 已确认的现状

FastWAM 相关入口：

- 数据集：`src/fastwam/datasets/lerobot/robot_video_dataset.py`
- LeRobot wrapper：`src/fastwam/datasets/lerobot/base_lerobot_dataset.py`
- 当前 sampler：`src/fastwam/utils/samplers.py`，只有 `ResumableEpochSampler`，按随机全局 index 采样。
- 训练器：`src/fastwam/trainer.py`，`Wan22Trainer._build_loader()` 固定使用 `ResumableEpochSampler`。
- 模型：`src/fastwam/models/wan22/fastwam.py`，`training_loss()` 里已有 `tokens_out["action"]`。
- action token 形状：`ActionDiT.pre_dit()` 把 action 序列 `[B, T, action_dim]` 编码成 `[B, T, hidden_dim]`，因此 KEM head 可以逐 action token 预测。
- 配置：`configs/train.yaml` 目前没有 sampler 配置；`configs/model/fastwam.yaml` 目前只有 `loss.lambda_action`。

数据可用性：

- FastWAM vendored LeRobot 的 `MultiLeRobotDataset.__getitem__()` 会把底层样本中的 `episode_index`、`frame_index`、`timestamp`、`index` 保留下来，并额外加入 `dataset_index`。
- RoboTwin-Mem 的 `meta/episodes.jsonl` 已经有 `keyframe_steps`，例如 `cover_blocks_hard` 每个 episode 有 `length` 和 `keyframe_steps`。
- `BaseLerobotDataset` 当前没有把 episode 级 `keyframe_steps` 缓存成查询接口，也没有生成 `chunk_keyframe_target`。

EventVLA 可借鉴点：

- sampler 输出 tuple：`dataset_index, trajectory_id, step_index, is_new_episode, is_last_sampled_step, anchor_index, prev_anchor_step, is_keyframe_approx`。
- sparse anchor 逻辑：总是从 0 开始；`sampling_interval > 1` 时第二个 anchor 用 seed/epoch/dataset/trajectory 的 hash 做确定性随机偏移；末尾尽量补一个 tail anchor。
- DDP 对齐：每个 rank 按轨迹切片，最后用局部 stream 循环补齐到全 rank 相同 batch 数。
- KEM target：对 chunk 内每个 step 计算到最近 `keyframe_steps` 的距离，可用 `raised_cosine` 或线性 dilation，默认 dilation 为 8。
- KEM loss：`BCEWithLogitsLoss(pos_weight=keyframe_positive_weight)`，EventVLA 默认正样本权重 7。
- event 选择：从 `event_future_min_offset` 之后选概率最大位置，超过 `event_commit_threshold` 才触发；推理时再加 NMS 和 cooldown。
- 历史关键帧注入：EventVLA 数据侧根据 `keyframe_steps <= current_step` 取可见关键帧图像，模型侧在 forward 开始调用 `_resolve_training_keyframe_inputs()`，按 `teacher/predict/union/teacher_to_predict` 选择 dataloader teacher memory 或 runtime memory，再由 `_augment_batch_images_with_keyframe_images()` 把这些图像作为 `memory_keyframe` 插入 Qwen-VL 图像输入列表。

## 数据顺序采样方案

### 1. 给 FastWAM 数据集补 episode 元信息接口

建议在 `BaseLerobotDataset` 初始化后构建以下缓存：

- `trajectory_ids`: 当前 split 内的全局 episode id。
- `trajectory_lengths`: 每个 episode 的长度。
- `trajectory_start_indices`: 当前 `MultiLeRobotDataset` 拼接后的全局起始 frame index。
- `episode_to_dataset_index`: episode 来自哪个 dataset dir。
- `episode_to_local_episode_index`: 在单个底层 dataset 中的 episode index。
- `keyframe_steps_by_episode`: 从每个 dataset 的 `meta.episodes` 或 `meta/episodes.jsonl` 读取 `keyframe_steps` / `inspect_keyframe_steps`。

新增方法：

- `get_keyframe_steps(trajectory_id) -> list[int]`
- `get_inspect_keyframe_steps(trajectory_id) -> list[int]`
- `has_inspect_keyframe_annotations() -> bool`
- `is_inspect_keyframe(trajectory_id, timestep) -> bool`
- `episode_step_to_global_index(trajectory_id, step_index) -> int`

注意：FastWAM 的 `RobotVideoDataset` 外层包了一层 `BaseLerobotDataset`，sampler 传入 trainer 的 dataset 是 `RobotVideoDataset`。因此 `RobotVideoDataset` 也需要透传这些方法，或者 sampler 内部约定使用 `dataset.lerobot_dataset` 作为真实 episode dataset。

### 2. 新增 FastWAM 版顺序 batch sampler

建议新增 `src/fastwam/utils/sequence_sampler.py`，实现 `SequentialEpisodeBatchSampler`，直接参考 EventVLA 结构，但输出可以更贴近 FastWAM：

```python
EpisodeSampleIndex = tuple[int, int, bool, bool, int, int, bool]
# global_index, trajectory_id, is_new_episode, is_last_sampled_step,
# anchor_index, prev_anchor_step, is_keyframe_approx
```

也可以保留 EventVLA 的 8 元 tuple：

```python
dataset_index, trajectory_id, step_index, is_new_episode,
is_last_sampled_step, anchor_index, prev_anchor_step, is_keyframe_approx
```

我更建议保留 8 元 tuple，因为它和 EventVLA 对齐，便于后续 memory bank 按 dataset/episode/step 管理。`RobotVideoDataset.__getitem__()` 收到 tuple 后再把它映射成底层 global index。

关键逻辑：

- `_build_sparse_anchors()` 与 EventVLA 保持一致，`max_valid_step = trajectory_length - action_horizon`。
- `action_horizon` 对 FastWAM 应使用 `num_frames - 1`，也就是底层 action chunk 长度；注意不是 video 下采样后的 `T_video - 1`。
- `sampling_interval` 可默认等于 `global_sample_stride` 或单独配置，建议单独配置。
- `_nearest_sampled_keyframes()` 用于给 sparse anchor 打 `is_keyframe_approx`，解决 keyframe 不刚好落在采样 anchor 上的问题。
- DDP 下沿用 EventVLA 的“轨迹池按 rank 切片、用 max steps per rank 决定 batch 数、本地循环补齐”策略，避免各 rank dataloader 长度不一致。
- 如果一个 rank 没拿到轨迹，用 fallback sample，避免 DDP hang。

### 3. 让 dataset 接受 tuple 索引

`RobotVideoDataset._get(idx)` 当前假设 `idx` 是 int。需要扩展为：

1. 如果 `idx` 是 int，保持现有行为。
2. 如果 `idx` 是 tuple，则解析 episode 顺序采样元信息：
   - `global_index = BaseLerobotDataset.episode_step_to_global_index(trajectory_id, step_index)`
   - 用 `global_index` 读取原始 sample。
   - 把采样元信息写入返回 batch：
     - `dataset_index`
     - `trajectory_id` / `episode_index`
     - `timestep` / `frame_index`
     - `is_new_episode`
     - `is_last_sampled_step`
     - `anchor_index`
     - `prev_anchor_step`
     - `is_keyframe_approx`

`BaseLerobotDataset.__getitem__()` 也应把底层 `episode_index`、`frame_index`、`timestamp` 明确放进 sample，避免 processor 后丢字段。

### 4. 构造 KEM 训练标签

参考 EventVLA 的 `_build_chunk_keyframe_supervision()`，在数据侧生成：

- `keyframe_steps`: episode 级关键帧列表。
- `has_keyframe_annotations`: 是否有标注。
- `use_keyframe_supervision`: 是否参与 KEM loss。
- `is_keyframe_exact`: 当前 step 是否刚好是 keyframe。
- `is_keyframe_proxy`: 当前 anchor 是否是最近采样关键帧，来自 sampler 的 `is_keyframe_approx`。
- `chunk_keyframe_target`: `[action_horizon]` float tensor。
- `chunk_keyframe_exact_steps`: 落在当前 chunk 内的绝对关键帧 step。
- `teacher_event_offset`: 当前 chunk 内最应该 commit 的 offset。
- `teacher_event_confidence`: target 在该 offset 上的值。
- `teacher_should_commit`: confidence 是否超过阈值。
- `teacher_commit_timestep`: `step_index + teacher_event_offset`，无触发时为 -1。

target 公式建议沿用 EventVLA：

- `chunk_steps = step_index + arange(action_horizon)`
- `dist = min(abs(chunk_steps - keyframe_step))`
- `target_dilation == 0` 时只标 exact keyframe。
- `raised_cosine` 时：`target = 0.5 * (1 + cos(pi * dist / dilation))`，只对 `dist <= dilation` 生效。
- 线性 fallback：`target = max(0, 1 - dist / dilation)`。

FastWAM 的 action 是 `[num_frames - 1, action_dim]`，默认 RobotWin 配置下 `num_frames=33`，所以 `action_horizon=32`。KEM head 输出也应是 32，而不是视频采样后的 8 个 transition。

### 5. 构造训练时历史关键帧 memory

这一步是必要项，不是推理阶段才补的功能。对每个训练 sample，数据侧要用关键帧标签构造当前时刻已经“看过”的历史关键帧输入：

- `memory_keyframe_video`: `[max_keyframes, 3, H, W]`，已经完成和当前 `video` 相同的相机拼接、resize、crop、normalize。
- `memory_keyframe_mask`: `[max_keyframes]`，有效 keyframe 为 true，padding 为 false。
- `memory_keyframe_steps`: `[max_keyframes]`，历史关键帧的 episode 内绝对 step，padding 可为 -1。
- `memory_keyframe_count`: 有效历史关键帧数量。
- `keyframe_input_memory_source`: 当前 sample 使用 `teacher`、`predict`、`union` 还是 `none`。

候选历史关键帧来自 episode 级 `keyframe_steps`：

```python
if include_current_keyframe:
    candidates = [kf for kf in keyframe_steps if kf <= current_step]
else:
    candidates = [kf for kf in keyframe_steps if kf < current_step]
candidates = sorted(set(candidates))
candidates = candidates[-max_keyframes:]  # latest
```

FastWAM 建议默认 `include_current_keyframe=false`，因为当前观测帧已经作为 `input_image / first_frame_latents` 进入模型，memory 更应该表示历史信息；如果要完全对齐 EventVLA，可配置为 true。

多相机处理要和当前视频一致。比如 `concat_multi_camera="robotwin"` 时，历史关键帧也要把 `cam_high/cam_left_wrist/cam_right_wrist` 合成同样的 384x320 画面，而不是只塞单个相机视角。这样 memory 图像和 FastWAM 当前视频帧分布一致。

padding 后保持固定 `max_keyframes`，便于 default collate；没有历史关键帧时传零图和全 false mask。KEM label 仍然按当前 chunk 构造，memory 则只使用当前 step 之前的关键帧，两者不要混淆。

## 历史关键帧塞入 FastWAM 模型的方案

### 1. 采用 Memory Prefix Video Tokens

这里采用 **Memory Prefix Video Tokens**，不再采用 context-token 压缩方案。核心要求是：历史关键帧 memory tokens 和 current first frame tokens 在处理方式上尽量一致。

具体约束：

- 历史关键帧先走和其他视频帧一致的 VAE image encode。
- VAE latent 走和其他视频帧一致的 `video_expert.patchify()` / `patch_embedding`。
- 不对 memory tokens 做额外 pooling、压缩、query summarization 或 token 数裁剪。
- memory tokens 的 RoPE temporal index 固定为 0。
- memory tokens 的 `t_mod` 和 current first frame tokens 一致，也就是使用 denoising timestep 0 的调制。
- 默认不加额外 slot embedding、age embedding、source embedding；如果后续要做消融，可以另开配置，但默认方案保持 memory token 和 current first frame 尽量同构。

数据流：

```text
memory_keyframe_video [B, K, 3, H, W]
  -> VAE encode, same as image/video frames
  -> memory_latents [B, C, K, h, w]
  -> video_expert.patchify(), same as normal video latent frames
  -> memory_video_tokens [B, K * tokens_per_frame, hidden_dim]
  -> prefix before current video tokens inside MoT video branch
```

如果 `memory_keyframe_mask` 中某个 keyframe 是 padding，则仍可生成对应 token，但这些 token 必须在 attention mask 中设为不可见，并且不能参与任何 loss。

### 2. Patchify 与 token 顺序

`FastWAM.build_inputs()` 增加：

- 读取 `sample["memory_keyframe_video"]`、`sample["memory_keyframe_mask"]`、`sample["memory_keyframe_steps"]`。
- 用 VAE 将 `[B*K, 3, H, W]` 编成 image latent，再 reshape 为 `[B, C, K, h, w]`。
- 使用 `video_expert.patchify(memory_latents)`，保持和其他帧相同的 patch size、Conv3D 权重和 token hidden dim。
- token 顺序建议为 `memory_0, memory_1, ..., memory_{K-1}, current_and_future_video_tokens`。
- memory keyframe 的组内空间顺序与普通视频帧 patch token 顺序完全一致。

伪代码：

```python
memory = sample["memory_keyframe_video"]      # [B, K, 3, H, W]
memory_mask = sample["memory_keyframe_mask"]  # [B, K]
B, K = memory.shape[:2]

memory_flat = memory.reshape(B * K, 3, H, W)
memory_latents_flat = self._encode_input_image_latents_tensor_batch(memory_flat)
memory_latents = memory_latents_flat.reshape(B, K, C, 1, h, w)
memory_latents = memory_latents.permute(0, 2, 1, 3, 4, 5).reshape(B, C, K, h, w)

memory_patch = self.video_expert.patchify(memory_latents)
memory_tokens = rearrange(memory_patch, "b c f h w -> b (f h w) c")
```

这里的 `_encode_input_image_latents_tensor_batch()` 可以复用现有 VAE image encode 逻辑，但要支持 batch，不改变 VAE 编码语义。

### 3. RoPE temporal index 固定为 0

memory tokens 的 RoPE 要和 current first frame 一致。最简单、最稳的实现方式是直接复制 current first frame 的 `freqs`：

```python
tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
first_frame_freqs = video_pre["freqs"][:tokens_per_frame]  # temporal index = 0
memory_freqs = first_frame_freqs.repeat(K, 1, 1)
video_pre["freqs"] = torch.cat([memory_freqs, video_pre["freqs"]], dim=0)
```

不要为不同历史关键帧分配 `-K, ..., -1` 或 `1, 2, ...` 之类的 temporal RoPE index。即使有多个 memory keyframe，它们的 temporal RoPE 都固定为 0，和 current first frame 一致。

### 4. t_mod 与 current first frame 一致

FastWAM 当前 `video_expert.pre_dit()` 在 `seperated_timestep && fuse_vae_embedding_in_latents` 下会把 first latent frame 的 token timestep 设为 0：

```python
token_timesteps[:, 0, :] = 0
```

memory tokens 也应该使用同样的 timestep 0 调制。实现上同样可以复制 current first frame 的 `t_mod`：

```python
tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
first_frame_t_mod = video_pre["t_mod"][:, :tokens_per_frame]  # [B, S, 6, D]
memory_t_mod = first_frame_t_mod.repeat(1, K, 1, 1)
video_pre["t_mod"] = torch.cat([memory_t_mod, video_pre["t_mod"]], dim=1)
```

这保证 memory tokens 和 current first frame 一样被模型视为 clean observation tokens，而不是 noisy future tokens。

### 5. Attention mask

不能直接调用原来的 `_build_mot_attention_mask()` 后完事，因为原逻辑只让 action attend 到 current first-frame tokens。加入 memory prefix 后，要让 action 同时 attend 到：

- 有效 memory tokens。
- current first frame tokens。
- action tokens 自身。

建议构造 condition token block：

- `condition_tokens = valid_memory_tokens + current_first_frame_tokens`
- memory/current-first-frame 查询只 attend 到 condition block，不 attend 到 future noisy video tokens。
- future video tokens attend 到全部 video tokens，包括 memory、current first frame 和 future noisy video tokens，保持原 `first_frame_causal` 中未来帧可看条件帧的性质。
- action tokens attend 到 condition block 和 action tokens。

padding memory tokens 必须在 mask 中全局不可见；如果某个 sample 的某个 keyframe 是 padding，其对应 `tokens_per_frame` 个 memory tokens 都不应被任何 query attend。

注意：`memory_keyframe_mask` 是逐样本不同的，而当前 MoT 路径里的 mixed self-attention mask 主要按 2D `[S, S]` 全 batch 共享。为了正确屏蔽 padding memory tokens，Memory Prefix Video Tokens 需要把 MoT/flash attention 的 self-attention mask 扩展到 batch-wise `[B, S, S]`，或在进入 attention 前构造等价的 per-sample 可见性 mask。不要用一个 2D mask 假装所有 sample 都有同样数量的有效 memory，否则 padding token 会被 action/video token 看到。

### 6. Post-DiT slicing 与 video loss

memory tokens 是条件 token，不是要预测的视频帧。因此：

- `self.mot()` 输入 video 分支时包含 `memory_tokens + normal_video_tokens`。
- `video_expert.post_dit()` 前必须切掉 memory prefix，只把 `normal_video_tokens` 传回 `post_dit()`。
- video loss 仍然只对原始训练视频 latent 计算，不包含 memory latent。
- KEM head 仍然接在 `tokens_out["action"]` 上，不接 memory tokens。

伪代码：

```python
memory_seq_len = K * tokens_per_frame
video_tokens_with_memory = torch.cat([memory_tokens, video_pre["tokens"]], dim=1)
video_pre["tokens"] = video_tokens_with_memory

tokens_out = self.mot(...)
normal_video_tokens_out = tokens_out["video"][:, memory_seq_len:]
pred_video = self.video_expert.post_dit(normal_video_tokens_out, video_pre_without_memory_meta)
```

这里需要保留一份不含 memory prefix 的 `video_pre` meta，或者在 `post_dit()` 前恢复 `grid_size` 对应的正常视频帧数。

### 7. 推理 KV cache

`infer_action()` 当前会 prefill current first frame 的 video KV cache。加入 memory prefix 后，prefill 输入要变为：

```text
memory video tokens + current first frame tokens
```

并且 action denoise 循环中使用同一份 video KV cache。由于 memory tokens 的 RoPE 和 `t_mod` 都与 current first frame 一致，推理时它们和训练时的条件 token 语义保持一致。

### 8. Teacher memory 与 predicted memory 的训练 schedule

参考 EventVLA，训练时的 memory source 应支持：

- `teacher`: 始终使用数据侧按 `keyframe_steps` 取出的历史关键帧。
- `predict`: 使用运行时 KEM 预测维护的 memory bank。
- `union`: teacher 与 predict 合并去重。
- `teacher_to_predict`: 前期 teacher，随后按概率逐步切到 predict。
- `none`: 消融实验，不塞历史关键帧。

FastWAM 的建议训练节奏：

1. **Warmup**：`keyframe_train_memory_source=teacher`。模型先学会在有 teacher 历史关键帧条件时预测动作/视频/KEM。
2. **Scheduled sampling**：切到 `teacher_to_predict`，用 `keyframe_schedule_teacher_prob` 从 1 降到 0。
3. **Student memory**：顺序 sampler 下维护 runtime memory bank，按 KEM 预测的 `should_trigger_event` 和 `pred_event_offset` 延迟写入未来观测。

如果还使用随机 sampler，就只能可靠地做 teacher memory，因为随机 batch 没有 episode 内连续状态；要训练 predicted runtime memory，必须使用顺序 sampler，并用 `is_new_episode` 重置 bank。

### 9. 训练/推理一致性

训练阶段用 teacher memory 时，输入模型的字段和推理时保持同名同形状：

- 训练：`memory_keyframe_video/mask/steps` 来自 label。
- 推理：同样字段来自 policy wrapper 维护的 memory bank。

这样 `FastWAM.infer_action()`、`infer_joint()` 只需要新增可选参数：

```python
memory_keyframe_video=None
memory_keyframe_mask=None
memory_keyframe_steps=None
```

然后走同一个 VAE encode + patchify + prefix mask 路径。推理中 KEM 输出只负责决定何时把新观测写入 bank，不改变模型 forward 的输入接口。

## KEM 模型方案

### 1. Head 放置位置

推荐在 `FastWAM.training_loss()` 的 MoT 输出后接：

```python
action_features = tokens_out["action"]  # [B, action_horizon, hidden_dim]
kem_logits = self.kem_head(action_features).squeeze(-1)  # [B, action_horizon]
```

原因：

- `tokens_out["action"]` 已经融合了文本、首帧视频、proprio、noisy action token 和 mixed attention 信息。
- token 序列长度天然等于 action horizon，和 `chunk_keyframe_target` 对齐。
- 不需要改 ActionDiT 的输入输出协议。

head 结构可参考 EventVLA：

```python
nn.Sequential(
    nn.LayerNorm(action_hidden_dim),
    nn.Linear(action_hidden_dim, action_hidden_dim),
    nn.GELU(),
    nn.Linear(action_hidden_dim, 1),
)
```

其中 `action_hidden_dim = self.action_expert.hidden_dim`。

### 2. Loss 与指标

新增模型参数：

- `kem_enabled: bool = false`
- `kem_loss_weight: float = 1.0`
- `kem_positive_weight: float = 7.0`
- `kem_threshold: float = 0.5`
- `event_future_min_offset: int = 1`
- `event_commit_threshold: float = 0.55`
- `kem_target_dilation: int = 8`
- `kem_target_kernel: raised_cosine`

训练时：

- 如果 `sample["chunk_keyframe_target"]` 存在且 `use_keyframe_supervision` 有真值，则计算 KEM loss。
- `F.binary_cross_entropy_with_logits(kem_logits, target, pos_weight=...)`
- 如果当前 batch 没有监督样本，返回零损失但让 head 保持在 graph 上，避免 DDP unused parameter。
- `loss_total = loss_video * lambda_video + loss_action * lambda_action + kem_loss_weight * loss_kem`

日志指标：

- `loss_kem`
- `kem_target_rate`
- `kem_pred_rate`
- `kem_accuracy`
- `kem_recall`
- `kem_precision`
- `event_commit_accuracy`
- `event_commit_recall`
- `event_commit_precision`
- `event_offset_mae`

### 3. 推理 event 选择

先实现与 EventVLA 等价的纯函数：

- `_select_chunk_event(kem_probs, threshold=None)`
- `_select_chunk_event_candidates(kem_probs, threshold=None, nms_window=None)`
- `_select_inference_chunk_event(kem_probs, episode_ids, current_steps)`

默认策略：

1. 跳过 `event_future_min_offset` 之前的位置，避免把当前帧立即写入 memory。
2. 选未来概率最大 offset。
3. confidence 超过 `event_commit_threshold` 才触发。
4. eval 阶段可用 `keyframe_inference_nms_window` 和 `keyframe_inference_cooldown_steps` 抑制重复触发。

第一阶段训练必须使用 teacher 历史关键帧作为模型输入；这里的“不写 memory”仅指暂时不要求模型用自己的预测维护 runtime bank。KEM 头先返回：

- `chunk_keyframe_prob`
- `pred_event_offset`
- `pred_event_confidence`
- `should_trigger_event`
- `teacher_event_offset`
- `teacher_event_confidence`
- `teacher_should_commit`

第二阶段再接入 `experiments/robotwin_mem/fastwam_policy/deploy_policy.py` 和顺序训练 runtime bank，根据 `should_trigger_event` 把未来 commit timestep 的观测加入 memory bank。

## 训练器与 optimizer 方案

当前 trainer 只训练：

```python
trainable_params = list(self.model.dit.parameters())
if proprio_encoder is not None:
    trainable_params.extend(list(proprio_encoder.parameters()))
```

采用 Memory Prefix Video Tokens 后，memory token 使用现有 VAE 和 `video_expert.patchify()` / `patch_embedding`，默认不新增独立 memory encoder 或 projector 参数。因此 optimizer 主要只需要额外加入 `kem_head`；如果后续做可选消融并引入额外 memory adapter，再把 adapter 参数加入 optimizer。

```python
kem_head = getattr(self.model, "kem_head", None)
if kem_head is not None:
    trainable_params.extend(list(kem_head.parameters()))
```

`_apply_dit_only_train_mode()` 也需要让 `kem_head.train()` 且 `requires_grad_(True)`。memory prefix 默认复用 `model.dit` 内的 video patch embedding，这部分已经在现有 DiT trainable 参数里。

checkpoint：

- `FastWAM.save_checkpoint()` 增加 `kem_head` state_dict 和 `kem_config`。
- 增加 `memory_config`，记录 prefix token 注入方式、`max_keyframes`、`include_current_keyframe` 等设置；默认没有额外 memory 权重需要保存。
- `load_checkpoint()` 有则加载，无则 warning 后随机初始化，兼容旧 checkpoint。

## 配置草案

`configs/train.yaml` 增加：

```yaml
sampler:
  type: random  # random | sequential_episode
  sampling_interval: 1
  shuffle_trajectories: true
  balance_dataset_step_counts: false
```

`configs/model/fastwam.yaml` 增加：

```yaml
kem:
  enabled: true
  loss_weight: 1.0
  positive_weight: 7.0
  threshold: 0.5
  event_future_min_offset: 1
  event_commit_threshold: 0.55
  inference_nms_window: 20
  inference_cooldown_steps: 20

keyframe_memory:
  enabled: true
  max_keyframes: 5
  include_current_keyframe: false
  selection: latest
  order: chronological
  source_train: teacher          # teacher | predict | union | teacher_to_predict | none
  source_eval: predict
  schedule_warmup_steps: 10000
  schedule_transition_steps: 30000
  schedule_teacher_prob_start: 1.0
  schedule_teacher_prob_end: 0.0
  schedule_mix_granularity: sample
  injection_mode: video_prefix_tokens
  rope_temporal_index: 0
  t_mod_source: current_first_frame
  compress_tokens: false

loss:
  lambda_video: 1.0
  lambda_action: 1.0
  lambda_kem: ${model.kem.loss_weight}
```

`configs/data/robotwin*.yaml` 的 train/val dataset 增加：

```yaml
keyframe_supervision:
  enabled: true
  target_dilation: 8
  target_kernel: raised_cosine
  event_future_min_offset: ${model.kem.event_future_min_offset}
  teacher_event_threshold: ${model.kem.event_commit_threshold}

keyframe_memory:
  enabled: true
  max_keyframes: ${model.keyframe_memory.max_keyframes}
  include_current_keyframe: ${model.keyframe_memory.include_current_keyframe}
  selection: ${model.keyframe_memory.selection}
  order: ${model.keyframe_memory.order}
```

注意 Hydra 里 `data` 引 `model` 可能形成解析依赖，实际实现时可选择在 data config 里重复这些值，或者放到顶层 `keyframe_supervision`。

## 推荐实施顺序

### 阶段 A：数据闭环

1. 在 `BaseLerobotDataset` 缓存 episode lengths、starts、keyframe steps。
2. 在 `RobotVideoDataset` 透传 episode/keyframe 查询方法。
3. 生成 `chunk_keyframe_target`、teacher event 字段，并确认 default random sampler 下也能正常返回这些字段。
4. 根据 `keyframe_steps < current_step` 读取历史关键帧，按当前视频相同的相机拼接和图像 transform 生成 `memory_keyframe_video/mask/steps`。
5. 新增顺序 sampler，但先只在单卡验证 `DataLoader` 能跑通。
6. 接入 `Wan22Trainer._build_loader()`，用 `cfg.sampler.type` 切换 random/sequential。

验收：

- 任取 RoboTwin-Mem 数据，打印一个 batch，确认 `episode_index/frame_index/chunk_keyframe_target` 对齐。
- 同一 batch 中 `memory_keyframe_steps` 必须都小于当前 `frame_index`，且图像 shape 与当前视频单帧一致。
- `chunk_keyframe_target.max()` 在接近 `keyframe_steps` 的 chunk 内接近 1，远离关键帧时为 0。
- DDP 下每个 rank 的 dataloader batch 数一致。

### 阶段 B：Teacher memory 注入模型

1. `FastWAM.__init__()` 加 memory prefix 配置，不新增默认 memory projector。
2. `build_inputs()` 接收并搬运 `memory_keyframe_video/mask/steps`。
3. `training_loss()` 中将历史关键帧经 VAE encode 和 `video_expert.patchify()` 变成 memory video tokens，prefix 到 `video_pre["tokens"]` 前。
4. memory tokens 的 `freqs` 复制 current first frame 的 `freqs`，即 RoPE temporal index 固定为 0。
5. memory tokens 的 `t_mod` 复制 current first frame 的 `t_mod`。
6. 修改 MoT attention mask，让 action attend 到有效 memory tokens 与 current first frame tokens。
7. `video_expert.post_dit()` 前切掉 memory prefix，只对正常视频 tokens 算 video loss。
8. `infer_action()` / `infer_joint()` 增加同名 memory 参数，复用同一个 VAE encode + patchify + prefix mask 路径。

验收：

- `keyframe_memory.enabled=false` 时旧训练行为完全不变。
- `keyframe_memory.enabled=true` 时 forward 能打印 `memory_keyframe_count`、`keyframe_input_memory_source=teacher`。
- memory tokens 数量必须等于 `valid_keyframes * tokens_per_frame`，不能被额外压缩。
- memory tokens 的 `freqs` 与 current first frame `freqs` 完全一致；`t_mod` 与 current first frame `t_mod` 完全一致。
- 没有历史关键帧的 early episode sample 不 crash，memory mask 全 false。

### 阶段 C：KEM 训练头

1. `FastWAM.__init__()` 加 `kem_head` 与 KEM 配置。
2. `runtime.create_fastwam()` 解析 `kem` 与 `keyframe_memory` 配置并传入模型。
3. `training_loss()` 在已经注入 memory 后的 `tokens_out["action"]` 上计算 `kem_logits`。
4. 加 `_compute_kem_loss()`、`_select_chunk_event()` 和指标。
5. trainer optimizer/freeze/checkpoint 加入 `kem_head`。

验收：

- `kem.enabled=false` 时仅训练 teacher memory 条件下的视频/动作损失。
- `kem.enabled=true` 且无 keyframe 标注数据时不 crash，KEM loss 为 0 或不计入。
- RoboTwin-Mem 数据上日志出现 `loss_kem`、`kem_target_rate`、`memory_keyframe_count`，且 loss 可下降。

### 阶段 D：Predict memory 与推理 commit

1. 在 `infer_action()` 或 policy wrapper 层返回 KEM 输出。
2. 在顺序 sampler 训练中维护 per-slot runtime bank，支持 `teacher_to_predict` schedule。
3. 在 `experiments/robotwin_mem/fastwam_policy/deploy_policy.py` 维护 per-env episode state：
   - 当前 step
   - pending event
   - last committed step
   - memory image bank
4. 当 `should_trigger_event` 为真时，按 `pred_event_offset` 延迟到未来 timestep 写入观测。
5. 加 NMS/cooldown，避免同一关键事件附近重复写。

验收：

- 可视化每个 episode 的 predicted event timestep，与 `keyframe_steps` 计算 offset MAE。
- eval policy 中 memory 写入次数接近 teacher keyframe 数，不出现每几步重复写。

## 主要风险与处理

- **action horizon 对齐风险**：FastWAM action horizon 是 `num_frames - 1`，video 用 `action_video_freq_ratio` 下采样后只有 9 帧；KEM target 必须按 action step 对齐，不能按 video transition 对齐。
- **tuple index collate 风险**：PyTorch sampler 输出 tuple 后，dataset 必须先转换为普通 tensor/标量字段，否则 default collate 可能生成难用的嵌套结构。
- **memory token 数量风险**：一张 384x320 图经 VAE 和 patch 后会产生完整 `tokens_per_frame` 个 token，`max_keyframes=5` 会显著增加 MoT self-attention 开销；本方案明确不压缩 memory tokens，因此需要通过限制 `max_keyframes`、分辨率或 batch size 控制显存。
- **当前帧语义风险**：memory prefix 不能替代 current first frame。attention mask 中应把有效 memory tokens 和 current first frame tokens 一起视为 condition block；`post_dit()` 前必须切掉 memory prefix。
- **RoPE/t_mod 一致性风险**：memory tokens 必须复制 current first frame 的 `freqs` 和 `t_mod`，不要给历史关键帧分配独立 temporal RoPE index 或 noisy timestep。
- **batch-wise mask 风险**：不同 sample 的有效 memory 数不同，padding memory token 必须逐样本屏蔽；MoT attention mask 需要支持 `[B, S, S]` 或等价机制，不能只用共享 2D mask。
- **训练/推理分布风险**：训练必须先用 teacher 历史关键帧作为输入；后续再 scheduled sampling 到 predict memory，否则推理 memory bank 会是模型从未见过的条件。
- **DDP unused parameter 风险**：如果 KEM head 在某些 batch 没有 loss，要加 zero loss 或让 head 始终参与 graph。
- **旧 checkpoint 兼容风险**：旧 checkpoint 没有 `kem_head`，load 时应 warning 而不是报错。
- **配置依赖风险**：`data` 配置引用 `model.kem` 可能受 Hydra 解析顺序影响，必要时把 keyframe supervision 配置放在顶层。
- **memory commit 语义风险**：EventVLA 是 VLA 原生 raw image memory；FastWAM 应把 raw image memory 转成 prefix video tokens，再进入 MoT，不要把 memory 当成需要预测的 video target。

## 最小可行版本

如果想尽快落地，建议 MVP 只做：

1. 从 `meta/episodes.jsonl` 读 `keyframe_steps`。
2. 在现有随机 sampler 下生成 `chunk_keyframe_target`。
3. 根据 `keyframe_steps < current_step` 取最近 `max_keyframes` 个历史关键帧，生成 `memory_keyframe_video/mask/steps`。
4. 给 FastWAM 加 Memory Prefix Video Tokens 注入，让历史关键帧经 VAE latent 和 `video_expert.patchify()` 变成完整 memory video tokens，并 prefix 到 video branch。
5. 给 FastWAM 加 `kem_head + loss_kem + metrics`。

这才是最小闭环：模型训练时已经在历史关键帧条件下学习动作、视频和 KEM。顺序 sampler、teacher-to-predict schedule 与推理 memory commit 随后加入，会更贴近 EventVLA 的完整行为。
