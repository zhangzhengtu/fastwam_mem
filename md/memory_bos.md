# FastWAM 结构化 Memory Block + Memory BOS 方案

## 目标

当前 FastWAM 已经支持把 `memory_keyframe_video` 作为 video prefix tokens 注入 MoT：历史关键帧或历史帧会和当前帧一样经过 VAE encode、`video_expert.patchify()`，再拼到普通 video tokens 前面。这个做法简单有效，但所有 memory tokens 都混在同一个 prefix 里，模型只能通过位置和训练分布隐式地区分：

- 当前帧 clean observation
- 固定历史帧，例如 `t-10`、`t-5`
- teacher/predicted keyframe
- padding memory

新的目标是把现有 `memory_keyframe_video` 改成 **结构化 memory block**，并在 keyframe block 和 recent-history block 前分别放一个可学习 BOS token。这样模型在 test time 能更清楚地把“当前观测”“关键帧记忆”和“短期历史”分开，为后续学习“需不需要读 memory / 需不需要写 keyframe / 需不需要读 recent history”打基础。

本方案借鉴两类思路：

- **MemoryWAM**：recent history 和 event/keyframe anchor 保留 full visual tokens，通过 3D RoPE 和 attention mask 区分 token 的时序角色。
- **NativeMEM**：在 memory 序列前放一个 learnable token，显式标记 memory block 的入口。这里进一步拆成 `MEM_BOS` 和 `RECENT_BOS`，分别标记 keyframe memory 与 recent-history memory。

本阶段 **不引入 gist tokens**。`t-10`、`t-5` 和 keyframe 都保留完整 video tokens。

## 总体结构

推荐 token 顺序：

```text
[MEM_BOS]
[keyframe full tokens: k1, k2, ...]
[RECENT_BOS]
[history full tokens: t-10]
[history full tokens: t-5]
[current and future video tokens]
[action tokens]
```

其中：

- `MEM_BOS` 是 keyframe memory block 的可学习入口 token，shape 为 `[1, 1, video_hidden_dim]`。
- `RECENT_BOS` 是 recent-history block 的可学习入口 token，shape 为 `[1, 1, video_hidden_dim]`。
- `history full tokens` 来自固定 offset，例如 `history_step_offsets = [-10, -5]`。
- `keyframe full tokens` 来自 teacher keyframe 或 predicted keyframe memory bank。
- `current and future video tokens` 保持现有 FastWAM video branch 逻辑，当前帧仍然是 first-frame clean observation。
- `action tokens` 仍由 ActionDiT 编码。

注意：`MEM_BOS` 和 `RECENT_BOS` 都不替代任何图像 token，它们只是两个 memory 子块的入口/边界 token。

## 为什么比 source embedding 更合适

source embedding 的作用是给 token 加身份标签，例如 `history`、`keyframe`、`current`。它很轻量，但仍然让所有 token 混在一个普通 prefix 序列里。

`MEM_BOS + RECENT_BOS + structured memory block` 的好处是：

1. **结构上分离 memory 和 current observation**
   模型可以通过 `MEM_BOS`、`RECENT_BOS` 和 attention mask 明确知道“这里是 keyframe memory”“这里是 recent history”，而不是只靠 embedding 猜。

2. **方便后续加 read gate**
   后续可以让 action/current tokens 对两个 BOS 的使用强度或额外 head 预测：

   ```text
   read_keyframe_memory_prob
   read_recent_history_prob
   read_keyframe_prob
   ```

   如果不需要 memory，可以 mask 掉整个 memory block，或把 memory contribution 乘 gate。

3. **保留 full visual detail**
   `t-10`、`t-5` 和 keyframe 仍然保留完整 patch tokens，不做 mean pooling 或 gist 压缩，避免关键细节丢失。

4. **和现有 FastWAM 改动兼容**
   现有 `memory_keyframe_video -> VAE -> patchify -> prefix tokens` 可以继续复用，只需要把输入字段结构化，并插入一个 learnable token。

## 数据字段设计

建议把当前 `memory_keyframe_*` 字段逐步改名为更通用的 `memory_block_*`。为了兼容旧代码，可以先同时保留旧字段名。

### 新字段

训练 batch 中新增：

```python
memory_block_video: FloatTensor[K, 3, H, W]
memory_block_mask: BoolTensor[K]
memory_block_steps: LongTensor[K]
memory_block_source: LongTensor[K]
memory_block_offsets: LongTensor[K]
memory_block_count: LongTensor[]
```

collate 后：

```python
memory_block_video: [B, K, 3, H, W]
memory_block_mask: [B, K]
memory_block_steps: [B, K]
memory_block_source: [B, K]
memory_block_offsets: [B, K]
memory_block_count: [B]
```

为了固定 token layout，推荐把 slots 按 source 分区：

```text
keyframe slots: [0, max_keyframe_slots)
recent slots: [max_keyframe_slots, max_keyframe_slots + max_recent_slots)
```

如果继续使用单个 `memory_block_video`，则 `memory_block_source` 必须能唯一标出每个 slot 属于 keyframe 还是 recent。更清晰的实现也可以拆成两组字段：

```python
memory_keyframe_video: [B, K_key, 3, H, W]
memory_keyframe_mask: [B, K_key]
memory_recent_video: [B, K_recent, 3, H, W]
memory_recent_mask: [B, K_recent]
```

模型内部再统一拼成：

```text
[MEM_BOS][keyframe tokens][RECENT_BOS][recent tokens]
```

### source id 约定

```python
MEMORY_SOURCE_PAD = 0
MEMORY_SOURCE_HISTORY = 1
MEMORY_SOURCE_KEYFRAME_TEACHER = 2
MEMORY_SOURCE_KEYFRAME_PREDICTED = 3
```

本阶段不一定要使用 source embedding，但 `memory_block_source` 必须保留。它用于：

- debug
- attention/read gate 训练
- 未来区分 teacher keyframe 和 predicted keyframe
- 未来可选 source embedding 或 source-specific adapter

### offsets 约定

```python
memory_block_offsets = memory_step - current_step
```

例如：

```text
t-10 -> -10
t-5  -> -5
keyframe at step 80, current step 100 -> -20
padding -> 0 or sentinel
```

如果 future keyframe 只作为监督 target，不作为可见 memory，不应进入 `memory_block_video`。

## 数据构造逻辑

在 `RobotVideoDataset._build_memory_keyframes()` 的基础上改成 `_build_memory_block()`。

候选顺序建议固定为：

1. keyframe memory：按原来的 `selection=latest` 取历史 keyframe。
2. 固定历史帧：按 `history_step_offsets` 顺序取，例如 `[-10, -5]`。
3. padding：不足 `max_memory_blocks` 的位置补 0。

伪代码：

```python
selected_keyframes = []
selected_recent = []

keyframes = get_keyframe_steps(trajectory_id)
keyframes = [k for k in keyframes if k < current_step]
keyframes = keyframes[-max_keyframe_slots:]

for k in keyframes:
    selected_keyframes.append({
        "step": k,
        "source": MEMORY_SOURCE_KEYFRAME_TEACHER,
        "offset": k - current_step,
    })

for offset in history_step_offsets:
    memory_step = current_step + offset
    if memory_step >= 0:
        selected_recent.append({
            "step": memory_step,
            "source": MEMORY_SOURCE_HISTORY,
            "offset": offset,
        })

selected = selected_keyframes + selected_recent
```

每个 selected step 都使用和当前 `video` 一样的 `_format_video(..., [0])` 路径：

```text
multi-camera tensor
-> robotwin mosaic
-> resize / crop / normalize
-> [3, H, W], range [-1, 1]
```

这样 `history/keyframe/current` 的视觉预处理完全一致。

## 模型侧改动

### 1. 增加 learnable BOS tokens

在 `FastWAM.__init__()` 中：

```python
self.memory_bos_token = nn.Parameter(
    torch.zeros(1, 1, int(self.video_expert.hidden_dim), dtype=torch_dtype)
)
self.recent_bos_token = nn.Parameter(
    torch.zeros(1, 1, int(self.video_expert.hidden_dim), dtype=torch_dtype)
)
nn.init.normal_(self.memory_bos_token, std=0.02)
nn.init.normal_(self.recent_bos_token, std=0.02)
```

`memory_bos_token` 只在存在 keyframe memory 时插入；`recent_bos_token` 只在存在 recent-history memory 时插入。两个 BOS 都没有对应图像，仅用于标记子块边界。

### 2. build_inputs 接收结构化字段

`build_inputs()` 中优先读取新字段：

```python
memory_block_video = sample.get("memory_block_video", sample.get("memory_keyframe_video"))
memory_block_mask = sample.get("memory_block_mask", sample.get("memory_keyframe_mask"))
memory_block_steps = sample.get("memory_block_steps", sample.get("memory_keyframe_steps"))
memory_block_source = sample.get("memory_block_source", None)
memory_block_offsets = sample.get("memory_block_offsets", None)
```

为了兼容现有代码，可以在返回 dict 中同时保留：

```python
"memory_block_video": memory_block_video,
"memory_block_mask": memory_block_mask,
"memory_block_steps": memory_block_steps,
"memory_block_source": memory_block_source,
"memory_block_offsets": memory_block_offsets,

# old aliases
"memory_keyframe_video": memory_block_video,
"memory_keyframe_mask": memory_block_mask,
"memory_keyframe_steps": memory_block_steps,
```

### 3. 替换 `_prefix_memory_video_tokens()`

建议把函数改名为：

```python
_prefix_memory_block_tokens()
```

输入：

```python
video_pre
memory_block_video
memory_block_mask
memory_block_source
memory_block_offsets
tiled
```

输出：

```python
{
    "memory_seq_len": int,
    "keyframe_seq_len": int,
    "recent_seq_len": int,
    "memory_bos_seq_len": 1 or 0,
    "recent_bos_seq_len": 1 or 0,
    "memory_token_mask": BoolTensor[B, memory_seq_len],
}
```

### 4. token 构造

当前已有逻辑：

```python
memory_flat = memory_block_video.reshape(B * K, 3, H, W)
memory_flat = memory_flat.unsqueeze(2)
memory_latents_flat = self._encode_video_latents(memory_flat)
memory_latents = reshape to [B, C, K, h, w]
memory_patch = self.video_expert.patchify(memory_latents)
memory_tokens = [B, K * tokens_per_frame, D]
```

然后根据 `memory_block_source` 分成两个子块：

```python
keyframe_tokens = tokens whose source is keyframe_teacher/keyframe_predicted
recent_tokens = tokens whose source is history
```

在 keyframe 子块前插入 `MEM_BOS`，在 recent 子块前插入 `RECENT_BOS`：

```python
pieces = []

if has_keyframe_memory:
    mem_bos = self.memory_bos_token.to(dtype=memory_tokens.dtype, device=memory_tokens.device)
    pieces.append(mem_bos.expand(B, 1, -1))
    pieces.append(keyframe_tokens)

if has_recent_history:
    recent_bos = self.recent_bos_token.to(dtype=memory_tokens.dtype, device=memory_tokens.device)
    pieces.append(recent_bos.expand(B, 1, -1))
    pieces.append(recent_tokens)

memory_tokens = torch.cat(pieces, dim=1)
```

最终：

```python
video_pre["tokens"] = torch.cat([memory_tokens, video_pre["tokens"]], dim=1)
```

如果一个 batch 中某个 sample 没有 keyframe，但另一个 sample 有 keyframe，仍然建议在 batch 维度保留固定 layout：

```text
[MEM_BOS][K_keyframe slots][RECENT_BOS][K_recent slots]
```

没有对应 memory 的 sample 通过 `memory_token_mask` mask 掉 BOS 和 visual tokens。这样 batch 内 token shape 稳定。

## 3D RoPE 设计

当前 FastWAM memory prefix 逻辑把 memory frame 的 RoPE temporal index 固定为 current first frame 的 0。这对“memory 只是额外条件图”是稳定的，但对于结构化 memory block，建议升级为：

```text
MEM_BOS: 使用 keyframe memory marker 坐标
RECENT_BOS: 使用 recent-history marker 坐标
history t-10: temporal index = -10 或离散 bucket
history t-5: temporal index = -5 或离散 bucket
keyframe: temporal index = clamp(keyframe_step - current_step)
current frame: temporal index = 0
future noisy video: 保持原 video_pre 的时间索引
```

### 最小稳定版

为了不大改 RoPE 生成代码，第一版可以：

- `MEM_BOS` 和 `RECENT_BOS` 都复制 current first-frame 的第一个 token RoPE。
- 所有 memory full tokens 继续复制 current first-frame RoPE。
- 通过 attention mask 和两个 BOS 先完成结构分离。

这是最稳的 warm start 方式。

### 推荐增强版

后续再加 relative temporal RoPE：

```python
relative_t = memory_block_offsets
relative_t = clamp(relative_t, min=-max_history, max=0)
```

然后为每个 memory frame 生成对应 temporal coordinate。这样 `t-10`、`t-5`、keyframe 不只靠顺序区分，还能在 positional space 中区分。

如果当前 `video_expert.freqs` 生成接口不支持负时间索引，可以先做 bucket：

```text
bucket 0: MEM_BOS
bucket 1: keyframe
bucket 2: RECENT_BOS
bucket 3: history -10
bucket 4: history -5
bucket 5: current
```

但从语义上，relative temporal index 更接近 MemoryWAM。

## Attention Mask 设计

记：

```text
M = memory_seq_len = keyframe block + recent block
M_key = 1 + K_key * tokens_per_frame
M_recent = 1 + K_recent * tokens_per_frame
V = normal_video_seq_len
A = action_seq_len
```

token 排列：

```text
[MEM_BOS][keyframe tokens][RECENT_BOS][recent tokens][normal video V][action A]
```

推荐第一版 mask：

### memory block 内部

```text
MEM_BOS 可以看所有有效 keyframe visual tokens
keyframe visual tokens 可以看 MEM_BOS 和同一个 keyframe frame 内 tokens
RECENT_BOS 可以看所有有效 recent visual tokens
recent visual tokens 可以看 RECENT_BOS 和同一个 recent frame 内 tokens
padding memory tokens 不可见
```

为了实现简单，也可以第一版让有效 memory block 内部全可见，但仍保持两个 BOS 的位置边界：

```python
mask[:M, :M] = valid_memory_token_mask
```

### current first frame

当前 first-frame tokens 可以看：

```text
current first-frame tokens
keyframe block
recent-history block
```

这让当前观测能够和 memory 对齐，但不会让 future noisy frames 污染 memory。

### future video tokens

future noisy video tokens 可以看：

```text
原 video-to-video causal mask
keyframe block
recent-history block
```

保持 video diffusion 的条件建模能力。

### action tokens

action tokens 可以看：

```text
action tokens
current first-frame tokens
keyframe block
recent-history block
```

这和当前 FastWAM “action 看 first frame + memory prefix” 的语义一致，只是 memory block 被显式结构化。

### padding

`memory_block_mask[B, K]` 需要扩展成 visual token mask：

```python
visual_mask = memory_block_mask.repeat_interleave(tokens_per_frame, dim=1)
keyframe_visual_mask = keyframe_mask.repeat_interleave(tokens_per_frame, dim=1)
recent_visual_mask = recent_mask.repeat_interleave(tokens_per_frame, dim=1)
mem_bos_mask = keyframe_visual_mask.any(dim=1, keepdim=True)
recent_bos_mask = recent_visual_mask.any(dim=1, keepdim=True)
memory_token_mask = torch.cat(
    [mem_bos_mask, keyframe_visual_mask, recent_bos_mask, recent_visual_mask],
    dim=1,
)
```

如果一个 sample 没有任何 memory：

- 可以不插入 `MEM_BOS` 和 `RECENT_BOS`，`memory_seq_len=0`。
- 或保留固定 layout，但 mask 掉对应 BOS 和 visual tokens。

建议第一版如果要最小改动：**没有任何有效 memory 时不插入两个 BOS**。如果要 batch shape 最稳定，则始终保留固定 layout，并用 mask 控制每个 sample 的有效性。

## 推理侧 RoboTwin-Mem 设计

当前 `deploy_policy.py` 里已经有：

- `history_observation_bank`
- `memory_bank`
- `_memory_tensors()`

建议改为 `_memory_block_tensors()`，返回：

```python
memory_block_video
memory_block_mask
memory_block_steps
memory_block_source
memory_block_offsets
```

推理时构造顺序：

1. 从 predicted keyframe `memory_bank` 取 keyframe slots：

```python
source = MEMORY_SOURCE_KEYFRAME_PREDICTED
offset = keyframe_step - current_step
```

2. 从 `history_observation_bank` 取 `step-10`、`step-5`：

```python
source = MEMORY_SOURCE_HISTORY
offset = target_step - current_step
```

3. padding。

然后传给：

```python
infer_action(..., memory_block_video=..., memory_block_mask=..., ...)
infer_joint(..., memory_block_video=..., memory_block_mask=..., ...)
```

为了兼容旧 checkpoint/旧函数签名，可以保留：

```python
if "memory_block_video" in infer_params:
    pass new fields
elif "memory_keyframe_video" in infer_params:
    pass old aliases
```

## 配置建议

```yaml
keyframe_memory:
  enabled: true
  max_keyframes: 5
  history_step_offsets: [-10, -5]
  structured_block: true
  use_memory_bos: true
  use_recent_bos: true
  memory_bos_insert_when_empty: false
  recent_bos_insert_when_empty: false
  use_relative_memory_rope: false
  max_keyframe_slots: 5
  max_recent_slots: 2
  full_token_sources: [history, keyframe_teacher, keyframe_predicted]
  source_field_enabled: true
  offset_field_enabled: true
```

旧字段可以继续兼容。

## 实现步骤

### Step 1. 数据侧

修改：

```text
src/fastwam/datasets/lerobot/robot_video_dataset.py
```

新增：

```python
_build_memory_block()
```

返回新字段，同时保留旧 alias。

### Step 2. 推理 adapter

修改：

```text
experiments/robotwin_mem/fastwam_policy/deploy_policy.py
```

新增：

```python
_memory_block_tensors()
```

并在 infer 参数支持时传 `memory_block_*`。

### Step 3. 模型输入

修改：

```text
src/fastwam/models/wan22/fastwam.py
src/fastwam/models/wan22/fastwam_idm.py
src/fastwam/models/wan22/fastwam_joint.py
```

新增：

```python
self.memory_bos_token
self.recent_bos_token
_prefix_memory_block_tokens()
```

旧 `_prefix_memory_video_tokens()` 可以包装新函数，降低改动面。

### Step 4. Attention mask

修改：

```python
_build_mot_attention_mask()
```

新增参数：

```python
memory_bos_seq_len: int = 0
recent_bos_seq_len: int = 0
keyframe_seq_len: int = 0
recent_seq_len: int = 0
memory_token_mask: Optional[torch.Tensor] = None
```

第一版可以把整个有效 memory block 当作 condition block：

```text
condition = keyframe block + recent-history block + current first frame
action attends condition
future video attends keyframe/recent blocks
```

### Step 5. 配置

修改：

```text
configs/model/fastwam.yaml
configs/model/fastwam_idm.yaml
configs/model/fastwam_joint.yaml
configs/data/robotwin.yaml
```

加入：

```yaml
structured_block: true
use_memory_bos: true
use_recent_bos: true
```

