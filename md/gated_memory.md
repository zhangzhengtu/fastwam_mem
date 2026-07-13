# FastWAM 双时间尺度视觉记忆方案

## Always-on Recent Context + Label-Free Gated Keyframe Memory

## 1. Proposal 概述

本方案在 FastWAM 中引入两种不同时间尺度的视觉历史：

1. **Recent memory**

   * 表示最近几个固定历史观测；
   * 每次推理始终输入；
   * 保留完整视觉 patch tokens；
   * 使用独立的 `RECENT_BOS` 标记；
   * 不经过 read gate。

2. **Keyframe memory**

   * 表示由 KEM 写入的稀疏关键事件帧；
   * 只有 read gate 打开时才输入；
   * 保留 memory bank 中全部有效 keyframe 的完整视觉 tokens；
   * 使用独立的 `MEM_BOS` 标记；
   * gate 关闭时完全跳过 keyframe 的读取和后续计算。

最终模型结构为：

```text
Always-on:
RECENT_BOS + recent full tokens + current visual tokens

Conditionally-on:
MEM_BOS + all keyframe full tokens
```

Read gate 只回答一个问题：

> 当前视觉状态是否通常需要长期 keyframe memory？

它不判断：

* 具体哪张 keyframe 有用；
* 当前 memory 内容是否与动作匹配；
* 是否应该读取某一个 keyframe；
* recent history 是否需要输入。

因此：

```text
gate OFF:
[MEM_BOS]
[RECENT_BOS][recent full tokens]
[current visual tokens]
→ Video DiT
→ action denoising

gate ON:
[MEM_BOS][all keyframe full tokens]
[RECENT_BOS][recent full tokens]
[current visual tokens]
→ Video DiT
→ action denoising
```

`MEM_BOS` 始终存在，但 gate OFF 时不存在任何 keyframe visual tokens。

---

# 2. 与当前实现的关系

当前 FastWAM 的低延迟推理流程是：

```text
current image
→ VAE
→ Video DiT 单次 prefill
→ 缓存逐层 video K/V
→ 多步 Action DiT denoising
```

Action denoising 期间不再重复运行 Video DiT，这是 FastWAM 推理效率的核心。

你当前已经实现或规划的内容包括：

* keyframe metadata；
* episode 顺序 sampler；
* KEM head；
* teacher keyframe memory；
* predicted runtime memory bank；
* teacher-to-predict schedule；
* full-token memory prefix；
* `MEM_BOS` 与 `RECENT_BOS`；
* recent history bank；
* batch-wise memory attention mask；
* memory prefix 在 video post-DiT 前切除。

现有结构化 memory 方案已经将 token 分成：

```text
[MEM_BOS]
[keyframe full tokens]
[RECENT_BOS]
[recent-history full tokens]
[current/future video tokens]
[action tokens]
```

并明确 recent 和 keyframe 都保留完整视觉 tokens，不做 pooling 或 gist 压缩。

本 proposal 不推翻这些内容，而是在此基础上增加：

> 只针对 keyframe block 的无标签 hard read gate。

---

# 3. 两类历史信息的职责

## 3.1 Recent memory：始终开启的短期上下文

Recent memory 用于提供连续运动和局部状态变化信息，例如：

* 机械臂刚才从哪个方向移动；
* 物体刚才是否发生位移；
* 当前抓取状态是否稳定；
* 最近一次接触前后的视觉变化；
* 单帧当前观测无法直接表达的速度和短期动态。

Recent memory 是基础 observation context，而不是可选的长期记忆。

因此：

```text
每次 policy inference：
RECENT_BOS 始终输入；
所有有效 recent tokens 始终输入；
action 始终允许读取 recent tokens。
```

它不受 keyframe gate 控制。

## 3.2 Keyframe memory：稀疏长期事件信息

Keyframe memory 用于保存相对长期、离散的重要事件，例如：

* 某个对象之前已经被操作；
* 某个容器之前已经打开；
* 某一步子任务已经完成；
* 某个关键状态曾经出现但当前画面中不再可见；
* 需要跨越较长时间进行历史消歧的信息。

Keyframe 由 KEM 写入。

Read gate 只负责决定：

```text
当前是否需要读取长期 keyframe block。
```

打开后不做 retrieval，直接读取 bounded memory bank 中全部有效 keyframe。

## 3.3 Gate 的准确语义

由于 recent tokens 始终存在，read gate 实际学习的是：

> 在已经拥有 current observation 和 recent context 的条件下，当前任务阶段是否通常还需要长期 keyframe history？

这是比“是否需要任何历史”更精确的定义。

记：

[
g_t=P(
\text{need long-term keyframe memory}
\mid
\text{current visual state}
)
]

而不是：

[
P(
\text{need any temporal context}
\mid
\text{current visual state}
)
]

---

# 4. 最终 Token Layout

## 4.1 Gate OFF

```text
[MEM_BOS]
[RECENT_BOS]
[recent frame 1 full tokens]
[recent frame 2 full tokens]
...
[current frame tokens]
[future noisy video tokens, training only]
[action tokens]
```

此时：

* `MEM_BOS` 作为“长期 memory block 未开启”的结构标记；
* 不创建 keyframe visual tokens；
* 不加载 keyframe latent；
* 不运行 keyframe patchify；
* Video DiT 序列中没有 keyframe tokens；
* video K/V cache 中没有 keyframe K/V；
* action attention 中没有 keyframe K/V。

## 4.2 Gate ON

```text
[MEM_BOS]
[keyframe 1 full tokens]
[keyframe 2 full tokens]
...
[RECENT_BOS]
[recent frame 1 full tokens]
[recent frame 2 full tokens]
...
[current frame tokens]
[future noisy video tokens, training only]
[action tokens]
```

此时 memory bank 中全部有效 keyframe 被输入。

这里的“全部”指：

```text
memory bank 当前保留的全部 keyframe，
受 max_keyframes 上限限制。
```

不做：

* top-k keyframe selection；
* learned retrieval；
* keyframe importance ranking；
* keyframe token compression。

## 4.3 为什么两个 BOS 都必须保留

`MEM_BOS` 和 `RECENT_BOS` 代表不同的语义结构：

```text
MEM_BOS:
长期、稀疏、事件驱动、可选读取

RECENT_BOS:
短期、连续、固定保留、始终读取
```

它们不仅是 source embedding，而是在序列中显式建立两个视觉条件区域。

这样模型可以区分：

* 当前观测；
* 最近视觉动态；
* 长期事件记忆。

现有 structured memory block 设计已经为这种分离提供了基础。

---

# 5. Read Gate 设计

## 5.1 Gate 输入

主方案中 gate 只读取：

```text
learnable MEM_BOS
+
current visual patch tokens
```

不读取：

* keyframe 内容；
* recent tokens；
* `RECENT_BOS`；
* KEM probability；
* memory count；
* instruction；
* proprio；
* previous gate state。

这是一个有意限制。

目标不是进行最优的 history-aware routing，而是学习一个：

> Current-state memory phase detector。

也就是说，只根据当前视觉场景判断这一类状态通常是否需要长期 memory。

## 5.2 MEM_BOS 的双重用途

只保留一个 learnable `MEM_BOS` 参数。

它有两个使用位置：

### Router query

```text
MEM_BOS
→ cross-attend current visual tokens
→ router hidden
→ read probability
```

### Main Video DiT marker

原始 learnable `MEM_BOS` 作为长期 memory block 的序列边界：

```text
[MEM_BOS][optional keyframe tokens]...
```

推荐不要把 router 更新后的 hidden 直接塞入主 Video DiT。

也就是说：

```text
同一个 learnable parameter 被共享；
router 输出只用于 gate；
main DiT 使用原始 MEM_BOS token。
```

这样可以避免 router hidden 变成额外的 current-summary side channel。

## 5.3 Gate 所在位置

Gate 必须放在：

```text
current VAE encode
→ current patch embedding
→ read gate
→ main Video DiT prefill
```

而不能放在完整 current-frame Video DiT prefill 之后。

最终流程：

```text
current image
    ↓
current VAE latent
    ↓
current patch tokens
    ↓
MEM_BOS router
    ↓
hard gate
    ├── OFF:
    │   recent + current
    │
    └── ON:
        keyframe + recent + current
    ↓
恰好一次 main Video DiT prefill
    ↓
Action DiT denoising
```

无论 gate 开关，都只运行一次主 Video DiT。

---

# 6. 推理流程

## 6.1 Always-on 部分

每次推理始终执行：

```text
1. 当前图像 VAE encode；
2. 当前 visual patchify；
3. 准备 RECENT_BOS；
4. 读取 recent latent bank；
5. patchify recent latents；
6. 运行 lightweight read router。
```

Recent tokens 属于基础推理成本。

## 6.2 Gate OFF

```text
current patch tokens
+
RECENT_BOS
+
recent full tokens
+
MEM_BOS
→ Video DiT prefill
→ current/recent video K/V
→ Action DiT denoising
```

此时不访问 keyframe latent bank。

## 6.3 Gate ON

```text
读取全部 cached keyframe latents
→ keyframe patchify
→ 拼入 MEM_BOS 后
→ 与 recent/current 一起运行 Video DiT
→ keyframe/recent/current K/V
→ Action DiT denoising
```

## 6.4 重要的成本定义

加入 recent 后，推理成本分为：

```text
基础成本：
current + recent

条件成本：
keyframe memory
```

因此 gate 节省的是：

```text
长期 keyframe block 的计算
```

而不是 recent-history 成本。

平均成本大致为：

[
C_{\mathrm{avg}}
================

C_{\mathrm{current}}
+
C_{\mathrm{recent}}
+
p_{\mathrm{read}}C_{\mathrm{keyframe}}
]

相对 always-on keyframe：

[
C_{\mathrm{always}}
===================

C_{\mathrm{current}}
+
C_{\mathrm{recent}}
+
C_{\mathrm{keyframe}}
]

所以 recent 数量越多，keyframe gate 带来的总加速比例会被 recent 的固定成本稀释，但长期 keyframe 数量较多时仍然有明显收益。

---

# 7. Recent Latent Bank

## 7.1 Recent 不应每次从 RGB 重复 VAE 编码

Recent tokens 每次推理都需要输入，因此 recent frame 应维护独立的 latent bank：

```text
Recent Latent Bank
```

每个 entry 至少保存：

```text
step
VAE latent
valid mask
relative offset
```

当前帧在完成 VAE encode 后，可以直接进入 rolling recent bank：

```text
current latent
→ 完成本次推理
→ detach
→ 写入 recent latent bank
```

下次推理时直接读取，不再编码。

## 7.2 Recent offset 的推荐定义

存在两种方案。

### 方案 A：按 policy inference / replan index

例如始终保存：

```text
previous replan -2
previous replan -1
```

优点：

* 这些图像本来就是历史 current frame；
* 已经完成 VAE encode；
* recent 不产生额外 VAE write cost；
* 与 FastWAM action chunk 推理方式最兼容。

这是推荐的第一版。

### 方案 B：按 environment step

例如当前设计中的：

```text
t-10
t-5
```

如果这些 step 不是之前的 replan frame，就需要在对应环境 step 额外执行一次 VAE，或者先保存 RGB 再延迟编码。

这会增加额外成本。

因此配置中应明确：

```text
history_offset_unit:
policy_call
或
environment_step
```

主实验建议先使用：

```text
history_offset_unit = policy_call
```

之后再消融精确环境步 offset。

## 7.3 Episode 开始

Episode 初期 recent frame 不足时：

* `RECENT_BOS` 仍然存在；
* 缺失的 recent slots 使用 padding；
* padding recent tokens 不允许被任何 query attend；
* 不建议使用当前帧复制填充，因为会制造虚假的时间变化。

---

# 8. Keyframe Latent Bank

## 8.1 写入机制

KEM 继续负责：

```text
什么时候写入 keyframe。
```

KEM head 接在 action tokens 上，预测 action chunk 内的关键事件概率和 future event offset。现有方案已经规划了：

* chunk keyframe supervision；
* soft target dilation；
* commit threshold；
* NMS；
* cooldown；
* teacher-to-predict schedule。

## 8.2 Keyframe 在写入时缓存 VAE latent

当某个观测被 commit 为 keyframe 时：

```text
keyframe observation
→ VAE encode 一次
→ 保存 VAE latent
```

后续每次 gate ON：

```text
直接读取 cached latent
→ patchify
→ Video DiT
```

不再重复运行 keyframe VAE。

如果 keyframe 恰好是当前 policy observation：

```text
直接复用本次已经计算的 current latent。
```

如果 keyframe 是 action chunk 执行中间的未来 observation：

```text
在实际到达 commit step 时，
额外运行一次 VAE 并写入 bank。
```

## 8.3 Keyframe bank 容量

建议：

```text
max_keyframes = 3–5
```

Gate 打开后读取 bank 中全部有效 keyframe。

Bank 满时采用确定性的 eviction：

```text
默认：移除最早 keyframe
```

第一版不要同时引入 learned eviction。

## 8.4 不缓存 joint Video-DiT KV

可以安全缓存：

```text
VAE latent
```

不建议缓存：

```text
最终 joint Video-DiT K/V
```

因为 keyframe token 经过主 Video DiT 后的表示可能依赖：

* 当前 visual tokens；
* recent tokens；
* instruction context；
* 当前 attention mask；
* 当前 token relative position。

因此 keyframe VAE 只编码一次，但 gate ON 时仍重新经过主 Video DiT。

---

# 9. 数据字段

建议不要再把 recent 与 keyframe 混在单一不透明字段中，而是显式拆开。

## 9.1 Keyframe 字段

```text
memory_keyframe_latents
memory_keyframe_mask
memory_keyframe_steps
memory_keyframe_source
memory_keyframe_count
```

Source 区分：

```text
teacher keyframe
predicted keyframe
padding
```

## 9.2 Recent 字段

```text
memory_recent_latents
memory_recent_mask
memory_recent_steps
memory_recent_offsets
memory_recent_count
```

## 9.3 兼容字段

现有实现已经使用或规划：

```text
memory_block_video
memory_block_mask
memory_block_steps
memory_block_source
memory_block_offsets
```

可以继续保留统一字段作为兼容层，但模型内部应尽早拆成：

```text
keyframe block
recent block
```

现有 structured memory 方案已经定义了 keyframe/history source id 和 offset 语义，可以直接复用。

---

# 10. Attention Mask

## 10.1 Condition block

定义：

```text
K = keyframe block
R = recent block
F0 = current clean frame
F+ = future noisy video
A = action tokens
```

结构：

```text
[MEM_BOS][K]
[RECENT_BOS][R]
[F0][F+][A]
```

## 10.2 Recent 始终可见

以下 query 始终允许读取有效 recent tokens：

```text
MEM_BOS
RECENT_BOS
current F0
future F+
action A
```

Recent tokens 本身可以读取：

```text
RECENT_BOS
其他有效 recent tokens
current F0
```

第一版也可以让整个 clean condition block 全连接，以简化实现。

## 10.3 Keyframe 受 gate 控制

Gate ON 时，keyframe block 与 clean condition block 正常连接：

```text
MEM_BOS
keyframe tokens
RECENT_BOS
recent tokens
current F0
```

Gate OFF 时：

* current 不得读取 keyframe；
* recent 不得读取 keyframe；
* action 不得读取 keyframe；
* future video 不得读取 keyframe；
* `MEM_BOS` 也不得读取 keyframe；
* keyframe 不得通过任何中间 token 间接泄漏到 action。

训练 soft gate 时，可以只对所有跨越 keyframe block 边界的 attention edges 加 gate bias：

[
b_{\mathrm{keyframe}}
=====================

\log(g+\epsilon)
]

Keyframe 内部 self-attention 可以保留，但其输出在 gate 接近 0 时不能影响任何非 keyframe token。

## 10.4 FastWAM 原始因果约束保持不变

保留：

```text
F0 不读取 F+
A 不读取 F+
A 可以读取 A
F+ 按原 first-frame-causal 方式读取视频条件
```

当前 FastWAM 的 action 本来只读取 clean first-frame video condition，不读取未来 noisy video。

加入 memory 后：

```text
A 始终读取 F0 + R；
gate ON 时额外读取 K。
```

## 10.5 Batch-wise mask

由于每个样本具有不同：

* keyframe count；
* recent count；
* keyframe gate；
* padding slots；

attention mask 必须支持 per-sample 可见性，例如：

```text
[B, S, S]
```

或等价的 block mask 机制。

现有方案已经明确指出不能只使用全 batch 共享二维 mask。

---

# 11. RoPE 与 Timestep Modulation

## 11.1 第一版稳定实现

建议第一版：

```text
MEM_BOS:
复制 current first-frame 的 marker RoPE

RECENT_BOS:
复制 current first-frame 的 marker RoPE

keyframe full tokens:
使用 clean condition timestep 0

recent full tokens:
使用 clean condition timestep 0

current:
保持现有 first-frame clean timestep 0
```

Keyframe 和 recent 都不是需要预测的 noisy video target。

## 11.2 Relative temporal information

Recent 对时序距离更加敏感，因此第二阶段优先给 recent 加 relative temporal coordinate：

```text
recent -1
recent -2
current 0
```

Keyframe 可以使用：

```text
clamped historical offset
```

但这不是 MVP 的必要条件。

推荐开发顺序：

```text
第一版：
两个 BOS + block mask + fixed temporal index

第二版：
recent relative temporal RoPE

第三版：
keyframe age/offset RoPE
```

不要同时修改所有位置编码机制，否则难以定位训练问题。

---

# 12. 无标签 Read Gate 训练

## 12.1 不构造任何 read label

禁止使用：

```text
memory-required label
keyframe proximity label
KEM label 作为 read label
有 memory / 无 memory loss 差
counterfactual utility target
gate BCE target
```

Gate 是一个 latent conditional-computation variable。

## 12.2 Gate 的优化目标

模型主任务 loss 为：

[
\mathcal L_{\mathrm{task}}
==========================

\lambda_a\mathcal L_{\mathrm{action}}
+
\lambda_v\mathcal L_{\mathrm{video}}
+
\lambda_k\mathcal L_{\mathrm{KEM}}
]

Read gate 只控制 keyframe block。

增加平均读取预算约束：

[
\mathbb E[g_t]\leq \rho
]

其中：

```text
ρ = 目标 keyframe read rate
```

例如：

```text
5%
10%
20%
```

推荐使用上界约束，而不是强制每个 batch 必须恰好读取 5%。

总目标可写成：

[
\mathcal L
==========

\mathcal L_{\mathrm{task}}
+
\mu\max(0,\bar p-\rho)
+
\lambda_b\mathcal L_{\mathrm{binary}}
]

其中：

[
\mathcal L_{\mathrm{binary}}
============================

\mathbb E[p(1-p)]
]

`binary loss` 只在训练后期逐渐加入。

## 12.3 Gate 如何得到任务梯度

训练早期可以始终构造 keyframe tokens，但通过 soft gate 控制：

```text
action/current/future 对 keyframe block 的 attention。
```

如果某种当前视觉状态下 keyframe 有助于降低 action loss：

```text
任务梯度会推动该状态的 gate probability 上升。
```

如果 recent + current 已经足够：

```text
读取预算约束会推动 gate probability 下降。
```

因此模型学习的是：

> 哪些视觉阶段值得把有限的长期 memory 计算预算花掉。

整个过程只有一次主任务 forward，不比较两个 policy 分支。

## 12.4 Recent 对 gate 学习的影响

由于 recent 始终输入，gate 不会为了简单的短期动态而打开 keyframe。

它只有在：

```text
current + recent 仍然不足
```

时才有动力开启长期 memory。

这会使 read gate 更接近真正的 long-term-memory gate，而不是通用 history gate。

---

# 13. 训练阶段

## Stage 1：Structured memory policy 预训练

输入始终包含：

```text
RECENT_BOS + recent
MEM_BOS + teacher keyframes
current
```

此时：

```text
keyframe gate = always ON
```

目标：

* 模型学会区分 recent 与 keyframe；
* 模型学会使用两种历史；
* KEM 学会预测 keyframe commit；
* 保证 keyframe-on 路径有效。

## Stage 2：Keyframe dropout warmup

Recent 始终保留。

随机隐藏整个 keyframe block：

```text
50% keyframe ON
50% keyframe OFF
```

随机值与样本内容无关，因此不是 label。

目标：

* 保持 recent + current 路径可用；
* 防止 policy 对 keyframe 形成绝对依赖；
* 为 router 提供稳定的两分支初始化。

## Stage 3：Soft label-free routing

启用：

```text
MEM_BOS + current tokens
→ soft read probability
```

Recent 始终正常输入。

初始 read budget 可以较宽松：

```text
20%
```

然后逐步下降：

```text
20% → 10% → 5%
```

此阶段训练中仍可构造 keyframe tokens，以获得稳定梯度。

## Stage 4：Hardening

逐步降低 gate temperature，使 soft gate 接近二值。

使用：

```text
hard forward
soft backward
```

或 straight-through binary gate。

如出现严重 collapse，可以使用 batch/global capacity top-k 作为稳定手段，但它不是第一版必须项。

## Stage 5：Teacher-to-predict keyframe memory

Recent bank 始终使用确定性的历史观测。

Keyframe memory source 按现有方案变化：

```text
teacher
→ teacher_to_predict
→ predicted
```

顺序 sampler 用于：

* episode reset；
* KEM future commit；
* runtime keyframe bank；
* predicted keyframe distribution；
* 防止跨 episode memory 泄漏。

现有方案已经明确，predicted memory 训练需要顺序 sampler，而随机 sampler 只能可靠构造 teacher memory。

## Stage 6：Hard inference fine-tuning

最终训练路径尽量接近部署：

```text
gate OFF:
不构造 keyframe block

gate ON:
输入全部 keyframe block

recent:
始终输入
```

对于 batch 内动态长度，工程上可按 gate 分成 ON/OFF 两个子 batch，或者仅在最后少量步骤进行真实 hard-routing 微调。

---

# 14. Video 与 Action Loss

Memory prefix 是 condition，不是 video target。

因此：

* recent tokens 不参与 video loss；
* keyframe tokens 不参与 video loss；
* `MEM_BOS` 不参与 video reconstruction；
* `RECENT_BOS` 不参与 video reconstruction；
* `video_expert.post_dit()` 前切掉全部 condition prefix；
* video loss 仍只对应原始 current/future video latent；
* KEM head 仍接在 action token 输出上。

当前 full-token memory 设计已经要求在 `post_dit()` 前切掉 memory prefix，并保持 KEM head 位于 action tokens。

---

# 15. 在线 Policy 状态

Policy wrapper 维护两个独立 bank。

## 15.1 Recent bank

```text
recent_latent_bank
```

特点：

* rolling；
* 固定 slot 数；
* 每次推理都读取；
* episode reset 清空；
* 不受 KEM 控制。

## 15.2 Keyframe bank

```text
keyframe_latent_bank
```

特点：

* event-driven；
* KEM 决定写入；
* gate 决定读取；
* bounded capacity；
* gate ON 时读取全部；
* episode reset 清空。

## 15.3 Pending commit

如果 KEM 预测未来 offset：

```text
current policy call
→ 产生 pending commit step
→ 环境执行到该 step
→ 编码该时刻 observation
→ 写入 keyframe latent bank
```

Recent bank 和 keyframe bank 可以包含同一个 frame，但语义不同：

```text
recent:
由于时间临近而保留

keyframe:
由于事件重要性而长期保留
```

第一版可以允许重复，后续再做 deduplication。

---

# 16. 推荐配置

```yaml
memory:
  structured_block: true

  recent:
    enabled: true
    use_recent_bos: true
    always_read: true
    max_recent_frames: 2

    # 推荐第一版
    offset_unit: policy_call
    offsets: [-2, -1]

    input_type: vae_latent
    full_visual_tokens: true

  keyframe:
    enabled: true
    use_memory_bos: true
    max_keyframes: 5
    selection: all_in_bank
    eviction: oldest

    input_type: vae_latent
    full_visual_tokens: true
    compression: none
    retrieval: none

  read_gate:
    enabled: true
    controls: keyframe_only

    input_mem_bos: true
    input_current_visual: true
    input_recent: false
    input_keyframe_content: false
    input_language: false
    input_proprio: false
    input_kem: false

    target_read_rate: 0.05
    budget_schedule: [0.20, 0.10, 0.05]

    soft_warmup: true
    hardening: true
    force_off_without_keyframes: true

  kem:
    enabled: true
    teacher_to_predict: true
    write_vae_latent: true
```

---

# 17. 代码改动范围

## 数据侧

主要修改：

```text
RobotVideoDataset
BaseLerobotDataset
sequence sampler
```

目标：

* 单独构造 recent 和 keyframe；
* 优先返回 cached latent；
* 保留旧 image path 作为 fallback；
* recent 固定可见；
* keyframe 按 teacher/predict source 构造。

## 模型侧

主要修改：

```text
fastwam.py
mot.py
wan_video_dit.py
```

新增：

* `memory_bos_token`；
* `recent_bos_token`；
* lightweight read router；
* recent latent patchify；
* keyframe latent conditional patchify；
* dynamic video prefill；
* batch-wise block attention mask。

## Trainer

新增日志：

```text
keyframe_read_prob
keyframe_hard_read_rate
keyframe_count
recent_count
read-budget loss
binary loss
gate temperature
```

## Policy adapter

将现有：

```text
history_observation_bank
memory_bank
```

明确拆成：

```text
recent_latent_bank
keyframe_latent_bank
pending_keyframe_commits
```

当前 policy 侧已经存在 history bank、memory bank 和 memory tensor 构造逻辑，因此这里主要是数据形态从 RGB/video tensor 迁移到分层 latent bank。

---

# 18. 实验设计

## 18.1 核心 Baselines

必须比较：

### Current only

```text
current
```

### Recent only

```text
recent + current
```

### Always-on keyframe

```text
keyframe + recent + current
```

### Random gated keyframe

```text
以相同 read rate 随机开启 keyframe
```

### Periodic keyframe read

```text
固定周期读取 keyframe
```

### Proposed gate

```text
MEM_BOS + current visual state
→ keyframe gate
→ recent 始终开启
```

## 18.2 Read rate

至少测试：

```text
2.5%
5%
10%
20%
100%
```

绘制：

```text
success rate
vs.
mean latency
```

## 18.3 关键消融

### Recent 作用

```text
无 recent
1 个 recent frame
2 个 recent frames
3 个 recent frames
```

### RECENT_BOS

```text
recent tokens without RECENT_BOS
recent tokens with RECENT_BOS
```

### Gate 输入

主方案：

```text
MEM_BOS + current
```

消融：

```text
MEM_BOS + current + recent
```

该消融可以确认：recent-aware gate 是否真的必要。

### Memory cache

```text
read 时从 RGB 编码
write 时缓存 VAE latent
```

### Keyframe gate

```text
soft gate
hard gate
random gate
periodic gate
```

---

# 19. 评测指标

由于没有 read labels，不报告：

```text
gate accuracy
gate precision
gate recall
```

这些指标没有合法 target。

报告：

## 任务表现

```text
success rate
per-task success
长时任务 success
不同 episode 长度下的 success
```

## Memory 使用

```text
keyframe read rate
平均连续开启长度
每次读取 keyframe 数
每 episode keyframe 写入数
recent frame 数
```

## 计算

```text
mean latency
gate-OFF latency
gate-ON latency
p50 / p95 / p99 latency
每 episode VAE 调用数
Video DiT token 数
video KV cache 长度
GPU peak memory
```

## 对比价值

最重要的比较是：

```text
Proposed 5% read
vs.
Random 5% read
vs.
Recent-only
vs.
Always-on keyframe
```

如果 Proposed 5% 明显优于 Random 5%，说明 gate 学到了视觉状态相关的长期 memory phase，而不是只依赖稀疏随机读取。

---

# 20. 主要风险

## 20.1 Recent 固定成本过高

Recent 始终输入，因此 recent frame 数量不能过多。

建议第一版：

```text
2 个 recent frames
```

否则 recent 固定成本可能吞噬 keyframe gating 的加速收益。

## 20.2 Gate 全开

处理：

* always-on policy 预训练后增加 random keyframe dropout；
* 使用 read budget；
* 逐步降低目标 read rate；
* 后期二值化。

## 20.3 Gate 全关

处理：

* 不从随机初始化直接训练 5% gate；
* 先训练强 keyframe-on policy；
* 先使用 20% budget；
* 再逐渐降低；
* 早期不使用强 binary penalty。

## 20.4 Keyframe 信息泄漏

Soft gate 训练时必须确保：

```text
gate OFF
=> keyframe 不能通过 MEM_BOS、
recent、current 或其他 residual 路径间接影响 action。
```

需要专门做 memory leakage 单元测试。

## 20.5 Recent offset 定义不一致

训练使用 environment-step offsets，而推理使用 policy-call offsets，会造成明显分布差异。

必须在配置和数据 metadata 中记录：

```text
offset_unit
```

并保持训练/推理一致。

## 20.6 VAE cache 失效

VAE latent cache 要求冻结：

* VAE；
* image preprocessing；
* camera mosaic；
* resize/crop；
* latent scaling。

任何变化都要重新生成 cache。

---

# 21. Proposal 的核心贡献

最终方案不是简单的：

```text
给 memory 加一个 gate。
```

而是：

## 双时间尺度视觉历史

```text
始终开启的短期 recent context
+
事件驱动的长期 keyframe memory
```

## 结构化视觉条件

```text
RECENT_BOS 标记短期历史
MEM_BOS 标记长期历史
```

## 无标签长期读取

```text
仅根据 MEM_BOS + current visual tokens
学习是否开启长期 keyframe block
```

## 完整视觉证据

```text
recent 和 keyframe 均保留 full visual tokens
```

## 真实条件计算

```text
gate OFF 时完全不实例化 keyframe tokens 和 K/V
```

## 写入时 latent cache

```text
keyframe 和 recent 均尽量复用或缓存 VAE latent
```

可以将整体方法概括为：

> **Dual-Timescale Structured Visual Memory for FastWAM**

或者更突出 read gate：

> **Label-Free Sparse Long-Term Memory Read with Always-On Recent Context**

---

# 22. 推荐实施顺序

## 第一步：Recent block 固化

完成：

```text
RECENT_BOS
recent latent bank
recent full-token prefix
recent always-on attention
```

先不引入 gate。

验证：

```text
recent + current
```

能够正常训练和推理。

## 第二步：Keyframe latent cache

把当前 keyframe memory 从：

```text
RGB → read 时 VAE
```

改成：

```text
write 时 VAE → cached latent
```

## 第三步：双 BOS structured layout

统一：

```text
[MEM_BOS][keyframe]
[RECENT_BOS][recent]
[current]
```

验证 video loss slicing 和 attention mask。

## 第四步：Router 只记录、不控制

增加：

```text
MEM_BOS + current → read probability
```

先记录 score 分布，不改变 forward。

## 第五步：Soft gate

Gate 只控制 keyframe attention。

Recent 始终保持开启。

## 第六步：Hard inference branch

实现：

```text
OFF:
recent + current

ON:
keyframe + recent + current
```

## 第七步：Teacher-to-predict

最后接入：

```text
顺序 sampler
KEM predicted commit
runtime keyframe latent bank
```

---

# 23. 最终架构定义

最终模型在每次推理时执行：

```text
Current frame
    ↓
VAE encode once
    ↓
Current patch tokens
    ├─────────────────────────────┐
    │                             │
    ▼                             ▼
MEM_BOS router              Recent latent bank
    │                             │
    │                             ▼
    │                    RECENT_BOS + recent tokens
    │
    ▼
Keyframe read gate
    │
    ├── OFF
    │     [MEM_BOS]
    │     [RECENT_BOS + recent]
    │     [current]
    │
    └── ON
          [MEM_BOS + all keyframes]
          [RECENT_BOS + recent]
          [current]
                │
                ▼
        One Video-DiT prefill
                │
                ▼
        Video K/V cache
                │
                ▼
        Action denoising
```

核心不变量是：

```text
Recent 每次都输入；
Keyframe 只有 gate ON 才输入；
Gate 只看 MEM_BOS 和 current visual tokens；
Keyframe 打开后全部输入；
没有任何 read label；
没有 counterfactual 双 forward；
每次推理只做一次主 Video DiT prefill。
```
