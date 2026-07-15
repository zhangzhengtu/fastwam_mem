# 推理时 MEM_BOS / Keyframe Attention 可视化方案

## 目标问题

可以看，而且建议把它做成一个轻量的 inference attention probe：

- 每个 environment step 触发一次 `infer_action()` 或 `infer_joint()`，内部有 `num_inference_steps` 个 denoising step。
- 在每个 denoising step 内，统计 query token group 对 memory key token group 的 attention 强度。
- 对同一个 environment step 的所有 denoising step 取平均，最后得到可以画曲线的标量：
  - `noisy_action -> MEM_BOS`
  - `noisy_action -> keyframe visual tokens`
  - `noisy_video -> MEM_BOS`
  - `noisy_video -> keyframe visual tokens`
  - 可选：`noisy_action/noisy_video -> RECENT_BOS`、`recent visual tokens`

最终可视化：

```text
x-axis: environment step
y-axis: attention strength
```

注意：`infer_action()` 的 action-only 推理路径只会产生 noisy action query，没有 noisy video query；`noisy_video -> ...` 需要走 `infer()`、`infer_joint()` 或 IDM/video denoising 路径。

## 指标定义

不要直接保存完整 attention matrix。完整矩阵形状接近：

```text
[num_layers, batch, num_heads, query_len, key_len]
```

评估时很容易占显存、拖慢速度。推荐只在 attention 内部即时聚合成少量标量。

设某一层某一 head 的 attention 权重为：

```python
attn = softmax(q @ k.transpose(-2, -1) * scale + mask)
# shape: [B, H, Q, K]
```

定义 query groups：

- `action_all`: 所有 noisy action tokens。
- `video_noisy`: normal video tokens 中排除 memory prefix 和 current first-frame tokens 后的 future/noisy video tokens。
- `video_current_first`: 当前观测 first-frame tokens，可选 debug。

定义 key groups：

- `mem_bos`: `MEM_BOS` token。
- `keyframe_visual`: keyframe full visual tokens，不含 `MEM_BOS`。
- `recent_bos`: `RECENT_BOS` token。
- `recent_visual`: recent-history full visual tokens。
- `memory_all`: `MEM_BOS + keyframe_visual + RECENT_BOS + recent_visual`。

核心标量建议用 mean-over-query-head-layer：

```python
score(query_group, key_group)
  = attn[..., query_indices, key_indices].sum(dim=-1).mean()
```

这里对 key 维度用 `sum`，表示 query token 分配给整个 key group 的注意力质量；再对 query token、head、layer、denoising step 求平均。不要对 key 维度再 `mean`，否则 keyframe visual tokens 数量越多，数值会被 token 数稀释。

同时可以额外记录一个 token-normalized 版本：

```python
score_per_key_token = score / max(len(key_indices), 1)
```

它适合比较 `MEM_BOS` 单 token 和大块 `keyframe_visual` 的单 token 平均吸引力。

## Token span 必须显式记录

当前 `_prefix_memory_block_tokens()` 已返回：

```python
memory_seq_len
keyframe_seq_len
recent_seq_len
memory_bos_seq_len
recent_bos_seq_len
memory_token_mask
```

但可视化最好不要靠这些长度临时硬推。建议把 `memory_info` 扩展出稳定的 span metadata：

```python
memory_info.update({
    "spans": {
        "mem_bos": (0, memory_bos_seq_len),
        "keyframe_visual": (
            memory_bos_seq_len,
            keyframe_seq_len,
        ),
        "recent_bos": (
            keyframe_seq_len,
            keyframe_seq_len + recent_bos_seq_len,
        ),
        "recent_visual": (
            keyframe_seq_len + recent_bos_seq_len,
            memory_seq_len,
        ),
        "normal_video": (
            memory_seq_len,
            video_seq_len,
        ),
        "current_first_frame": (
            memory_seq_len,
            memory_seq_len + tokens_per_frame,
        ),
        "noisy_video": (
            memory_seq_len + tokens_per_frame,
            video_seq_len,
        ),
        "action": (
            video_seq_len,
            video_seq_len + action_seq_len,
        ),
    }
})
```

对非 structured block 的旧 `memory_keyframe_video` prefix，可以把：

```python
mem_bos = empty
keyframe_visual = (0, memory_seq_len)
recent_bos = empty
recent_visual = empty
```

这样同一套 probe 能兼容旧版和新版 memory。

## Hook 位置

### 1. action-only 推理

`FastWAM.infer_action()` 当前流程是：

```text
video_pre = video_expert.pre_dit(first_frame_latents)
memory_info = _prefix_memory_video_tokens(...)
attention_mask = _build_mot_attention_mask(...)
video_kv_cache = mot.prefill_video_cache(...)

for action denoising step:
    _predict_action_noise_with_cache(...)
        mot.forward_action_with_video_cache(...)
```

这里最适合统计：

```text
noisy_action -> MEM_BOS
noisy_action -> keyframe_visual
noisy_action -> RECENT_BOS
noisy_action -> recent_visual
```

需要在 `MoT.forward_action_with_video_cache()` 增加可选参数：

```python
attention_probe: Optional[AttentionProbe] = None
probe_step: Optional[int] = None
token_spans: Optional[dict[str, tuple[int, int]]] = None
```

在每层构造出：

```python
k_cat = torch.cat([k_video, k_action], dim=1)
v_cat = torch.cat([v_video, v_action], dim=1)
```

之后，用同一份 `q_action/k_cat/attention_mask` 计算统计值。正常 forward 仍然调用 `flash_attention()` 输出 mixed attention，不改变模型结果。

### 2. joint/video 推理

如果要看 noisy video token：

```text
noisy_video -> MEM_BOS
noisy_video -> keyframe_visual
```

需要在 `MoT.forward()` 或 joint/video denoising路径中统计 mixed self-attention。此时 query 是 `[video tokens + action tokens]`，所以可以同时统计：

```text
video_noisy -> memory groups
action_all -> memory groups
```

`infer_action()` 没有 future noisy video tokens，因此不能从 action-only cache 路径得到 `noisy_video` 曲线。

## AttentionProbe 设计

新增一个轻量对象，例如：

```python
class AttentionProbe:
    def __init__(self, enabled: bool, groups: list[tuple[str, str]]):
        self.enabled = enabled
        self.groups = groups
        self.rows = []

    def add(self, *, env_step, denoise_step, layer_idx, values):
        self.rows.append({
            "env_step": int(env_step),
            "denoise_step": int(denoise_step),
            "layer_idx": int(layer_idx),
            **{k: float(v) for k, v in values.items()},
        })
```

在 attention 内部只保存聚合值：

```python
def collect_attention_stats(q, k, mask, *, num_heads, query_spans, key_spans):
    bsz, q_len, hidden = q.shape
    head_dim = hidden // num_heads
    qh = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    kh = k.view(bsz, k.shape[1], num_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(qh.float(), kh.float().transpose(-2, -1)) * (head_dim ** -0.5)

    if mask is not None:
        # mask=True means visible. Convert invisible positions to -inf.
        visible = mask
        if visible.ndim == 2:
            visible = visible.unsqueeze(0).unsqueeze(0)
        elif visible.ndim == 3:
            visible = visible.unsqueeze(1)
        scores = scores.masked_fill(~visible.to(torch.bool), torch.finfo(scores.dtype).min)

    attn = torch.softmax(scores, dim=-1)
    out = {}
    for q_name, q_span in query_spans.items():
        q0, q1 = q_span
        if q1 <= q0:
            continue
        for k_name, k_span in key_spans.items():
            k0, k1 = k_span
            if k1 <= k0:
                continue
            mass = attn[:, :, q0:q1, k0:k1].sum(dim=-1).mean()
            out[f"{q_name}_to_{k_name}"] = mass.detach().cpu()
            out[f"{q_name}_to_{k_name}_per_key"] = (mass / max(k1 - k0, 1)).detach().cpu()
    return out
```

实际实现时要注意 action cache 路径里的 `action_attention_mask` 形状是 `[Sa, Sv+Sa]`，对应 query span 应该从 action-local 坐标开始：

```python
query_spans = {"action_all": (0, action_seq_len)}
key_spans = {
    "mem_bos": memory_info["spans"]["mem_bos"],
    "keyframe_visual": memory_info["spans"]["keyframe_visual"],
    "recent_bos": memory_info["spans"]["recent_bos"],
    "recent_visual": memory_info["spans"]["recent_visual"],
}
```

joint `MoT.forward()` 路径中 query/key 都是 joint 坐标，可以直接用 `memory_info["spans"]`。

## 每个 environment step 怎么聚合

每次 policy 重新规划 action chunk 时，记一次 environment step，例如 RoboTwin-Mem 的：

```python
experiments/robotwin_mem/fastwam_policy/deploy_policy.py
  _infer_action_chunk()
```

给 `infer_action()` 传：

```python
return_attention_stats=True
attention_env_step=self._env_step_or_plan_step
```

模型返回：

```python
pred["attention_stats"] = {
    "per_denoise_layer": rows,
    "summary": {
        "action_to_mem_bos": ...,
        "action_to_keyframe_visual": ...,
    },
}
```

聚合规则：

```python
summary_for_env_step =
    mean over denoise_step
    mean over selected layers
```

推荐默认层选择：

- `all_layers_mean`: 全层平均，作为主曲线。
- `last_4_layers_mean`: 可选，更接近输出决策。
- `per_layer`: 保存到 csv，后续画 heatmap。

保存为一行 JSONL/CSV：

```json
{
  "episode_id": 0,
  "env_step": 48,
  "plan_index": 12,
  "num_denoise_steps": 20,
  "memory_count": 3,
  "action_to_mem_bos": 0.031,
  "action_to_mem_bos_per_key": 0.031,
  "action_to_keyframe_visual": 0.214,
  "action_to_keyframe_visual_per_key": 0.00042,
  "action_to_recent_bos": 0.018,
  "action_to_recent_visual": 0.117
}
```

建议保存路径：

```text
evaluate_results/robotwin_mem/<ckpt_tag>/<run_ts>/<task_name>/attention_stats/
  episode_000.jsonl
  episode_000_curves.png
  episode_000_layers.png
```

## 曲线绘制

最小画图脚本读取 JSONL 后：

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

rows = [json.loads(line) for line in open("episode_000.jsonl")]
df = pd.DataFrame(rows).sort_values("env_step")

plt.figure(figsize=(10, 4))
plt.plot(df["env_step"], df["action_to_mem_bos"], label="action -> MEM_BOS")
plt.plot(df["env_step"], df["action_to_keyframe_visual"], label="action -> keyframe visual")
plt.plot(df["env_step"], df["action_to_recent_bos"], label="action -> RECENT_BOS")
plt.plot(df["env_step"], df["action_to_recent_visual"], label="action -> recent visual")
plt.xlabel("environment step")
plt.ylabel("attention mass")
plt.legend()
plt.tight_layout()
plt.savefig("episode_000_curves.png", dpi=200)
```

如果使用 `infer_joint()` 并启用 video probe，再额外画：

```python
plt.plot(df["env_step"], df["video_noisy_to_mem_bos"], label="noisy video -> MEM_BOS")
plt.plot(df["env_step"], df["video_noisy_to_keyframe_visual"], label="noisy video -> keyframe visual")
```

建议把 `attention mass` 和 `per_key` 两种图分开画：

- `attention mass`: 看模型整体把多少注意力分配给一个 memory group。
- `per_key`: 看单个 BOS/token 的平均吸引力，适合比较 BOS vs 大块 visual tokens。

## 实现步骤

1. 扩展 `memory_info`
   - 在 `_prefix_memory_video_tokens()` / `_prefix_memory_block_tokens()` 中返回 `spans`。
   - structured block 下显式返回 `mem_bos`、`keyframe_visual`、`recent_bos`、`recent_visual`。
   - 非 structured prefix 下把全部 memory prefix 视为 `keyframe_visual`。

2. 新增 attention probe 工具
   - 建议放在 `src/fastwam/models/wan22/attention_probe.py` 或 `src/fastwam/utils/attention_probe.py`。
   - 只计算并保存聚合标量，不保存完整 attention matrix。

3. 改 `MoT.forward_action_with_video_cache()`
   - 增加可选 `attention_probe`、`probe_context` 参数。
   - 每层对 `q_action/k_cat/action_attention_mask` 统计 action query 到 memory key spans。
   - 默认关闭，不影响训练和普通评估。

4. 改 `MoT.forward()`
   - 用于 joint/video 路径。
   - 每层对 joint attention 统计 `video_noisy` 和 `action_all` 到 memory key spans。

5. 改 `FastWAM.infer_action()` / `infer()` / `infer_joint()`
   - 增加参数：
     ```python
     return_attention_stats: bool = False
     attention_env_step: Optional[int] = None
     attention_layer_mode: str = "all"
     ```
   - 每个 denoising step 调 probe。
   - 推理结束后把 probe summary 放进返回 dict。

6. 改 policy/eval 保存
   - 在 `deploy_policy.py::_infer_action_chunk()` 中，从 `pred["attention_stats"]` 取 summary。
   - 追加写入 `attention_stats/episode_xxx.jsonl`。
   - 评估结束后调用画图脚本生成 PNG。

## 校验 checklist

- 没有 memory 时，所有 memory attention 字段应为 0 或缺省，不应报错。
- `MEM_BOS` mask 为 false 的 sample 不应被统计进有效 attention。
- `action_to_mem_bos + action_to_keyframe_visual + action_to_recent_bos + action_to_recent_visual` 不要求等于 1，因为 action 还会 attend current first frame 和 action tokens。
- `keyframe_visual` 用 key 维度 sum，`keyframe_visual_per_key` 才除以 token 数。
- action-only 路径不能声称有 `noisy_video_to_*`。
- 开启 probe 前后，固定 seed 的 action 输出应保持一致或在数值误差范围内一致。
- probe 默认关闭，普通训练/评估速度不受影响。

## RoboTwin-Mem 评估并自动画图命令

下面命令会在 RoboTwin-Mem 评估时开启 attention probe，并在每个 episode 结束后自动保存：

- `*_attention.jsonl`: 每次 replan/environment step 的 attention summary。
- `*_attention.png`: attention 曲线图。

当前落盘和绘图只保留：

- `env_step`
- `action_to_keyframe_visual`
- `action_to_recent_visual`
- `action_to_current_visual`

```bash
cd /nav-oss/yangganlin/tzz_workspace/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"

RUN_TS="attn_$(date +%Y%m%d_%H%M%S)"
CKPT_TAG="robotwin_cover_blocks_hard_fastwam_3cam_384_1e-4_0713_step_020000"

python experiments/robotwin_mem/eval_robotwin_mem_single.py \
  task=robotwin_cover_blocks_hard_fastwam_3cam_384_1e-4 \
  ckpt=./runs/robotwin_cover_blocks_hard_fastwam_3cam_384_1e-4_0713/checkpoints/weights/step_020000.pt \
  EVALUATION.robotwin_root=third_party/RoboTwin-Mem \
  EVALUATION.task_name=cover_blocks_hard \
  EVALUATION.task_config=demo_clean \
  EVALUATION.dataset_stats_path=./runs/robotwin_cover_blocks_hard_fastwam_3cam_384_1e-4_0713/dataset_stats.json \
  EVALUATION.eval_num_episodes=10 \
  EVALUATION.eval_video_log=True \
  EVALUATION.attention_stats_enabled=True \
  EVALUATION.attention_stats_layer_mode=all \
  EVALUATION.output_dir=./evaluate_results/robotwin_mem/${RUN_TS} \
  gpu_id=0

ATTN_DIR="./evaluate_results/robotwin_mem/${CKPT_TAG}/${RUN_TS}/cover_blocks_hard/attention_stats"
ls -lh "${ATTN_DIR}"
```

输出示例：

```text
evaluate_results/robotwin_mem/${CKPT_TAG}/${RUN_TS}/cover_blocks_hard/attention_stats/
  episode0_demo-clean_success-true_attention.jsonl
  episode0_demo-clean_success-true_attention.png
  episode0_demo-clean_success-true_attention_contribution.png
  episode1_demo-clean_success-false_attention.jsonl
  episode1_demo-clean_success-false_attention.png
  episode1_demo-clean_success-false_attention_contribution.png
```

JSONL 每行包含当前 environment step、每个 keyframe slot 的 attention mass、每个 keyframe slot 的真实 value contribution，以及 recent/current visual attention：

```json
{
  "env_step": 24,
  "keyframe_00_step": 12,
  "action_to_keyframe_00_visual": 0.1031,
  "action_to_keyframe_00_visual_contrib": 0.8427,
  "keyframe_01_step": -1,
  "action_to_keyframe_01_visual": 0.0,
  "action_to_keyframe_01_visual_contrib": 0.0,
  "action_to_recent_visual": 0.5729,
  "action_to_current_visual": 0.1832
}
```

`*_attention.png` 绘制：

```text
action_to_keyframe_00_visual
action_to_keyframe_01_visual
...
action_to_recent_visual
action_to_current_visual
```

`*_attention_contribution.png` 绘制：

```text
action_to_keyframe_00_visual_contrib
action_to_keyframe_01_visual_contrib
...
```

如果某个 episode/step 还没有对应 keyframe slot，则该 slot 的 attention 和 contribution 都记为 0，`keyframe_xx_step` 记为 `-1`。

如果只想看最后一次运行的 attention 图：

```bash
find "./evaluate_results/robotwin_mem/${CKPT_TAG}/${RUN_TS}/cover_blocks_hard/attention_stats" \
  -name '*_attention.png' \
  -print
```
