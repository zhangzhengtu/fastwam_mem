from __future__ import annotations

from typing import Any, Optional

import torch


Span = tuple[int, int]


def _to_jsonable_spans(spans: dict[str, Span]) -> dict[str, list[int]]:
    return {name: [int(start), int(end)] for name, (start, end) in spans.items()}


def collect_attention_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    v: Optional[torch.Tensor] = None,
    num_heads: int,
    query_spans: dict[str, Span],
    key_spans: dict[str, Span],
) -> dict[str, float]:
    """Collect group-level attention mass without saving the full matrix."""
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"`q` and `k` must be [B,S,D], got {tuple(q.shape)} and {tuple(k.shape)}")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"`q`/`k` shape mismatch: {tuple(q.shape)} vs {tuple(k.shape)}")
    if v is not None and (v.ndim != 3 or v.shape != k.shape):
        raise ValueError(f"`v` must match `k` shape [B,S,D], got {tuple(v.shape)} vs {tuple(k.shape)}")

    num_heads = int(num_heads)
    if num_heads <= 0 or q.shape[2] % num_heads != 0:
        raise ValueError(f"Invalid num_heads={num_heads} for hidden dim {q.shape[2]}")
    head_dim = q.shape[2] // num_heads

    qh = q.float().reshape(q.shape[0], q.shape[1], num_heads, head_dim).transpose(1, 2)
    kh = k.float().reshape(k.shape[0], k.shape[1], num_heads, head_dim).transpose(1, 2)
    vh = None
    if v is not None:
        vh = v.float().reshape(v.shape[0], v.shape[1], num_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * (head_dim ** -0.5)

    if mask is not None:
        visible = mask.to(device=scores.device, dtype=torch.bool)
        if visible.ndim == 2:
            visible = visible.unsqueeze(0).unsqueeze(0)
        elif visible.ndim == 3:
            visible = visible.unsqueeze(1)
        elif visible.ndim != 4:
            raise ValueError(f"`mask` must be 2D, 3D, or 4D, got {tuple(mask.shape)}")
        scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)

    attn = torch.softmax(scores, dim=-1)
    out: dict[str, float] = {}
    for query_name, (q_start, q_end) in query_spans.items():
        q_start = max(int(q_start), 0)
        q_end = min(int(q_end), attn.shape[-2])
        if q_end <= q_start:
            continue
        for key_name, (k_start, k_end) in key_spans.items():
            k_start = max(int(k_start), 0)
            k_end = min(int(k_end), attn.shape[-1])
            if k_end <= k_start:
                continue
            mass = attn[:, :, q_start:q_end, k_start:k_end].sum(dim=-1).mean()
            key = f"{query_name}_to_{key_name}"
            value = float(mass.detach().cpu().item())
            out[key] = value
            out[f"{key}_per_key"] = value / float(max(k_end - k_start, 1))
            if vh is not None:
                contribution = torch.matmul(attn[:, :, q_start:q_end, k_start:k_end], vh[:, :, k_start:k_end])
                out[f"{key}_contrib"] = float(contribution.norm(dim=-1).mean().detach().cpu().item())
    return out


class AttentionProbe:
    def __init__(
        self,
        *,
        env_step: Optional[int] = None,
        layer_mode: str = "all",
        num_layers: Optional[int] = None,
        query_spans: Optional[dict[str, Span]] = None,
        key_spans: Optional[dict[str, Span]] = None,
    ):
        self.env_step = env_step
        self.layer_mode = str(layer_mode)
        self.num_layers = None if num_layers is None else int(num_layers)
        self.query_spans = dict(query_spans or {})
        self.key_spans = dict(key_spans or {})
        self.rows: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return bool(self.query_spans and self.key_spans)

    def add(self, *, denoise_step: int, layer_idx: int, values: dict[str, float]) -> None:
        row: dict[str, Any] = {
            "denoise_step": int(denoise_step),
            "layer_idx": int(layer_idx),
        }
        if self.env_step is not None:
            row["env_step"] = int(self.env_step)
        row.update({key: float(value) for key, value in values.items()})
        self.rows.append(row)

    def _row_selected(self, row: dict[str, Any]) -> bool:
        mode = self.layer_mode.lower()
        if mode in {"all", "all_layers", "all_layers_mean"}:
            return True
        if self.num_layers is None:
            return True
        layer_idx = int(row["layer_idx"])
        if mode in {"last", "last_layer"}:
            return layer_idx == self.num_layers - 1
        if mode in {"last_4", "last4", "last_4_layers"}:
            return layer_idx >= max(self.num_layers - 4, 0)
        if mode.startswith("last_"):
            try:
                count = int(mode.split("_", 1)[1])
            except ValueError:
                return True
            return layer_idx >= max(self.num_layers - count, 0)
        return True

    def summary(self) -> dict[str, float]:
        rows = [row for row in self.rows if self._row_selected(row)]
        if not rows:
            return {}
        skip_keys = {"env_step", "denoise_step", "layer_idx"}
        value_keys = sorted({key for row in rows for key in row if key not in skip_keys})
        out: dict[str, float] = {}
        for key in value_keys:
            values = [float(row[key]) for row in rows if key in row]
            if values:
                out[key] = float(sum(values) / len(values))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_mode": self.layer_mode,
            "num_layers": self.num_layers,
            "query_spans": _to_jsonable_spans(self.query_spans),
            "key_spans": _to_jsonable_spans(self.key_spans),
            "per_denoise_layer": list(self.rows),
            "summary": self.summary(),
        }
