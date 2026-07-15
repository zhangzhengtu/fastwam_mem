from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .attention_probe import AttentionProbe
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)

MEMORY_SOURCE_PAD = 0
MEMORY_SOURCE_HISTORY = 1
MEMORY_SOURCE_KEYFRAME_TEACHER = 2
MEMORY_SOURCE_KEYFRAME_PREDICTED = 3


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        kem_config: Optional[dict[str, Any]] = None,
        keyframe_memory_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.kem_config = {
            "enabled": False,
            "loss_weight": 1.0,
            "positive_weight": 7.0,
            "threshold": 0.5,
            "event_future_min_offset": 1,
            "event_commit_threshold": 0.55,
            "inference_nms_window": None,
            "inference_cooldown_steps": 0,
            "debug_log_samples": False,
            "debug_log_max_samples": 1,
            "debug_log_rank": 0,
        }
        if kem_config:
            self.kem_config.update(dict(kem_config))
        self.kem_enabled = bool(self.kem_config.get("enabled", False))
        self.kem_loss_weight = float(self.kem_config.get("loss_weight", 1.0))
        self.kem_positive_weight = float(self.kem_config.get("positive_weight", 7.0))
        self.kem_threshold = float(self.kem_config.get("threshold", 0.5))
        self.event_future_min_offset = int(self.kem_config.get("event_future_min_offset", 1))
        self.event_commit_threshold = float(self.kem_config.get("event_commit_threshold", 0.55))
        self.kem_debug_log_samples = bool(self.kem_config.get("debug_log_samples", False))
        self.kem_debug_log_max_samples = max(int(self.kem_config.get("debug_log_max_samples", 1)), 1)
        self.kem_debug_log_rank = int(self.kem_config.get("debug_log_rank", 0))
        self.kem_head = None
        if self.kem_enabled:
            action_hidden_dim = int(self.action_expert.hidden_dim)
            self.kem_head = nn.Sequential(
                nn.LayerNorm(action_hidden_dim),
                nn.Linear(action_hidden_dim, action_hidden_dim),
                nn.GELU(),
                nn.Linear(action_hidden_dim, 1),
            ).to(dtype=torch_dtype)

        self.keyframe_memory_config = {
            "enabled": False,
            "max_keyframes": 0,
            "history_step_offsets": [],
            "include_current_keyframe": False,
            "selection": "latest",
            "order": "chronological",
            "source_train": "teacher",
            "source_eval": "predict",
            "injection_mode": "video_prefix_tokens",
            "rope_temporal_index": 0,
            "t_mod_source": "current_first_frame",
            "compress_tokens": False,
            "structured_block": False,
            "use_memory_bos": True,
            "use_recent_bos": True,
            "memory_bos_insert_when_empty": False,
            "recent_bos_insert_when_empty": False,
            "use_relative_memory_rope": False,
            "relative_memory_rope_max_offset": 128,
        }
        if keyframe_memory_config:
            self.keyframe_memory_config.update(dict(keyframe_memory_config))
        self.keyframe_memory_enabled = bool(self.keyframe_memory_config.get("enabled", False))
        self.keyframe_memory_max_keyframes = int(self.keyframe_memory_config.get("max_keyframes", 0))
        self.keyframe_memory_structured_block = bool(
            self.keyframe_memory_config.get("structured_block", False)
        )
        self.use_memory_bos = bool(self.keyframe_memory_config.get("use_memory_bos", True))
        self.use_recent_bos = bool(self.keyframe_memory_config.get("use_recent_bos", True))
        self.memory_bos_insert_when_empty = bool(
            self.keyframe_memory_config.get("memory_bos_insert_when_empty", False)
        )
        self.recent_bos_insert_when_empty = bool(
            self.keyframe_memory_config.get("recent_bos_insert_when_empty", False)
        )
        self.use_relative_memory_rope = bool(
            self.keyframe_memory_config.get("use_relative_memory_rope", False)
        )
        self.relative_memory_rope_max_offset = max(
            int(self.keyframe_memory_config.get("relative_memory_rope_max_offset", 128)),
            0,
        )
        self.memory_bos_token = nn.Parameter(
            torch.zeros(1, 1, int(self.video_expert.hidden_dim), dtype=torch_dtype)
        )
        self.recent_bos_token = nn.Parameter(
            torch.zeros(1, 1, int(self.video_expert.hidden_dim), dtype=torch_dtype)
        )
        nn.init.normal_(self.memory_bos_token, std=0.02)
        nn.init.normal_(self.recent_bos_token, std=0.02)

        self.to(self.device)

    @staticmethod
    def _empty_memory_attention_spans() -> dict[str, tuple[int, int]]:
        return {
            "mem_bos": (0, 0),
            "keyframe_visual": (0, 0),
            "recent_bos": (0, 0),
            "recent_visual": (0, 0),
            "memory_all": (0, 0),
        }

    @staticmethod
    def _action_attention_probe_spans(
        memory_info: dict[str, Any],
        action_seq_len: int,
        current_visual_span: tuple[int, int] = (0, 0),
    ) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
        memory_spans = dict(memory_info.get("spans") or FastWAM._empty_memory_attention_spans())
        key_names = {"mem_bos", "keyframe_visual", "recent_bos", "recent_visual", "memory_all"}
        key_spans = {
            name: tuple(span)
            for name, span in memory_spans.items()
            if name in key_names
            or (name.startswith("keyframe_") and name.endswith("_visual"))
            or (name.startswith("recent_") and name.endswith("_visual"))
        }
        key_spans["current_visual"] = tuple(current_visual_span)
        return {"action": (0, int(action_seq_len))}, key_spans

    @staticmethod
    def _joint_attention_probe_spans(
        memory_info: dict[str, Any],
        *,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
    ) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
        memory_seq_len = int(memory_info.get("memory_seq_len", 0))
        video_seq_len = int(video_seq_len)
        action_seq_len = int(action_seq_len)
        first_frame_end = min(memory_seq_len + int(video_tokens_per_frame), video_seq_len)
        memory_spans = dict(memory_info.get("spans") or FastWAM._empty_memory_attention_spans())
        key_names = {"mem_bos", "keyframe_visual", "recent_bos", "recent_visual", "memory_all"}
        key_spans = {
            name: tuple(span)
            for name, span in memory_spans.items()
            if name in key_names
            or (name.startswith("keyframe_") and name.endswith("_visual"))
            or (name.startswith("recent_") and name.endswith("_visual"))
        }
        key_spans["current_visual"] = (memory_seq_len, first_frame_end)
        query_spans = {
            "noisy_video": (first_frame_end, video_seq_len),
            "action": (video_seq_len, video_seq_len + action_seq_len),
        }
        return query_spans, key_spans

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        kem_config: Optional[dict[str, Any]] = None,
        keyframe_memory_config: Optional[dict[str, Any]] = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            kem_config=kem_config,
            keyframe_memory_config=keyframe_memory_config,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        memory_block_video = sample.get("memory_block_video", sample.get("memory_keyframe_video", None))
        memory_block_mask = sample.get("memory_block_mask", sample.get("memory_keyframe_mask", None))
        memory_block_steps = sample.get("memory_block_steps", sample.get("memory_keyframe_steps", None))
        memory_block_source = sample.get("memory_block_source", None)
        memory_block_offsets = sample.get("memory_block_offsets", None)
        if memory_block_video is not None:
            if memory_block_video.ndim != 5:
                raise ValueError(
                    "`sample['memory_block_video']` must be [B,K,3,H,W], "
                    f"got shape {tuple(memory_block_video.shape)}"
                )
            if memory_block_video.shape[0] != batch_size or memory_block_video.shape[2] != 3:
                raise ValueError(
                    "`memory_block_video` shape mismatch: "
                    f"got {tuple(memory_block_video.shape)}, expected B={batch_size}, C=3."
                )
            memory_block_video = memory_block_video.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            if memory_block_mask is None:
                memory_block_mask = torch.ones(
                    memory_block_video.shape[:2],
                    dtype=torch.bool,
                    device=self.device,
                )
            else:
                memory_block_mask = memory_block_mask.to(
                    device=self.device,
                    dtype=torch.bool,
                    non_blocking=True,
                )
            if memory_block_mask.shape != memory_block_video.shape[:2]:
                raise ValueError(
                    "`memory_block_mask` shape mismatch: "
                    f"got {tuple(memory_block_mask.shape)} vs expected {tuple(memory_block_video.shape[:2])}"
                )
            if memory_block_steps is not None:
                memory_block_steps = memory_block_steps.to(
                    device=self.device,
                    dtype=torch.long,
                    non_blocking=True,
                )
            if memory_block_source is not None:
                memory_block_source = memory_block_source.to(
                    device=self.device,
                    dtype=torch.long,
                    non_blocking=True,
                )
                if memory_block_source.shape != memory_block_video.shape[:2]:
                    raise ValueError(
                        "`memory_block_source` shape mismatch: "
                        f"got {tuple(memory_block_source.shape)} vs expected {tuple(memory_block_video.shape[:2])}"
                    )
            if memory_block_offsets is not None:
                memory_block_offsets = memory_block_offsets.to(
                    device=self.device,
                    dtype=torch.long,
                    non_blocking=True,
                )
                if memory_block_offsets.shape != memory_block_video.shape[:2]:
                    raise ValueError(
                        "`memory_block_offsets` shape mismatch: "
                        f"got {tuple(memory_block_offsets.shape)} vs expected {tuple(memory_block_video.shape[:2])}"
                    )

        chunk_keyframe_target = sample.get("chunk_keyframe_target", None)
        if chunk_keyframe_target is not None:
            chunk_keyframe_target = chunk_keyframe_target.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
        use_keyframe_supervision = sample.get("use_keyframe_supervision", None)
        if use_keyframe_supervision is not None:
            use_keyframe_supervision = use_keyframe_supervision.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
        frame_index = sample.get("frame_index", sample.get("timestep", None))
        if frame_index is not None:
            frame_index = frame_index.to(device=self.device, dtype=torch.long, non_blocking=True)
        sample_stride = sample.get("sample_stride", None)
        if sample_stride is not None:
            sample_stride = sample_stride.to(device=self.device, dtype=torch.long, non_blocking=True)
        teacher_commit_timestep = sample.get("teacher_commit_timestep", None)
        if teacher_commit_timestep is not None:
            teacher_commit_timestep = teacher_commit_timestep.to(
                device=self.device,
                dtype=torch.long,
                non_blocking=True,
            )

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "memory_block_video": memory_block_video,
            "memory_block_mask": memory_block_mask,
            "memory_block_steps": memory_block_steps,
            "memory_block_source": memory_block_source,
            "memory_block_offsets": memory_block_offsets,
            "memory_block_count": sample.get("memory_block_count", sample.get("memory_keyframe_count", None)),
            "memory_keyframe_video": memory_block_video,
            "memory_keyframe_mask": memory_block_mask,
            "memory_keyframe_steps": memory_block_steps,
            "memory_keyframe_count": sample.get("memory_keyframe_count", sample.get("memory_block_count", None)),
            "chunk_keyframe_target": chunk_keyframe_target,
            "use_keyframe_supervision": use_keyframe_supervision,
            "teacher_event_offset": sample.get("teacher_event_offset", None),
            "teacher_event_confidence": sample.get("teacher_event_confidence", None),
            "teacher_should_commit": sample.get("teacher_should_commit", None),
            "teacher_commit_timestep": teacher_commit_timestep,
            "sample_timestep": frame_index,
            "sample_stride": sample_stride,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        memory_seq_len: int = 0,
        memory_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        memory_seq_len = int(memory_seq_len)
        if memory_seq_len <= 0:
            mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

            # video -> video
            mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
            )
            # action -> action
            mask[video_seq_len:, video_seq_len:] = True
            # action -> first-frame video only
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            mask[video_seq_len:, :first_frame_tokens] = True
            return mask

        normal_video_seq_len = video_seq_len - memory_seq_len
        if normal_video_seq_len <= 0:
            raise ValueError(
                f"Memory prefix leaves no normal video tokens: video_seq_len={video_seq_len}, "
                f"memory_seq_len={memory_seq_len}."
            )
        first_frame_tokens = min(video_tokens_per_frame, normal_video_seq_len)
        condition_end = memory_seq_len + first_frame_tokens

        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        normal_video_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=normal_video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

        # memory/current first-frame queries only see the condition block.
        mask[:condition_end, :condition_end] = True
        # normal video queries keep the original video mask and can see memory keys.
        mask[memory_seq_len:video_seq_len, memory_seq_len:video_seq_len] = normal_video_mask
        mask[condition_end:video_seq_len, :memory_seq_len] = True
        # action sees condition tokens and action tokens.
        mask[video_seq_len:, :condition_end] = True
        mask[video_seq_len:, video_seq_len:] = True

        if memory_token_mask is None:
            return mask
        if memory_token_mask.ndim != 2 or memory_token_mask.shape[1] != memory_seq_len:
            raise ValueError(
                "`memory_token_mask` must be [B, memory_seq_len], "
                f"got {tuple(memory_token_mask.shape)} with memory_seq_len={memory_seq_len}."
            )
        batch_mask = mask.unsqueeze(0).expand(memory_token_mask.shape[0], -1, -1).clone()
        memory_token_mask = memory_token_mask.to(device=device, dtype=torch.bool)
        batch_mask[:, :, :memory_seq_len] &= memory_token_mask.unsqueeze(1)
        batch_mask[:, :memory_seq_len, :] &= memory_token_mask.unsqueeze(2)
        invalid_query = ~memory_token_mask
        if invalid_query.any():
            current_first_token = memory_seq_len
            batch_mask[:, :memory_seq_len, current_first_token] |= invalid_query
        return batch_mask.unsqueeze(1)

    def _prefix_memory_video_tokens(
        self,
        video_pre: dict[str, Any],
        memory_keyframe_video: Optional[torch.Tensor],
        memory_keyframe_mask: Optional[torch.Tensor],
        memory_block_source: Optional[torch.Tensor] = None,
        memory_block_offsets: Optional[torch.Tensor] = None,
        tiled: bool = False,
    ) -> dict[str, Any]:
        if (
            not self.keyframe_memory_enabled
            or memory_keyframe_video is None
            or memory_keyframe_video.shape[1] == 0
        ):
            return {
                "memory_seq_len": 0,
                "keyframe_seq_len": 0,
                "recent_seq_len": 0,
                "memory_bos_seq_len": 0,
                "recent_bos_seq_len": 0,
                "memory_token_mask": None,
                "spans": self._empty_memory_attention_spans(),
            }

        if self.keyframe_memory_structured_block:
            return self._prefix_memory_block_tokens(
                video_pre=video_pre,
                memory_block_video=memory_keyframe_video,
                memory_block_mask=memory_keyframe_mask,
                memory_block_source=memory_block_source,
                memory_block_offsets=memory_block_offsets,
                tiled=tiled,
            )

        if memory_keyframe_mask is None:
            memory_keyframe_mask = torch.ones(
                memory_keyframe_video.shape[:2],
                dtype=torch.bool,
                device=memory_keyframe_video.device,
            )
        if not bool(memory_keyframe_mask.any().item()):
            return {
                "memory_seq_len": 0,
                "keyframe_seq_len": 0,
                "recent_seq_len": 0,
                "memory_bos_seq_len": 0,
                "recent_bos_seq_len": 0,
                "memory_token_mask": None,
                "spans": self._empty_memory_attention_spans(),
            }
        batch_size, num_keyframes, channels, height, width = memory_keyframe_video.shape
        if channels != 3:
            raise ValueError(f"`memory_keyframe_video` channel dim must be 3, got {channels}.")
        if batch_size != video_pre["tokens"].shape[0]:
            raise ValueError(
                "`memory_keyframe_video` batch mismatch: "
                f"{batch_size} vs video tokens batch {video_pre['tokens'].shape[0]}."
            )

        memory_flat = memory_keyframe_video.reshape(batch_size * num_keyframes, channels, height, width)
        memory_flat = memory_flat.unsqueeze(2)
        memory_latents_flat = self._encode_video_latents(memory_flat, tiled=tiled)
        if memory_latents_flat.shape[2] != 1:
            raise ValueError(
                "Single-frame memory VAE encode must produce one latent frame, "
                f"got {memory_latents_flat.shape[2]}."
            )
        latent_channels, latent_h, latent_w = memory_latents_flat.shape[1], memory_latents_flat.shape[3], memory_latents_flat.shape[4]
        memory_latents = memory_latents_flat[:, :, 0].reshape(
            batch_size,
            num_keyframes,
            latent_channels,
            latent_h,
            latent_w,
        ).permute(0, 2, 1, 3, 4).contiguous()

        memory_patch = self.video_expert.patchify(memory_latents)
        memory_tokens = memory_patch.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, memory_patch.shape[1])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        memory_seq_len = int(memory_tokens.shape[1])
        expected_seq_len = int(num_keyframes * tokens_per_frame)
        if memory_seq_len != expected_seq_len:
            raise ValueError(
                "Memory token count mismatch: "
                f"got {memory_seq_len}, expected {expected_seq_len} from K={num_keyframes} and tokens_per_frame={tokens_per_frame}."
            )

        first_frame_freqs = video_pre["freqs"][:tokens_per_frame]
        memory_freqs = first_frame_freqs.repeat(num_keyframes, 1, 1)
        first_frame_t_mod = video_pre["t_mod"][:, :tokens_per_frame]
        memory_t_mod = first_frame_t_mod.repeat(1, num_keyframes, 1, 1)
        first_frame_context_mask = video_pre["context_mask"][:, :tokens_per_frame]
        memory_context_mask = first_frame_context_mask.repeat(1, num_keyframes, 1)

        video_pre["tokens"] = torch.cat([memory_tokens, video_pre["tokens"]], dim=1)
        video_pre["freqs"] = torch.cat([memory_freqs, video_pre["freqs"]], dim=0)
        video_pre["t_mod"] = torch.cat([memory_t_mod, video_pre["t_mod"]], dim=1)
        video_pre["context_mask"] = torch.cat([memory_context_mask, video_pre["context_mask"]], dim=1)
        memory_token_mask = memory_keyframe_mask.repeat_interleave(tokens_per_frame, dim=1)
        spans = {
            "mem_bos": (0, 0),
            "keyframe_visual": (0, memory_seq_len),
            "recent_bos": (0, 0),
            "recent_visual": (0, 0),
            "memory_all": (0, memory_seq_len),
        }
        for keyframe_idx in range(num_keyframes):
            start = keyframe_idx * tokens_per_frame
            spans[f"keyframe_{keyframe_idx:02d}_visual"] = (start, start + tokens_per_frame)
        return {
            "memory_seq_len": memory_seq_len,
            "keyframe_seq_len": memory_seq_len,
            "recent_seq_len": 0,
            "memory_bos_seq_len": 0,
            "recent_bos_seq_len": 0,
            "memory_token_mask": memory_token_mask,
            "spans": spans,
        }

    def _memory_freqs_for_offsets(
        self,
        offsets: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        if offsets.ndim != 2:
            raise ValueError(f"`offsets` must be [B,K], got {tuple(offsets.shape)}")
        batch_size, num_frames = offsets.shape
        temporal_cache, height_cache, width_cache = self.video_expert.freqs
        max_temporal_index = int(temporal_cache.shape[0]) - 1
        max_relative = int(self.relative_memory_rope_max_offset)
        if max_relative > 0:
            offsets = offsets.clamp(min=-max_relative, max=0)
        else:
            offsets = offsets.clamp(max=0)
        abs_offsets = offsets.abs().clamp(max=max_temporal_index).long().to(device=temporal_cache.device)
        temporal_freqs = temporal_cache[abs_offsets].to(device=device)
        temporal_freqs = torch.where(
            offsets.to(device=device).unsqueeze(-1) < 0,
            temporal_freqs.conj(),
            temporal_freqs,
        )
        temporal_freqs = temporal_freqs.view(batch_size, num_frames, 1, 1, -1).expand(
            batch_size, num_frames, h, w, -1
        )
        height_freqs = height_cache[:h].to(device=device).view(1, 1, h, 1, -1).expand(
            batch_size, num_frames, h, w, -1
        )
        width_freqs = width_cache[:w].to(device=device).view(1, 1, 1, w, -1).expand(
            batch_size, num_frames, h, w, -1
        )
        return torch.cat([temporal_freqs, height_freqs, width_freqs], dim=-1).reshape(
            batch_size,
            num_frames * h * w,
            1,
            -1,
        )

    def _prefix_memory_block_tokens(
        self,
        video_pre: dict[str, Any],
        memory_block_video: torch.Tensor,
        memory_block_mask: Optional[torch.Tensor],
        memory_block_source: Optional[torch.Tensor],
        memory_block_offsets: Optional[torch.Tensor],
        tiled: bool = False,
    ) -> dict[str, Any]:
        if memory_block_mask is None:
            memory_block_mask = torch.ones(
                memory_block_video.shape[:2],
                dtype=torch.bool,
                device=memory_block_video.device,
            )
        if not bool(memory_block_mask.any().item()):
            return {
                "memory_seq_len": 0,
                "keyframe_seq_len": 0,
                "recent_seq_len": 0,
                "memory_bos_seq_len": 0,
                "recent_bos_seq_len": 0,
                "memory_token_mask": None,
                "spans": self._empty_memory_attention_spans(),
            }

        batch_size, num_slots, channels, height, width = memory_block_video.shape
        if channels != 3:
            raise ValueError(f"`memory_block_video` channel dim must be 3, got {channels}.")
        if batch_size != video_pre["tokens"].shape[0]:
            raise ValueError(
                "`memory_block_video` batch mismatch: "
                f"{batch_size} vs video tokens batch {video_pre['tokens'].shape[0]}."
            )
        if memory_block_source is not None and memory_block_source.shape != memory_block_mask.shape:
            raise ValueError(
                "`memory_block_source` shape mismatch: "
                f"{tuple(memory_block_source.shape)} vs {tuple(memory_block_mask.shape)}"
            )
        if memory_block_offsets is not None and memory_block_offsets.shape != memory_block_mask.shape:
            raise ValueError(
                "`memory_block_offsets` shape mismatch: "
                f"{tuple(memory_block_offsets.shape)} vs {tuple(memory_block_mask.shape)}"
            )

        memory_flat = memory_block_video.reshape(batch_size * num_slots, channels, height, width)
        memory_flat = memory_flat.unsqueeze(2)
        memory_latents_flat = self._encode_video_latents(memory_flat, tiled=tiled)
        if memory_latents_flat.shape[2] != 1:
            raise ValueError(
                "Single-frame memory VAE encode must produce one latent frame, "
                f"got {memory_latents_flat.shape[2]}."
            )
        latent_channels = int(memory_latents_flat.shape[1])
        latent_h = int(memory_latents_flat.shape[3])
        latent_w = int(memory_latents_flat.shape[4])
        memory_latents = memory_latents_flat[:, :, 0].reshape(
            batch_size,
            num_slots,
            latent_channels,
            latent_h,
            latent_w,
        ).permute(0, 2, 1, 3, 4).contiguous()

        memory_patch = self.video_expert.patchify(memory_latents)
        _, hidden_dim, patched_slots, patch_h, patch_w = memory_patch.shape
        memory_slot_tokens = memory_patch.permute(0, 2, 3, 4, 1).reshape(
            batch_size,
            patched_slots,
            patch_h * patch_w,
            hidden_dim,
        )
        if patched_slots != num_slots:
            raise ValueError(
                "Memory patchify changed the number of memory frames: "
                f"got {patched_slots}, expected {num_slots}."
            )
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        if tokens_per_frame != patch_h * patch_w:
            raise ValueError(
                "Memory tokens-per-frame mismatch: "
                f"got {patch_h * patch_w}, expected {tokens_per_frame}."
            )

        if memory_block_source is None:
            keyframe_slot_layout = torch.ones((num_slots,), dtype=torch.bool, device=memory_block_mask.device)
            recent_slot_layout = torch.zeros((num_slots,), dtype=torch.bool, device=memory_block_mask.device)
            keyframe_slot_mask = memory_block_mask
            recent_slot_mask = torch.zeros_like(memory_block_mask)
        else:
            is_keyframe = (
                (memory_block_source == MEMORY_SOURCE_KEYFRAME_TEACHER)
                | (memory_block_source == MEMORY_SOURCE_KEYFRAME_PREDICTED)
            ) & memory_block_mask
            is_recent = (memory_block_source == MEMORY_SOURCE_HISTORY) & memory_block_mask
            keyframe_slot_layout = is_keyframe.any(dim=0)
            recent_slot_layout = is_recent.any(dim=0)
            keyframe_slot_mask = is_keyframe
            recent_slot_mask = is_recent

        first_frame_freqs = video_pre["freqs"][:tokens_per_frame]
        batched_freqs = bool(self.use_relative_memory_rope and memory_block_offsets is not None)
        first_frame_t_mod = video_pre["t_mod"][:, :tokens_per_frame]
        first_frame_context_mask = video_pre["context_mask"][:, :tokens_per_frame]

        token_pieces: list[torch.Tensor] = []
        freq_pieces: list[torch.Tensor] = []
        t_mod_pieces: list[torch.Tensor] = []
        context_mask_pieces: list[torch.Tensor] = []
        mask_pieces: list[torch.Tensor] = []
        keyframe_seq_len = 0
        recent_seq_len = 0
        memory_bos_seq_len = 0
        recent_bos_seq_len = 0

        def add_bos(token: torch.Tensor, active_mask: torch.Tensor) -> tuple[int, int]:
            bos = token.to(device=memory_block_video.device, dtype=memory_slot_tokens.dtype).expand(batch_size, 1, -1)
            token_pieces.append(bos)
            if batched_freqs:
                freq_pieces.append(first_frame_freqs[:1].unsqueeze(0).expand(batch_size, -1, -1, -1))
            else:
                freq_pieces.append(first_frame_freqs[:1])
            t_mod_pieces.append(video_pre["t_mod"][:, :1])
            context_mask_pieces.append(video_pre["context_mask"][:, :1])
            mask_pieces.append(active_mask)
            return 1, 1

        def add_visual_slots(slot_layout: torch.Tensor, slot_mask: torch.Tensor) -> int:
            slot_count = int(slot_layout.sum().item())
            if slot_count == 0:
                return 0
            tokens = memory_slot_tokens[:, slot_layout].reshape(batch_size, slot_count * tokens_per_frame, hidden_dim)
            token_pieces.append(tokens)
            if batched_freqs:
                offsets = memory_block_offsets[:, slot_layout]
                freq_pieces.append(
                    self._memory_freqs_for_offsets(
                        offsets=offsets,
                        h=patch_h,
                        w=patch_w,
                        device=memory_block_video.device,
                    )
                )
            else:
                freq_pieces.append(first_frame_freqs.repeat(slot_count, 1, 1))
            t_mod_pieces.append(
                first_frame_t_mod.unsqueeze(1)
                .expand(batch_size, slot_count, tokens_per_frame, 6, int(self.video_expert.hidden_dim))
                .reshape(batch_size, slot_count * tokens_per_frame, 6, int(self.video_expert.hidden_dim))
            )
            context_mask_pieces.append(
                first_frame_context_mask.unsqueeze(1)
                .expand(batch_size, slot_count, tokens_per_frame, first_frame_context_mask.shape[-1])
                .reshape(batch_size, slot_count * tokens_per_frame, first_frame_context_mask.shape[-1])
            )
            mask_pieces.append(slot_mask[:, slot_layout].repeat_interleave(tokens_per_frame, dim=1))
            return slot_count * tokens_per_frame

        keyframe_active = keyframe_slot_mask[:, keyframe_slot_layout].any(dim=1, keepdim=True) if bool(keyframe_slot_layout.any().item()) else torch.zeros((batch_size, 1), dtype=torch.bool, device=memory_block_mask.device)
        keyframe_frame_spans: dict[str, tuple[int, int]] = {}
        if bool(keyframe_slot_layout.any().item()):
            if self.use_memory_bos:
                active = keyframe_active | bool(self.memory_bos_insert_when_empty)
                memory_bos_seq_len, added = add_bos(self.memory_bos_token, active)
                keyframe_seq_len += added
            keyframe_visual_start = int(keyframe_seq_len)
            keyframe_visual_len = add_visual_slots(keyframe_slot_layout, keyframe_slot_mask)
            keyframe_seq_len += keyframe_visual_len
            for keyframe_idx in range(keyframe_visual_len // tokens_per_frame):
                start = keyframe_visual_start + keyframe_idx * tokens_per_frame
                keyframe_frame_spans[f"keyframe_{keyframe_idx:02d}_visual"] = (
                    start,
                    start + tokens_per_frame,
                )

        recent_active = recent_slot_mask[:, recent_slot_layout].any(dim=1, keepdim=True) if bool(recent_slot_layout.any().item()) else torch.zeros((batch_size, 1), dtype=torch.bool, device=memory_block_mask.device)
        recent_frame_spans: dict[str, tuple[int, int]] = {}
        if bool(recent_slot_layout.any().item()):
            if self.use_recent_bos:
                active = recent_active | bool(self.recent_bos_insert_when_empty)
                recent_bos_seq_len, added = add_bos(self.recent_bos_token, active)
                recent_seq_len += added
            recent_visual_start = int(keyframe_seq_len + recent_seq_len)
            recent_visual_len = add_visual_slots(recent_slot_layout, recent_slot_mask)
            recent_seq_len += recent_visual_len
            for recent_idx in range(recent_visual_len // tokens_per_frame):
                start = recent_visual_start + recent_idx * tokens_per_frame
                recent_frame_spans[f"recent_{recent_idx:02d}_visual"] = (
                    start,
                    start + tokens_per_frame,
                )

        if not token_pieces:
            return {
                "memory_seq_len": 0,
                "keyframe_seq_len": 0,
                "recent_seq_len": 0,
                "memory_bos_seq_len": 0,
                "recent_bos_seq_len": 0,
                "memory_token_mask": None,
                "spans": self._empty_memory_attention_spans(),
            }

        memory_tokens = torch.cat(token_pieces, dim=1)
        if batched_freqs:
            normal_freqs = video_pre["freqs"].unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            normal_freqs = video_pre["freqs"]
        memory_freqs = torch.cat(freq_pieces, dim=1 if batched_freqs else 0)
        memory_t_mod = torch.cat(t_mod_pieces, dim=1)
        memory_context_mask = torch.cat(context_mask_pieces, dim=1)
        memory_token_mask = torch.cat(mask_pieces, dim=1).to(device=memory_block_video.device, dtype=torch.bool)

        video_pre["tokens"] = torch.cat([memory_tokens, video_pre["tokens"]], dim=1)
        video_pre["freqs"] = torch.cat([memory_freqs, normal_freqs], dim=1 if batched_freqs else 0)
        video_pre["t_mod"] = torch.cat([memory_t_mod, video_pre["t_mod"]], dim=1)
        video_pre["context_mask"] = torch.cat([memory_context_mask, video_pre["context_mask"]], dim=1)

        memory_seq_len = int(memory_tokens.shape[1])
        recent_bos_start = int(keyframe_seq_len)
        recent_bos_end = recent_bos_start + int(recent_bos_seq_len)
        spans = {
            "mem_bos": (0, int(memory_bos_seq_len)),
            "keyframe_visual": (int(memory_bos_seq_len), int(keyframe_seq_len)),
            "recent_bos": (recent_bos_start, recent_bos_end),
            "recent_visual": (recent_bos_end, memory_seq_len),
            "memory_all": (0, memory_seq_len),
        }
        spans.update(keyframe_frame_spans)
        spans.update(recent_frame_spans)
        return {
            "memory_seq_len": memory_seq_len,
            "keyframe_seq_len": int(keyframe_seq_len),
            "recent_seq_len": int(recent_seq_len),
            "memory_bos_seq_len": int(memory_bos_seq_len),
            "recent_bos_seq_len": int(recent_bos_seq_len),
            "memory_token_mask": memory_token_mask,
            "spans": spans,
        }

    def _select_chunk_event(self, kem_probs: torch.Tensor, threshold: Optional[float] = None) -> dict[str, torch.Tensor]:
        if threshold is None:
            threshold = self.event_commit_threshold
        if kem_probs.ndim == 1:
            kem_probs = kem_probs.unsqueeze(0)
            squeeze = True
        elif kem_probs.ndim == 2:
            squeeze = False
        else:
            raise ValueError(f"`kem_probs` must be [T] or [B,T], got {tuple(kem_probs.shape)}")

        batch_size, horizon = kem_probs.shape
        min_offset = max(int(self.event_future_min_offset), 0)
        offsets = torch.full((batch_size,), -1, dtype=torch.long, device=kem_probs.device)
        confidence = torch.zeros((batch_size,), dtype=kem_probs.dtype, device=kem_probs.device)
        should_commit = torch.zeros((batch_size,), dtype=torch.bool, device=kem_probs.device)
        if min_offset < horizon:
            future = kem_probs[:, min_offset:]
            best_rel = torch.argmax(future, dim=1)
            offsets = best_rel + min_offset
            confidence = future.gather(1, best_rel.unsqueeze(1)).squeeze(1)
            should_commit = confidence >= float(threshold)
            offsets = torch.where(should_commit, offsets, torch.full_like(offsets, -1))

        if squeeze:
            return {
                "pred_event_offset": offsets[0],
                "pred_event_confidence": confidence[0],
                "should_trigger_event": should_commit[0],
            }
        return {
            "pred_event_offset": offsets,
            "pred_event_confidence": confidence,
            "should_trigger_event": should_commit,
        }

    @staticmethod
    def _debug_tensor_value(tensor: Optional[torch.Tensor], index: int, default: int = -1) -> int:
        if tensor is None:
            return int(default)
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return int(default)
        if tensor.ndim == 0:
            return int(tensor.detach().cpu().item())
        if index >= tensor.shape[0]:
            return int(default)
        value = tensor[index]
        if value.numel() == 0:
            return int(default)
        return int(value.flatten()[0].detach().cpu().item())

    def _log_kem_debug_samples(
        self,
        *,
        event_pred: dict[str, torch.Tensor],
        inputs: dict[str, Any],
    ) -> None:
        if not self.kem_debug_log_samples:
            return
        if self.kem_debug_log_rank >= 0:
            current_rank = 0
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                current_rank = int(torch.distributed.get_rank())
            if current_rank != self.kem_debug_log_rank:
                return

        pred_offset = event_pred["pred_event_offset"].detach()
        pred_should = event_pred["should_trigger_event"].detach().to(dtype=torch.bool)
        batch_size = int(pred_offset.shape[0]) if pred_offset.ndim > 0 else 1
        max_samples = min(batch_size, self.kem_debug_log_max_samples)

        sample_timestep = inputs.get("sample_timestep")
        sample_stride = inputs.get("sample_stride")
        memory_keyframe_steps = inputs.get("memory_keyframe_steps")
        teacher_commit_timestep = inputs.get("teacher_commit_timestep")

        for sample_idx in range(max_samples):
            timestep = self._debug_tensor_value(sample_timestep, sample_idx, default=-1)
            stride = max(self._debug_tensor_value(sample_stride, sample_idx, default=1), 1)
            offset = self._debug_tensor_value(pred_offset, sample_idx, default=-1)
            should_commit = bool(pred_should.flatten()[sample_idx].detach().cpu().item())
            pred_kf = int(timestep + offset * stride) if should_commit and timestep >= 0 and offset >= 0 else -1
            gt_keyframe = self._debug_tensor_value(teacher_commit_timestep, sample_idx, default=-1)

            input_keyframe_steps: list[int] = []
            if isinstance(memory_keyframe_steps, torch.Tensor) and memory_keyframe_steps.ndim >= 2:
                steps = memory_keyframe_steps[sample_idx].detach().cpu().flatten().tolist()
                input_keyframe_steps = [int(step) for step in steps if int(step) >= 0]

            logger.info(
                ">>  sample_timestep=%s\n"
                "    input_keyframe_steps=%s\n"
                "    pred_kf=%s gt_keyframe=%s",
                timestep,
                input_keyframe_steps,
                pred_kf,
                gt_keyframe,
            )

    def _compute_kem_loss(
        self,
        action_features: torch.Tensor,
        inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.kem_enabled or self.kem_head is None:
            return action_features.sum() * 0.0, {}

        kem_logits = self.kem_head(action_features).squeeze(-1)
        target = inputs.get("chunk_keyframe_target")
        use_supervision = inputs.get("use_keyframe_supervision")
        if target is None or use_supervision is None:
            return kem_logits.sum() * 0.0, {"loss_kem": 0.0}
        if target.shape != kem_logits.shape:
            raise ValueError(
                "`chunk_keyframe_target` shape must match KEM logits: "
                f"target={tuple(target.shape)} logits={tuple(kem_logits.shape)}"
            )

        valid = use_supervision.view(-1, 1).expand_as(target)
        action_is_pad = inputs.get("action_is_pad")
        if action_is_pad is not None:
            valid = valid & (~action_is_pad)

        if not bool(valid.any().item()):
            return kem_logits.sum() * 0.0, {"loss_kem": 0.0}

        pos_weight = torch.tensor(
            self.kem_positive_weight,
            dtype=kem_logits.dtype,
            device=kem_logits.device,
        )
        loss_token = F.binary_cross_entropy_with_logits(
            kem_logits.float(),
            target.float(),
            pos_weight=pos_weight.float(),
            reduction="none",
        ).to(dtype=kem_logits.dtype)
        valid_f = valid.to(dtype=loss_token.dtype)
        loss_kem = (loss_token * valid_f).sum() / valid_f.sum().clamp(min=1.0)

        with torch.no_grad():
            probs = torch.sigmoid(kem_logits.float())
            pred_binary = probs >= float(self.kem_threshold)
            target_binary = target.float() >= 0.5
            valid_bool = valid.bool()
            pred_valid = pred_binary & valid_bool
            target_valid = target_binary & valid_bool
            true_positive = (pred_valid & target_valid).sum().float()
            pred_positive = pred_valid.sum().float()
            target_positive = target_valid.sum().float()
            correct = ((pred_binary == target_binary) & valid_bool).sum().float()
            valid_count = valid_bool.sum().float().clamp(min=1.0)
            event_pred = self._select_chunk_event(probs)
            self._log_kem_debug_samples(event_pred=event_pred, inputs=inputs)

            teacher_should = inputs.get("teacher_should_commit")
            teacher_offset = inputs.get("teacher_event_offset")
            event_metrics: dict[str, float] = {}
            if teacher_should is not None and teacher_offset is not None:
                teacher_should = teacher_should.to(device=kem_logits.device, dtype=torch.bool)
                teacher_offset = teacher_offset.to(device=kem_logits.device, dtype=torch.long)
                pred_should = event_pred["should_trigger_event"].to(dtype=torch.bool)
                pred_offset = event_pred["pred_event_offset"].to(dtype=torch.long)
                event_tp = (pred_should & teacher_should).sum().float()
                event_pred_pos = pred_should.sum().float()
                event_target_pos = teacher_should.sum().float()
                event_correct = (pred_should == teacher_should).sum().float()
                both = pred_should & teacher_should
                if bool(both.any().item()):
                    offset_mae = (pred_offset[both] - teacher_offset[both]).abs().float().mean()
                else:
                    offset_mae = torch.zeros((), device=kem_logits.device)
                event_metrics = {
                    "event_commit_accuracy": float((event_correct / max(teacher_should.numel(), 1)).item()),
                    "event_commit_precision": float((event_tp / event_pred_pos.clamp(min=1.0)).item()),
                    "event_commit_recall": float((event_tp / event_target_pos.clamp(min=1.0)).item()),
                    "event_offset_mae": float(offset_mae.item()),
                }

            metrics = {
                "loss_kem": float(loss_kem.detach().item()),
                "kem_target_rate": float((target_valid.sum().float() / valid_count).item()),
                "kem_pred_rate": float((pred_valid.sum().float() / valid_count).item()),
                "kem_accuracy": float((correct / valid_count).item()),
                "kem_precision": float((true_positive / pred_positive.clamp(min=1.0)).item()),
                "kem_recall": float((true_positive / target_positive.clamp(min=1.0)).item()),
            }
            metrics.update(event_metrics)
        return loss_kem, metrics

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        memory_info = self._prefix_memory_video_tokens(
            video_pre=video_pre,
            memory_keyframe_video=inputs.get("memory_block_video"),
            memory_keyframe_mask=inputs.get("memory_block_mask"),
            memory_block_source=inputs.get("memory_block_source"),
            memory_block_offsets=inputs.get("memory_block_offsets"),
            tiled=tiled,
        )
        video_tokens = video_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
            memory_seq_len=int(memory_info["memory_seq_len"]),
            memory_token_mask=memory_info["memory_token_mask"],
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video_tokens = tokens_out["video"]
        if int(memory_info["memory_seq_len"]) > 0:
            pred_video_tokens = pred_video_tokens[:, int(memory_info["memory_seq_len"]):]
        pred_video = self.video_expert.post_dit(pred_video_tokens, video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        loss_kem, kem_metrics = self._compute_kem_loss(tokens_out["action"], inputs)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
            + self.kem_loss_weight * loss_kem
        )
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        if self.kem_enabled:
            loss_dict["loss_kem"] = self.kem_loss_weight * float(loss_kem.detach().item())
            for key, value in kem_metrics.items():
                if key != "loss_kem":
                    loss_dict[key] = value
        memory_count = inputs.get("memory_keyframe_count")
        if memory_count is not None:
            if isinstance(memory_count, torch.Tensor):
                loss_dict["memory_keyframe_count"] = float(memory_count.float().mean().item())
            else:
                loss_dict["memory_keyframe_count"] = float(memory_count)
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        memory_keyframe_video: Optional[torch.Tensor] = None,
        memory_keyframe_mask: Optional[torch.Tensor] = None,
        memory_block_source: Optional[torch.Tensor] = None,
        memory_block_offsets: Optional[torch.Tensor] = None,
        tiled: bool = False,
        attention_probe: Optional[AttentionProbe] = None,
        probe_denoise_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        memory_info = self._prefix_memory_video_tokens(
            video_pre=video_pre,
            memory_keyframe_video=memory_keyframe_video,
            memory_keyframe_mask=memory_keyframe_mask,
            memory_block_source=memory_block_source,
            memory_block_offsets=memory_block_offsets,
            tiled=tiled,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            memory_seq_len=int(memory_info["memory_seq_len"]),
            memory_token_mask=memory_info["memory_token_mask"],
        )
        if attention_probe is not None:
            probe_query_spans, probe_key_spans = self._joint_attention_probe_spans(
                memory_info=memory_info,
                video_seq_len=int(video_pre["tokens"].shape[1]),
                action_seq_len=int(action_pre["tokens"].shape[1]),
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            )
            attention_probe.query_spans = {
                name: span for name, span in probe_query_spans.items() if int(span[1]) > int(span[0])
            }
            attention_probe.key_spans = {
                name: span for name, span in probe_key_spans.items() if int(span[1]) > int(span[0])
            }

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            attention_probe=attention_probe,
            probe_denoise_step=probe_denoise_step,
        )

        pred_video_tokens = tokens_out["video"]
        if int(memory_info["memory_seq_len"]) > 0:
            pred_video_tokens = pred_video_tokens[:, int(memory_info["memory_seq_len"]):]
        pred_video = self.video_expert.post_dit(pred_video_tokens, video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        return_tokens: bool = False,
        attention_probe: Optional[AttentionProbe] = None,
        probe_denoise_step: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            attention_probe=attention_probe,
            probe_denoise_step=probe_denoise_step,
        )
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)
        if return_tokens:
            return pred_action, action_tokens
        return pred_action

    def _prepare_inference_memory_keyframes(
        self,
        memory_keyframe_video: Optional[torch.Tensor],
        memory_keyframe_mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if memory_keyframe_video is None:
            return None, None
        if memory_keyframe_video.ndim == 4:
            memory_keyframe_video = memory_keyframe_video.unsqueeze(0)
        if memory_keyframe_video.ndim != 5 or memory_keyframe_video.shape[0] != 1 or memory_keyframe_video.shape[2] != 3:
            raise ValueError(
                "`memory_keyframe_video` must be [K,3,H,W] or [1,K,3,H,W], "
                f"got shape {tuple(memory_keyframe_video.shape)}"
            )
        memory_keyframe_video = memory_keyframe_video.to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        if memory_keyframe_mask is None:
            memory_keyframe_mask = torch.ones(
                memory_keyframe_video.shape[:2],
                dtype=torch.bool,
                device=self.device,
            )
        else:
            if memory_keyframe_mask.ndim == 1:
                memory_keyframe_mask = memory_keyframe_mask.unsqueeze(0)
            if memory_keyframe_mask.shape != memory_keyframe_video.shape[:2]:
                raise ValueError(
                    "`memory_keyframe_mask` shape mismatch: "
                    f"{tuple(memory_keyframe_mask.shape)} vs {tuple(memory_keyframe_video.shape[:2])}"
                )
            memory_keyframe_mask = memory_keyframe_mask.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
        return memory_keyframe_video, memory_keyframe_mask

    def _prepare_inference_memory_block(
        self,
        memory_block_video: Optional[torch.Tensor] = None,
        memory_block_mask: Optional[torch.Tensor] = None,
        memory_block_source: Optional[torch.Tensor] = None,
        memory_block_offsets: Optional[torch.Tensor] = None,
        memory_keyframe_video: Optional[torch.Tensor] = None,
        memory_keyframe_mask: Optional[torch.Tensor] = None,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        memory_video = memory_block_video if memory_block_video is not None else memory_keyframe_video
        memory_mask = memory_block_mask if memory_block_mask is not None else memory_keyframe_mask
        memory_video, memory_mask = self._prepare_inference_memory_keyframes(memory_video, memory_mask)
        if memory_video is None:
            return None, None, None, None

        if memory_block_source is not None:
            if memory_block_source.ndim == 1:
                memory_block_source = memory_block_source.unsqueeze(0)
            if memory_block_source.shape != memory_video.shape[:2]:
                raise ValueError(
                    "`memory_block_source` shape mismatch: "
                    f"{tuple(memory_block_source.shape)} vs {tuple(memory_video.shape[:2])}"
                )
            memory_block_source = memory_block_source.to(
                device=self.device,
                dtype=torch.long,
                non_blocking=True,
            )
        if memory_block_offsets is not None:
            if memory_block_offsets.ndim == 1:
                memory_block_offsets = memory_block_offsets.unsqueeze(0)
            if memory_block_offsets.shape != memory_video.shape[:2]:
                raise ValueError(
                    "`memory_block_offsets` shape mismatch: "
                    f"{tuple(memory_block_offsets.shape)} vs {tuple(memory_video.shape[:2])}"
                )
            memory_block_offsets = memory_block_offsets.to(
                device=self.device,
                dtype=torch.long,
                non_blocking=True,
            )
        return memory_video, memory_mask, memory_block_source, memory_block_offsets

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        memory_keyframe_video: Optional[torch.Tensor] = None,
        memory_keyframe_mask: Optional[torch.Tensor] = None,
        memory_keyframe_steps: Optional[torch.Tensor] = None,
        memory_block_video: Optional[torch.Tensor] = None,
        memory_block_mask: Optional[torch.Tensor] = None,
        memory_block_steps: Optional[torch.Tensor] = None,
        memory_block_source: Optional[torch.Tensor] = None,
        memory_block_offsets: Optional[torch.Tensor] = None,
        return_attention_stats: bool = False,
        attention_env_step: Optional[int] = None,
        attention_layer_mode: str = "all",
    ) -> dict[str, Any]:
        self.eval()
        memory_keyframe_video, memory_keyframe_mask, memory_block_source, memory_block_offsets = self._prepare_inference_memory_block(
            memory_block_video=memory_block_video,
            memory_block_mask=memory_block_mask,
            memory_block_source=memory_block_source,
            memory_block_offsets=memory_block_offsets,
            memory_keyframe_video=memory_keyframe_video,
            memory_keyframe_mask=memory_keyframe_mask,
        )
        action_only_pred = None
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_pred = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
                memory_keyframe_video=memory_keyframe_video.clone() if memory_keyframe_video is not None else None,
                memory_keyframe_mask=memory_keyframe_mask.clone() if memory_keyframe_mask is not None else None,
                memory_keyframe_steps=memory_keyframe_steps.clone() if memory_keyframe_steps is not None else None,
                memory_block_source=memory_block_source.clone() if memory_block_source is not None else None,
                memory_block_offsets=memory_block_offsets.clone() if memory_block_offsets is not None else None,
            )
            action_only_out = action_only_pred["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        attention_probe = None
        if return_attention_stats:
            attention_probe = AttentionProbe(
                env_step=attention_env_step,
                layer_mode=attention_layer_mode,
                num_layers=int(self.mot.num_layers),
            )
        for denoise_step, (step_t_video, step_delta_video, step_t_action, step_delta_action) in enumerate(zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        )):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
                memory_keyframe_video=memory_keyframe_video,
                memory_keyframe_mask=memory_keyframe_mask,
                memory_block_source=memory_block_source,
                memory_block_offsets=memory_block_offsets,
                tiled=tiled,
                attention_probe=attention_probe,
                probe_denoise_step=denoise_step,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        result = {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }
        if attention_probe is not None:
            attention_stats = attention_probe.to_dict()
            attention_stats.update(
                {
                    "mode": "joint",
                    "num_denoise_steps": int(num_inference_steps),
                }
            )
            result["attention_stats"] = attention_stats
        if action_only_pred is not None:
            for key in (
                "chunk_keyframe_prob",
                "pred_event_offset",
                "pred_event_confidence",
                "should_trigger_event",
            ):
                if key in action_only_pred:
                    result[key] = action_only_pred[key]
        return result

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        memory_keyframe_video: Optional[torch.Tensor] = None,
        memory_keyframe_mask: Optional[torch.Tensor] = None,
        memory_keyframe_steps: Optional[torch.Tensor] = None,
        memory_block_video: Optional[torch.Tensor] = None,
        memory_block_mask: Optional[torch.Tensor] = None,
        memory_block_steps: Optional[torch.Tensor] = None,
        memory_block_source: Optional[torch.Tensor] = None,
        memory_block_offsets: Optional[torch.Tensor] = None,
        return_attention_stats: bool = False,
        attention_env_step: Optional[int] = None,
        attention_layer_mode: str = "all",
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )
        memory_keyframe_video, memory_keyframe_mask, memory_block_source, memory_block_offsets = self._prepare_inference_memory_block(
            memory_block_video=memory_block_video,
            memory_block_mask=memory_block_mask,
            memory_block_source=memory_block_source,
            memory_block_offsets=memory_block_offsets,
            memory_keyframe_video=memory_keyframe_video,
            memory_keyframe_mask=memory_keyframe_mask,
        )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        memory_info = self._prefix_memory_video_tokens(
            video_pre=video_pre,
            memory_keyframe_video=memory_keyframe_video,
            memory_keyframe_mask=memory_keyframe_mask,
            memory_block_source=memory_block_source,
            memory_block_offsets=memory_block_offsets,
            tiled=tiled,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            memory_seq_len=int(memory_info["memory_seq_len"]),
            memory_token_mask=memory_info["memory_token_mask"],
        )
        attention_mask_for_cache = attention_mask[0, 0] if attention_mask.ndim == 4 else attention_mask
        attention_probe = None
        if return_attention_stats:
            probe_query_spans, probe_key_spans = self._action_attention_probe_spans(
                memory_info=memory_info,
                action_seq_len=int(latents_action.shape[1]),
                current_visual_span=(
                    int(memory_info["memory_seq_len"]),
                    int(memory_info["memory_seq_len"]) + int(video_pre["meta"]["tokens_per_frame"]),
                ),
            )
            probe_key_spans = {
                name: span for name, span in probe_key_spans.items() if int(span[1]) > int(span[0])
            }
            attention_probe = AttentionProbe(
                env_step=attention_env_step,
                layer_mode=attention_layer_mode,
                num_layers=int(self.mot.num_layers),
                query_spans=probe_query_spans,
                key_spans=probe_key_spans,
            )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask_for_cache[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        last_action_tokens = None
        for denoise_step, (step_t_action, step_delta_action) in enumerate(zip(infer_timesteps_action, infer_deltas_action)):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_result = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask_for_cache,
                video_seq_len=video_seq_len,
                return_tokens=self.kem_enabled,
                attention_probe=attention_probe,
                probe_denoise_step=denoise_step,
            )
            if self.kem_enabled:
                pred_action_posi, last_action_tokens = pred_action_result
            else:
                pred_action_posi = pred_action_result
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        result = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        if attention_probe is not None:
            attention_stats = attention_probe.to_dict()
            attention_stats.update(
                {
                    "memory_seq_len": int(memory_info["memory_seq_len"]),
                    "keyframe_seq_len": int(memory_info.get("keyframe_seq_len", 0)),
                    "recent_seq_len": int(memory_info.get("recent_seq_len", 0)),
                    "num_denoise_steps": int(num_inference_steps),
                }
            )
            result["attention_stats"] = attention_stats
        if self.kem_enabled and self.kem_head is not None and last_action_tokens is not None:
            kem_logits = self.kem_head(last_action_tokens).squeeze(-1)
            kem_probs = torch.sigmoid(kem_logits.float())
            event = self._select_chunk_event(kem_probs)
            result.update(
                {
                    "chunk_keyframe_prob": kem_probs[0].detach().to(device="cpu", dtype=torch.float32),
                    "pred_event_offset": int(event["pred_event_offset"][0].detach().cpu().item()),
                    "pred_event_confidence": float(event["pred_event_confidence"][0].detach().cpu().item()),
                    "should_trigger_event": bool(event["should_trigger_event"][0].detach().cpu().item()),
                }
            )
        return result

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        memory_keyframe_video: Optional[torch.Tensor] = None,
        memory_keyframe_mask: Optional[torch.Tensor] = None,
        memory_keyframe_steps: Optional[torch.Tensor] = None,
        memory_block_video: Optional[torch.Tensor] = None,
        memory_block_mask: Optional[torch.Tensor] = None,
        memory_block_steps: Optional[torch.Tensor] = None,
        memory_block_source: Optional[torch.Tensor] = None,
        memory_block_offsets: Optional[torch.Tensor] = None,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            memory_keyframe_video=memory_keyframe_video,
            memory_keyframe_mask=memory_keyframe_mask,
            memory_keyframe_steps=memory_keyframe_steps,
            memory_block_video=memory_block_video,
            memory_block_mask=memory_block_mask,
            memory_block_steps=memory_block_steps,
            memory_block_source=memory_block_source,
            memory_block_offsets=memory_block_offsets,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "kem_config": self.kem_config,
            "keyframe_memory_config": self.keyframe_memory_config,
        }
        if hasattr(self, "memory_bos_token"):
            payload["memory_bos_token"] = self.memory_bos_token.detach().cpu()
        if hasattr(self, "recent_bos_token"):
            payload["recent_bos_token"] = self.recent_bos_token.detach().cpu()
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.kem_head is not None:
            payload["kem_head"] = self.kem_head.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")
        if self.kem_head is not None:
            if "kem_head" in payload:
                self.kem_head.load_state_dict(payload["kem_head"], strict=True)
            else:
                logger.warning("Checkpoint has no `kem_head` weights; keeping current `kem_head` params.")
        elif "kem_head" in payload:
            logger.warning("Checkpoint contains `kem_head` weights but current model has KEM disabled; ignoring.")

        for param_name in ("memory_bos_token", "recent_bos_token"):
            param = getattr(self, param_name, None)
            if not isinstance(param, torch.nn.Parameter):
                continue
            if param_name in payload:
                value = payload[param_name].to(device=param.device, dtype=param.dtype)
                if value.shape != param.shape:
                    raise ValueError(
                        f"Checkpoint `{param_name}` shape mismatch: "
                        f"{tuple(value.shape)} vs current {tuple(param.shape)}."
                    )
                with torch.no_grad():
                    param.copy_(value)
            elif self.keyframe_memory_enabled and (
                (param_name == "memory_bos_token" and self.use_memory_bos)
                or (param_name == "recent_bos_token" and self.use_recent_bos)
            ):
                logger.warning("Checkpoint has no `%s`; keeping current initialized value.", param_name)

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
