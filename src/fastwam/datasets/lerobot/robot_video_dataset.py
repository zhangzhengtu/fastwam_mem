import hashlib
import os
from typing import Any, Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        keyframe_supervision: Optional[dict[str, Any]] = None,
        keyframe_memory: Optional[dict[str, Any]] = None,
        allow_random_fallback_on_error: bool = False,
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            allow_random_fallback_on_error=allow_random_fallback_on_error,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.allow_random_fallback_on_error = bool(allow_random_fallback_on_error)
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        if isinstance(keyframe_supervision, DictConfig):
            keyframe_supervision = OmegaConf.to_container(keyframe_supervision, resolve=True)
        if isinstance(keyframe_memory, DictConfig):
            keyframe_memory = OmegaConf.to_container(keyframe_memory, resolve=True)
        self.keyframe_supervision_cfg = keyframe_supervision or {}
        self.keyframe_memory_cfg = keyframe_memory or {}
        self.keyframe_supervision_enabled = bool(self.keyframe_supervision_cfg.get("enabled", False))
        self.keyframe_memory_enabled = bool(self.keyframe_memory_cfg.get("enabled", False))
        self.kem_target_dilation = int(self.keyframe_supervision_cfg.get("target_dilation", 8))
        self.kem_target_kernel = str(self.keyframe_supervision_cfg.get("target_kernel", "raised_cosine"))
        self.teacher_event_min_offset = int(self.keyframe_supervision_cfg.get("event_future_min_offset", 1))
        self.teacher_event_threshold = float(self.keyframe_supervision_cfg.get("teacher_event_threshold", 0.55))
        self.max_memory_keyframes = int(self.keyframe_memory_cfg.get("max_keyframes", 0))
        self.include_current_keyframe = bool(self.keyframe_memory_cfg.get("include_current_keyframe", False))
        self.memory_selection = str(self.keyframe_memory_cfg.get("selection", "latest"))
        self.memory_order = str(self.keyframe_memory_cfg.get("order", "chronological"))
        self.keyframe_input_memory_source = str(self.keyframe_memory_cfg.get("source", "teacher"))

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    work_stats_path = os.path.join(work_dir, "dataset_stats.json")
                    if os.path.abspath(pretrained_norm_stats) != os.path.abspath(work_stats_path):
                        save_dataset_stats_to_json(dataset_stats, work_stats_path)

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        return len(self.lerobot_dataset)

    @property
    def action_horizon(self) -> int:
        return int(self.num_frames - 1)

    @staticmethod
    def _to_int(value: Any, default: int = -1) -> int:
        if value is None:
            return int(default)
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return int(default)
            return int(value.flatten()[0].item())
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return int(default)
            return int(value.reshape(-1)[0].item())
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _parse_sequence_index(self, idx):
        if not isinstance(idx, tuple):
            return None
        if len(idx) == 8:
            (
                dataset_index,
                trajectory_id,
                step_index,
                is_new_episode,
                is_last_sampled_step,
                anchor_index,
                prev_anchor_step,
                is_keyframe_approx,
            ) = idx
            global_index = self.lerobot_dataset.episode_step_to_global_index(trajectory_id, step_index)
            return {
                "global_index": int(global_index),
                "dataset_index": int(dataset_index),
                "trajectory_id": int(trajectory_id),
                "step_index": int(step_index),
                "is_new_episode": bool(is_new_episode),
                "is_last_sampled_step": bool(is_last_sampled_step),
                "anchor_index": int(anchor_index),
                "prev_anchor_step": int(prev_anchor_step),
                "is_keyframe_approx": bool(is_keyframe_approx),
            }
        if len(idx) == 7:
            (
                global_index,
                trajectory_id,
                is_new_episode,
                is_last_sampled_step,
                anchor_index,
                prev_anchor_step,
                is_keyframe_approx,
            ) = idx
            _, step_index = self.lerobot_dataset.global_index_to_trajectory_step(global_index)
            return {
                "global_index": int(global_index),
                "dataset_index": int(self.lerobot_dataset.episode_to_dataset_index[int(trajectory_id)]),
                "trajectory_id": int(trajectory_id),
                "step_index": int(step_index),
                "is_new_episode": bool(is_new_episode),
                "is_last_sampled_step": bool(is_last_sampled_step),
                "anchor_index": int(anchor_index),
                "prev_anchor_step": int(prev_anchor_step),
                "is_keyframe_approx": bool(is_keyframe_approx),
            }
        raise ValueError(f"Unsupported sequence sampler index tuple with length {len(idx)}: {idx}")

    def get_keyframe_steps(self, trajectory_id: int) -> list[int]:
        return self.lerobot_dataset.get_keyframe_steps(trajectory_id)

    def get_inspect_keyframe_steps(self, trajectory_id: int) -> list[int]:
        return self.lerobot_dataset.get_inspect_keyframe_steps(trajectory_id)

    def has_inspect_keyframe_annotations(self) -> bool:
        return self.lerobot_dataset.has_inspect_keyframe_annotations()

    def is_inspect_keyframe(self, trajectory_id: int, timestep: int) -> bool:
        return self.lerobot_dataset.is_inspect_keyframe(trajectory_id, timestep)

    def episode_step_to_global_index(self, trajectory_id: int, step_index: int) -> int:
        return self.lerobot_dataset.episode_step_to_global_index(trajectory_id, step_index)

    def _format_video(self, pixel_values: torch.Tensor, image_is_pad: Optional[torch.Tensor], sample_indices):
        video = pixel_values
        if video.ndim == 5:
            video = video[:, sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
            num_cameras = 1
        selected_image_is_pad = None if image_is_pad is None else image_is_pad[sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]
        return video, selected_image_is_pad

    def _padded_keyframe_steps(self, keyframe_steps: list[int]) -> torch.Tensor:
        pad_len = max(int(getattr(self.lerobot_dataset, "max_keyframe_steps_per_episode", 0)), 1)
        out = torch.full((pad_len,), -1, dtype=torch.long)
        if keyframe_steps:
            values = torch.as_tensor(keyframe_steps[:pad_len], dtype=torch.long)
            out[: values.numel()] = values
        return out

    def _build_chunk_keyframe_supervision(
        self,
        trajectory_id: int,
        step_index: int,
        is_keyframe_approx: bool,
    ) -> dict[str, Any]:
        action_horizon = self.action_horizon
        keyframe_steps = self.get_keyframe_steps(trajectory_id)
        has_annotations = len(keyframe_steps) > 0
        sample_stride = int(getattr(self.lerobot_dataset, "global_sample_stride", 1))
        chunk_steps = torch.arange(action_horizon, dtype=torch.long) * sample_stride + int(step_index)
        target = torch.zeros((action_horizon,), dtype=torch.float32)
        exact_steps = torch.full((action_horizon,), -1, dtype=torch.long)

        if self.keyframe_supervision_enabled and has_annotations:
            keyframes = torch.as_tensor(keyframe_steps, dtype=torch.long)
            distances = (chunk_steps[:, None] - keyframes[None, :]).abs().amin(dim=1).float()
            dilation = int(self.kem_target_dilation)
            if dilation <= 0:
                target = (distances == 0).float()
            elif self.kem_target_kernel == "raised_cosine":
                in_window = distances <= float(dilation)
                target = torch.zeros_like(distances)
                target[in_window] = 0.5 * (
                    1.0 + torch.cos(torch.pi * distances[in_window] / float(dilation))
                )
            else:
                target = torch.clamp(1.0 - distances / float(max(dilation, 1)), min=0.0)

            for offset, absolute_step in enumerate(chunk_steps.tolist()):
                if absolute_step in keyframe_steps:
                    exact_steps[offset] = int(absolute_step)

        future_min = max(int(self.teacher_event_min_offset), 0)
        if future_min >= action_horizon:
            teacher_event_offset = -1
            teacher_event_confidence = 0.0
            teacher_should_commit = False
        else:
            future_target = target[future_min:]
            best_rel = int(torch.argmax(future_target).item()) if future_target.numel() > 0 else 0
            teacher_event_offset = int(future_min + best_rel)
            teacher_event_confidence = float(target[teacher_event_offset].item())
            teacher_should_commit = bool(
                self.keyframe_supervision_enabled
                and has_annotations
                and teacher_event_confidence >= self.teacher_event_threshold
            )
            if not teacher_should_commit:
                teacher_event_offset = -1

        teacher_commit_timestep = int(step_index + teacher_event_offset * sample_stride) if teacher_should_commit else -1
        return {
            "keyframe_steps": self._padded_keyframe_steps(keyframe_steps),
            "keyframe_steps_count": torch.tensor(len(keyframe_steps), dtype=torch.long),
            "has_keyframe_annotations": torch.tensor(has_annotations, dtype=torch.bool),
            "use_keyframe_supervision": torch.tensor(
                bool(self.keyframe_supervision_enabled and has_annotations),
                dtype=torch.bool,
            ),
            "is_keyframe_exact": torch.tensor(int(step_index) in set(keyframe_steps), dtype=torch.bool),
            "is_keyframe_proxy": torch.tensor(bool(is_keyframe_approx), dtype=torch.bool),
            "chunk_keyframe_target": target,
            "chunk_keyframe_exact_steps": exact_steps,
            "teacher_event_offset": torch.tensor(teacher_event_offset, dtype=torch.long),
            "teacher_event_confidence": torch.tensor(teacher_event_confidence, dtype=torch.float32),
            "teacher_should_commit": torch.tensor(teacher_should_commit, dtype=torch.bool),
            "teacher_commit_timestep": torch.tensor(teacher_commit_timestep, dtype=torch.long),
        }

    def _build_memory_keyframes(self, trajectory_id: int, step_index: int) -> dict[str, Any]:
        max_keyframes = max(int(self.max_memory_keyframes), 0)
        height, width = int(self.video_size[0]), int(self.video_size[1])
        if not self.keyframe_memory_enabled or max_keyframes == 0:
            return {}

        memory_video = torch.zeros((max_keyframes, 3, height, width), dtype=torch.float32)
        memory_mask = torch.zeros((max_keyframes,), dtype=torch.bool)
        memory_steps = torch.full((max_keyframes,), -1, dtype=torch.long)

        keyframe_steps = self.get_keyframe_steps(trajectory_id)
        if self.include_current_keyframe:
            candidates = [kf for kf in keyframe_steps if kf <= int(step_index)]
        else:
            candidates = [kf for kf in keyframe_steps if kf < int(step_index)]
        candidates = sorted(set(candidates))
        if self.memory_selection != "latest":
            logger.warning("Unsupported keyframe memory selection `%s`; falling back to latest.", self.memory_selection)
        candidates = candidates[-max_keyframes:]
        if self.memory_order == "reverse_chronological":
            candidates = list(reversed(candidates))

        for slot, keyframe_step in enumerate(candidates):
            global_index = self.episode_step_to_global_index(trajectory_id, keyframe_step)
            keyframe_sample = self.lerobot_dataset[global_index]
            keyframe_video, _ = self._format_video(
                keyframe_sample["pixel_values"],
                keyframe_sample.get("image_is_pad"),
                [0],
            )
            memory_video[slot] = keyframe_video[:, 0]
            memory_mask[slot] = True
            memory_steps[slot] = int(keyframe_step)

        return {
            "memory_keyframe_video": memory_video,
            "memory_keyframe_mask": memory_mask,
            "memory_keyframe_steps": memory_steps,
            "memory_keyframe_count": torch.tensor(len(candidates), dtype=torch.long),
            "keyframe_input_memory_source": self.keyframe_input_memory_source,
        }

    def _get(self, idx):
        sequence_meta = self._parse_sequence_index(idx)
        sample_idx = sequence_meta["global_index"] if sequence_meta is not None else idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible or sequence_meta is not None:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        video, image_is_pad = self._format_video(
            sample["pixel_values"],
            sample["image_is_pad"],
            self.video_sample_indices,
        )

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }

        if sequence_meta is None:
            trajectory_id = self._to_int(sample.get("trajectory_id"), default=-1)
            step_index = self._to_int(sample.get("frame_index", sample.get("timestep")), default=0)
            if trajectory_id < 0:
                trajectory_id, step_index = self.lerobot_dataset.global_index_to_trajectory_step(sample_idx)
            dataset_index = self._to_int(sample.get("dataset_index"), default=-1)
            episode_index = self._to_int(sample.get("episode_index"), default=-1)
            sequence_meta = {
                "dataset_index": dataset_index,
                "trajectory_id": trajectory_id,
                "step_index": step_index,
                "is_new_episode": step_index == 0,
                "is_last_sampled_step": False,
                "anchor_index": -1,
                "prev_anchor_step": -1,
                "is_keyframe_approx": False,
            }
        else:
            dataset_index = int(sequence_meta["dataset_index"])
            episode_index = int(self.lerobot_dataset.episode_to_local_episode_index[sequence_meta["trajectory_id"]])
            step_index = int(sequence_meta["step_index"])

        data.update(
            {
                "idx": torch.tensor(int(sample_idx), dtype=torch.long),
                "dataset_index": torch.tensor(int(dataset_index), dtype=torch.long),
                "trajectory_id": torch.tensor(int(sequence_meta["trajectory_id"]), dtype=torch.long),
                "episode_index": torch.tensor(int(episode_index), dtype=torch.long),
                "frame_index": torch.tensor(int(step_index), dtype=torch.long),
                "timestep": torch.tensor(int(step_index), dtype=torch.long),
                "sample_stride": torch.tensor(
                    int(getattr(self.lerobot_dataset, "global_sample_stride", 1)),
                    dtype=torch.long,
                ),
                "is_new_episode": torch.tensor(bool(sequence_meta["is_new_episode"]), dtype=torch.bool),
                "is_last_sampled_step": torch.tensor(bool(sequence_meta["is_last_sampled_step"]), dtype=torch.bool),
                "anchor_index": torch.tensor(int(sequence_meta["anchor_index"]), dtype=torch.long),
                "prev_anchor_step": torch.tensor(int(sequence_meta["prev_anchor_step"]), dtype=torch.long),
                "is_keyframe_approx": torch.tensor(bool(sequence_meta["is_keyframe_approx"]), dtype=torch.bool),
            }
        )
        data.update(
            self._build_chunk_keyframe_supervision(
                trajectory_id=int(sequence_meta["trajectory_id"]),
                step_index=int(step_index),
                is_keyframe_approx=bool(sequence_meta["is_keyframe_approx"]),
            )
        )
        data.update(
            self._build_memory_keyframes(
                trajectory_id=int(sequence_meta["trajectory_id"]),
                step_index=int(step_index),
            )
        )
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            return self._get(idx)
        except Exception as e:
            if self.allow_random_fallback_on_error:
                logger.warning(
                    "Error processing sample idx %s: %s. Returning a random sample because "
                    "`allow_random_fallback_on_error=True`.",
                    idx,
                    e,
                )
                print(traceback.format_exc())
                random_idx = np.random.randint(len(self))
                return self._get(random_idx)
            raise RuntimeError(f"Failed to process sample idx {idx}; preserving failure to avoid label/data misalignment.") from e
