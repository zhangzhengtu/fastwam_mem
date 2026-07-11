import logging
import os
import shutil
import sys
import time
import inspect
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.video_io import save_mp4

logger = logging.getLogger(__name__)
EXPECTED_ROBOTWIN_MEM_QPOS_DIM = 14
ROBOTWIN_HEAD_SIZE_WH = (320, 256)
ROBOTWIN_WRIST_SIZE_WH = (160, 128)
VIS_CELL_SIZE_WH = (160, 128)
DEFAULT_EVAL_VIDEO_FPS = 10
MIN_PRED_KEYFRAME_COMMIT_GAP_STEPS = 200


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected RGB image [H,W,3], got shape {tuple(array.shape)}")
    array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _model_video_to_uint8_frames(video: Any) -> list[np.ndarray]:
    if video is None:
        return []

    if isinstance(video, torch.Tensor):
        tensor = video.detach().to(device="cpu", dtype=torch.float32)
        if tensor.ndim == 5:
            if tensor.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 for predicted video, got {tuple(tensor.shape)}")
            tensor = tensor[0]
        if tensor.ndim != 4:
            raise ValueError(f"Expected predicted video [3,T,H,W] or [T,3,H,W], got {tuple(tensor.shape)}")
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 3, 0)
        elif tensor.shape[1] == 3:
            tensor = tensor.permute(0, 2, 3, 1)
        else:
            raise ValueError(f"Predicted video has no RGB channel dimension: {tuple(tensor.shape)}")

        min_value = float(tensor.min().item()) if tensor.numel() > 0 else 0.0
        max_value = float(tensor.max().item()) if tensor.numel() > 0 else 0.0
        if min_value >= -1.05 and max_value <= 1.05:
            tensor = (tensor.clamp(-1, 1) + 1.0) * 127.5
        else:
            tensor = tensor.clamp(0, 255)
        frames = tensor.round().to(dtype=torch.uint8).numpy()
        return [np.ascontiguousarray(frame) for frame in frames]

    frames = []
    for frame in video:
        if isinstance(frame, Image.Image):
            frames.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))
        else:
            frames.append(_to_uint8_rgb(frame))
    return frames


class WorldActionRobotWinMemPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        action_video_freq_ratio: int,
        eval_video_enabled: bool,
        eval_video_fps: int,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self.action_video_freq_ratio = int(max(1, action_video_freq_ratio))
        self.eval_video_enabled = bool(eval_video_enabled)
        self.eval_video_fps = int(max(1, eval_video_fps))
        keyframe_memory_cfg = getattr(self.model, "keyframe_memory_config", {}) or {}
        kem_cfg = getattr(self.model, "kem_config", {}) or {}
        self.memory_max_keyframes = int(keyframe_memory_cfg.get("max_keyframes", 0))
        self.memory_enabled = bool(keyframe_memory_cfg.get("enabled", False)) and self.memory_max_keyframes > 0
        self.memory_bank: deque[tuple[int, torch.Tensor]] = deque(maxlen=max(self.memory_max_keyframes, 1))
        self._pending_memory_commit_step: Optional[int] = None
        self._pending_memory_confidence: float = 0.0
        self._last_committed_memory_step: int = -10**9
        self._memory_cooldown_steps = int(kem_cfg.get("inference_cooldown_steps", 0) or 0)
        self._memory_cooldown_steps = max(
            self._memory_cooldown_steps,
            MIN_PRED_KEYFRAME_COMMIT_GAP_STEPS,
        )
        self._memory_nms_window = int(kem_cfg.get("inference_nms_window", 0) or 0)

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._eval_video_frames: list[Image.Image] = []
        self._active_plan_pred_frames: list[np.ndarray] = []
        self._pred_keyframe_snapshots: list[dict[str, Any]] = []
        self._plan_actions_executed = 0
        self._plan_saved_pred_indices: set[int] = set()
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized WorldActionRobotWinMemPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )

    def _memory_tensors(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.memory_enabled or not self.memory_bank:
            return None, None, None
        bank = list(self.memory_bank)[-self.memory_max_keyframes:]
        _, first_image = bank[0]
        memory_video = torch.zeros(
            (self.memory_max_keyframes,) + tuple(first_image.shape),
            dtype=first_image.dtype,
        )
        memory_mask = torch.zeros((self.memory_max_keyframes,), dtype=torch.bool)
        memory_steps = torch.full((self.memory_max_keyframes,), -1, dtype=torch.long)
        for slot, (step, image_tensor) in enumerate(bank):
            memory_video[slot] = image_tensor
            memory_mask[slot] = True
            memory_steps[slot] = int(step)
        return memory_video, memory_mask, memory_steps

    def _commit_memory_observation(
        self,
        observation: Dict[str, Any],
        step: int,
        confidence: Optional[float] = None,
    ) -> None:
        if not self.memory_enabled:
            return
        if step - self._last_committed_memory_step < self._memory_cooldown_steps:
            return
        image_tensor = self._build_robotwin_image_tensor(observation)[0].detach().to(device="cpu", dtype=torch.float32)
        self.memory_bank.append((int(step), image_tensor))
        head_image = _to_uint8_rgb(self._require_camera_rgb(observation["observation"], "head_camera")).copy()
        self._pred_keyframe_snapshots.append(
            {
                "step": int(step),
                "confidence": float(confidence or 0.0),
                "image": head_image,
            }
        )
        self._last_committed_memory_step = int(step)
        logger.debug("Committed FastWAM keyframe memory at step=%d count=%d", step, len(self.memory_bank))

    def _maybe_commit_pending_memory(self, observation: Optional[Dict[str, Any]]) -> None:
        if observation is None or self._pending_memory_commit_step is None:
            return
        if self.step_count < self._pending_memory_commit_step:
            return
        self._commit_memory_observation(
            observation,
            self.step_count,
            confidence=self._pending_memory_confidence,
        )
        self._pending_memory_commit_step = None
        self._pending_memory_confidence = 0.0

    def _schedule_predicted_memory_event(self, pred: Dict[str, Any]) -> None:
        if not self.memory_enabled or not bool(pred.get("should_trigger_event", False)):
            return
        pred_offset = int(pred.get("pred_event_offset", -1))
        if pred_offset < 0:
            return
        commit_step = int(self.step_count + pred_offset)
        confidence = float(pred.get("pred_event_confidence", 0.0))
        if commit_step - self._last_committed_memory_step < self._memory_cooldown_steps:
            return
        if self._pending_memory_commit_step is not None and self._memory_nms_window > 0:
            if abs(commit_step - self._pending_memory_commit_step) <= self._memory_nms_window:
                if confidence <= self._pending_memory_confidence:
                    return
        self._pending_memory_commit_step = commit_step
        self._pending_memory_confidence = confidence
        logger.debug(
            "Scheduled FastWAM keyframe memory commit at step=%d confidence=%.4f",
            commit_step,
            confidence,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _require_camera_rgb(self, obs_data: Dict[str, Any], camera_name: str) -> np.ndarray:
        if camera_name not in obs_data:
            raise KeyError(
                f"RoboTwin-Mem observation is missing `{camera_name}`. "
                "This FastWAM checkpoint expects head, left wrist, and right wrist RGB cameras."
            )
        camera_data = obs_data[camera_name]
        if "rgb" not in camera_data:
            raise KeyError(f"RoboTwin-Mem observation `{camera_name}` is missing `rgb`.")
        return camera_data["rgb"]

    def _camera_triplet_from_observation(self, observation: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_data = observation["observation"]
        head = _resize_rgb(self._require_camera_rgb(obs_data, "head_camera"), VIS_CELL_SIZE_WH)
        left = _resize_rgb(self._require_camera_rgb(obs_data, "left_camera"), VIS_CELL_SIZE_WH)
        right = _resize_rgb(self._require_camera_rgb(obs_data, "right_camera"), VIS_CELL_SIZE_WH)
        return head, left, right

    def _build_robotwin_composite_image(self, observation: Dict[str, Any]) -> np.ndarray:
        obs_data = observation["observation"]
        head = _resize_rgb(self._require_camera_rgb(obs_data, "head_camera"), ROBOTWIN_HEAD_SIZE_WH)
        left = _resize_rgb(self._require_camera_rgb(obs_data, "left_camera"), ROBOTWIN_WRIST_SIZE_WH)
        right = _resize_rgb(self._require_camera_rgb(obs_data, "right_camera"), ROBOTWIN_WRIST_SIZE_WH)
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]
        return image

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        image = self._build_robotwin_composite_image(observation)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _split_robotwin_composite(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = _to_uint8_rgb(image)
        height, width = image.shape[:2]
        head_h = int(round(height * 2.0 / 3.0))
        head_h = max(1, min(head_h, height - 1))
        mid_w = width // 2
        head = _resize_rgb(image[:head_h, :, :], VIS_CELL_SIZE_WH)
        left = _resize_rgb(image[head_h:, :mid_w, :], VIS_CELL_SIZE_WH)
        right = _resize_rgb(image[head_h:, mid_w:, :], VIS_CELL_SIZE_WH)
        return head, left, right

    def _make_eval_compare_frame(
        self,
        imagined: tuple[np.ndarray, np.ndarray, np.ndarray],
        executed: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> Image.Image:
        top = np.concatenate([_to_uint8_rgb(frame) for frame in imagined], axis=1)
        bottom = np.concatenate([_to_uint8_rgb(frame) for frame in executed], axis=1)
        return Image.fromarray(np.concatenate([top, bottom], axis=0), mode="RGB")

    def _set_active_plan_video(self, video: Any) -> None:
        self._active_plan_pred_frames = _model_video_to_uint8_frames(video)
        self._plan_actions_executed = 0
        self._plan_saved_pred_indices.clear()

    def _maybe_append_eval_frame(self, observation: Optional[Dict[str, Any]]) -> None:
        if not self.eval_video_enabled or observation is None or not self._active_plan_pred_frames:
            return
        if self._plan_actions_executed % self.action_video_freq_ratio != 0:
            return

        pred_index = self._plan_actions_executed // self.action_video_freq_ratio
        if pred_index >= len(self._active_plan_pred_frames) or pred_index in self._plan_saved_pred_indices:
            return

        imagined = self._split_robotwin_composite(self._active_plan_pred_frames[pred_index])
        executed = self._camera_triplet_from_observation(observation)
        self._eval_video_frames.append(self._make_eval_compare_frame(imagined=imagined, executed=executed))
        self._plan_saved_pred_indices.add(pred_index)

    def requires_observation_for_eval_video(self) -> bool:
        return self.eval_video_enabled

    def start_eval_video(self, episode_idx: Optional[int] = None) -> None:
        del episode_idx
        self._eval_video_frames.clear()
        self._active_plan_pred_frames.clear()
        self._pred_keyframe_snapshots.clear()
        self._plan_actions_executed = 0
        self._plan_saved_pred_indices.clear()

    def save_eval_video(self, path: str, fps: Optional[int] = None) -> Optional[str]:
        if not self.eval_video_enabled:
            return None
        self._save_pred_keyframe_snapshots(path)
        if not self._eval_video_frames:
            logger.warning("No eval comparison frames were collected; skip saving %s", path)
            return None
        final_path = Path(path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.gettempdir()) / "fastwam_robotwin_mem_compare_video" / str(os.getpid())
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / final_path.name
        save_mp4(self._eval_video_frames, str(tmp_path), fps=int(fps or self.eval_video_fps))
        shutil.copyfile(tmp_path, final_path)
        tmp_path.unlink(missing_ok=True)
        return str(final_path)

    def _save_pred_keyframe_snapshots(self, video_path: str) -> Optional[str]:
        if not self._pred_keyframe_snapshots:
            return None

        final_path = Path(video_path)
        keyframe_dir = final_path.with_name(f"{final_path.stem}_pred_keyframes")
        keyframe_dir.mkdir(parents=True, exist_ok=True)
        for index, snapshot in enumerate(self._pred_keyframe_snapshots):
            step = int(snapshot["step"])
            confidence = float(snapshot["confidence"])
            image = Image.fromarray(_to_uint8_rgb(snapshot["image"]), mode="RGB")
            image.save(keyframe_dir / f"{index:03d}_step_{step:06d}_conf_{confidence:.4f}.png")
        logger.info("Saved %d predicted keyframe images to %s", len(self._pred_keyframe_snapshots), keyframe_dir)
        return str(keyframe_dir)

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        if state_vector.shape[-1] != EXPECTED_ROBOTWIN_MEM_QPOS_DIM:
            raise ValueError(
                "This RoboTwin-Mem FastWAM checkpoint expects a 14-D aloha-agilex dual-arm "
                f"qpos vector, got shape {tuple(state_vector.shape)}."
            )
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        infer_method = self.model.infer_joint if hasattr(self.model, "infer_joint") else self.model.infer_action
        infer_params = inspect.signature(infer_method).parameters
        if "num_video_frames" in infer_params:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        if "memory_keyframe_video" in infer_params:
            memory_video, memory_mask, memory_steps = self._memory_tensors()
            if memory_video is not None:
                infer_kwargs["memory_keyframe_video"] = memory_video
                infer_kwargs["memory_keyframe_mask"] = memory_mask
                infer_kwargs["memory_keyframe_steps"] = memory_steps
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = infer_method(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        if self.eval_video_enabled:
            self._set_active_plan_video(pred.get("video"))
        self._schedule_predicted_memory_event(pred)

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return (not self.pending_actions) or self._pending_memory_commit_step is not None

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        self._maybe_commit_pending_memory(observation)
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            self._maybe_append_eval_frame(observation)
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        self._maybe_append_eval_frame(observation)
        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self._plan_actions_executed += 1
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self.memory_bank.clear()
        self._pred_keyframe_snapshots.clear()
        self._pending_memory_commit_step = None
        self._pending_memory_confidence = 0.0
        self._last_committed_memory_step = -10**9
        self.episode_count += 1
        self.step_count = 0
        self.start_eval_video(episode_idx=self.episode_count)
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    eval_video_enabled = _parse_bool(usr_args.get("eval_video_log", False))
    eval_video_fps = int(usr_args.get("eval_video_fps", DEFAULT_EVAL_VIDEO_FPS))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)

    policy = WorldActionRobotWinMemPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        action_video_freq_ratio=action_video_freq_ratio,
        eval_video_enabled=eval_video_enabled,
        eval_video_fps=eval_video_fps,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
