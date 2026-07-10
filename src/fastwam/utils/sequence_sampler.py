from __future__ import annotations

import hashlib
from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


EpisodeSampleIndex = tuple[int, int, int, bool, bool, int, int, bool]


class SequentialEpisodeSampler(Sampler[EpisodeSampleIndex]):
    """Yield deterministic trajectory-ordered anchors for episode-aware training.

    Each item is:
        dataset_index, trajectory_id, step_index, is_new_episode,
        is_last_sampled_step, anchor_index, prev_anchor_step, is_keyframe_approx
    """

    def __init__(
        self,
        dataset: Sized,
        *,
        seed: int,
        batch_size: int,
        num_processes: int,
        action_horizon: int,
        sample_stride: int = 1,
        sampling_interval: int = 1,
        shuffle_trajectories: bool = True,
        balance_dataset_step_counts: bool = False,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.action_horizon = int(action_horizon)
        self.sample_stride = max(int(sample_stride), 1)
        self.sampling_interval = max(int(sampling_interval), 1)
        self.shuffle_trajectories = bool(shuffle_trajectories)
        self.balance_dataset_step_counts = bool(balance_dataset_step_counts)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0
        self._cached_epoch: int | None = None
        self._cached_indices: list[EpisodeSampleIndex] = []

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._cached_epoch = None

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)
        self._cached_epoch = None

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    @staticmethod
    def _stable_hash_int(*parts: object) -> int:
        payload = "::".join(str(part) for part in parts).encode("utf-8")
        return int(hashlib.sha1(payload).hexdigest()[:16], 16)

    def _build_sparse_anchors(self, trajectory_id: int, trajectory_length: int) -> list[int]:
        max_valid_step = max(int(trajectory_length) - 1 - self.action_horizon * self.sample_stride, 0)
        if max_valid_step <= 0:
            return [0]
        if self.sampling_interval <= 1:
            return list(range(0, max_valid_step + 1))

        anchors = [0]
        span = min(self.sampling_interval, max_valid_step)
        offset = 1 + (
            self._stable_hash_int(self.seed, self.epoch, self.epoch_offset, trajectory_id) % span
        )
        anchors.extend(range(offset, max_valid_step + 1, self.sampling_interval))
        if anchors[-1] != max_valid_step:
            anchors.append(max_valid_step)
        return sorted(set(int(anchor) for anchor in anchors))

    @staticmethod
    def _nearest_sampled_keyframes(anchors: list[int], keyframe_steps: list[int]) -> set[int]:
        if not anchors or not keyframe_steps:
            return set()
        marked = set()
        for keyframe_step in keyframe_steps:
            nearest = min(anchors, key=lambda anchor: (abs(anchor - keyframe_step), anchor))
            marked.add(int(nearest))
        return marked

    def _trajectory_order(self) -> list[int]:
        trajectory_ids = list(getattr(self.dataset, "lerobot_dataset", self.dataset).trajectory_ids)
        if self.shuffle_trajectories:
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed + self.epoch + self.epoch_offset)
            perm = torch.randperm(len(trajectory_ids), generator=g).tolist()
            trajectory_ids = [trajectory_ids[i] for i in perm]
        return trajectory_ids

    def _split_trajectories_by_rank(self, trajectory_ids: list[int]) -> list[list[int]]:
        num_processes = max(self.num_processes, 1)
        if num_processes == 1:
            return [list(trajectory_ids)]
        if not self.balance_dataset_step_counts:
            return [list(trajectory_ids[rank::num_processes]) for rank in range(num_processes)]

        episode_dataset = getattr(self.dataset, "lerobot_dataset", self.dataset)
        rank_trajectories: list[list[int]] = [[] for _ in range(num_processes)]
        rank_loads = [0 for _ in range(num_processes)]
        scored = []
        for order, trajectory_id in enumerate(trajectory_ids):
            trajectory_length = int(episode_dataset.trajectory_lengths[trajectory_id])
            scored.append((len(self._build_sparse_anchors(trajectory_id, trajectory_length)), order, trajectory_id))
        for anchor_count, _, trajectory_id in sorted(scored, key=lambda item: (-item[0], item[1])):
            rank = min(range(num_processes), key=lambda i: (rank_loads[i], i))
            rank_trajectories[rank].append(int(trajectory_id))
            rank_loads[rank] += int(anchor_count)
        return rank_trajectories

    def _build_stream_for_trajectories(
        self,
        episode_dataset,
        trajectory_ids: list[int],
    ) -> list[EpisodeSampleIndex]:
        stream: list[EpisodeSampleIndex] = []
        for trajectory_id in trajectory_ids:
            trajectory_length = int(episode_dataset.trajectory_lengths[trajectory_id])
            dataset_index = int(episode_dataset.episode_to_dataset_index[trajectory_id])
            anchors = self._build_sparse_anchors(trajectory_id, trajectory_length)
            keyframe_steps = episode_dataset.get_keyframe_steps(trajectory_id)
            keyframe_anchor_steps = self._nearest_sampled_keyframes(anchors, keyframe_steps)
            for anchor_index, step_index in enumerate(anchors):
                prev_anchor_step = -1 if anchor_index == 0 else int(anchors[anchor_index - 1])
                stream.append(
                    (
                        dataset_index,
                        int(trajectory_id),
                        int(step_index),
                        anchor_index == 0,
                        anchor_index == len(anchors) - 1,
                        int(anchor_index),
                        prev_anchor_step,
                        int(step_index) in keyframe_anchor_steps,
                    )
                )
        return stream

    @staticmethod
    def _pad_stream_to_batches(
        stream: list[EpisodeSampleIndex],
        fallback: EpisodeSampleIndex,
        *,
        batch_size: int,
        target_batches: int,
    ) -> list[EpisodeSampleIndex]:
        batch_size = max(int(batch_size), 1)
        target_len = max(int(target_batches), 1) * batch_size
        if not stream:
            stream = [fallback]
        if len(stream) < target_len:
            repeats = (target_len - len(stream) + len(stream) - 1) // len(stream)
            stream = stream + (stream * repeats)[: target_len - len(stream)]
        return stream[:target_len]

    def _build_epoch_indices(self) -> list[EpisodeSampleIndex]:
        if self._cached_epoch == self.epoch and self._cached_indices:
            return list(self._cached_indices)

        episode_dataset = getattr(self.dataset, "lerobot_dataset", self.dataset)
        rank_trajectories = self._split_trajectories_by_rank(self._trajectory_order())
        rank_streams = [
            self._build_stream_for_trajectories(episode_dataset, trajectories)
            for trajectories in rank_trajectories
        ]

        fallback = (0, 0, 0, True, True, 0, -1, False)
        for rank_stream in rank_streams:
            if rank_stream:
                fallback = rank_stream[0]
                break

        batch_size = max(self.batch_size, 1)
        max_batches = max(
            ((len(stream) + batch_size - 1) // batch_size for stream in rank_streams),
            default=1,
        )
        rank_streams = [
            self._pad_stream_to_batches(
                stream,
                fallback,
                batch_size=batch_size,
                target_batches=max_batches,
            )
            for stream in rank_streams
        ]

        stream: list[EpisodeSampleIndex] = []
        for batch_index in range(max_batches):
            start = batch_index * batch_size
            end = start + batch_size
            for rank_stream in rank_streams:
                stream.extend(rank_stream[start:end])

        self._cached_epoch = self.epoch
        self._cached_indices = list(stream)
        return stream

    def __iter__(self) -> Iterator[EpisodeSampleIndex]:
        indices = self._build_epoch_indices()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self._build_epoch_indices())
