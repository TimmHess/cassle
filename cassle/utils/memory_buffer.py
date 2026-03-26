"""
Replay / rehearsal memory buffer for continual learning.

After training each task, call MemoryBuffer.update() to randomly sample and
persist a subset of that task's data (as global indices into the full training
dataset).  When preparing the dataloader for the next task, call
build_replay_dataloader() which transparently merges the memory into the
current task's data while keeping the batch size constant.

Two memory-sizing modes (controlled via MemoryBuffer.update()):

  Per-task budget (samples_per_task, default)
      A fixed number of samples is stored for every task independently.
      Total memory grows as samples_per_task * num_tasks_seen.

  Fixed total budget (total_budget)
      The total number of stored samples is capped.  Slots are distributed
      evenly across all seen tasks: per_task = total_budget // num_tasks_seen.
      When a new task arrives, existing task files are trimmed to stay within
      the budget (samples are evicted randomly).

Two batch-interleaving modes (controlled via replay_ratio in build_replay_dataloader()):

  replay_ratio = None (default)
      Memory is concatenated with the task dataset and sampled proportionally,
      i.e. exactly as if the memory were part of the task's own dataset.
      Effective memory fraction per batch ≈ n_mem / (n_task + n_mem).

  replay_ratio = r  (float in (0, 1))
      Each batch contains approximately r * batch_size memory samples.
      Achieved via WeightedRandomSampler — the batch size is unchanged.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
)


class MemoryBuffer:
    """
    Stores random subsets (rehearsal samples) from previously seen tasks.

    Indices are persisted as .pt files so they survive across the separate
    subprocess calls that main_continual.py makes for each task.
    """

    def __init__(self, save_dir: Path, samples_per_task: int = 200):
        self.save_dir = Path(save_dir)
        self.samples_per_task = samples_per_task
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _task_file(self, task_idx: int) -> Path:
        return self.save_dir / f"memory_task{task_idx}.pt"

    def _save_task(self, task_dataset: Dataset, task_idx: int, n: int) -> None:
        """Sample *n* global indices from *task_dataset* and write to disk."""
        if isinstance(task_dataset, Subset):
            # .indices may be 2-D [N, 1] when built from tensor.nonzero()
            indices = torch.as_tensor(task_dataset.indices).flatten()
        else:
            indices = torch.arange(len(task_dataset))

        n = min(n, len(indices))
        perm = torch.randperm(len(indices))[:n]
        torch.save(indices[perm], self._task_file(task_idx))
        print(
            f"[MemoryBuffer] Saved {n} samples from task {task_idx}"
            f" → {self._task_file(task_idx)}"
        )

    def _trim_task_file(self, task_idx: int, n: int) -> None:
        """Randomly trim an existing task file to at most *n* samples."""
        f = self._task_file(task_idx)
        if not f.exists():
            return
        indices = torch.load(f)
        if len(indices) <= n:
            return
        kept = indices[torch.randperm(len(indices))[:n]]
        torch.save(kept, f)
        print(
            f"[MemoryBuffer] Trimmed task {task_idx} to {n} samples"
            f" (budget rebalance)"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        task_dataset: Dataset,
        task_idx: int,
        total_budget: Optional[int] = None,
    ) -> None:
        """Store a random subset of *task_dataset* and, if using a fixed total
        budget, rebalance existing task files.

        Args:
            task_dataset:  The dataset for the task just trained.
            task_idx:      Zero-based index of the task just trained.
            total_budget:  If given, cap the *total* number of stored samples
                across all tasks seen so far.  Slots are split evenly:
                ``per_task = total_budget // (task_idx + 1)``.  Existing task
                files are trimmed to fit the new per-task quota.
                If ``None``, use the fixed ``samples_per_task`` set at
                construction time (total memory grows with each task).
        """
        if total_budget is not None:
            n_tasks = task_idx + 1
            per_task = max(1, total_budget // n_tasks)
            self._save_task(task_dataset, task_idx, per_task)
            for t in range(task_idx):
                self._trim_task_file(t, per_task)
        else:
            self._save_task(task_dataset, task_idx, self.samples_per_task)

    def get_memory_dataset(
        self, full_dataset: Dataset, up_to_task: int
    ) -> Optional[Dataset]:
        """Return a ``Subset`` of *full_dataset* covering all stored samples
        from tasks ``0 .. up_to_task - 1``.  Returns ``None`` if no files
        are found (e.g. on task 0).
        """
        all_indices = []
        for t in range(up_to_task):
            f = self._task_file(t)
            if f.exists():
                all_indices.append(torch.load(f))
            else:
                print(f"[MemoryBuffer] Warning: no memory file for task {t}, skipping.")

        if not all_indices:
            return None

        return Subset(full_dataset, torch.cat(all_indices).tolist())


# ---------------------------------------------------------------------------
# Dataloader builder
# ---------------------------------------------------------------------------


def build_replay_dataloader(
    task_dataset: Dataset,
    memory_dataset: Optional[Dataset],
    batch_size: int,
    num_workers: int,
    replay_ratio: Optional[float] = None,
    collate_fn=None,
) -> DataLoader:
    """Build a DataLoader that interleaves *task_dataset* with *memory_dataset*
    while keeping *batch_size* constant.

    Args:
        task_dataset:   Current task's dataset.
        memory_dataset: Rehearsal samples from previous tasks, or ``None``.
        batch_size:     Total batch size — never increased by replay.
        num_workers:    DataLoader worker processes.
        replay_ratio:   Controls how memory samples are mixed into each batch.

            ``None`` — Proportional mode.  Memory is appended to the task
            dataset and sampled uniformly (``ConcatDataset`` + shuffle).
            The fraction of memory samples per batch is naturally
            ``n_mem / (n_task + n_mem)``.

            ``float`` — Fixed-ratio mode.  Each batch contains approximately
            ``round(replay_ratio * batch_size)`` memory samples.  Implemented
            via ``WeightedRandomSampler`` so that batch size is unchanged.
            One "epoch" is defined as ``len(task_dataset)`` sampled steps.

        collate_fn: Optional custom collate function (forwarded to DataLoader).

    Returns:
        A configured ``DataLoader``.
    """
    if memory_dataset is None or len(memory_dataset) == 0:
        return DataLoader(
            task_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    combined = ConcatDataset([task_dataset, memory_dataset])
    n_task = len(task_dataset)
    n_mem = len(memory_dataset)

    if replay_ratio is None:
        # Proportional: treat memory as part of the task dataset.
        return DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    # Fixed-ratio via WeightedRandomSampler.
    #
    # We want the expected fraction of memory samples in each batch to equal
    # replay_ratio.  With w_task = 1 for all task samples:
    #
    #   w_mem * n_mem / (1 * n_task + w_mem * n_mem) = replay_ratio
    #   ⟹  w_mem = replay_ratio * n_task / ((1 - replay_ratio) * n_mem)
    #
    if not (0.0 < replay_ratio < 1.0):
        raise ValueError(f"replay_ratio must be in (0, 1), got {replay_ratio}")

    w_mem = replay_ratio * n_task / ((1.0 - replay_ratio) * n_mem)
    weights = torch.cat([
        torch.ones(n_task),
        torch.full((n_mem,), w_mem),
    ])
    # num_samples = n_task: one epoch = one pass through the current task data.
    sampler = WeightedRandomSampler(weights, num_samples=n_task, replacement=True)

    return DataLoader(
        combined,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
