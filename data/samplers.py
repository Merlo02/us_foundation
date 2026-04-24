"""Custom samplers for the ultrasound foundation model DataModules.

The ETL pipeline produces strongly-imbalanced splits — especially for the
``lateral_gastrocnemius_verasonics`` dataset which dominates the corpus.
``EpochSubsetSampler`` implements Experiment B3: at every epoch it draws a
fresh random subset of size ``epoch_k`` from the over-represented dataset's
indices and concatenates it with the (usually much smaller) ``other_indices``
pool. The resulting iteration order is re-shuffled per epoch and sharded
across DDP ranks.
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class EpochSubsetSampler(Sampler[int]):
    """Yield ``other_indices`` plus a fresh random subset of ``lg_indices``.

    Parameters
    ----------
    lg_indices :
        Pre-computed integer indices of the over-represented dataset (e.g.
        lateral gastrocnemius) within the parent ``HDF5Dataset``.
    other_indices :
        All remaining sample indices that should always be seen.
    epoch_k :
        Number of lg samples to draw at each epoch. If ``epoch_k >=
        len(lg_indices)`` the full pool is used (sampling becomes a no-op).
    seed :
        Base seed — the per-epoch RNG is seeded with ``seed + epoch`` so that
        each rank draws the *same* subset (essential for DDP consistency).
    num_replicas, rank :
        DDP sharding. Set by the DataModule at runtime. When ``num_replicas
        is None`` the sampler yields the full (non-sharded) subset.
    drop_last :
        If ``True`` and the epoch length is not divisible by ``num_replicas``,
        trailing samples are discarded so every rank sees the same count.
    shuffle :
        Whether to re-shuffle the concatenated (lg-subset + other) list.
    """

    def __init__(
        self,
        lg_indices: Sequence[int],
        other_indices: Sequence[int],
        epoch_k: int = 500_000,
        seed: int = 42,
        num_replicas: int | None = None,
        rank: int | None = None,
        drop_last: bool = True,
        shuffle: bool = True,
    ) -> None:
        self.lg_indices = np.asarray(lg_indices, dtype=np.int64)
        self.other_indices = np.asarray(other_indices, dtype=np.int64)
        self.epoch_k = min(int(epoch_k), int(self.lg_indices.size))
        self.seed = int(seed)
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.shuffle = shuffle
        self._epoch = 0

    # ------------------------------------------------------------------
    # Lightning / DDP hooks
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Mirror of ``DistributedSampler.set_epoch`` — called at each epoch start."""
        self._epoch = int(epoch)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _build_epoch_order(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed + self._epoch)
        if self.epoch_k >= self.lg_indices.size:
            lg_subset = self.lg_indices
        else:
            lg_subset = rng.choice(
                self.lg_indices, size=self.epoch_k, replace=False,
            )
        order = np.concatenate([self.other_indices, lg_subset])
        if self.shuffle:
            rng.shuffle(order)
        return order

    def _shard_for_rank(self, order: np.ndarray) -> np.ndarray:
        if self.num_replicas is None or self.rank is None:
            return order
        total = order.size
        if self.drop_last:
            per_rank = total // self.num_replicas
            usable = per_rank * self.num_replicas
            order = order[:usable]
        else:
            per_rank = (total + self.num_replicas - 1) // self.num_replicas
            pad = per_rank * self.num_replicas - total
            if pad > 0:
                order = np.concatenate([order, order[:pad]])
        return order[self.rank :: self.num_replicas]

    def __iter__(self) -> Iterator[int]:
        order = self._shard_for_rank(self._build_epoch_order())
        yield from (int(i) for i in order)

    def __len__(self) -> int:
        total = int(self.epoch_k + self.other_indices.size)
        if self.num_replicas is None:
            return total
        if self.drop_last:
            return (total // self.num_replicas)
        return (total + self.num_replicas - 1) // self.num_replicas
