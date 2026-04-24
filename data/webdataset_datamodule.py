"""PyTorch Lightning DataModule for WebDataset ``.tar`` shards.

Each sample inside a shard has two tar entries:

- ``<key>.signal.npy``       — 1-D float32 array (native length)
- ``<key>.metadata.json``    — dict with top-level ``sampling_frequency_hz``,
  ``dataset_source`` and ``is_filler`` fields (written by ``etl.runner``).

The DataModule uses ``wds.split_by_node`` + ``wds.split_by_worker`` for
DDP-safe sharding. No epoch-based subsampling is performed here —
``EpochSubsetSampler`` is HDF5-only by design (sequential shard reads
cannot be randomly subsampled without violating the stream contract).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    import lightning.pytorch as pl  # type: ignore[no-redef]

import webdataset as wds
from torch.utils.data import DataLoader

from .hdf5_datamodule import (
    collate_variable_length,
    compute_patch_timestamps_us,
    select_branch,
)

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Shard discovery
# ------------------------------------------------------------------

def _discover_shards(shard_root: Path, split: str) -> list[str]:
    split_dir = shard_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing WebDataset split dir: {split_dir}")
    shards = sorted(split_dir.glob("shard-*.tar"))
    if not shards:
        raise FileNotFoundError(f"No shards found under {split_dir}")
    return [str(s) for s in shards]


# ------------------------------------------------------------------
# Per-sample decoding (native-length mode — one input → one output)
# ------------------------------------------------------------------

def _decode_sample(
    sample: dict,
    window_sizes: Sequence[int],
    target_patch_mm: float,
    skip_fillers: bool = False,
) -> Optional[dict]:
    """Convert one raw WebDataset sample dict to the model-side batch format."""
    meta = sample["metadata.json"]
    if skip_fillers and bool(meta.get("is_filler", False)):
        return None

    signal = np.asarray(sample["signal.npy"], dtype=np.float32).ravel()
    fs = float(meta.get("sampling_frequency_hz", 0.0) or 0.0)
    W = select_branch(fs, window_sizes, target_patch_mm)
    ts = compute_patch_timestamps_us(signal.size, fs, W)

    return {
        "signal": torch.from_numpy(signal),
        "sampling_frequency_hz": fs,
        "dataset_source": str(meta.get("dataset_source", "")),
        "window_size": int(W),
        "patch_timestamps_us": torch.from_numpy(ts),
        "length": int(signal.size),
    }


# ------------------------------------------------------------------
# Fixed-S chunking (one input → one or more outputs)
# ------------------------------------------------------------------

def _iter_chunks(
    sample: dict,
    window_sizes: Sequence[int],
    target_patch_mm: float,
    target_patches: int,
    min_valid_patches: int,
    skip_fillers: bool = False,
):
    """Yield one or more fixed-S chunks from a single WebDataset sample.

    Mirrors :class:`HDF5Dataset`'s fixed-S path exactly: contiguous
    ``target_patches · W*`` signal slices, trailing chunks with fewer than
    ``min_valid_patches`` valid patches are dropped. Short acquisitions
    yield a single partial chunk (length < ``target_T``); the collate and
    the tokenizer will treat the missing tail as padding.
    """
    meta = sample["metadata.json"]
    if skip_fillers and bool(meta.get("is_filler", False)):
        return

    signal = np.asarray(sample["signal.npy"], dtype=np.float32).ravel()
    fs = float(meta.get("sampling_frequency_hz", 0.0) or 0.0)
    W = select_branch(fs, window_sizes, target_patch_mm)
    target_T = int(target_patches) * int(W)
    length = int(signal.size)
    if length < min_valid_patches * W:
        return

    n_chunks = max(1, (length + target_T - 1) // target_T)
    source_str = str(meta.get("dataset_source", ""))
    for c in range(n_chunks):
        chunk_start = c * target_T
        remaining = length - chunk_start
        valid = min(remaining, target_T) // W
        if valid < min_valid_patches:
            continue
        chunk_end = chunk_start + min(remaining, target_T)
        chunk = signal[chunk_start:chunk_end]
        ts = compute_patch_timestamps_us(
            chunk.size, fs, int(W), sample_offset=chunk_start,
        )
        yield {
            "signal": torch.from_numpy(chunk.copy()),
            "sampling_frequency_hz": fs,
            "dataset_source": source_str,
            "window_size": int(W),
            "patch_timestamps_us": torch.from_numpy(ts),
            "length": int(chunk.size),
        }


@wds.pipelinefilter
def _chunk_and_yield(
    data,
    window_sizes: Sequence[int],
    target_patch_mm: float,
    target_patches: int,
    min_valid_patches: int,
    skip_fillers: bool,
):
    """WebDataset pipeline filter: ``sample → 1..N fixed-S chunks``."""
    for sample in data:
        try:
            for chunk in _iter_chunks(
                sample,
                window_sizes=window_sizes,
                target_patch_mm=target_patch_mm,
                target_patches=target_patches,
                min_valid_patches=min_valid_patches,
                skip_fillers=skip_fillers,
            ):
                yield chunk
        except Exception as e:  # pragma: no cover
            log.warning("Skipping sample due to error: %s", e)
            continue


# ------------------------------------------------------------------
# DataModule
# ------------------------------------------------------------------

class WebDatasetDataModule(pl.LightningDataModule):
    """DDP-safe WebDataset loader.

    Parameters
    ----------
    shard_root :
        Directory containing ``{split}/shard-*.tar`` subfolders (written by
        ``etl.writers.WebDatasetWriter``).
    batch_size, num_workers : standard PyTorch DataLoader options.
    samples_per_shard :
        Must match the value used at ETL time. Used to estimate the epoch
        length so Lightning can display progress bars and schedulers.
    shuffle_buffer :
        Intra-shard shuffle buffer (samples). ``0`` disables shuffling.
    window_sizes, target_patch_mm :
        Must match the multi-tokenizer config.
    skip_fillers_val :
        If ``True``, the validation loader drops shard-padding filler
        samples (``is_filler=True``) so metrics reflect real data only.
    """

    def __init__(
        self,
        shard_root: str | Path,
        batch_size: int = 64,
        num_workers: int = 4,
        samples_per_shard: int = 1024,
        shuffle_buffer: int = 1000,
        window_sizes: Sequence[int] = (8, 16, 32),
        target_patch_mm: float = 0.6,
        target_patches: Optional[int] = None,
        min_valid_patches: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        skip_fillers_val: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["shard_root"])
        self.shard_root = Path(shard_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.samples_per_shard = samples_per_shard
        self.shuffle_buffer = shuffle_buffer
        self.window_sizes = tuple(window_sizes)
        self.target_patch_mm = target_patch_mm
        self.target_patches = target_patches
        self.min_valid_patches = int(min_valid_patches)
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.skip_fillers_val = skip_fillers_val

        self._train_shards: list[str] = []
        self._val_shards: list[str] = []
        self._test_shards: list[str] = []

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self._train_shards = _discover_shards(self.shard_root, "train")
            self._val_shards = _discover_shards(self.shard_root, "val")
        if stage in (None, "test"):
            try:
                self._test_shards = _discover_shards(self.shard_root, "test")
            except FileNotFoundError:
                self._test_shards = []

    # ------------------------------------------------------------------
    # Pipeline builder
    # ------------------------------------------------------------------
    def _build_pipeline(
        self,
        shards: list[str],
        shuffle: bool,
        skip_fillers: bool,
        epoch_size: Optional[int],
    ) -> wds.DataPipeline:
        stages: list = [
            wds.SimpleShardList(shards),
        ]

        # DDP: split shards across nodes first, then across workers within a node.
        stages += [
            wds.split_by_node,
            wds.split_by_worker,
        ]

        # Epoch control: ``wds.shuffle`` only shuffles within the buffer; shard
        # order is shuffled at the ShardList level if requested.
        if shuffle:
            stages.insert(1, wds.shuffle(len(shards)))

        stages += [
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
        ]

        if shuffle and self.shuffle_buffer > 0:
            stages.append(wds.shuffle(self.shuffle_buffer))

        stages += [
            wds.decode(handler=wds.warn_and_continue),
        ]

        if self.target_patches is not None:
            # Fixed-S mode: one shard sample can expand into multiple chunks,
            # so we use a pipeline filter (1→N) instead of ``wds.map`` (1→1).
            stages.append(
                _chunk_and_yield(
                    window_sizes=self.window_sizes,
                    target_patch_mm=self.target_patch_mm,
                    target_patches=int(self.target_patches),
                    min_valid_patches=self.min_valid_patches,
                    skip_fillers=skip_fillers,
                )
            )
        else:
            stages += [
                wds.map(
                    lambda s: _decode_sample(
                        s, self.window_sizes, self.target_patch_mm,
                        skip_fillers=skip_fillers,
                    ),
                    handler=wds.warn_and_continue,
                ),
                wds.select(lambda s: s is not None),
            ]

        stages.append(
            wds.batched(
                self.batch_size,
                collation_fn=collate_variable_length,
                partial=False,
            )
        )

        pipeline = wds.DataPipeline(*stages)
        if epoch_size is not None:
            pipeline = pipeline.with_epoch(epoch_size)
        return pipeline

    # ------------------------------------------------------------------
    # Epoch size estimation
    # ------------------------------------------------------------------
    def _estimated_num_batches(self, shards: list[str]) -> int:
        """Approximate batches/epoch assuming full shards (ETL guarantees this).

        In fixed-S mode the true count is ``Σ_i ceil(length_i / (S·W_i))``
        which cannot be derived from shard count alone; the approximation
        below ignores chunk expansion. The WebDataset pipeline uses
        ``.with_epoch(...)`` as a soft bound so this only affects the
        Lightning progress bar / step scheduler, not correctness.
        """
        total_samples = len(shards) * self.samples_per_shard
        world_size = self.trainer.world_size if self.trainer is not None else 1
        per_rank = total_samples // max(1, world_size)
        return per_rank // self.batch_size

    # ------------------------------------------------------------------
    # Public DataLoaders
    # ------------------------------------------------------------------
    def _make_loader(self, pipeline: wds.DataPipeline) -> DataLoader:
        return wds.WebLoader(
            pipeline,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            batch_size=None,  # batching already handled inside the pipeline
        )

    def train_dataloader(self) -> DataLoader:
        epoch_batches = self._estimated_num_batches(self._train_shards)
        pipeline = self._build_pipeline(
            self._train_shards, shuffle=True,
            skip_fillers=False, epoch_size=epoch_batches,
        )
        return self._make_loader(pipeline)

    def val_dataloader(self) -> DataLoader:
        epoch_batches = self._estimated_num_batches(self._val_shards)
        pipeline = self._build_pipeline(
            self._val_shards, shuffle=False,
            skip_fillers=self.skip_fillers_val, epoch_size=epoch_batches,
        )
        return self._make_loader(pipeline)

    def test_dataloader(self) -> Optional[DataLoader]:
        if not self._test_shards:
            return None
        epoch_batches = self._estimated_num_batches(self._test_shards)
        pipeline = self._build_pipeline(
            self._test_shards, shuffle=False,
            skip_fillers=self.skip_fillers_val, epoch_size=epoch_batches,
        )
        return self._make_loader(pipeline)
