"""PyTorch Lightning DataModule for the CSR-style HDF5 output of the ETL.

Layout of each ``{split}.h5`` file (written by ``etl.writers.HDF5Writer``):

- ``data``                 ``(M,) float32``  — concatenated signals
- ``offsets``              ``(N+1,) int64``
- ``sampling_frequencies`` ``(N,) float32``
- ``dataset_sources``      ``(N,) vlen utf-8 str``

The sample ``i`` is ``data[offsets[i]:offsets[i+1]]``.

At ``__init__`` time the dataset loads ``offsets``, ``sampling_frequencies``
and ``dataset_sources`` into RAM (≤ ~200 MB at N≈17M), and keeps the file
handle lazily opened in ``__getitem__`` for DDP-fork safety.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    import lightning.pytorch as pl  # type: ignore[no-redef]

from .samplers import EpochSubsetSampler

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Physical-routing helper (must match model.tokenizer.multi_tokenizer)
# ------------------------------------------------------------------

SPEED_OF_SOUND_MM_PER_US = 1.54  # c = 1540 m/s → 1.54 mm/µs

def select_branch(fs_hz: float, window_sizes: Sequence[int], target_mm: float = 0.6) -> int:
    """Pick ``W*`` from *window_sizes* minimising ``|W·c/(2·fs) − target_mm|``."""
    if fs_hz <= 0 or not window_sizes:
        return int(window_sizes[0])
    return int(min(
        window_sizes,
        key=lambda W: abs(W * 1_540_000.0 / (2.0 * fs_hz) - target_mm),
    ))


def compute_patch_timestamps_us(
    length: int,
    fs_hz: float,
    window_size: int,
    sample_offset: int = 0,
) -> np.ndarray:
    """Midpoint timestamps (µs) of each patch for CT-RoPE.

    ``t_i = ((i·W + W/2) + sample_offset) / fs`` seconds → multiply by 1e6
    for microseconds. Only full patches are used; trailing samples shorter
    than ``W`` are discarded (the tokenizer pads at the token level).

    ``sample_offset`` is the position, in *signal samples*, of the first
    sample of the current chunk within the original (un-chunked) signal.
    It keeps CT-RoPE's continuous time coherent across chunks of the same
    acquisition.
    """
    if fs_hz <= 0 or window_size <= 0:
        return np.zeros((0,), dtype=np.float32)
    n_patches = length // window_size
    if n_patches <= 0:
        return np.zeros((0,), dtype=np.float32)
    i = np.arange(n_patches, dtype=np.float64)
    t_s = (i * window_size + window_size / 2.0 + float(sample_offset)) / float(fs_hz)
    return (t_s * 1e6).astype(np.float32)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class HDF5Dataset(Dataset):
    """Random-access CSR-style HDF5 dataset.

    Parameters
    ----------
    h5_path :
        Path to ``{split}.h5``.
    window_sizes :
        Multi-tokenizer branch sizes ``(8, 16, 32)``. Used to pre-compute
        ``patch_timestamps_us`` and ``window_size`` per sample so the
        collate function can pad sequences of different lengths.
    target_patch_mm :
        Physical target depth (mm) used by ``select_branch``.
    target_patches :
        Optional **fixed-S** batching mode. When ``None`` (default) the
        dataset returns native-length signals and the number of patches
        per sample varies. When set (e.g. ``50``) each acquisition is split
        into deterministic chunks of exactly ``target_patches * W*`` signal
        samples; short acquisitions (or the last chunk of a long one) are
        returned as-is (collate zero-pads within the batch) with
        ``valid_patches = chunk_length // W*``. This guarantees the
        tokenizer outputs an exactly ``(B, target_patches, E)`` tensor at
        every batch once ``MultiTokenizer`` is called with
        ``fixed_num_patches=target_patches``.
    min_valid_patches :
        When ``target_patches`` is set, drop last chunks with fewer than
        ``min_valid_patches`` real patches (so an acquisition producing a
        tiny trailing chunk is not propagated as a near-empty sample).
    """

    def __init__(
        self,
        h5_path: str | Path,
        window_sizes: Sequence[int] = (8, 16, 32),
        target_patch_mm: float = 0.6,
        target_patches: Optional[int] = None,
        min_valid_patches: int = 1,
    ) -> None:
        self.h5_path = str(h5_path)
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.target_patch_mm = float(target_patch_mm)
        self.target_patches = int(target_patches) if target_patches is not None else None
        self.min_valid_patches = int(min_valid_patches)
        if self.target_patches is not None and self.target_patches <= 0:
            raise ValueError(f"target_patches must be > 0, got {self.target_patches}")

        with h5py.File(self.h5_path, "r") as f:
            self.offsets = f["offsets"][:].astype(np.int64)
            self.freqs = f["sampling_frequencies"][:].astype(np.float32)
            # vlen str loads as object ndarray → decode to Python strings
            sources = f["dataset_sources"][:]
            self.sources = np.asarray([
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in sources
            ], dtype=object)

        self._n = int(self.offsets.size - 1)
        # Pre-compute the branch (W*) picked for each sample to avoid redoing
        # it in every worker.
        self.window_for_sample = np.asarray([
            select_branch(float(fs), self.window_sizes, self.target_patch_mm)
            for fs in self.freqs
        ], dtype=np.int64)

        # Fixed-S mode: pre-compute the (sample_idx, offset_in_samples) flat
        # chunk map. The lists are stored as numpy arrays for cheap indexing
        # inside __getitem__.
        if self.target_patches is not None:
            chunk_sample = []
            chunk_offset = []
            for i in range(self._n):
                length_i = int(self.offsets[i + 1] - self.offsets[i])
                W = int(self.window_for_sample[i])
                target_T = self.target_patches * W
                if length_i < self.min_valid_patches * W:
                    # Too short to yield even one valid chunk — skip.
                    continue
                # Deterministic contiguous chunking: ceil(length / target_T).
                n_chunks = max(1, (length_i + target_T - 1) // target_T)
                for c in range(n_chunks):
                    chunk_start = c * target_T
                    remaining = length_i - chunk_start
                    valid = min(remaining, target_T) // W
                    if valid >= self.min_valid_patches:
                        chunk_sample.append(i)
                        chunk_offset.append(chunk_start)
            self._chunk_sample_idx = np.asarray(chunk_sample, dtype=np.int64)
            self._chunk_start_offset = np.asarray(chunk_offset, dtype=np.int64)
        else:
            self._chunk_sample_idx = None
            self._chunk_start_offset = None

        # Lazy-opened per-worker file handle (see ``__getitem__``).
        self._file: Optional[h5py.File] = None

    def __len__(self) -> int:
        if self.target_patches is not None:
            return int(self._chunk_sample_idx.size)
        return self._n

    def _ensure_open(self) -> h5py.File:
        if self._file is None:
            # Fork-safe: each worker opens its own handle on first access.
            self._file = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
        return self._file

    def __getitem__(self, idx: int) -> dict:
        f = self._ensure_open()

        if self.target_patches is None:
            # Native-length path (back-compat).
            start = int(self.offsets[idx])
            end = int(self.offsets[idx + 1])
            signal = np.asarray(f["data"][start:end], dtype=np.float32)
            fs = float(self.freqs[idx])
            W = int(self.window_for_sample[idx])
            ts = compute_patch_timestamps_us(signal.size, fs, W)
            n = int(signal.size)
            return {
                "signal": torch.from_numpy(signal),
                "sampling_frequency_hz": fs,
                "dataset_source": str(self.sources[idx]),
                "window_size": W,
                "patch_timestamps_us": torch.from_numpy(ts),
                "length": n,
                "full_length_samples": n,
                "chunk_index": -1,
                "num_chunks": -1,
            }

        # Fixed-S path: resolve chunk → (sample, offset) and slice the HDF5
        # ``data`` buffer for exactly (at most) target_patches · W* samples.
        sample_idx = int(self._chunk_sample_idx[idx])
        chunk_offset_in_sample = int(self._chunk_start_offset[idx])
        W = int(self.window_for_sample[sample_idx])
        target_T = self.target_patches * W

        sample_start = int(self.offsets[sample_idx])
        sample_end = int(self.offsets[sample_idx + 1])
        abs_start = sample_start + chunk_offset_in_sample
        abs_end = min(sample_start + chunk_offset_in_sample + target_T, sample_end)
        signal = np.asarray(f["data"][abs_start:abs_end], dtype=np.float32)

        fs = float(self.freqs[sample_idx])
        # Timestamps continue the original signal's clock so CT-RoPE sees
        # consistent continuous-time positions across chunks.
        ts = compute_patch_timestamps_us(
            signal.size, fs, W, sample_offset=chunk_offset_in_sample,
        )

        length_i = sample_end - sample_start
        num_chunks = max(1, (length_i + target_T - 1) // target_T)
        chunk_index = chunk_offset_in_sample // target_T if target_T > 0 else 0

        return {
            "signal": torch.from_numpy(signal),
            "sampling_frequency_hz": fs,
            "dataset_source": str(self.sources[sample_idx]),
            "window_size": W,
            "patch_timestamps_us": torch.from_numpy(ts),
            "length": int(signal.size),
            "full_length_samples": int(length_i),
            "chunk_index": int(chunk_index),
            "num_chunks": int(num_chunks),
        }

    # Pickling helper — drop the opaque h5py handle (re-opened lazily).
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_file"] = None
        return state


# ------------------------------------------------------------------
# Collate
# ------------------------------------------------------------------

def _pad_1d(tensors: list[torch.Tensor], pad_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of 1-D tensors to the longest length; return (padded, mask)."""
    max_len = max(t.size(0) for t in tensors) if tensors else 0
    B = len(tensors)
    out = torch.full((B, max_len), pad_value, dtype=tensors[0].dtype)
    mask = torch.zeros((B, max_len), dtype=torch.bool)
    for i, t in enumerate(tensors):
        n = t.size(0)
        out[i, :n] = t
        mask[i, :n] = True
    return out, mask


def collate_variable_length(batch: list[dict]) -> dict:
    """Pad signals and patch timestamps to a common length within the batch.

    The padding mask ``signal_mask`` is ``True`` on valid positions; the
    downstream tokenizer uses it (together with ``window_size``) to build
    the token-level padding mask.
    """
    signals = [b["signal"] for b in batch]
    ts = [b["patch_timestamps_us"] for b in batch]

    padded_signal, signal_mask = _pad_1d(signals, pad_value=0.0)
    padded_ts, ts_mask = _pad_1d(ts, pad_value=0.0)

    out = {
        "signal": padded_signal,
        "signal_mask": signal_mask,
        "sampling_frequency_hz": torch.tensor(
            [b["sampling_frequency_hz"] for b in batch], dtype=torch.float32,
        ),
        "window_size": torch.tensor(
            [b["window_size"] for b in batch], dtype=torch.long,
        ),
        "patch_timestamps_us": padded_ts,
        "patch_mask": ts_mask,
        "dataset_source": [b["dataset_source"] for b in batch],
        "length": torch.tensor([b["length"] for b in batch], dtype=torch.long),
    }
    if batch and "full_length_samples" in batch[0]:
        out["full_length_samples"] = torch.tensor(
            [int(b["full_length_samples"]) for b in batch], dtype=torch.long,
        )
        out["chunk_index"] = torch.tensor(
            [int(b["chunk_index"]) for b in batch], dtype=torch.long,
        )
        out["num_chunks"] = torch.tensor(
            [int(b["num_chunks"]) for b in batch], dtype=torch.long,
        )
    return out


# ------------------------------------------------------------------
# DataModule
# ------------------------------------------------------------------

class HDF5DataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping the CSR-style HDF5 output.

    Supports four sampling strategies:

    - ``naive``          — iterate every sample once per epoch.
    - ``static``         — caps applied at ETL-time (nothing to do here).
    - ``dynamic_epoch``  — at every epoch draw ``epoch_k`` random indices
      from the over-represented dataset (``lg_dataset_name``) and keep all
      other samples (Experiment B3).
    - ``proportional``   — MOIRAI-style threshold ratio: cap
      ``N_i ≤ threshold_ratio · sum_j N_j``, subsampled once at
      ``setup()`` time (deterministic per seed).
    """

    def __init__(
        self,
        hdf5_dir: str | Path,
        batch_size: int = 64,
        num_workers: int = 4,
        window_sizes: Sequence[int] = (8, 16, 32),
        target_patch_mm: float = 0.6,
        target_patches: Optional[int] = None,
        min_valid_patches: int = 1,
        sampling_strategy: str = "naive",
        epoch_k: int = 500_000,
        threshold_ratio: float = 0.1,
        lg_dataset_name: str = "lateral_gastrocnemius_verasonics",
        seed: int = 42,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["hdf5_dir"])
        self.hdf5_dir = Path(hdf5_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_sizes = tuple(window_sizes)
        self.target_patch_mm = target_patch_mm
        self.target_patches = target_patches
        self.min_valid_patches = int(min_valid_patches)
        self.sampling_strategy = sampling_strategy
        self.epoch_k = epoch_k
        self.threshold_ratio = threshold_ratio
        self.lg_dataset_name = lg_dataset_name
        self.seed = seed
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        assert sampling_strategy in ("naive", "static", "dynamic_epoch", "proportional"), (
            f"Unknown sampling_strategy: {sampling_strategy!r}"
        )

        self.train_ds: Optional[HDF5Dataset] = None
        self.val_ds: Optional[HDF5Dataset] = None
        self.test_ds: Optional[HDF5Dataset] = None
        self._train_sampler: Optional[EpochSubsetSampler] = None
        self._train_indices: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def _open_split(self, split: str) -> HDF5Dataset:
        path = self.hdf5_dir / f"{split}.h5"
        if not path.exists():
            raise FileNotFoundError(f"Missing HDF5 file: {path}")
        return HDF5Dataset(
            path,
            window_sizes=self.window_sizes,
            target_patch_mm=self.target_patch_mm,
            target_patches=self.target_patches,
            min_valid_patches=self.min_valid_patches,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_ds = self._open_split("train")
            self.val_ds = self._open_split("val")
            self._configure_train_sampler()
        if stage in (None, "test"):
            try:
                self.test_ds = self._open_split("test")
            except FileNotFoundError:
                self.test_ds = None

    # ------------------------------------------------------------------
    # Sampler configuration
    # ------------------------------------------------------------------
    def _base_dataset_of(self, ds: HDF5Dataset, idx: int) -> str:
        src = str(ds.sources[idx])
        return src.split("::", 1)[0]

    def _configure_train_sampler(self) -> None:
        assert self.train_ds is not None
        if self.sampling_strategy in ("naive", "static"):
            return

        # Split indices: LG vs rest.
        #   - Native-length mode: one dataset index per sample → sample sources.
        #   - Fixed-S mode:       one dataset index per chunk   → source of the
        #                         parent sample of each chunk.
        if self.train_ds.target_patches is not None:
            parent = self.train_ds._chunk_sample_idx
            sources_per_index = self.train_ds.sources[parent]
        else:
            sources_per_index = self.train_ds.sources
        lg_mask = np.asarray([
            str(s).split("::", 1)[0] == self.lg_dataset_name
            for s in sources_per_index
        ], dtype=bool)
        lg_idx = np.where(lg_mask)[0].astype(np.int64)
        other_idx = np.where(~lg_mask)[0].astype(np.int64)

        if self.sampling_strategy == "dynamic_epoch":
            log.info(
                "HDF5DataModule: dynamic_epoch sampling — lg=%d, other=%d, epoch_k=%d",
                lg_idx.size, other_idx.size, self.epoch_k,
            )
            self._train_sampler = EpochSubsetSampler(
                lg_indices=lg_idx,
                other_indices=other_idx,
                epoch_k=self.epoch_k,
                seed=self.seed,
                num_replicas=self.trainer.world_size if self.trainer is not None else None,
                rank=self.trainer.global_rank if self.trainer is not None else None,
            )
            return

        if self.sampling_strategy == "proportional":
            total = int(lg_mask.size)
            cap = int(self.threshold_ratio * total)
            if lg_idx.size > cap:
                rng = np.random.default_rng(self.seed)
                chosen = rng.choice(lg_idx, size=cap, replace=False)
                self._train_indices = np.sort(np.concatenate([chosen, other_idx]))
                log.info(
                    "HDF5DataModule: proportional sampling — cap=%d (ratio=%.3f · %d)",
                    cap, self.threshold_ratio, total,
                )
            else:
                self._train_indices = np.arange(total, dtype=np.int64)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    def _make_loader(
        self,
        ds: Dataset,
        shuffle: bool,
        sampler=None,
    ) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=collate_variable_length,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        if self._train_sampler is not None:
            return self._make_loader(self.train_ds, shuffle=False, sampler=self._train_sampler)
        if self._train_indices is not None:
            from torch.utils.data import Subset
            return self._make_loader(
                Subset(self.train_ds, self._train_indices.tolist()),
                shuffle=True,
            )
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return self._make_loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_ds is None:
            return None
        return self._make_loader(self.test_ds, shuffle=False)

    def on_train_epoch_start(self) -> None:
        if self._train_sampler is not None and self.trainer is not None:
            self._train_sampler.set_epoch(int(self.trainer.current_epoch))
