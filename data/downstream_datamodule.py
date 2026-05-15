"""Skeleton DataModule for labeled multi-channel downstream tasks.

This is a **contract** — drop in a real labeled dataset later by either
(a) wiring an existing labeled HDF5 file to this loader, or (b) replacing
:class:`DownstreamDataset` with a project-specific reader.

Expected HDF5 layout (one file per split, lazy-opened per worker — same
pattern as :class:`~data.hdf5_datamodule.HDF5Dataset`)::

    /signal                 : (N, C, T)    float32  — multi-channel signals
    /label                  : (N,)         int64 (classification)
                                            float32 (regression; can be (N, K))
    /sampling_frequency_hz  : (N,)         float32  (or scalar attribute)
    /dataset_source         : (N,)         vlen utf-8  (optional)

The collate batches signals to a common ``T_max`` and provides:

- ``signal``: ``(B, C, T_max)``
- ``signal_mask``: ``(B, T_max)`` — bool, validity per time-step (shared
  across channels)
- ``sampling_frequency_hz``: ``(B,)``
- ``window_size``: ``(B,)`` long — pre-routed ``W*`` per sample
- ``patch_timestamps_us``: ``(B, S_max)``
- ``label``: ``(B,)`` int64 or ``(B,)`` / ``(B, K)`` float32
- ``dataset_source``: list[str] (or ``"unknown"`` if not present)

The wrapper :class:`~model.downstream.UltrasonicEncoderWrapper` accepts
this batch directly — no further re-shape is required at the model
boundary.
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

import h5py
from torch.utils.data import DataLoader, Dataset

from model.tokenizer.multi_tokenizer import (
    SPEED_OF_SOUND_MM_S,
    select_branch,
)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers (kept minimal; mirror collate_variable_length in spirit)
# ----------------------------------------------------------------------

def _pad_2d(
    tensors: list[torch.Tensor], pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad ``(C, T_i)`` tensors to common ``T_max``. Returns ``(B, C, T_max)``
    and a shared ``(B, T_max)`` validity mask (validity is per time-step)."""
    if not tensors:
        return torch.empty(0), torch.empty(0)
    C = tensors[0].size(0)
    T_max = max(t.size(1) for t in tensors)
    B = len(tensors)
    out = torch.full((B, C, T_max), pad_value, dtype=tensors[0].dtype)
    mask = torch.zeros((B, T_max), dtype=torch.bool)
    for i, t in enumerate(tensors):
        n = t.size(1)
        out[i, :, :n] = t
        mask[i, :n] = True
    return out, mask


def _pad_1d(
    tensors: list[torch.Tensor], pad_value: float = 0.0,
) -> torch.Tensor:
    if not tensors:
        return torch.empty(0)
    T_max = max(t.size(0) for t in tensors)
    B = len(tensors)
    out = torch.full((B, T_max), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.size(0)] = t
    return out


def _compute_patch_timestamps(
    length: int, window_size: int, fs_hz: float,
) -> np.ndarray:
    """Midpoints (µs) of non-overlapping patches of size ``W``.

    ``t_i = (i·W + W/2) / fs * 1e6`` for ``i = 0, …, length // W - 1``.
    Mirrors :meth:`MultiTokenizer._compute_timestamps`.
    """
    n = max(int(length) // int(window_size), 0)
    if n == 0 or fs_hz <= 0:
        return np.zeros((0,), dtype=np.float32)
    i_idx = np.arange(n, dtype=np.float32)
    ts = (i_idx * window_size + 0.5 * window_size) / float(fs_hz) * 1e6
    return ts.astype(np.float32)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class DownstreamDataset(Dataset):
    """HDF5-backed labeled multi-channel dataset (lazy open per worker).

    Parameters
    ----------
    h5_path :
        Path to the per-split HDF5 file (see module docstring for layout).
    window_sizes :
        Allowed patch widths. ``W*`` is picked per sample via
        :func:`~model.tokenizer.multi_tokenizer.select_branch`.
    target_patch_mm :
        Physical patch depth target for routing (default 0.6 mm).
    label_dtype :
        ``"int64"`` for classification, ``"float32"`` for regression.
    sampling_frequency_hz_fallback :
        Used when the HDF5 does not store a per-sample ``fs``.
    """

    def __init__(
        self,
        h5_path: str | Path,
        window_sizes: Sequence[int] = (8, 16, 32),
        target_patch_mm: float = 0.6,
        label_dtype: str = "int64",
        sampling_frequency_hz_fallback: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.h5_path = str(h5_path)
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.target_patch_mm = float(target_patch_mm)
        self.label_dtype = str(label_dtype)
        self.fs_fallback = (
            float(sampling_frequency_hz_fallback)
            if sampling_frequency_hz_fallback is not None
            else None
        )

        # Cheap metadata only at __init__ — heavy arrays loaded lazily per worker.
        with h5py.File(self.h5_path, "r") as f:
            if "signal" not in f:
                raise KeyError(
                    f"{self.h5_path}: expected dataset '/signal' (N, C, T)"
                )
            if "label" not in f:
                raise KeyError(f"{self.h5_path}: expected dataset '/label'")
            self.n_samples = int(f["signal"].shape[0])
            self.num_channels = int(f["signal"].shape[1])
            self.signal_dtype = f["signal"].dtype
            self.has_fs_array = "sampling_frequency_hz" in f
            self.has_dataset_source = "dataset_source" in f

        self._h5: Optional[h5py.File] = None  # populated lazily per worker

    # ------------------------------------------------------------------
    def _ensure_open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
        return self._h5

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        f = self._ensure_open()
        signal = np.asarray(f["signal"][idx], dtype=np.float32)   # (C, T)
        C, T = signal.shape

        label_raw = f["label"][idx]
        if self.label_dtype == "int64":
            label = np.int64(label_raw)
        else:
            label = np.asarray(label_raw, dtype=np.float32)

        if self.has_fs_array:
            fs_hz = float(f["sampling_frequency_hz"][idx])
        elif self.fs_fallback is not None:
            fs_hz = self.fs_fallback
        else:
            raise RuntimeError(
                f"{self.h5_path}: no '/sampling_frequency_hz' and no fallback set"
            )

        W = select_branch(fs_hz, self.window_sizes, self.target_patch_mm)
        ts = _compute_patch_timestamps(T, W, fs_hz)

        ds_src = (
            f["dataset_source"][idx].decode("utf-8")
            if self.has_dataset_source
            else "unknown"
        )

        return {
            "signal": torch.from_numpy(signal),                    # (C, T)
            "label": label,
            "sampling_frequency_hz": fs_hz,
            "window_size": int(W),
            "patch_timestamps_us": torch.from_numpy(ts),
            "length": int(T),
            "dataset_source": ds_src,
        }


# ----------------------------------------------------------------------
# Collate
# ----------------------------------------------------------------------

def collate_downstream(batch: list[dict]) -> dict:
    """Pad ``signal`` along T (shared mask across channels), stack labels."""
    signals = [b["signal"] for b in batch]
    ts = [b["patch_timestamps_us"] for b in batch]

    padded_signal, signal_mask = _pad_2d(signals, pad_value=0.0)   # (B, C, T_max)
    padded_ts = _pad_1d(ts, pad_value=0.0)                          # (B, S_max)

    labels_raw = [b["label"] for b in batch]
    if isinstance(labels_raw[0], np.ndarray):
        labels = torch.from_numpy(np.stack(labels_raw, axis=0))
    else:
        labels = torch.tensor(labels_raw)

    return {
        "signal": padded_signal,
        "signal_mask": signal_mask,
        "sampling_frequency_hz": torch.tensor(
            [b["sampling_frequency_hz"] for b in batch], dtype=torch.float32,
        ),
        "window_size": torch.tensor(
            [b["window_size"] for b in batch], dtype=torch.long,
        ),
        "patch_timestamps_us": padded_ts,
        "label": labels,
        "length": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "dataset_source": [b["dataset_source"] for b in batch],
    }


# ----------------------------------------------------------------------
# DataModule
# ----------------------------------------------------------------------

class DownstreamDataModule(pl.LightningDataModule):
    """Lightning DataModule for labeled multi-channel HDF5 splits."""

    def __init__(
        self,
        train_path: str | Path,
        val_path: Optional[str | Path] = None,
        test_path: Optional[str | Path] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        window_sizes: Sequence[int] = (8, 16, 32),
        target_patch_mm: float = 0.6,
        label_dtype: str = "int64",
        sampling_frequency_hz_fallback: Optional[float] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        shuffle_train: bool = True,
    ) -> None:
        super().__init__()
        self.train_path = str(train_path)
        self.val_path = str(val_path) if val_path is not None else None
        self.test_path = str(test_path) if test_path is not None else None
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.target_patch_mm = float(target_patch_mm)
        self.label_dtype = str(label_dtype)
        self.fs_fallback = sampling_frequency_hz_fallback
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and self.num_workers > 0
        self.shuffle_train = bool(shuffle_train)

        self._train_ds: Optional[DownstreamDataset] = None
        self._val_ds: Optional[DownstreamDataset] = None
        self._test_ds: Optional[DownstreamDataset] = None

    # ------------------------------------------------------------------
    def _make_dataset(self, path: str) -> DownstreamDataset:
        return DownstreamDataset(
            h5_path=path,
            window_sizes=self.window_sizes,
            target_patch_mm=self.target_patch_mm,
            label_dtype=self.label_dtype,
            sampling_frequency_hz_fallback=self.fs_fallback,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self._train_ds = self._make_dataset(self.train_path)
            if self.val_path is not None:
                self._val_ds = self._make_dataset(self.val_path)
        if stage in (None, "test") and self.test_path is not None:
            self._test_ds = self._make_dataset(self.test_path)

    # ------------------------------------------------------------------
    def _make_loader(
        self, ds: DownstreamDataset, shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_downstream,
            drop_last=shuffle,  # drop last only for train to keep DDP shapes consistent
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "Call setup('fit') first"
        return self._make_loader(self._train_ds, shuffle=self.shuffle_train)

    def val_dataloader(self) -> Optional[DataLoader]:
        if self._val_ds is None:
            return None
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self._test_ds is None:
            return None
        return self._make_loader(self._test_ds, shuffle=False)


__all__ = [
    "DownstreamDataModule",
    "DownstreamDataset",
    "collate_downstream",
]
