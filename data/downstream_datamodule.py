"""Downstream DataModule (single file — replaces the old ``data/downstream/``).

Consumes a single ``all.h5`` produced by :mod:`etl_downstream` with the
fixed schema::

    /signal       (N, C, T) float32
    /labels       (N,)      int64
    /session_id   (N,)      int64
    /patient_id   (N,)      int64

    root attrs: sampling_frequency_hz, dataset_name, label_type,
                num_classes, num_channels, samples_per_frame

Two split modes:

- ``intra_session`` — test = rows where ``session_id == test_id``;
  remaining rows form a train/val pool split randomly by ``val_ratio``.
- ``intra_patient`` — test = rows where ``patient_id == test_id``;
  remaining rows form a train/val pool split randomly by ``val_ratio``.

The batch dict matches the contract documented in
:class:`model.downstream.encoder_wrapper.UltrasonicEncoderWrapper`::

    signal              (B, C, T)  float32
    signal_mask         (B, T)     bool      (always True — T is fixed)
    sampling_frequency_hz (B,)     float32
    window_size         (B,)       int64
    patch_timestamps_us (B, S)     float32
    label               (B,)       int64
    session_id          (B,)       int64     (logging only)
    patient_id          (B,)       int64     (logging only)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    import lightning.pytorch as pl  # type: ignore[no-redef]

from .hdf5_datamodule import compute_patch_timestamps_us, select_branch

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class DownstreamDataset(Dataset):
    """Random-access view over the downstream ``all.h5``.

    The h5 file is opened lazily per worker for DDP-fork safety (same
    pattern as :class:`data.hdf5_datamodule.HDF5Dataset`). Per-sample
    metadata (``labels``, ``session_id``, ``patient_id``) is loaded into
    RAM at construction time (24 bytes per row).

    Because every row has the same ``(C, T)`` shape and the same
    ``sampling_frequency_hz``, ``window_size`` and ``patch_timestamps_us``
    are computed once at ``__init__`` and shared across all rows.
    """

    def __init__(
        self,
        h5_path: str | Path,
        window_sizes: Sequence[int] = (8, 16, 32),
        target_patch_mm: float = 0.6,
    ) -> None:
        self.h5_path = str(h5_path)
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.target_patch_mm = float(target_patch_mm)

        with h5py.File(self.h5_path, "r") as f:
            attrs = f.attrs
            self.sampling_frequency_hz = float(attrs["sampling_frequency_hz"])
            self.dataset_name = str(attrs.get("dataset_name", Path(self.h5_path).stem))
            self.label_type = str(attrs.get("label_type", "label"))
            self.num_classes = int(attrs["num_classes"])
            self.num_channels = int(attrs["num_channels"])
            self.samples_per_frame = int(attrs["samples_per_frame"])

            self.labels = np.asarray(f["labels"][:], dtype=np.int64)
            self.session_id = np.asarray(f["session_id"][:], dtype=np.int64)
            self.patient_id = np.asarray(f["patient_id"][:], dtype=np.int64)

            sig_shape = f["signal"].shape
            if sig_shape[1:] != (self.num_channels, self.samples_per_frame):
                raise ValueError(
                    f"signal shape {sig_shape} inconsistent with attrs "
                    f"(C={self.num_channels}, T={self.samples_per_frame}).",
                )
            self._n = int(sig_shape[0])

        self.window_size = int(select_branch(
            self.sampling_frequency_hz, self.window_sizes, self.target_patch_mm,
        ))
        ts_np = compute_patch_timestamps_us(
            self.samples_per_frame, self.sampling_frequency_hz, self.window_size,
        )
        self.patch_timestamps_us = torch.from_numpy(np.asarray(ts_np, dtype=np.float32))

        self._file: Optional[h5py.File] = None

    def _ensure_open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r", libver="latest")
        return self._file

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        f = self._ensure_open()
        frame = np.asarray(f["signal"][idx], dtype=np.float32)
        return {
            "signal": torch.from_numpy(frame),
            "label": int(self.labels[idx]),
            "session_id": int(self.session_id[idx]),
            "patient_id": int(self.patient_id[idx]),
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "window_size": self.window_size,
            "patch_timestamps_us": self.patch_timestamps_us,
        }

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_file"] = None
        return state


# ----------------------------------------------------------------------
# Collate
# ----------------------------------------------------------------------

def collate_downstream(batch: list[dict]) -> dict:
    B = len(batch)
    signals = torch.stack([b["signal"] for b in batch], dim=0)  # (B, C, T)
    T = signals.shape[-1]
    signal_mask = torch.ones((B, T), dtype=torch.bool)
    return {
        "signal": signals,
        "signal_mask": signal_mask,
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "session_id": torch.tensor([b["session_id"] for b in batch], dtype=torch.long),
        "patient_id": torch.tensor([b["patient_id"] for b in batch], dtype=torch.long),
        "sampling_frequency_hz": torch.tensor(
            [b["sampling_frequency_hz"] for b in batch], dtype=torch.float32,
        ),
        "window_size": torch.tensor(
            [b["window_size"] for b in batch], dtype=torch.long,
        ),
        "patch_timestamps_us": torch.stack(
            [b["patch_timestamps_us"] for b in batch], dim=0,
        ),
    }


# ----------------------------------------------------------------------
# DataModule
# ----------------------------------------------------------------------

_VALID_SPLIT_MODES = ("intra_session", "intra_patient")


class DownstreamDataModule(pl.LightningDataModule):
    """LightningDataModule wrapping a single downstream ``all.h5``."""

    def __init__(
        self,
        h5_path: str | Path,
        split_mode: str,
        test_id: int,
        val_ratio: float = 0.1,
        seed: int = 42,
        batch_size: int = 64,
        num_workers: int = 4,
        window_sizes: Sequence[int] = (8, 16, 32),
        target_patch_mm: float = 0.6,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        shuffle_train: bool = True,
    ) -> None:
        super().__init__()
        if split_mode not in _VALID_SPLIT_MODES:
            raise ValueError(
                f"split_mode must be one of {_VALID_SPLIT_MODES}, got {split_mode!r}.",
            )
        if not 0.0 <= float(val_ratio) < 1.0:
            raise ValueError(
                f"val_ratio must be in [0, 1), got {val_ratio!r}.",
            )
        self.save_hyperparameters(ignore=["h5_path"])
        self.h5_path = Path(h5_path)
        self.split_mode = split_mode
        self.test_id = int(test_id)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.target_patch_mm = float(target_patch_mm)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.shuffle_train = bool(shuffle_train)

        self.dataset: Optional[DownstreamDataset] = None
        self.train_ds: Optional[Subset] = None
        self.val_ds: Optional[Subset] = None
        self.test_ds: Optional[Subset] = None

        # Public metadata (read from h5 attrs by the Dataset at setup()).
        self.num_channels: Optional[int] = None
        self.samples_per_frame: Optional[int] = None
        self.num_classes: Optional[int] = None
        self.label_type: Optional[str] = None
        self.sampling_frequency_hz: Optional[float] = None

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is not None:
            return
        ds = DownstreamDataset(
            self.h5_path,
            window_sizes=self.window_sizes,
            target_patch_mm=self.target_patch_mm,
        )
        self.dataset = ds
        self.num_channels = ds.num_channels
        self.samples_per_frame = ds.samples_per_frame
        self.num_classes = ds.num_classes
        self.label_type = ds.label_type
        self.sampling_frequency_hz = ds.sampling_frequency_hz

        id_arr = ds.session_id if self.split_mode == "intra_session" else ds.patient_id
        test_idx = np.where(id_arr == self.test_id)[0].astype(np.int64)
        if test_idx.size == 0:
            raise ValueError(
                f"split_mode={self.split_mode!r} test_id={self.test_id!r} matches "
                f"zero rows in {self.h5_path}. Available values: "
                f"{sorted(np.unique(id_arr).tolist())}",
            )
        pool = np.where(id_arr != self.test_id)[0].astype(np.int64)
        if pool.size == 0:
            raise ValueError(
                f"split_mode={self.split_mode!r} test_id={self.test_id!r} leaves "
                f"no rows for train+val.",
            )

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(pool)
        n_val = int(round(self.val_ratio * perm.size))
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])

        self.train_ds = Subset(ds, train_idx.tolist())
        self.val_ds = Subset(ds, val_idx.tolist())
        self.test_ds = Subset(ds, test_idx.tolist())

        log.info(
            "DownstreamDataModule: dataset=%s label_type=%s split_mode=%s test_id=%d | "
            "train=%d val=%d test=%d (total=%d, num_classes=%d, C=%d, T=%d, fs=%g Hz)",
            ds.dataset_name, ds.label_type, self.split_mode, self.test_id,
            len(self.train_ds), len(self.val_ds), len(self.test_ds), len(ds),
            ds.num_classes, ds.num_channels, ds.samples_per_frame,
            ds.sampling_frequency_hz,
        )

    # ------------------------------------------------------------------
    def _make_loader(self, ds: Dataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=collate_downstream,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return self._make_loader(
            self.train_ds, shuffle=self.shuffle_train, drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return self._make_loader(self.val_ds, shuffle=False, drop_last=False)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_ds is None or len(self.test_ds) == 0:
            return None
        return self._make_loader(self.test_ds, shuffle=False, drop_last=False)


__all__ = [
    "DownstreamDataModule",
    "DownstreamDataset",
    "collate_downstream",
]
