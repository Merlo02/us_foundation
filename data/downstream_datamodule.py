"""Downstream DataModule (single file — replaces the old ``data/downstream/``).

Consumes a single ``all.h5`` produced by :mod:`etl_downstream`. The
``/labels`` dataset shape/dtype depends on the task::

    classification:  /labels (N,)   int64    (one class id per frame)
    regression:      /labels (N, K) float32  (K continuous targets per frame)

Full schema::

    /signal       (N, C, T)        float32
    /labels       (N,) | (N, K)    int64 | float32
    /session_id   (N,)             int64
    /patient_id   (N,)             int64

    root attrs: sampling_frequency_hz, dataset_name, label_type,
                task_type, num_classes, num_channels, samples_per_frame
                (+ num_outputs, label_names when task_type='regression')

A missing ``task_type`` attr is treated as ``"classification"`` so older
files keep working unchanged.

Three split modes:

- ``intra_session`` — test = rows where ``session_id == test_id``. The
  remaining rows form the *non-test pool* which is split into train/val
  depending on ``grouped_val``:
    * ``grouped_val=True`` (default) — one *other* session is held out
      **as a whole** for validation (``val_id`` if given, else one
      auto-selected deterministically from the seed). Validation is then
      grouped, never a random slice of the train sessions — consecutive
      frames within a session are highly autocorrelated, so a random val
      slice leaks from train and reports an over-optimistic accuracy that
      does not track the held-out test session.
    * ``grouped_val=False`` — val is a random ``val_ratio`` slice of the
      non-test pool (same iid split logic as ``random`` mode, only the
      test set is grouped). Useful when the dataset has too few sessions
      to spare a whole one for validation; accept that val/train share
      sessions and will overestimate accuracy.
- ``intra_patient`` — same logic as ``intra_session``, partitioned by
  ``patient_id``.
- ``random`` — iid split across all rows ignoring session/patient: test
  takes ``test_ratio`` of the rows, the remainder is split into train/val
  by ``val_ratio``. Note: consecutive frames within a session are highly
  correlated, so this mode tends to overestimate accuracy (train↔test
  leakage). Useful as a baseline / sanity check.
Signals are interpolated (when ``apply_interpolate``) and then normalized
**per channel** with the same ``normalize_signal_numpy`` formula used by
the pretraining loader. Normalization statistics are computed **per
channel on the train split only** (each channel's ``mean/std/min/max`` is
aggregated over all train frames at that channel index, then applied
identically to train/val/test). Stats are a property of the channel, not
of the individual frame — this matches the pretraining ETL regime
(per-signal global stats) and avoids val/test amplitude leaking into the
normalization scale.

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

from transforms.normalization import normalize_signal_numpy, validate_normalization_type
from transforms.signal_processing import compute_interpolation_numpy

from .hdf5_datamodule import compute_patch_timestamps_us, select_branch
from .signal_tracer import SignalTracer

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
        apply_interpolate: bool = False,
        target_length: Optional[int] = None,
        normalization_type: str = "none",
        norm_eps_z: float = 1e-6,
        norm_eps_mm: float = 1e-10,
    ) -> None:
        self.h5_path = str(h5_path)
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.target_patch_mm = float(target_patch_mm)
        self.apply_interpolate = bool(apply_interpolate)
        self.normalization_type = validate_normalization_type(normalization_type)
        self.norm_eps_z = float(norm_eps_z)
        self.norm_eps_mm = float(norm_eps_mm)
        if self.apply_interpolate and (target_length is None or int(target_length) <= 0):
            raise ValueError(
                "apply_interpolate=True requires a positive target_length "
                "(typically the strict_target_length of the pretraining run "
                "so the encoder sees the same sequence length).",
            )

        with h5py.File(self.h5_path, "r") as f:
            attrs = f.attrs
            self.sampling_frequency_hz = float(attrs["sampling_frequency_hz"])
            self.dataset_name = str(attrs.get("dataset_name", Path(self.h5_path).stem))
            self.label_type = str(attrs.get("label_type", "label"))
            self.task_type = str(attrs.get("task_type", "classification"))
            self.is_regression = self.task_type == "regression"
            self.num_classes = int(attrs.get("num_classes", 0))
            self.num_outputs = int(attrs.get("num_outputs", 0))
            self.label_names = [str(s) for s in attrs.get("label_names", [])] or None
            self.num_channels = int(attrs["num_channels"])
            self.samples_per_frame = int(attrs["samples_per_frame"])

            # Classification → (N,) int64 class ids. Regression → (N, K)
            # float32 continuous targets (collate stacks them to (B, K)).
            if self.is_regression:
                self.labels = np.asarray(f["labels"][:], dtype=np.float32)
                if self.labels.ndim == 1:
                    self.labels = self.labels[:, None]
                if self.num_outputs == 0:
                    self.num_outputs = int(self.labels.shape[1])
            else:
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

        # ``effective_T`` is what the model sees per frame. With interpolation
        # we report the resampled length so collate / signal_mask use it.
        self._n_raw = self.samples_per_frame
        if self.apply_interpolate:
            self.effective_T = int(target_length)
        else:
            self.effective_T = self.samples_per_frame
        # Public field kept for downstream code that reads ``samples_per_frame``
        # (e.g. signal_mask = ones((B, T))). After interpolation it is the
        # resampled length, not the on-disk one.
        self.samples_per_frame = self.effective_T

        self.window_size = int(select_branch(
            self.sampling_frequency_hz, self.window_sizes, self.target_patch_mm,
        ))
        # When interpolation is applied, pass n_raw so patch midpoints map
        # back through np.interp's index grid → CT-RoPE timestamps remain
        # coherent with the original physical fs (same regime the encoder
        # saw during pretraining).
        ts_np = compute_patch_timestamps_us(
            self.effective_T,
            self.sampling_frequency_hz,
            self.window_size,
            n_raw=self._n_raw if self.apply_interpolate else None,
        )
        self.patch_timestamps_us = torch.from_numpy(np.asarray(ts_np, dtype=np.float32))

        # Per-channel normalization stats, shape ``(C,)``. Injected by the
        # DataModule via :meth:`set_channel_stats` after the train/val/test
        # split is decided (computed on the train rows only). ``None`` until
        # set; required when ``normalization_type != 'none'``.
        self.channel_mean: Optional[np.ndarray] = None
        self.channel_std: Optional[np.ndarray] = None
        self.channel_min: Optional[np.ndarray] = None
        self.channel_max: Optional[np.ndarray] = None

        self._file: Optional[h5py.File] = None

    def _ensure_open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r", libver="latest")
        return self._file

    def compute_channel_stats(
        self, indices: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-channel ``(mean, std, min, max)`` aggregated over ``indices``.

        Statistics are computed on the same pipeline stage that
        :meth:`_normalize_frame` later normalizes (i.e. after interpolation
        when ``apply_interpolate`` is set). Returns four ``(C,)`` float32
        arrays. Uses a local h5 handle (``with``) so ``self._file`` stays
        ``None`` and the lazy per-worker open pattern remains fork-safe.
        """
        idx = np.sort(np.asarray(indices, dtype=np.int64))
        if idx.size == 0:
            raise ValueError("compute_channel_stats: empty index set.")
        C = self.num_channels
        s = np.zeros(C, dtype=np.float64)
        sq = np.zeros(C, dtype=np.float64)
        count = 0
        cmin = np.full(C, np.inf, dtype=np.float64)
        cmax = np.full(C, -np.inf, dtype=np.float64)
        chunk = 2048
        with h5py.File(self.h5_path, "r", libver="latest") as f:
            dset = f["signal"]
            for start in range(0, idx.size, chunk):
                sel = idx[start:start + chunk]
                block = np.asarray(dset[sel], dtype=np.float32)  # (b, C, T_raw)
                if self.apply_interpolate:
                    block = np.stack(
                        [self._maybe_interpolate(fr) for fr in block], axis=0,
                    )
                b = block.astype(np.float64)
                s += b.sum(axis=(0, 2))
                sq += (b * b).sum(axis=(0, 2))
                count += b.shape[0] * b.shape[2]
                cmin = np.minimum(cmin, b.min(axis=(0, 2)))
                cmax = np.maximum(cmax, b.max(axis=(0, 2)))
        mean = s / count
        var = np.maximum(sq / count - mean * mean, 0.0)
        std = np.sqrt(var)
        return (
            mean.astype(np.float32),
            std.astype(np.float32),
            cmin.astype(np.float32),
            cmax.astype(np.float32),
        )

    def set_channel_stats(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        vmin: np.ndarray,
        vmax: np.ndarray,
    ) -> None:
        """Store per-channel normalization stats (each ``(C,)``)."""
        C = self.num_channels
        for name, arr in (("mean", mean), ("std", std), ("min", vmin), ("max", vmax)):
            if np.asarray(arr).shape != (C,):
                raise ValueError(
                    f"channel {name} stats must have shape ({C},), got "
                    f"{np.asarray(arr).shape}.",
                )
        self.channel_mean = np.asarray(mean, dtype=np.float32)
        self.channel_std = np.asarray(std, dtype=np.float32)
        self.channel_min = np.asarray(vmin, dtype=np.float32)
        self.channel_max = np.asarray(vmax, dtype=np.float32)

    def __len__(self) -> int:
        return self._n

    def _maybe_interpolate(self, frame: np.ndarray) -> np.ndarray:
        """Force-resample each channel to ``effective_T`` (matches the
        pretraining strict-interpolation regime). No-op when interpolation
        is disabled or the frame already has the target length."""
        if not self.apply_interpolate or frame.shape[-1] == self.effective_T:
            return frame
        resampled = np.empty((frame.shape[0], self.effective_T), dtype=np.float32)
        for c in range(frame.shape[0]):
            resampled[c] = compute_interpolation_numpy(
                frame[c], self.effective_T, force_resample=True,
            )
        return resampled

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Per-channel normalization with **train-global** stats.

        Each channel ``c`` is normalized with its own ``(mean, std, min, max)``
        aggregated over the train rows (see :meth:`compute_channel_stats` /
        :meth:`set_channel_stats`), applied via the same
        ``normalize_signal_numpy`` formula as the pretraining loader. Stats
        are a property of the channel (computed once over all train frames),
        not of the individual frame.
        """
        if self.normalization_type == "none":
            return frame
        if self.channel_mean is None:
            raise RuntimeError(
                "Per-channel normalization stats are not set. The DataModule "
                "must call set_channel_stats() (with stats from the train "
                "split) before any frame is read.",
            )
        out = np.empty_like(frame, dtype=np.float32)
        for c in range(frame.shape[0]):
            out[c] = normalize_signal_numpy(
                frame[c], self.normalization_type,
                float(self.channel_mean[c]), float(self.channel_std[c]),
                float(self.channel_min[c]), float(self.channel_max[c]),
                eps_z=self.norm_eps_z, eps_mm=self.norm_eps_mm,
            )
        return out

    def __getitem__(self, idx: int) -> dict:
        f = self._ensure_open()
        frame = np.asarray(f["signal"][idx], dtype=np.float32)  # (C, T_raw)
        frame = self._maybe_interpolate(frame)
        frame = self._normalize_frame(frame)
        # Regression → (K,) float32 vector; classification → python int.
        if self.is_regression:
            label = np.ascontiguousarray(self.labels[idx], dtype=np.float32)
        else:
            label = int(self.labels[idx])
        return {
            "signal": torch.from_numpy(np.ascontiguousarray(frame)),
            "label": label,
            "session_id": int(self.session_id[idx]),
            "patient_id": int(self.patient_id[idx]),
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "window_size": self.window_size,
            "patch_timestamps_us": self.patch_timestamps_us,
        }

    def load_trace_stages(
        self, idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(raw, preprocessed, normalized)`` frames for diagnostics.

        - ``raw``          — the on-disk ``(C, T_raw)`` frame.
        - ``preprocessed`` — after interpolation ``(C, effective_T)`` (== raw
          if interpolation is disabled).
        - ``normalized``   — after per-frame normalization (== preprocessed if
          ``normalization_type='none'``); this is exactly what enters the
          tokenizer.

        Used by :meth:`DownstreamDataModule._dump_signal_traces` so the shared
        :class:`~data.signal_tracer.SignalTracer` can plot every pipeline stage.
        """
        f = self._ensure_open()
        raw = np.asarray(f["signal"][idx], dtype=np.float32)
        pp = self._maybe_interpolate(raw)
        norm = self._normalize_frame(pp)
        return raw, pp, norm

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
    # Classification labels are python ints → (B,) long. Regression labels
    # are (K,) float vectors → (B, K) float32. Detect by the first item's
    # type so collate stays dataset-agnostic.
    first_label = batch[0]["label"]
    if isinstance(first_label, (int, np.integer)):
        label = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    else:
        label = torch.stack(
            [torch.as_tensor(np.asarray(b["label"]), dtype=torch.float32) for b in batch],
            dim=0,
        )  # (B, K)
    return {
        "signal": signals,
        "signal_mask": signal_mask,
        "label": label,
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

_VALID_SPLIT_MODES = ("intra_session", "intra_patient", "random")


class DownstreamDataModule(pl.LightningDataModule):
    """LightningDataModule wrapping a single downstream ``all.h5``."""

    def __init__(
        self,
        h5_path: str | Path,
        split_mode: str,
        test_id: Optional[int] = None,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        val_id: Optional[int] = None,
        grouped_val: bool = True,
        seed: int = 42,
        batch_size: int = 64,
        num_workers: int = 4,
        window_sizes: Sequence[int] = (8, 16, 32),
        target_patch_mm: float = 0.6,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        shuffle_train: bool = True,
        signal_trace_enabled: bool = False,
        signal_trace_dir: str | Path = "signal_traces",
        test_every_epoch: bool = False,
        apply_interpolate: bool = False,
        target_length: Optional[int] = None,
        normalization_type: str = "none",
        norm_eps_z: float = 1e-6,
        norm_eps_mm: float = 1e-10,
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
        if split_mode == "random":
            if not 0.0 < float(test_ratio) < 1.0:
                raise ValueError(
                    f"split_mode='random' requires test_ratio in (0, 1), "
                    f"got {test_ratio!r}.",
                )
            if float(test_ratio) + float(val_ratio) >= 1.0:
                raise ValueError(
                    f"test_ratio + val_ratio must be < 1, got "
                    f"{test_ratio!r} + {val_ratio!r}.",
                )
        else:
            if test_id is None:
                raise ValueError(
                    f"split_mode={split_mode!r} requires test_id (int).",
                )
            if not bool(grouped_val) and val_id is not None:
                raise ValueError(
                    f"grouped_val=False is incompatible with val_id={val_id!r}: "
                    f"val_id only makes sense when validation is grouped. "
                    f"Set grouped_val=true to use val_id, or leave val_id=null "
                    f"to take a random val_ratio slice of the non-test pool.",
                )
        self.save_hyperparameters(ignore=["h5_path"])
        self.h5_path = Path(h5_path)
        self.split_mode = split_mode
        self.test_id = int(test_id) if test_id is not None else None
        self.test_ratio = float(test_ratio)
        self.val_ratio = float(val_ratio)
        self.val_id = int(val_id) if val_id is not None else None
        self.grouped_val = bool(grouped_val)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.target_patch_mm = float(target_patch_mm)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.shuffle_train = bool(shuffle_train)
        self.signal_trace_enabled = bool(signal_trace_enabled)
        self.signal_trace_dir = Path(signal_trace_dir)
        self.test_every_epoch = bool(test_every_epoch)
        self._traced = False
        self.apply_interpolate = bool(apply_interpolate)
        self.target_length = int(target_length) if target_length is not None else None
        if self.apply_interpolate and self.target_length is None:
            raise ValueError(
                "data.apply_interpolate=true requires data.target_length "
                "(set it to the strict_target_length of the pretraining run).",
            )
        self.normalization_type = validate_normalization_type(normalization_type)
        self.norm_eps_z = float(norm_eps_z)
        self.norm_eps_mm = float(norm_eps_mm)

        # Shared pretraining diagnostic: plots raw -> interpolated -> normalized
        # signals exactly as they enter the tokenizer (rank-0, epoch 0).
        self.signal_tracer: Optional[SignalTracer] = (
            SignalTracer(True, str(self.signal_trace_dir))
            if self.signal_trace_enabled else None
        )

        self.dataset: Optional[DownstreamDataset] = None
        self.train_ds: Optional[Subset] = None
        self.val_ds: Optional[Subset] = None
        self.test_ds: Optional[Subset] = None

        # Public metadata (read from h5 attrs by the Dataset at setup()).
        self.num_channels: Optional[int] = None
        self.samples_per_frame: Optional[int] = None
        self.num_classes: Optional[int] = None
        self.num_outputs: Optional[int] = None
        self.task_type: Optional[str] = None
        self.label_names: Optional[list[str]] = None
        self.label_type: Optional[str] = None
        self.sampling_frequency_hz: Optional[float] = None

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is not None:
            if stage in (None, "fit"):
                self._dump_signal_traces()
            return
        ds = DownstreamDataset(
            self.h5_path,
            window_sizes=self.window_sizes,
            target_patch_mm=self.target_patch_mm,
            apply_interpolate=self.apply_interpolate,
            target_length=self.target_length,
            normalization_type=self.normalization_type,
            norm_eps_z=self.norm_eps_z,
            norm_eps_mm=self.norm_eps_mm,
        )
        self.dataset = ds
        self.num_channels = ds.num_channels
        self.samples_per_frame = ds.samples_per_frame
        self.num_classes = ds.num_classes
        self.num_outputs = ds.num_outputs
        self.task_type = ds.task_type
        self.label_names = ds.label_names
        self.label_type = ds.label_type
        self.sampling_frequency_hz = ds.sampling_frequency_hz

        rng = np.random.default_rng(self.seed)
        group_kind: Optional[str] = None
        val_group_id: Optional[int] = None

        if self.split_mode == "random":
            perm = rng.permutation(len(ds))
            n_test = int(round(self.test_ratio * perm.size))
            n_val = int(round(self.val_ratio * perm.size))
            if n_test == 0 or n_val == 0 or n_test + n_val >= perm.size:
                raise ValueError(
                    f"split_mode='random' produced a degenerate split: "
                    f"n_test={n_test} n_val={n_val} total={perm.size} "
                    f"(test_ratio={self.test_ratio}, val_ratio={self.val_ratio}).",
                )
            test_idx = np.sort(perm[:n_test])
            val_idx = np.sort(perm[n_test:n_test + n_val])
            train_idx = np.sort(perm[n_test + n_val:])
        else:
            group_kind = "session_id" if self.split_mode == "intra_session" else "patient_id"
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

            # Validation source depends on ``grouped_val``:
            #   * True  → hold out one *other* group as a whole (grouped val;
            #             matches the test-group generalization regime). A
            #             random slice would leak (adjacent frames within a
            #             session are near-duplicates) and report an
            #             over-optimistic val accuracy.
            #   * False → random ``val_ratio`` slice of the non-test pool
            #             (iid within the pool). Accept the train↔val leak;
            #             only the test split is grouped.
            pool_group_ids = np.unique(id_arr[pool])
            if self.val_ratio == 0.0 and self.val_id is None:
                val_idx = np.zeros(0, dtype=np.int64)
                train_idx = np.sort(pool)
            elif not self.grouped_val:
                if not 0.0 < self.val_ratio < 1.0:
                    raise ValueError(
                        f"grouped_val=False requires val_ratio in (0, 1), "
                        f"got {self.val_ratio!r}. Set val_ratio=0 to skip "
                        f"validation, or grouped_val=true to hold out a group.",
                    )
                pool_perm = rng.permutation(pool)
                n_val = int(round(self.val_ratio * pool_perm.size))
                if n_val == 0 or n_val >= pool_perm.size:
                    raise ValueError(
                        f"grouped_val=False produced a degenerate val split: "
                        f"n_val={n_val} pool_size={pool_perm.size} "
                        f"(val_ratio={self.val_ratio}).",
                    )
                val_idx = np.sort(pool_perm[:n_val])
                train_idx = np.sort(pool_perm[n_val:])
            else:
                if self.val_id is not None:
                    if self.val_id == self.test_id:
                        raise ValueError(
                            f"data.val_id ({self.val_id}) must differ from "
                            f"test_id ({self.test_id}).",
                        )
                    if self.val_id not in pool_group_ids:
                        raise ValueError(
                            f"split_mode={self.split_mode!r} val_id={self.val_id!r} "
                            f"matches no non-test rows. Available {group_kind} values "
                            f"(excluding test_id={self.test_id}): "
                            f"{sorted(pool_group_ids.tolist())}",
                        )
                    val_group_id = int(self.val_id)
                else:
                    if pool_group_ids.size < 2:
                        raise ValueError(
                            f"split_mode={self.split_mode!r} needs >= 2 non-test "
                            f"{group_kind} groups to hold out a grouped validation "
                            f"set, but only {pool_group_ids.size} is available "
                            f"({sorted(pool_group_ids.tolist())}). Add more data, set "
                            f"data.val_id explicitly, set data.val_ratio=0 to skip "
                            f"validation, or use split_mode='random'.",
                        )
                    val_group_id = int(rng.choice(pool_group_ids))
                val_idx = np.sort(
                    np.where(id_arr == val_group_id)[0].astype(np.int64),
                )
                train_idx = np.sort(
                    np.where(
                        (id_arr != self.test_id) & (id_arr != val_group_id),
                    )[0].astype(np.int64),
                )
                if train_idx.size == 0:
                    raise ValueError(
                        f"Grouped validation ({group_kind}={val_group_id}) leaves "
                        f"no training rows. Choose a different data.val_id or add "
                        f"more groups.",
                    )

        # Per-channel normalization stats: computed ONCE on the train split
        # and applied identically to train/val/test. Keeping the calculation
        # here (after the split is decided) guarantees that val/test
        # amplitudes never leak into the normalization scale. Skip when
        # ``normalization_type='none'`` — no stats are ever read in that case.
        if self.normalization_type != "none":
            ch_mean, ch_std, ch_min, ch_max = ds.compute_channel_stats(
                train_idx.tolist(),
            )
            ds.set_channel_stats(ch_mean, ch_std, ch_min, ch_max)
            log.info(
                "DownstreamDataModule: per-channel %s stats from %d train rows "
                "→ mean=%s std=%s min=%s max=%s",
                self.normalization_type, train_idx.size,
                np.array2string(ch_mean, precision=4),
                np.array2string(ch_std, precision=4),
                np.array2string(ch_min, precision=4),
                np.array2string(ch_max, precision=4),
            )

        self.train_ds = Subset(ds, train_idx.tolist())
        self.val_ds = Subset(ds, val_idx.tolist())
        self.test_ds = Subset(ds, test_idx.tolist())

        if self.split_mode == "random":
            test_desc = f"random(test_ratio={self.test_ratio})"
            val_desc = f"random(val_ratio={self.val_ratio})"
        else:
            test_desc = f"{group_kind}={self.test_id}"
            if val_group_id is not None:
                val_desc = f"{group_kind}={val_group_id}"
            elif not self.grouped_val and len(self.val_ds) > 0:
                val_desc = f"random(val_ratio={self.val_ratio}, within non-test pool)"
            else:
                val_desc = "none"

        target_desc = (
            f"num_outputs={ds.num_outputs} ({ds.label_names})"
            if ds.is_regression else f"num_classes={ds.num_classes}"
        )
        log.info(
            "DownstreamDataModule: dataset=%s task=%s label_type=%s split_mode=%s | "
            "test=[%s] val=[%s] | train=%d val=%d test=%d "
            "(total=%d, %s, C=%d, T=%d, fs=%g Hz, norm=%s)",
            ds.dataset_name, ds.task_type, ds.label_type, self.split_mode,
            test_desc, val_desc,
            len(self.train_ds), len(self.val_ds), len(self.test_ds), len(ds),
            target_desc, ds.num_channels, ds.samples_per_frame,
            ds.sampling_frequency_hz, self.normalization_type,
        )

        if stage in (None, "fit"):
            self._dump_signal_traces()

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

    def val_dataloader(self):
        """Return the val loader, or ``[val, test]`` when ``test_every_epoch``.

        Returning a list triggers Lightning's multi-dataloader validation:
        ``LightningModule.validation_step`` is called once per loader with
        an extra ``dataloader_idx`` argument (0 = val, 1 = test). The module
        dispatches that to its ``_step`` with the right stage so metrics
        land under ``val/...`` and ``test/...`` independently.
        """
        assert self.val_ds is not None
        val_loader = self._make_loader(self.val_ds, shuffle=False, drop_last=False)
        if not self.test_every_epoch:
            return val_loader
        if self.test_ds is None or len(self.test_ds) == 0:
            return val_loader
        test_loader = self._make_loader(self.test_ds, shuffle=False, drop_last=False)
        return [val_loader, test_loader]

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_ds is None or len(self.test_ds) == 0:
            return None
        return self._make_loader(self.test_ds, shuffle=False, drop_last=False)

    # ------------------------------------------------------------------
    # Signal trace (rank 0): one PNG per (label, session, channel) from the
    # train split, via the shared pretraining ``SignalTracer``. Each PNG
    # shows raw -> interpolated -> normalized, i.e. exactly what enters the
    # tokenizer.
    # ------------------------------------------------------------------
    def _dump_signal_traces(self) -> None:
        if self.signal_tracer is None or self._traced:
            return
        if self.trainer is not None and int(self.trainer.global_rank) != 0:
            self._traced = True
            return
        if self.dataset is None or self.train_ds is None:
            return

        ds = self.dataset
        train_idx = np.asarray(self.train_ds.indices, dtype=np.int64)
        sessions = ds.session_id[train_idx]

        # Classification → one frame per (class, session). Regression has no
        # classes, so trace one frame per session instead (label set to -1
        # and omitted from the filename below).
        seen: set[tuple[int, int]] = set()
        picks: list[tuple[int, int, int]] = []  # (label, session, h5_idx)
        for i, h5_idx in enumerate(train_idx):
            label = -1 if ds.is_regression else int(ds.labels[h5_idx])
            key = (label, int(sessions[i]))
            if key in seen:
                continue
            seen.add(key)
            picks.append((key[0], key[1], int(h5_idx)))

        log.info(
            "signal_trace: dumping %d (label, session) pair(s) x %d channel(s) to "
            "%s — raw -> interpolated -> normalized (%s), as they enter the tokenizer",
            len(picks), ds.num_channels, self.signal_tracer.output_dir,
            self.normalization_type,
        )

        # Train-global per-channel stats actually used by ``_normalize_frame``.
        # If normalization is disabled they may be None; report NaN in that
        # case so the plot box still renders.
        has_stats = (
            self.normalization_type != "none" and ds.channel_mean is not None
        )
        for label, session, h5_idx in picks:
            raw, pp, norm = ds.load_trace_stages(h5_idx)  # each (C, T)
            patient = int(ds.patient_id[h5_idx])
            cls_tag = "reg" if ds.is_regression else f"cls{label:02d}"
            for c in range(raw.shape[0]):
                src_key = (
                    f"{ds.dataset_name}_{cls_tag}_sess{session:03d}"
                    f"_pat{patient:03d}_ch{c}"
                )
                if not self.signal_tracer.should_trace(src_key):
                    continue
                if has_stats:
                    stats = {
                        "signal_mean": float(ds.channel_mean[c]),
                        "signal_std": float(ds.channel_std[c]),
                        "signal_min": float(ds.channel_min[c]),
                        "signal_max": float(ds.channel_max[c]),
                    }
                else:
                    stats = {
                        "signal_mean": float("nan"),
                        "signal_std": float("nan"),
                        "signal_min": float("nan"),
                        "signal_max": float("nan"),
                    }
                self.signal_tracer.trace(
                    raw_etl=raw[c],
                    pp_full=pp[c],
                    norm_chunks=[norm[c]],
                    normalization_type=self.normalization_type,
                    stats=stats,
                    dataset_source=src_key,
                    target_patches=None,
                    window_size=ds.window_size,
                )

        self._traced = True


__all__ = [
    "DownstreamDataModule",
    "DownstreamDataset",
    "collate_downstream",
]
