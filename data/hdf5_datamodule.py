"""PyTorch Lightning DataModule for the CSR-style HDF5 output of the ETL.

Layout of each ``{split}.h5`` file (written by ``etl.writers.HDF5Writer``):

- ``data``                 ``(M,) float32``  — concatenated signals
- ``offsets``              ``(N+1,) int64``
- ``sampling_frequencies`` ``(N,) float32``
- ``dataset_sources``      ``(N,) vlen utf-8 str``
- ``signal_means`` … ``signal_maxs`` ``(N,) float32`` — global stats (optional until ETL refresh)

The sample ``i`` is ``data[offsets[i]:offsets[i+1]]``.

At ``__init__`` time the dataset loads ``offsets``, ``sampling_frequencies``
and ``dataset_sources`` into RAM (≤ ~200 MB at N≈17M), and keeps the file
handle lazily opened in ``__getitem__`` for DDP-fork safety.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
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
from transforms.signal_processing import (
    bandpass_edges_from_center_frequency,
    compute_bandpass_numpy,
    compute_envelope_numpy,
    compute_interpolation_numpy,
)

from .samplers import EpochSubsetSampler
from .signal_tracer import SignalTracer, set_signal_trace_epoch, trace_dataloader_worker_init

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


def _base_dataset_name(src: object) -> str:
    return str(src).split("::", 1)[0]


def _raw_row_counts_by_base(sources: np.ndarray, n: int) -> dict[str, int]:
    if n <= 0:
        return {}
    return dict(sorted(Counter(_base_dataset_name(sources[i]) for i in range(n)).items()))


def _format_count_dict(d: dict[str, int]) -> str:
    if not d:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items()))


# ------------------------------------------------------------------
# Online preprocessing helpers (Experiment D — variable-S path)
# ------------------------------------------------------------------

_VALID_PREPROCESSING_MODES = ("raw", "bandpass", "envelope")


@dataclass
class _PreprocessingParams:
    """Resolved preprocessing parameters passed from DataModule to Dataset.

    Built once in ``setup()`` from the ETL config YAML and then shared across
    all ``HDF5Dataset`` / WebDataset decode calls for that run.
    """
    mode: str                              # "raw" | "bandpass" | "envelope"
    apply_interpolate: bool
    dataset_fc_hz: dict[str, float] = field(default_factory=dict)
    dataset_bw: dict[str, float] = field(default_factory=dict)
    bandpass_order: int = 4
    target_length: Optional[int] = None
    dataset_truncate_mode: dict[str, str] = field(default_factory=dict)


def _build_preprocessing_params(
    preprocessing_mode: str,
    apply_interpolate: bool,
    etl_config_path: str,
) -> _PreprocessingParams:
    """Parse *etl_config_path* and return a resolved ``_PreprocessingParams``.

    Uses the same YAML-loading logic as ``runners/run_etl.py``.
    """
    import yaml
    from etl.config import DatasetConfig, ETLConfig

    cfg_path = Path(etl_config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"etl_config_path not found: {cfg_path}. "
            "Set data.etl_config_path to the ETL config used to produce the dataset."
        )
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    # Mirror the key-normalisation done in run_etl.py
    if raw.get("rf_bandwidth_fraction") is None and raw.get("envelope_bandpass_fraction") is not None:
        raw["rf_bandwidth_fraction"] = raw.pop("envelope_bandpass_fraction")
    else:
        raw.pop("envelope_bandpass_fraction", None)
    raw.pop("bandpass_low_hz", None)
    raw.pop("bandpass_high_hz", None)

    datasets = [DatasetConfig(**ds) for ds in raw.pop("datasets", [])]
    etl_cfg = ETLConfig(datasets=datasets, **raw)

    fc_hz: dict[str, float] = {}
    bw: dict[str, float] = {}
    truncate_mode: dict[str, str] = {}

    for ds in etl_cfg.datasets:
        ex = ds.extra or {}
        fc = ex.get("transmit_center_frequency_hz") or ex.get("tx_fc_hz")
        if preprocessing_mode in ("bandpass", "envelope"):
            if fc is None or float(fc) <= 0:
                raise ValueError(
                    f"Dataset {ds.name!r}: extra.transmit_center_frequency_hz is required "
                    f"for preprocessing_mode={preprocessing_mode!r} but is missing or zero "
                    f"in ETL config {cfg_path}."
                )
            fc_hz[ds.name] = float(fc)
        bw[ds.name] = float(ex.get("rf_bandwidth_fraction", etl_cfg.rf_bandwidth_fraction))
        tm = ex.get("interpolation_truncate_mode") or etl_cfg.interpolation_truncate_mode or "left"
        truncate_mode[ds.name] = str(tm)

    target_length: Optional[int] = None
    if apply_interpolate:
        if etl_cfg.target_length is None or int(etl_cfg.target_length) <= 0:
            raise ValueError(
                f"apply_interpolate=True but ETL config {cfg_path} has no "
                "target_length set (or it is non-positive)."
            )
        target_length = int(etl_cfg.target_length)

    return _PreprocessingParams(
        mode=preprocessing_mode,
        apply_interpolate=apply_interpolate,
        dataset_fc_hz=fc_hz,
        dataset_bw=bw,
        bandpass_order=int(etl_cfg.bandpass_order),
        target_length=target_length,
        dataset_truncate_mode=truncate_mode,
    )


def _apply_online_preprocessing(
    signal: np.ndarray,
    dataset_source: str,
    fs: float,
    pp: _PreprocessingParams,
) -> np.ndarray:
    """Apply the ``bandpass → envelope → interpolate`` chain to *signal*.

    Called per-sample inside ``__getitem__`` (HDF5) or stream-decode (WDS),
    before normalisation, on the full native-length signal.
    """
    base = _base_dataset_name(dataset_source)
    if pp.mode != "raw":
        lo, hi = bandpass_edges_from_center_frequency(
            pp.dataset_fc_hz[base], pp.dataset_bw[base], fs,
        )
        signal = compute_bandpass_numpy(signal, fs, lo, hi, pp.bandpass_order)
        if pp.mode == "envelope":
            signal = compute_envelope_numpy(signal)
    if pp.apply_interpolate:
        tmode = pp.dataset_truncate_mode.get(base, "left")
        signal = compute_interpolation_numpy(signal, pp.target_length, tmode)  # type: ignore[arg-type]
    return signal


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
    preprocessing_mode :
        Online signal preprocessing applied **before** normalisation in the
        variable-S path only (``target_patches`` must be ``None``).
        ``"raw"`` (default) — no transformation.
        ``"bandpass"`` — zero-phase RF bandpass filter.
        ``"envelope"`` — bandpass then Hilbert amplitude envelope.
        Requires *_pp* to be provided by the parent ``HDF5DataModule``.
    apply_interpolate :
        If ``True``, resample/truncate the signal to ``_pp.target_length``
        after the ``preprocessing_mode`` step.  The per-dataset truncation
        mode (left/right/center) is read from ``_pp.dataset_truncate_mode``.
        Requires *_pp*. Incompatible with fixed-S (``target_patches`` set).
    _pp :
        ``_PreprocessingParams`` instance built by ``HDF5DataModule.setup()``
        from the ETL config YAML.  Must be provided when
        ``preprocessing_mode != "raw"`` or ``apply_interpolate=True``.
    """

    def __init__(
        self,
        h5_path: str | Path,
        window_sizes: Sequence[int] = (8, 16, 32),
        target_patch_mm: float = 0.6,
        target_patches: Optional[int] = None,
        min_valid_patches: int = 1,
        normalization_type: str = "none",
        norm_eps_z: float = 1e-6,
        norm_eps_mm: float = 1e-10,
        signal_tracer: Optional["SignalTracer"] = None,
        preprocessing_mode: str = "raw",
        apply_interpolate: bool = False,
        _pp: Optional[_PreprocessingParams] = None,
    ) -> None:
        self.h5_path = str(h5_path)
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.target_patch_mm = float(target_patch_mm)
        self.target_patches = int(target_patches) if target_patches is not None else None
        self.min_valid_patches = int(min_valid_patches)
        self.normalization_type = validate_normalization_type(normalization_type)
        self.norm_eps_z = float(norm_eps_z)
        self.norm_eps_mm = float(norm_eps_mm)
        self.signal_tracer = signal_tracer

        if preprocessing_mode not in _VALID_PREPROCESSING_MODES:
            raise ValueError(
                f"preprocessing_mode must be one of {_VALID_PREPROCESSING_MODES}, "
                f"got {preprocessing_mode!r}."
            )
        if preprocessing_mode != "raw" and self.target_patches is not None:
            raise ValueError(
                f"preprocessing_mode={preprocessing_mode!r} is not supported with "
                f"HDF5 fixed-S (target_patches={target_patches}). "
                "Run ETL offline with the desired preprocessing mode and point "
                "hdf5_dir at that output directory."
            )
        if apply_interpolate and self.target_patches is not None:
            raise ValueError(
                f"apply_interpolate=True is not supported with HDF5 fixed-S "
                f"(target_patches={target_patches})."
            )
        needs_pp = preprocessing_mode != "raw" or apply_interpolate
        if needs_pp and _pp is None:
            raise ValueError(
                f"preprocessing_mode={preprocessing_mode!r} / apply_interpolate={apply_interpolate} "
                "requires _pp (_PreprocessingParams) to be provided. "
                "Use HDF5DataModule with etl_config_path set."
            )
        self.preprocessing_mode = preprocessing_mode
        self.apply_interpolate = apply_interpolate
        self._pp = _pp

        if self.target_patches is not None and self.target_patches <= 0:
            raise ValueError(f"target_patches must be > 0, got {self.target_patches}")

        self.signal_means: Optional[np.ndarray] = None
        self.signal_stds: Optional[np.ndarray] = None
        self.signal_mins: Optional[np.ndarray] = None
        self.signal_maxs: Optional[np.ndarray] = None

        with h5py.File(self.h5_path, "r") as f:
            self.offsets = f["offsets"][:].astype(np.int64)
            self.freqs = f["sampling_frequencies"][:].astype(np.float32)
            # vlen str loads as object ndarray → decode to Python strings
            sources = f["dataset_sources"][:]
            self.sources = np.asarray([
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in sources
            ], dtype=object)
            if self.normalization_type != "none":
                required = ("signal_means", "signal_stds", "signal_mins", "signal_maxs")
                missing = [k for k in required if k not in f]
                if missing:
                    raise ValueError(
                        f"HDF5 {self.h5_path!r} missing datasets {missing}. "
                        "Re-run ETL with updated writers or use normalization_type='none'.",
                    )
                self.signal_means = f["signal_means"][:].astype(np.float32)
                self.signal_stds = f["signal_stds"][:].astype(np.float32)
                self.signal_mins = f["signal_mins"][:].astype(np.float32)
                self.signal_maxs = f["signal_maxs"][:].astype(np.float32)

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

    def _finalize_signal_chunk(
        self,
        signal: np.ndarray,
        sample_idx: int,
        dataset_source: str,
        raw_etl: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Online preprocessing (variable-S only; _pp is None in fixed-S / raw mode)
        if self._pp is not None and (self._pp.mode != "raw" or self._pp.apply_interpolate):
            fs = float(self.freqs[sample_idx])
            signal = _apply_online_preprocessing(signal, dataset_source, fs, self._pp)

        mean = std = vmin = vmax = 0.0
        if self.normalization_type != "none":
            assert self.signal_means is not None
            mean = float(self.signal_means[sample_idx])
            std = float(self.signal_stds[sample_idx])
            vmin = float(self.signal_mins[sample_idx])
            vmax = float(self.signal_maxs[sample_idx])
        stats_map = {
            "signal_mean": mean,
            "signal_std": std,
            "signal_min": vmin,
            "signal_max": vmax,
        }
        out = normalize_signal_numpy(
            signal,
            self.normalization_type,
            mean,
            std,
            vmin,
            vmax,
            eps_z=self.norm_eps_z,
            eps_mm=self.norm_eps_mm,
        )
        # Variable-S tracing: raw_etl is provided only from the variable-S path.
        if (
            raw_etl is not None
            and self.signal_tracer is not None
            and self.signal_tracer.should_trace(dataset_source)
        ):
            W = int(self.window_for_sample[sample_idx])
            self.signal_tracer.trace(
                raw_etl=raw_etl,
                pp_full=signal,
                norm_chunks=[np.asarray(out, dtype=np.float32)],
                normalization_type=self.normalization_type,
                stats=stats_map,
                dataset_source=dataset_source,
                target_patches=None,
                window_size=W,
            )
        return out

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
            raw_buf = np.asarray(f["data"][start:end], dtype=np.float32)
            src = str(self.sources[idx])
            signal = self._finalize_signal_chunk(raw_buf.copy(), idx, src, raw_etl=raw_buf)
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
        length_i = sample_end - sample_start
        src_fixed = str(self.sources[sample_idx])
        fs = float(self.freqs[sample_idx])

        # Fixed-S tracing: load the full signal once, compute all chunks, plot.
        # Only executed when signal_trace_enabled=true AND this source is new.
        if self.signal_tracer is not None and self.signal_tracer.should_trace(src_fixed):
            full_raw = np.asarray(f["data"][sample_start:sample_end], dtype=np.float32)
            full_pp = full_raw
            if self._pp is not None and (self._pp.mode != "raw" or self._pp.apply_interpolate):
                full_pp = _apply_online_preprocessing(full_raw, src_fixed, fs, self._pp)
            mean_t = std_t = vmin_t = vmax_t = 0.0
            if self.normalization_type != "none" and self.signal_means is not None:
                mean_t = float(self.signal_means[sample_idx])
                std_t = float(self.signal_stds[sample_idx])
                vmin_t = float(self.signal_mins[sample_idx])
                vmax_t = float(self.signal_maxs[sample_idx])
            n_chunks_full = max(1, (len(full_pp) + target_T - 1) // target_T)
            norm_chunks: list[np.ndarray] = []
            for c in range(n_chunks_full):
                chunk_sl = full_pp[c * target_T: min((c + 1) * target_T, len(full_pp))]
                nc = normalize_signal_numpy(
                    chunk_sl,
                    self.normalization_type,
                    mean_t, std_t, vmin_t, vmax_t,
                    eps_z=self.norm_eps_z,
                    eps_mm=self.norm_eps_mm,
                )
                norm_chunks.append(np.asarray(nc, dtype=np.float32))
            self.signal_tracer.trace(
                raw_etl=full_raw,
                pp_full=full_pp,
                norm_chunks=norm_chunks,
                normalization_type=self.normalization_type,
                stats={
                    "signal_mean": mean_t, "signal_std": std_t,
                    "signal_min": vmin_t, "signal_max": vmax_t,
                },
                dataset_source=src_fixed,
                target_patches=self.target_patches,
                window_size=W,
            )

        abs_start = sample_start + chunk_offset_in_sample
        abs_end = min(sample_start + chunk_offset_in_sample + target_T, sample_end)
        signal = np.asarray(f["data"][abs_start:abs_end], dtype=np.float32)
        signal = self._finalize_signal_chunk(signal, sample_idx, src_fixed)

        # Timestamps continue the original signal's clock so CT-RoPE sees
        # consistent continuous-time positions across chunks.
        ts = compute_patch_timestamps_us(
            signal.size, fs, W, sample_offset=chunk_offset_in_sample,
        )

        num_chunks = max(1, (length_i + target_T - 1) // target_T)
        chunk_index = chunk_offset_in_sample // target_T if target_T > 0 else 0

        return {
            "signal": torch.from_numpy(signal),
            "sampling_frequency_hz": fs,
            "dataset_source": src_fixed,
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


def _item_counts_for_flat_indices(ds: HDF5Dataset, flat_indices: np.ndarray) -> dict[str, int]:
    """Count training items (chunk or native index) by base dataset name."""
    if flat_indices.size == 0:
        return {}
    idx = flat_indices.astype(np.int64, copy=False)
    if ds.target_patches is not None:
        parent = ds._chunk_sample_idx[idx]
        srcs = ds.sources[parent]
    else:
        srcs = ds.sources[idx]
    return dict(sorted(Counter(_base_dataset_name(s) for s in srcs).items()))


def _group_indices_by_base(
    base_names: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """Return ``[(base_dataset_name, flat_index_array), …]`` sorted by name.

    *base_names* is a 1-D object array of base dataset name strings, one per
    flat training index (i.e. already resolved through ``_chunk_sample_idx``
    when in fixed-S mode).
    """
    unique = sorted(set(base_names.tolist()))
    result = []
    for name in unique:
        mask = base_names == name
        result.append((name, np.where(mask)[0].astype(np.int64)))
    return result


def _split_lg_budget(total: int, ratios: Sequence[float]) -> tuple[int, int, int]:
    """Split ``total`` into three integer counts (train, val, test) via largest remainder.

    ``ratios`` are normalised to sum to 1; the three returned values sum to ``total``.
    """
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    r = np.asarray(ratios, dtype=np.float64).ravel()
    if r.size != 3:
        raise ValueError(f"Expected three ratios (train, val, test), got {r.size}")
    s = float(r.sum())
    if s <= 0:
        raise ValueError(f"lg_budget_split_ratios must sum to a positive value, got {s}")
    r = r / s
    if total == 0:
        return 0, 0, 0
    exact = total * r
    floors = np.floor(exact).astype(int)
    rem = int(total) - int(floors.sum())
    frac = exact - floors.astype(np.float64)
    order = np.argsort(-frac)
    out = floors.copy()
    for i in range(rem):
        out[order[i % 3]] += 1
    return int(out[0]), int(out[1]), int(out[2])


def _post_chunk_item_counts_by_base(ds: HDF5Dataset) -> dict[str, int]:
    if ds.target_patches is None:
        return _raw_row_counts_by_base(ds.sources, ds._n)
    if ds._chunk_sample_idx is None or ds._chunk_sample_idx.size == 0:
        return {}
    parents = ds.sources[ds._chunk_sample_idx.astype(np.int64, copy=False)]
    return dict(sorted(Counter(_base_dataset_name(p) for p in parents).items()))


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
    - ``dynamic_epoch``  — ``epoch_k`` is a **global** LG budget split across
      train / val / test (default 0.8 / 0.1 / 0.1, see ``lg_budget_split_ratios``):
      each training epoch draws its train share from LG in ``train.h5``;
      val/test use a fixed random subset of that size from their splits' LG
      pools (all non-LG items are always included).
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
        lg_budget_split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        normalization_type: str = "none",
        norm_eps_z: float = 1e-6,
        norm_eps_mm: float = 1e-10,
        signal_trace_enabled: bool = False,
        signal_trace_dir: str | Path = "debug_plots",
        preprocessing_mode: str = "raw",
        apply_interpolate: bool = False,
        etl_config_path: Optional[str] = None,
        dataset_caps: Optional[dict] = None,
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
        self.lg_budget_split_ratios = tuple(float(x) for x in lg_budget_split_ratios)
        self.seed = seed
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.normalization_type = validate_normalization_type(normalization_type)
        self.norm_eps_z = float(norm_eps_z)
        self.norm_eps_mm = float(norm_eps_mm)
        self.signal_trace_enabled = bool(signal_trace_enabled)
        self.signal_trace_dir = Path(signal_trace_dir)

        if preprocessing_mode not in _VALID_PREPROCESSING_MODES:
            raise ValueError(
                f"preprocessing_mode must be one of {_VALID_PREPROCESSING_MODES}, "
                f"got {preprocessing_mode!r}."
            )
        needs_etl_cfg = preprocessing_mode != "raw" or apply_interpolate
        if needs_etl_cfg and not etl_config_path:
            raise ValueError(
                f"preprocessing_mode={preprocessing_mode!r} / apply_interpolate={apply_interpolate} "
                "requires etl_config_path to be set."
            )
        self.preprocessing_mode = preprocessing_mode
        self.apply_interpolate = bool(apply_interpolate)
        self.etl_config_path = etl_config_path
        self._pp: Optional[_PreprocessingParams] = None
        self.dataset_caps: dict[str, int] = {
            str(k): int(v) for k, v in (dataset_caps or {}).items()
        }

        self.signal_tracer: Optional[SignalTracer] = None
        if self.signal_trace_enabled:
            self.signal_tracer = SignalTracer(True, str(self.signal_trace_dir))

        assert sampling_strategy in ("naive", "static", "dynamic_epoch", "proportional"), (
            f"Unknown sampling_strategy: {sampling_strategy!r}"
        )

        self.train_ds: Optional[HDF5Dataset] = None
        self.val_ds: Optional[HDF5Dataset] = None
        self.test_ds: Optional[HDF5Dataset] = None
        self._train_sampler: Optional[EpochSubsetSampler] = None
        self._train_indices: Optional[np.ndarray] = None
        self._val_eval_ds: Optional[Dataset] = None
        self._test_eval_ds: Optional[Dataset] = None
        # Total training samples per epoch (all ranks). Set in _log_effective_per_epoch_data().
        self._epoch_train_samples: Optional[int] = None

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
            normalization_type=self.normalization_type,
            norm_eps_z=self.norm_eps_z,
            norm_eps_mm=self.norm_eps_mm,
            signal_tracer=self.signal_tracer,
            preprocessing_mode=self.preprocessing_mode,
            apply_interpolate=self.apply_interpolate,
            _pp=self._pp,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        # Build preprocessing params once (shared across all splits).
        if self._pp is None and (self.preprocessing_mode != "raw" or self.apply_interpolate):
            assert self.etl_config_path is not None
            self._pp = _build_preprocessing_params(
                self.preprocessing_mode,
                self.apply_interpolate,
                self.etl_config_path,
            )
            log.info(
                "HDF5DataModule: online preprocessing mode=%r apply_interpolate=%s "
                "target_length=%s (etl_config=%s)",
                self._pp.mode,
                self._pp.apply_interpolate,
                self._pp.target_length,
                self.etl_config_path,
            )
        if stage in (None, "fit"):
            self.train_ds = self._open_split("train")
            self.val_ds = self._open_split("val")
            self._val_eval_ds = None
            self._configure_train_sampler()
            if self.sampling_strategy == "dynamic_epoch":
                k_tr, k_val, k_te = _split_lg_budget(
                    int(self.epoch_k), self.lg_budget_split_ratios,
                )
                self._val_eval_ds = self._make_lg_capped_eval_subset(
                    self.val_ds, k_val, rng_salt=100_003,
                )
                log.info(
                    "HDF5DataModule: dynamic_epoch LG budget (train/val/test) = %d / %d / %d "
                    "| epoch_k=%d ratios=%s",
                    k_tr,
                    k_val,
                    k_te,
                    int(self.epoch_k),
                    self.lg_budget_split_ratios,
                )
            self._log_effective_per_epoch_data()
        if stage in (None, "test"):
            self._test_eval_ds = None
            try:
                self.test_ds = self._open_split("test")
                if self.sampling_strategy == "dynamic_epoch":
                    *_, k_te = _split_lg_budget(
                        int(self.epoch_k), self.lg_budget_split_ratios,
                    )
                    self._test_eval_ds = self._make_lg_capped_eval_subset(
                        self.test_ds, k_te, rng_salt=100_019,
                    )
                    te_idx = np.asarray(self._test_eval_ds.indices, dtype=np.int64)
                    te_by = _item_counts_for_flat_indices(self.test_ds, te_idx)
                    log.info(
                        "HDF5DataModule: test (LG-capped eval) — total=%d | %s",
                        te_idx.size, _format_count_dict(te_by),
                    )
                else:
                    vb = _post_chunk_item_counts_by_base(self.test_ds)
                    log.info(
                        "HDF5DataModule: test — items per epoch (post-chunk): total=%d | %s",
                        len(self.test_ds), _format_count_dict(vb),
                    )
            except FileNotFoundError:
                self.test_ds = None

    def _effective_train_items_by_dataset(self) -> tuple[dict[str, int], int]:
        """Counts actually seen **per training epoch** (chunk items + train-time sampling)."""
        assert self.train_ds is not None
        if self.sampling_strategy == "naive":
            by = _post_chunk_item_counts_by_base(self.train_ds)
            return by, sum(by.values())

        if self.sampling_strategy == "static":
            if self._train_indices is not None:
                by = dict(_item_counts_for_flat_indices(
                    self.train_ds, self._train_indices.astype(np.int64, copy=False),
                ))
            else:
                by = _post_chunk_item_counts_by_base(self.train_ds)
            return by, sum(by.values())

        if self.sampling_strategy == "dynamic_epoch":
            assert self._train_sampler is not None
            sam = self._train_sampler
            by = dict(_item_counts_for_flat_indices(self.train_ds, sam.other_indices))
            lg = self.lg_dataset_name
            k = int(sam.epoch_k)
            by[lg] = by.get(lg, 0) + k
            by = dict(sorted(by.items()))
            return by, sum(by.values())

        if self.sampling_strategy == "proportional":
            assert self._train_indices is not None
            by = _item_counts_for_flat_indices(
                self.train_ds, self._train_indices.astype(np.int64, copy=False),
            )
            return by, sum(by.values())

        raise RuntimeError(f"Unhandled sampling_strategy: {self.sampling_strategy!r}")

    def _log_effective_per_epoch_data(self) -> None:
        assert self.train_ds is not None and self.val_ds is not None
        ws = self.trainer.world_size if self.trainer is not None else 1
        val_by = (
            _item_counts_for_flat_indices(
                self.val_ds, np.asarray(self._val_eval_ds.indices, dtype=np.int64),
            )
            if self._val_eval_ds is not None
            else _post_chunk_item_counts_by_base(self.val_ds)
        )
        val_n = (
            len(self._val_eval_ds.indices)
            if self._val_eval_ds is not None
            else len(self.val_ds)
        )
        tr_by, tr_n = self._effective_train_items_by_dataset()
        self._epoch_train_samples = tr_n

        ddp_note = ""
        if self._train_sampler is not None and ws > 1:
            ddp_note = f" | DDP ~{tr_n // ws} indices/rank (global {tr_n})"

        log.info(
            "HDF5DataModule: train per epoch — strategy=%s | total=%d | by_dataset: %s%s",
            self.sampling_strategy, tr_n, _format_count_dict(tr_by), ddp_note,
        )
        log.info(
            "HDF5DataModule: val per epoch — total=%d | by_dataset: %s",
            val_n, _format_count_dict(val_by),
        )

    # ------------------------------------------------------------------
    # Sampler configuration
    # ------------------------------------------------------------------
    def _lg_other_flat_indices(self, ds: HDF5Dataset) -> tuple[np.ndarray, np.ndarray]:
        if ds.target_patches is not None:
            parent = ds._chunk_sample_idx
            sources_per_index = ds.sources[parent]
        else:
            sources_per_index = ds.sources
        lg_mask = np.asarray([
            str(s).split("::", 1)[0] == self.lg_dataset_name
            for s in sources_per_index
        ], dtype=bool)
        lg_idx = np.where(lg_mask)[0].astype(np.int64)
        other_idx = np.where(~lg_mask)[0].astype(np.int64)
        return lg_idx, other_idx

    def _make_lg_capped_eval_subset(
        self, ds: HDF5Dataset, lg_k: int, rng_salt: int,
    ) -> Dataset:
        lg_idx, other_idx = self._lg_other_flat_indices(ds)
        if lg_k <= 0:
            chosen_lg = np.zeros((0,), dtype=np.int64)
        elif lg_k >= lg_idx.size:
            chosen_lg = lg_idx
        else:
            rng = np.random.default_rng(self.seed + int(rng_salt))
            chosen_lg = rng.choice(lg_idx, size=int(lg_k), replace=False)
        order = np.sort(np.concatenate([other_idx, chosen_lg]))
        return Subset(ds, order.astype(np.int64).tolist())

    def _base_dataset_of(self, ds: HDF5Dataset, idx: int) -> str:
        src = str(ds.sources[idx])
        return src.split("::", 1)[0]

    def _configure_train_sampler(self) -> None:
        assert self.train_ds is not None
        if self.sampling_strategy == "naive":
            return

        if self.sampling_strategy == "static":
            if not self.dataset_caps:
                # No caps provided → behave identically to naive.
                return
            ds = self.train_ds
            if ds.target_patches is not None:
                base_names = np.asarray(
                    [_base_dataset_name(s) for s in ds.sources[ds._chunk_sample_idx]],
                    dtype=object,
                )
            else:
                base_names = np.asarray(
                    [_base_dataset_name(s) for s in ds.sources],
                    dtype=object,
                )
            rng = np.random.default_rng(self.seed)
            chosen: list[np.ndarray] = []
            for dset_name, indices in _group_indices_by_base(base_names):
                cap = self.dataset_caps.get(dset_name)
                if cap is None or cap >= len(indices):
                    chosen.append(indices)
                else:
                    chosen.append(rng.choice(indices, size=int(cap), replace=False))
            self._train_indices = (
                np.sort(np.concatenate(chosen)).astype(np.int64)
                if chosen else np.zeros(0, dtype=np.int64)
            )
            return

        lg_idx, other_idx = self._lg_other_flat_indices(self.train_ds)

        if self.sampling_strategy == "dynamic_epoch":
            k_train, _, _ = _split_lg_budget(
                int(self.epoch_k), self.lg_budget_split_ratios,
            )
            self._train_sampler = EpochSubsetSampler(
                lg_indices=lg_idx,
                other_indices=other_idx,
                epoch_k=k_train,
                seed=self.seed,
                num_replicas=self.trainer.world_size if self.trainer is not None else None,
                rank=self.trainer.global_rank if self.trainer is not None else None,
            )
            return

        if self.sampling_strategy == "proportional":
            total = int(lg_idx.size + other_idx.size)
            cap = int(self.threshold_ratio * total)
            if lg_idx.size > cap:
                rng = np.random.default_rng(self.seed)
                chosen = rng.choice(lg_idx, size=cap, replace=False)
                self._train_indices = np.sort(np.concatenate([chosen, other_idx]))
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
        wi = (
            trace_dataloader_worker_init
            if self.signal_tracer is not None and self.num_workers > 0
            else None
        )
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
            worker_init_fn=wi,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        if self._train_sampler is not None:
            return self._make_loader(self.train_ds, shuffle=False, sampler=self._train_sampler)
        if self._train_indices is not None:
            return self._make_loader(
                Subset(self.train_ds, self._train_indices.tolist()),
                shuffle=True,
            )
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        ds = self._val_eval_ds if self._val_eval_ds is not None else self.val_ds
        return self._make_loader(ds, shuffle=False)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_ds is None:
            return None
        ds = self._test_eval_ds if self._test_eval_ds is not None else self.test_ds
        return self._make_loader(ds, shuffle=False)

    def _sync_trace_epoch_mp(self) -> None:
        if self.signal_tracer is not None and self.trainer is not None:
            set_signal_trace_epoch(int(self.trainer.current_epoch))

    def on_train_epoch_start(self) -> None:
        self._sync_trace_epoch_mp()
        if self._train_sampler is not None and self.trainer is not None:
            self._train_sampler.set_epoch(int(self.trainer.current_epoch))
        if self._epoch_train_samples is not None and self.trainer is not None:
            ws = self.trainer.world_size if self.trainer is not None else 1
            per_rank = self._epoch_train_samples // max(ws, 1)
            log.info(
                "Epoch %d/%d — train samples: %d per rank × %d rank(s) = %d total",
                self.trainer.current_epoch + 1,
                self.trainer.max_epochs,
                per_rank,
                ws,
                self._epoch_train_samples,
            )

    def on_validation_epoch_start(self) -> None:
        self._sync_trace_epoch_mp()
