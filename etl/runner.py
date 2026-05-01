from __future__ import annotations

import json
import logging
import tarfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

from .config import ETLConfig, FORMAT_OUTPUT_SUBDIR
from .debug import FormatOutputDebugQA, LoadStageDebug
from .processors import BaseDatasetProcessor
from .standardize import (
    bandpass_edges_from_center_frequency,
    compute_bandpass,
    compute_envelope,
    compute_interpolation,
    is_dead_signal,
    sanitize_signal,
    validate_sample,
)
from .writers import HDF5Writer, WebDatasetWriter

log = logging.getLogger(__name__)


@dataclass
class _KeptSample:
    """Lightweight container for one output sample."""
    sample_id: str
    source_dataset: str
    channel_idx: int
    signal: np.ndarray
    sampling_frequency_hz: float | None
    metadata: dict = field(default_factory=dict)

    @property
    def base_dataset(self) -> str:
        """Strip the optional ``::sub`` suffix used for per-session debug splitting."""
        return self.source_dataset.split("::", 1)[0]


@dataclass
class _FormatWriters:
    """HDF5 / WebDataset writers for one logical output format."""
    fmt: str
    output_root: str
    wds: WebDatasetWriter | None
    hdf5: HDF5Writer | None


def _split_sample_counts(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """How many samples go to each split name (``split_ratios`` must sum to 1).

    Uses a **largest remainder** method so the counts are non-negative, sum
    to *n* exactly, and follow the target proportions as closely as possible
    (unlike naive per-split rounding, which can overshoot/undershoot for small
    *n*).
    """
    names = list(ratios.keys())
    if not names:
        return {}
    rsum = float(sum(ratios.values()))
    if abs(rsum - 1.0) > 1e-6:
        # Fall back: normalise in-place copy for robustness in debug runs.
        norm = {k: float(ratios[k]) / rsum for k in names}
    else:
        norm = {k: float(ratios[k]) for k in names}

    raw = {k: norm[k] * n for k in names}
    base = {k: int(np.floor(raw[k])) for k in names}
    rem = n - int(sum(base.values()))
    if rem < 0:
        # Should not happen; clamp defensively.
        for k in sorted(names, key=lambda x: base[x], reverse=True):
            if rem == 0:
                break
            if base[k] > 0:
                base[k] -= 1
                rem += 1
    elif rem > 0:
        order = sorted(
            names, key=lambda k: (raw[k] - base[k], k), reverse=True,
        )
        for k in order:
            if rem == 0:
                break
            base[k] += 1
            rem -= 1
    return base


def _stratified_split_indices(
    all_samples: list[_KeptSample],
    split_ratios: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Stratified split by :attr:`_KeptSample.base_dataset`.

    For every base dataset separately, apply ``split_ratios`` (via
    :func:`_split_sample_counts`) on that group's samples, shuffle within the
    group, then concatenate the per-group slices for each split name.

    This ensures (up to per-group rounding) that each source dataset
    contributes the same train/val/test fractions.
    """
    split_names = list(split_ratios.keys())
    by_base: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(all_samples):
        by_base[s.base_dataset].append(i)

    per_split: dict[str, list[np.ndarray]] = {name: [] for name in split_names}

    for base, idxs in sorted(by_base.items(), key=lambda kv: kv[0]):
        n = len(idxs)
        if n == 0:
            continue
        perm = rng.permutation(np.asarray(idxs, dtype=np.int64))
        counts = _split_sample_counts(n, split_ratios)
        pos = 0
        for name in split_names:
            k = counts[name]
            per_split[name].append(perm[pos : pos + k])
            pos += k
        log.info(
            "Stratified split for base dataset '%s' (n=%d): %s",
            base, n, counts,
        )

    out: dict[str, np.ndarray] = {}
    for name in split_names:
        parts = per_split[name]
        if not parts:
            out[name] = np.asarray([], dtype=np.int64)
        else:
            out[name] = np.concatenate(parts, axis=0)
    return out


def _verify_shard_divisibility(
    config: ETLConfig, shard_counts: dict[str, int],
) -> bool:
    total_workers = config.total_workers
    ok = True
    for split, n_shards in shard_counts.items():
        if n_shards == 0:
            continue
        if n_shards % total_workers != 0:
            log.warning(
                "Split '%s': %d shards is NOT divisible by total_workers=%d. "
                "Consider adjusting samples_per_shard.",
                split,
                n_shards,
                total_workers,
            )
            ok = False
        else:
            log.info(
                "Split '%s': %d shards — divisible by %d (ok)",
                split,
                n_shards,
                total_workers,
            )
    return ok


def _count_samples_in_wds_shard_tar(tar_path: Path) -> int:
    """Number of WebDataset samples in one shard tar (``.signal.npy`` members)."""
    with tarfile.open(tar_path, "r:*") as tf:
        return sum(
            1
            for m in tf.getmembers()
            if m.isfile() and Path(m.name).name.endswith(".signal.npy")
        )


def _collect_shard_sample_counts(format_output_dir: str) -> dict[str, dict]:
    """Walk ``<format_output_dir>/wds/`` and collect per-shard acquisition counts.

    Includes filler samples when ``pad_last_shard`` padded the final shard.
    """
    wds_root = Path(format_output_dir) / "wds"
    if not wds_root.exists():
        return {}

    result: dict[str, dict] = {}
    for split_dir in sorted(wds_root.iterdir()):
        if not split_dir.is_dir():
            continue
        shards = sorted(split_dir.glob("shard-*.tar"))
        per_shard = {s.name: _count_samples_in_wds_shard_tar(s) for s in shards}
        counts = list(per_shard.values())
        if not counts:
            continue
        result[split_dir.name] = {
            "num_shards": len(counts),
            "total_samples_on_disk": sum(counts),
            "min_samples_per_shard": min(counts),
            "max_samples_per_shard": max(counts),
            "avg_samples_per_shard": round(sum(counts) / len(counts)),
            "per_shard": per_shard,
        }
    return result


def _collect_all_format_shard_sample_counts(
    config: ETLConfig,
) -> dict[str, dict[str, dict]]:
    """Per enabled format, per-shard sample counts under that format's output root."""
    if config.output_format not in ("webdataset", "both"):
        return {}
    out: dict[str, dict[str, dict]] = {}
    for fmt in config.enabled_format_keys():
        root = str(Path(config.output_dir) / FORMAT_OUTPUT_SUBDIR[fmt])
        counts = _collect_shard_sample_counts(root)
        if counts:
            out[fmt] = counts
    return out


def _build_interpolation_truncate_by_dataset(
    processors: Sequence[BaseDatasetProcessor],
    default: str,
) -> dict[str, str]:
    """Per YAML ``name`` (``dataset_name()``): effective mode for interpolate path."""
    out: dict[str, str] = {}
    for p in processors:
        v = p.config.extra.get("interpolation_truncate_mode")
        out[p.dataset_name()] = str(v).lower() if v is not None else default
    return out


def _build_transmit_center_hz_by_dataset(
    processors: Sequence[BaseDatasetProcessor],
) -> dict[str, float]:
    """Per YAML ``name``: design / central RF frequency (Hz) when declared."""
    out: dict[str, float] = {}
    for p in processors:
        ex = p.config.extra or {}
        v = ex.get("transmit_center_frequency_hz")
        if v is None:
            v = ex.get("tx_fc_hz")
        if v is not None:
            out[p.dataset_name()] = float(v)
    return out


def _build_rf_bandwidth_fraction_by_dataset(
    processors: Sequence[BaseDatasetProcessor],
    default: float,
) -> dict[str, float]:
    """Per dataset: ``extra.rf_bandwidth_fraction`` or global default."""
    out: dict[str, float] = {}
    d = float(default)
    for p in processors:
        ex = p.config.extra or {}
        v = ex.get("rf_bandwidth_fraction")
        out[p.dataset_name()] = float(v) if v is not None else d
    return out


def _warn_missing_fs_for_rf_filter(s: _KeptSample, missing_fs_warned: set[str]) -> None:
    if s.sampling_frequency_hz is None and s.source_dataset not in missing_fs_warned:
        log.warning(
            "Dataset '%s': bandpass/envelope output requested but "
            "sampling_frequency_hz missing — filter is a no-op (raw passed through)",
            s.source_dataset,
        )
        missing_fs_warned.add(s.source_dataset)


def _compute_enabled_format_signals(
    s: _KeptSample,
    config: ETLConfig,
    enabled_formats: frozenset[str],
    missing_fs_warned: set[str],
    missing_tx_fc_warned: set[str],
    interpolation_truncate_mode: str,
    tx_fc_by_dataset: dict[str, float],
    rf_bw_by_dataset: dict[str, float],
) -> dict[str, np.ndarray]:
    """Bandpass / envelope from per-dataset ``tx_fc`` + ``rf_bandwidth_fraction`` + ``fs``."""
    out: dict[str, np.ndarray] = {}
    raw = s.signal
    if "raw" in enabled_formats:
        out["raw"] = raw
    need_bp = "bandpass" in enabled_formats
    need_env = "envelope" in enabled_formats

    if need_bp or need_env:
        _warn_missing_fs_for_rf_filter(s, missing_fs_warned)
        sig_f: np.ndarray | None = None
        fs = s.sampling_frequency_hz
        if fs is not None and fs > 0:
            fc = tx_fc_by_dataset.get(s.base_dataset)
            if fc is None:
                if s.base_dataset not in missing_tx_fc_warned:
                    log.warning(
                        "Dataset '%s': transmit_center_frequency_hz missing in "
                        "processor extra — bandpass/envelope use unfiltered signal",
                        s.base_dataset,
                    )
                    missing_tx_fc_warned.add(s.base_dataset)
            else:
                bw = rf_bw_by_dataset.get(
                    s.base_dataset, config.rf_bandwidth_fraction,
                )
                lo, hi = bandpass_edges_from_center_frequency(fc, bw, fs)
                sig_f = compute_bandpass(
                    raw, fs, lo, hi, order=config.bandpass_order,
                )
        if sig_f is not None:
            if need_bp:
                out["bandpass"] = sanitize_signal(sig_f)
            if need_env:
                out["envelope"] = sanitize_signal(compute_envelope(sig_f))
        else:
            sig_u = np.asarray(raw, dtype=np.float32)
            if need_bp:
                out["bandpass"] = sanitize_signal(sig_u)
            if need_env:
                out["envelope"] = sanitize_signal(compute_envelope(sig_u))

    if "interpolate" in enabled_formats:
        assert config.target_length is not None
        out["interpolate"] = sanitize_signal(
            compute_interpolation(
                raw,
                int(config.target_length),
                truncate_mode=interpolation_truncate_mode,
            ),
        )
    return out


def _build_format_writers(
    config: ETLConfig,
    split_name: str,
    use_wds: bool,
    use_hdf5: bool,
) -> list[_FormatWriters]:
    packs: list[_FormatWriters] = []
    for fmt in config.enabled_format_keys():
        root = str(Path(config.output_dir) / FORMAT_OUTPUT_SUBDIR[fmt])
        wds_w = (
            WebDatasetWriter(root, split_name, config.samples_per_shard)
            if use_wds
            else None
        )
        h5_w = (
            HDF5Writer(
                root, split_name,
                chunk_mb=config.hdf5_chunk_mb,
            )
            if use_hdf5
            else None
        )
        packs.append(_FormatWriters(fmt=fmt, output_root=root, wds=wds_w, hdf5=h5_w))
    return packs


def _close_format_writers_and_shards(
    packs: list[_FormatWriters],
    config: ETLConfig,
    split_name: str,
    shard_counts: dict[str, dict[str, int]],
) -> None:
    for pack in packs:
        if pack.wds is not None:
            pack.wds.close()
            n_shards = pack.wds.count // config.samples_per_shard
            if pack.wds.count % config.samples_per_shard:
                n_shards += 1
            shard_counts.setdefault(pack.fmt, {})[split_name] = n_shards
        if pack.hdf5 is not None:
            pack.hdf5.close()


def _collect_excluded_channel_samples(
    processors: Sequence[BaseDatasetProcessor],
    all_files: list[tuple[BaseDatasetProcessor, str]],
    load_debug: LoadStageDebug,
    max_files_per_dataset: int = 3,
) -> None:
    """Load a few files per dataset and yield signals from excluded channels."""
    datasets_with_exclusions = {}
    for proc in processors:
        excl = proc.config.channels_to_exclude
        keep = proc.config.channels_to_keep
        if excl or keep is not None:
            datasets_with_exclusions[proc.dataset_name()] = proc

    if not datasets_with_exclusions:
        return

    files_per_ds: dict[str, list[str]] = defaultdict(list)
    for proc, fpath in all_files:
        ds = proc.dataset_name()
        if ds in datasets_with_exclusions and len(files_per_ds[ds]) < max_files_per_dataset:
            files_per_ds[ds].append(fpath)

    for ds_name, proc in datasets_with_exclusions.items():
        for fpath in files_per_ds.get(ds_name, []):
            try:
                for sample in proc.load_all_channels(fpath):
                    if proc.should_keep_channel(sample.channel_idx):
                        continue
                    signal = sanitize_signal(sample.signal)
                    load_debug.add_excluded(sample, signal)
            except Exception:
                log.exception(
                    "Error loading excluded channels from %s — skipping", fpath
                )


def _apply_static_subsampling(
    all_samples: list[_KeptSample],
    caps: dict[str, int],
    rng: np.random.Generator,
) -> list[_KeptSample]:
    """Cap the number of samples per base dataset (Experiment B2).

    Samples for capped datasets are first shuffled, then truncated. Other
    datasets pass through unchanged.
    """
    if not caps:
        return all_samples

    by_ds: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(all_samples):
        by_ds[s.base_dataset].append(i)

    keep_idx: list[int] = []
    for ds, idxs in by_ds.items():
        cap = caps.get(ds)
        if cap is None or len(idxs) <= cap:
            keep_idx.extend(idxs)
        else:
            chosen = rng.choice(np.asarray(idxs), size=cap, replace=False)
            keep_idx.extend(int(x) for x in chosen)
            log.info(
                "Static subsampling: dataset '%s' capped at %d (out of %d)",
                ds, cap, len(idxs),
            )

    keep_idx.sort()
    return [all_samples[i] for i in keep_idx]


def _write_manifest(
    config: ETLConfig,
    split_stats: dict[str, dict[str, dict]],
    load_debug: LoadStageDebug,
    shard_counts: dict[str, dict[str, int]],
    shard_samples: dict[str, dict[str, dict]],
    shard_divisibility: dict[str, bool],
    elapsed_s: float,
    interpolation_truncate_by_dataset: dict[str, str],
    transmit_center_frequency_hz_by_dataset: dict[str, float],
    rf_bandwidth_fraction_by_dataset: dict[str, float],
    *,
    samples_after_static_cap: int,
    per_source_dataset_after_cap: dict[str, int],
    per_base_dataset_after_cap: dict[str, int],
) -> None:
    """Manifest aggregates **two stages**: load/filter (load_debug) vs final pool after caps."""

    kept_before_cap = sum(load_debug.count_kept.values())
    total_discarded = sum(load_debug.count_discarded.values())

    per_dataset_before_cap: dict[str, dict[str, int]] = defaultdict(
        lambda: {"samples_kept": 0, "samples_discarded": 0},
    )
    for (ds, _ch), v in load_debug.count_kept.items():
        per_dataset_before_cap[ds]["samples_kept"] += v
    for (ds, _ch), v in load_debug.count_discarded.items():
        per_dataset_before_cap[ds]["samples_discarded"] += v

    manifest = {
        "config": {
            "samples_per_shard": config.samples_per_shard,
            "batch_size": config.batch_size,
            "world_size": config.world_size,
            "num_workers": config.num_workers,
            "seed": config.seed,
            "output_format": config.output_format,
            "hdf5_chunk_mb": config.hdf5_chunk_mb,
            "min_signal_energy": config.min_signal_energy,
            "output_formats": dict(config.normalized_output_formats()),
            "target_length": config.target_length,
            "interpolation_truncate_mode": config.interpolation_truncate_mode,
            "interpolation_truncate_by_dataset": dict(
                sorted(interpolation_truncate_by_dataset.items()),
            ),
            "bandpass_order": config.bandpass_order,
            "rf_bandwidth_fraction": config.rf_bandwidth_fraction,
            "transmit_center_frequency_hz_by_dataset": dict(
                sorted(transmit_center_frequency_hz_by_dataset.items()),
            ),
            "rf_bandwidth_fraction_by_dataset": dict(
                sorted(rf_bandwidth_fraction_by_dataset.items()),
            ),
            "max_samples_per_dataset": dict(config.max_samples_per_dataset),
            "pad_last_shard": config.pad_last_shard,
            "debug_output_dir": config.debug_output_dir,
        },
        "totals": {
            "samples_after_static_cap": samples_after_static_cap,
            "samples_kept_before_static_cap": kept_before_cap,
            "samples_discarded_at_load": total_discarded,
        },
        "per_source_dataset_after_static_cap": dict(
            sorted(per_source_dataset_after_cap.items()),
        ),
        "per_base_dataset_after_static_cap": dict(
            sorted(per_base_dataset_after_cap.items()),
        ),
        "per_dataset_before_static_cap": dict(
            sorted(per_dataset_before_cap.items()),
        ),
        "per_format_split": split_stats,
        "per_dataset_per_channel_load_stage": load_debug.get_stats(),
        "shard_counts": shard_counts,
        "shard_samples": shard_samples,
        "shard_divisibility_ok": shard_divisibility,
        "elapsed_seconds": round(elapsed_s, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path(config.output_dir) / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log.info("Manifest written to %s", out_path)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def run_etl(config: ETLConfig, processors: Sequence[BaseDatasetProcessor]) -> None:
    """Execute the full ETL pipeline.

    Parameters
    ----------
    config : global ETL configuration.
    processors : already-instantiated processor objects (one per dataset).
    """
    config.validate()
    t0 = time.time()

    rng = np.random.default_rng(config.seed)

    tx_fc_by_dataset = _build_transmit_center_hz_by_dataset(processors)
    rf_bw_by_dataset = _build_rf_bandwidth_fraction_by_dataset(
        processors, config.rf_bandwidth_fraction,
    )

    # 1. Discover files from ALL processors
    all_files: list[tuple[BaseDatasetProcessor, str]] = []
    for proc in processors:
        files = proc.discover_files()
        log.info("Dataset '%s': found %d files", proc.dataset_name(), len(files))
        for f in files:
            all_files.append((proc, f))

    if not all_files:
        log.error("No files found across any dataset — nothing to do.")
        return

    per_ds_plot_limits: dict[str, int] = {}
    for p in processors:
        v = p.config.extra.get("num_ch_debug_plot_limit")
        if v is not None:
            per_ds_plot_limits[p.dataset_name()] = int(v)

    load_debug = LoadStageDebug(config, per_dataset_channel_plot_limits=per_ds_plot_limits)
    format_debug = {
        fmt: FormatOutputDebugQA(config, fmt, per_dataset_channel_plot_limits=per_ds_plot_limits)
        for fmt in config.enabled_format_keys()
    }
    split_stats: dict[str, dict[str, dict]] = defaultdict(dict)
    shard_counts: dict[str, dict[str, int]] = defaultdict(dict)

    use_wds = config.output_format in ("webdataset", "both")
    use_hdf5 = config.output_format in ("hdf5", "both")
    interp_truncate_by_dataset = _build_interpolation_truncate_by_dataset(
        processors,
        config.interpolation_truncate_mode,
    )

    # 2. Load ALL samples: raw native-length only (preprocessing at write time).
    all_samples: list[_KeptSample] = []

    for proc, filepath in tqdm(all_files, desc="load", unit="file"):
        try:
            for raw_sample in proc.load_and_yield(filepath):
                signal = sanitize_signal(raw_sample.signal)

                if is_dead_signal(signal, config.min_signal_energy):
                    load_debug.add_discarded(raw_sample, signal)
                    continue
                if not validate_sample(signal):
                    load_debug.add_discarded(raw_sample, signal)
                    continue

                load_debug.add_kept(raw_sample)
                all_samples.append(_KeptSample(
                    sample_id=raw_sample.sample_id,
                    source_dataset=raw_sample.source_dataset,
                    channel_idx=raw_sample.channel_idx,
                    signal=signal,
                    sampling_frequency_hz=raw_sample.sampling_frequency_hz,
                    metadata=raw_sample.metadata,
                ))
        except Exception:
            log.exception("Error processing file %s — skipping", filepath)

    # 3. Static subsampling (Exp B2): cap the number of samples per dataset
    all_samples = _apply_static_subsampling(
        all_samples, dict(config.max_samples_per_dataset), rng,
    )
    cap_by_base = Counter(s.base_dataset for s in all_samples)
    log.info(
        "After static subsampling: %d samples; per base_dataset: %s",
        len(all_samples),
        dict(sorted(cap_by_base.items())),
    )

    n_total = len(all_samples)
    total_bytes = sum(s.signal.nbytes for s in all_samples)
    log.info(
        "Loaded %d samples (%.2f GB in memory)",
        n_total, total_bytes / 1e9,
    )
    if n_total == 0:
        log.error("No samples kept — aborting.")
        return

    # 4. Stratified split by base dataset (same ratios applied *within* each source)
    split_names = list(config.split_ratios.keys())
    split_indices = _stratified_split_indices(
        all_samples, config.split_ratios, rng,
    )
    split_counts = {name: int(split_indices[name].size) for name in split_names}
    log.info("Global split counts (sum over datasets): %s", split_counts)

    missing_fs_warned: set[str] = set()
    missing_tx_fc_warned: set[str] = set()
    enabled_formats_frozen = frozenset(config.enabled_format_keys())

    # 5. Write each split (each enabled format → dedicated writers under subdirs)
    for split_name in split_names:
        idxs = split_indices[split_name].copy()
        rng.shuffle(idxs)
        n_split = len(idxs)
        log.info("--- Writing split: %s (%d samples) ---", split_name, n_split)

        packs = _build_format_writers(config, split_name, use_wds, use_hdf5)

        for pos, sample_i in enumerate(tqdm(idxs, desc=f"{split_name} [write]", unit="sample")):
            s = all_samples[int(sample_i)]
            itr_mode = interp_truncate_by_dataset.get(
                s.base_dataset,
                config.interpolation_truncate_mode,
            )

            outs = _compute_enabled_format_signals(
                s,
                config,
                enabled_formats_frozen,
                missing_fs_warned,
                missing_tx_fc_warned,
                itr_mode,
                tx_fc_by_dataset,
                rf_bw_by_dataset,
            )
            for pack in packs:
                signal_out = outs[pack.fmt]
                format_debug[pack.fmt].add_written_sample(
                    s.sample_id, s.source_dataset, s.channel_idx, signal_out,
                )
                metadata = {
                    **s.metadata,
                    "sampling_frequency_hz": s.sampling_frequency_hz,
                    "dataset_source": s.source_dataset,
                    "base_dataset": s.base_dataset,
                    "channel_idx": int(s.channel_idx),
                    "length": int(signal_out.shape[0]),
                    "is_filler": False,
                }
                if pack.wds is not None:
                    pack.wds.write(s.sample_id, signal_out, metadata)
                if pack.hdf5 is not None:
                    pack.hdf5.write(
                        signal_out, s.sampling_frequency_hz, s.source_dataset,
                    )

        # 5b. Auto-fill last WebDataset shard (if partial) — per format
        if config.pad_last_shard and n_split > 0:
            for pack in packs:
                if pack.wds is None:
                    continue
                deficit = (-pack.wds.count) % config.samples_per_shard
                if deficit <= 0:
                    continue
                pool_size = min(max(deficit * 2, 32), n_split)
                pool_ids = idxs[-pool_size:]
                filler_ids = rng.choice(
                    pool_ids, size=deficit, replace=(pool_size < deficit),
                )
                log.info(
                    "Split '%s' format '%s': padding last shard with %d filler samples",
                    split_name, pack.fmt, int(deficit),
                )
                for j, sample_i in enumerate(filler_ids):
                    s = all_samples[int(sample_i)]
                    itr_mode = interp_truncate_by_dataset.get(
                        s.base_dataset,
                        config.interpolation_truncate_mode,
                    )
                    outs_f = _compute_enabled_format_signals(
                        s,
                        config,
                        enabled_formats_frozen,
                        missing_fs_warned,
                        missing_tx_fc_warned,
                        itr_mode,
                        tx_fc_by_dataset,
                        rf_bw_by_dataset,
                    )
                    signal_out = outs_f[pack.fmt]
                    filler_metadata = {
                        **s.metadata,
                        "sampling_frequency_hz": s.sampling_frequency_hz,
                        "dataset_source": s.source_dataset,
                        "base_dataset": s.base_dataset,
                        "channel_idx": int(s.channel_idx),
                        "length": int(signal_out.shape[0]),
                        "is_filler": True,
                    }
                    pack.wds.write(
                        f"{s.sample_id}__filler{j}",
                        signal_out,
                        filler_metadata,
                    )

        _close_format_writers_and_shards(packs, config, split_name, shard_counts)

        for pack in packs:
            fd = format_debug[pack.fmt]
            if use_wds:
                fd.write_shard_mixing_wds(split_name, pack.output_root)
            if use_hdf5:
                fd.write_shard_mixing_hdf5(split_name, pack.output_root)

        for pack in packs:
            if split_name not in shard_counts.get(pack.fmt, {}):
                n_sh = 0
            else:
                n_sh = shard_counts[pack.fmt][split_name]
            split_stats[pack.fmt][split_name] = {
                "samples": n_split,
                "num_shards": n_sh,
                "samples_per_shard": config.samples_per_shard,
            }
        log.info("Split '%s' done — samples=%d", split_name, n_split)

    cnt_after_cap_src = Counter(s.source_dataset for s in all_samples)
    cnt_after_cap_base = Counter(s.base_dataset for s in all_samples)
    n_after_static_cap = len(all_samples)

    del all_samples

    # 6. Debug: collect samples from excluded channels
    log.info("Collecting excluded channel samples for debug plots ...")
    _collect_excluded_channel_samples(
        processors, all_files, load_debug,
    )

    # 7. Generate QA reports (per-format written grids + load-stage discarded/excluded)
    log.info("Generating QA debug reports ...")
    for _fmt, fd in format_debug.items():
        fd.generate_written_reports()
    load_debug.generate_reports()

    # 8. Post-processing
    shard_samples = _collect_all_format_shard_sample_counts(config) if use_wds else {}
    shard_divisibility: dict[str, bool] = {}
    if use_wds:
        for fmt in config.enabled_format_keys():
            sc = dict(shard_counts.get(fmt, {}))
            if sc:
                shard_divisibility[fmt] = _verify_shard_divisibility(config, sc)
            else:
                shard_divisibility[fmt] = True
    elapsed = time.time() - t0
    _write_manifest(
        config,
        {k: dict(v) for k, v in split_stats.items()},
        load_debug,
        {k: dict(v) for k, v in shard_counts.items()},
        shard_samples,
        shard_divisibility,
        elapsed,
        interp_truncate_by_dataset,
        tx_fc_by_dataset,
        rf_bw_by_dataset,
        samples_after_static_cap=n_after_static_cap,
        per_source_dataset_after_cap=dict(cnt_after_cap_src),
        per_base_dataset_after_cap=dict(cnt_after_cap_base),
    )
    log.info("ETL pipeline finished in %.1f s", elapsed)
