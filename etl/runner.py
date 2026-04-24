from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

from .config import ETLConfig
from .debug import DebugQA
from .processors import BaseDatasetProcessor, RawSample
from .standardize import (
    compute_bandpass,
    compute_envelope,
    is_dead_signal,
    sanitize_signal,
    standardize_length,
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


def _split_sample_counts(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """How many samples go to each split name (``split_ratios`` must sum to 1)."""
    names = list(ratios.keys())
    allocated = 0
    counts: dict[str, int] = {}
    for i, name in enumerate(names):
        if i == len(names) - 1:
            counts[name] = n - allocated
        else:
            c = int(round(ratios[name] * n))
            counts[name] = c
            allocated += c
    return counts


def _verify_shard_divisibility(config: ETLConfig, shard_counts: dict[str, int]) -> bool:
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


def _collect_shard_sizes(output_dir: str) -> dict[str, dict]:
    """Walk the wds/ output tree and collect per-shard file sizes."""
    wds_root = Path(output_dir) / "wds"
    if not wds_root.exists():
        return {}

    result: dict[str, dict] = {}
    for split_dir in sorted(wds_root.iterdir()):
        if not split_dir.is_dir():
            continue
        shards = sorted(split_dir.glob("shard-*.tar"))
        sizes = [s.stat().st_size for s in shards]
        if not sizes:
            continue
        result[split_dir.name] = {
            "num_shards": len(sizes),
            "total_bytes": sum(sizes),
            "min_shard_bytes": min(sizes),
            "max_shard_bytes": max(sizes),
            "avg_shard_bytes": round(sum(sizes) / len(sizes)),
            "per_shard": {s.name: s.stat().st_size for s in shards},
        }
    return result


def _preprocess_signal(
    signal: np.ndarray,
    mode: str,
    sampling_frequency_hz: float | None,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    """Apply the configured preprocessing variant (raw | envelope | bandpass)."""
    if mode == "raw":
        return signal
    if mode == "envelope":
        return compute_envelope(signal)
    if mode == "bandpass":
        return compute_bandpass(
            signal, sampling_frequency_hz, low_hz, high_hz, order=order,
        )
    raise ValueError(f"Unknown preprocessing_mode: {mode!r}")


def _collect_excluded_channel_samples(
    processors: Sequence[BaseDatasetProcessor],
    all_files: list[tuple[BaseDatasetProcessor, str]],
    debug_qa: DebugQA,
    target_length: int,
    truncation_mode: str,
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
                    signal = standardize_length(
                        sample.signal, target_length, mode=truncation_mode,
                    )
                    debug_qa.add_excluded(sample, signal)
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
    split_stats: dict,
    debug_qa: DebugQA,
    shard_counts: dict[str, int],
    shard_sizes: dict[str, dict],
    divisibility_ok: bool,
    elapsed_s: float,
) -> None:
    total_samples = sum(debug_qa.count_kept.values())
    total_discarded = sum(debug_qa.count_discarded.values())

    per_dataset: dict[str, dict] = defaultdict(lambda: {"samples": 0, "discarded": 0})
    for (ds, _ch), v in debug_qa.count_kept.items():
        per_dataset[ds]["samples"] += v
    for (ds, _ch), v in debug_qa.count_discarded.items():
        per_dataset[ds]["discarded"] += v

    manifest = {
        "config": {
            "target_length": config.target_length,
            "truncation_mode": config.truncation_mode,
            "samples_per_shard": config.samples_per_shard,
            "batch_size": config.batch_size,
            "world_size": config.world_size,
            "num_workers": config.num_workers,
            "seed": config.seed,
            "output_format": config.output_format,
            "hdf5_chunk_mb": config.hdf5_chunk_mb,
            "min_signal_energy": config.min_signal_energy,
            "preprocessing_mode": config.preprocessing_mode,
            "bandpass_low_hz": config.bandpass_low_hz,
            "bandpass_high_hz": config.bandpass_high_hz,
            "max_samples_per_dataset": dict(config.max_samples_per_dataset),
            "pad_last_shard": config.pad_last_shard,
        },
        "totals": {
            "samples": total_samples,
            "samples_discarded": total_discarded,
        },
        "per_dataset": dict(per_dataset),
        "per_split": split_stats,
        "per_dataset_per_channel": debug_qa.get_stats(),
        "shard_sizes": shard_sizes,
        "shard_divisibility_ok": divisibility_ok,
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

    debug_qa = DebugQA(config, per_dataset_channel_plot_limits=per_ds_plot_limits)
    split_stats: dict[str, dict] = {}
    shard_counts: dict[str, int] = {}

    use_wds = config.output_format in ("webdataset", "both")
    use_hdf5 = config.output_format in ("hdf5", "both")

    # 2. Load ALL samples from ALL files, apply preprocessing, collect in RAM
    all_samples: list[_KeptSample] = []
    missing_fs_warned: set[str] = set()

    for proc, filepath in tqdm(all_files, desc="load", unit="file"):
        try:
            for raw_sample in proc.load_and_yield(filepath):
                # Truncation-only (no interpolation, no padding)
                signal = standardize_length(
                    raw_sample.signal,
                    config.target_length,
                    mode=config.truncation_mode,
                )
                signal = sanitize_signal(signal)

                # Preprocessing variant (Experiment D)
                if config.preprocessing_mode != "raw":
                    if (
                        config.preprocessing_mode == "bandpass"
                        and raw_sample.sampling_frequency_hz is None
                        and raw_sample.source_dataset not in missing_fs_warned
                    ):
                        log.warning(
                            "Dataset '%s': bandpass requested but "
                            "sampling_frequency_hz missing — passing through raw",
                            raw_sample.source_dataset,
                        )
                        missing_fs_warned.add(raw_sample.source_dataset)
                    signal = _preprocess_signal(
                        signal,
                        config.preprocessing_mode,
                        raw_sample.sampling_frequency_hz,
                        config.bandpass_low_hz,
                        config.bandpass_high_hz,
                        config.bandpass_order,
                    )
                    signal = sanitize_signal(signal)

                if is_dead_signal(signal, config.min_signal_energy):
                    debug_qa.add_discarded(raw_sample, signal)
                    continue
                if not validate_sample(signal, config.target_length):
                    debug_qa.add_discarded(raw_sample, signal)
                    continue

                debug_qa.add_kept(raw_sample, signal)
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

    n_total = len(all_samples)
    total_bytes = sum(s.signal.nbytes for s in all_samples)
    log.info(
        "Loaded %d samples (%.2f GB in memory)",
        n_total, total_bytes / 1e9,
    )
    if n_total == 0:
        log.error("No samples kept — aborting.")
        return

    # 4. Split at *sample* level
    split_counts = _split_sample_counts(n_total, config.split_ratios)
    log.info("Sample-level split counts: %s", split_counts)

    perm = rng.permutation(n_total)
    split_names = list(config.split_ratios.keys())
    pos = 0
    split_indices: dict[str, np.ndarray] = {}
    for name in split_names:
        k = split_counts[name]
        split_indices[name] = perm[pos : pos + k]
        pos += k

    # 5. Write each split
    for split_name in split_names:
        idxs = split_indices[split_name].copy()
        rng.shuffle(idxs)
        n_split = len(idxs)
        log.info("--- Writing split: %s (%d samples) ---", split_name, n_split)

        wds_writer = (
            WebDatasetWriter(config.output_dir, split_name, config.samples_per_shard)
            if use_wds
            else None
        )
        hdf5_writer = (
            HDF5Writer(
                config.output_dir, split_name,
                chunk_mb=config.hdf5_chunk_mb,
            )
            if use_hdf5
            else None
        )

        for pos, sample_i in enumerate(tqdm(idxs, desc=f"{split_name} [write]", unit="sample")):
            s = all_samples[int(sample_i)]

            shard_idx = pos // config.samples_per_shard
            debug_qa.add_shard_sample(
                split_name, shard_idx,
                s.sample_id, s.source_dataset, s.signal,
                config.samples_per_shard,
            )

            metadata = {
                **s.metadata,
                "sampling_frequency_hz": s.sampling_frequency_hz,
                "dataset_source": s.source_dataset,
                "base_dataset": s.base_dataset,
                "channel_idx": int(s.channel_idx),
                "length": int(s.signal.shape[0]),
                "is_filler": False,
            }

            if wds_writer is not None:
                wds_writer.write(s.sample_id, s.signal, metadata)
            if hdf5_writer is not None:
                hdf5_writer.write(
                    s.signal, s.sampling_frequency_hz, s.source_dataset,
                )

        # 5b. Auto-fill last WebDataset shard (if partial)
        if wds_writer is not None and config.pad_last_shard and n_split > 0:
            deficit = (-wds_writer.count) % config.samples_per_shard
            if deficit > 0:
                # Pool the last few samples of this split for filler duplication.
                pool_size = min(max(deficit * 2, 32), n_split)
                pool_ids = idxs[-pool_size:]
                filler_ids = rng.choice(
                    pool_ids, size=deficit, replace=(pool_size < deficit),
                )
                log.info(
                    "Split '%s': padding last shard with %d filler samples",
                    split_name, int(deficit),
                )
                for j, sample_i in enumerate(filler_ids):
                    s = all_samples[int(sample_i)]
                    filler_metadata = {
                        **s.metadata,
                        "sampling_frequency_hz": s.sampling_frequency_hz,
                        "dataset_source": s.source_dataset,
                        "base_dataset": s.base_dataset,
                        "channel_idx": int(s.channel_idx),
                        "length": int(s.signal.shape[0]),
                        "is_filler": True,
                    }
                    wds_writer.write(
                        f"{s.sample_id}__filler{j}",
                        s.signal,
                        filler_metadata,
                    )

        if wds_writer is not None:
            wds_writer.close()
            n_shards = wds_writer.count // config.samples_per_shard
            if wds_writer.count % config.samples_per_shard:
                n_shards += 1
            shard_counts[split_name] = n_shards
        if hdf5_writer is not None:
            hdf5_writer.close()

        split_stats[split_name] = {
            "samples": n_split,
            "num_shards": shard_counts.get(split_name, 0),
            "samples_per_shard": config.samples_per_shard,
        }
        log.info("Split '%s' done — samples=%d", split_name, n_split)

    del all_samples

    # 6. Debug: collect samples from excluded channels
    log.info("Collecting excluded channel samples for debug plots ...")
    _collect_excluded_channel_samples(
        processors, all_files, debug_qa,
        config.target_length, config.truncation_mode,
    )

    # 7. Generate QA reports
    log.info("Generating QA debug reports ...")
    debug_qa.generate_reports()

    # 8. Post-processing
    shard_sizes = _collect_shard_sizes(config.output_dir) if use_wds else {}
    div_ok = _verify_shard_divisibility(config, shard_counts) if use_wds else True

    elapsed = time.time() - t0
    _write_manifest(config, split_stats, debug_qa, shard_counts, shard_sizes, div_ok, elapsed)
    log.info("ETL pipeline finished in %.1f s", elapsed)
