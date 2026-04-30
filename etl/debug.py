from __future__ import annotations

import io
import json
import logging
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for HPC / SSH
import matplotlib.pyplot as plt
import numpy as np

from .config import ETLConfig, FORMAT_OUTPUT_SUBDIR

if TYPE_CHECKING:
    from .processors import RawSample

log = logging.getLogger(__name__)

_MAX_SHARDS_FOR_QA = 10
_MAX_TRACES_PER_SHARD = 10


def _read_wds_samples_from_tar(tar_path: Path, max_samples: int) -> list[tuple[str, str, np.ndarray]]:
    """Decode up to *max_samples* from one WebDataset shard ``.tar``.

    Returns list of ``(sample_id, dataset_source, signal)`` read from disk.
    """
    out: list[tuple[str, str, np.ndarray]] = []
    with tarfile.open(tar_path, "r") as tar:
        names = [m.name for m in tar.getmembers() if m.isfile()]
        by_key: dict[str, dict[str, str]] = {}
        for n in names:
            if n.endswith(".signal.npy"):
                key = n[: -len(".signal.npy")]
                by_key.setdefault(key, {})["signal"] = n
            elif n.endswith(".metadata.json"):
                key = n[: -len(".metadata.json")]
                by_key.setdefault(key, {})["meta"] = n

        for key in sorted(by_key.keys()):
            if len(out) >= max_samples:
                break
            ent = by_key[key]
            sig_name = ent.get("signal")
            meta_name = ent.get("meta")
            if not sig_name:
                continue
            sig_m = tar.getmember(sig_name)
            sig_f = tar.extractfile(sig_m)
            if sig_f is None:
                continue
            signal = np.load(io.BytesIO(sig_f.read()), allow_pickle=False)
            signal = np.asarray(signal, dtype=np.float32).reshape(-1)
            ds_name = ""
            if meta_name:
                meta_m = tar.getmember(meta_name)
                meta_f = tar.extractfile(meta_m)
                if meta_f is not None:
                    try:
                        meta = json.loads(meta_f.read().decode("utf-8"))
                        ds_name = str(meta.get("dataset_source", ""))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
            out.append((key, ds_name, signal))
    return out


def _read_hdf5_shard_traces(
    h5_path: Path,
    samples_per_shard: int,
    shard_idx: int,
    max_traces: int,
) -> list[tuple[str, str, np.ndarray]]:
    """Return up to *max_traces* samples from HDF5 rows belonging to virtual shard *shard_idx*."""
    if not h5_path.is_file():
        return []
    out: list[tuple[str, str, np.ndarray]] = []
    with h5py.File(str(h5_path), "r") as f:
        offsets = np.asarray(f["offsets"][:], dtype=np.int64)
        data = f["data"]
        sources_raw = f["dataset_sources"][:]
        n = int(offsets.shape[0] - 1)
        lo = shard_idx * samples_per_shard
        hi = min(n, lo + samples_per_shard)
        if lo >= n or hi <= lo:
            return []
        end_i = min(lo + max_traces, hi)
        for i in range(lo, end_i):
            a, b = int(offsets[i]), int(offsets[i + 1])
            sig = np.asarray(data[a:b], dtype=np.float32)
            src = sources_raw[i]
            if isinstance(src, bytes):
                ds_name = src.decode("utf-8", errors="replace")
            else:
                ds_name = str(src)
            out.append((f"idx{i}", ds_name, sig))
    return out


def _save_shard_mixing_figure(
    samples: list[tuple[str, str, np.ndarray]],
    out_path: Path,
    suptitle: str,
) -> None:
    if not samples:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_colors: dict[str, str] = {}
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.8 * n), squeeze=False)
    for i, (sid, ds_name, sig) in enumerate(samples):
        if ds_name not in ds_colors:
            ds_colors[ds_name] = palette[len(ds_colors) % len(palette)]
        ax = axes[i][0]
        ax.plot(sig, linewidth=0.6, color=ds_colors[ds_name])
        ax.set_title(f"{sid}  [{ds_name}]", fontsize=7)
        ax.tick_params(labelsize=5)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


class LoadStageDebug:
    """Load-stage counters and optional plots (discarded / excluded only; no kept grids)."""

    def __init__(
        self,
        config: ETLConfig,
        per_dataset_channel_plot_limits: dict[str, int] | None = None,
    ) -> None:
        self.config = config
        self.max_per_class = config.debug_samples_per_class
        self._per_dataset_channel_plot_limits = per_dataset_channel_plot_limits or {}
        self._root = Path(config.debug_output_dir) / "_load"

        self.count_kept: dict[tuple[str, int], int] = defaultdict(int)
        self.count_discarded: dict[tuple[str, int], int] = defaultdict(int)
        self._discarded: dict[tuple[str, int], list] = defaultdict(list)
        self._excluded: dict[tuple[str, int], list] = defaultdict(list)

    def add_kept(self, sample: RawSample) -> None:
        key = (sample.source_dataset, sample.channel_idx)
        self.count_kept[key] += 1

    def add_excluded(self, sample: RawSample, signal: np.ndarray) -> None:
        key = (sample.source_dataset, sample.channel_idx)
        buf = self._excluded[key]
        if len(buf) < self.max_per_class:
            buf.append((sample.sample_id, signal.copy()))

    def add_discarded(self, sample: RawSample, signal: np.ndarray) -> None:
        key = (sample.source_dataset, sample.channel_idx)
        self.count_discarded[key] += 1
        buf = self._discarded[key]
        if len(buf) < self.max_per_class:
            buf.append((sample.sample_id, signal.copy()))

    def get_stats(self) -> dict:
        result: dict = {}
        all_ch_keys = sorted(
            set(self.count_kept.keys()) | set(self.count_discarded.keys()),
        )
        for key in all_ch_keys:
            ds, ch = key
            ds_dict = result.setdefault(ds, {})
            ds_dict[f"ch{ch}"] = {
                "samples": self.count_kept.get(key, 0),
                "discarded": self.count_discarded.get(key, 0),
            }
        for key in sorted(self._excluded.keys()):
            ds, ch = key
            ds_dict = result.setdefault(ds, {})
            ds_dict[f"ch{ch}"] = {
                "samples": 0,
                "note": "excluded by config",
            }
        return result

    def generate_reports(self) -> None:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        log.info("Writing load-stage QA under %s", out.resolve())

        all_keys = sorted(
            set(self.count_kept.keys()) | set(self.count_discarded.keys()),
        )
        datasets = sorted({k[0] for k in all_keys})

        for ds in datasets:
            ds_dir = out / ds
            ds_dir.mkdir(parents=True, exist_ok=True)
            ds_keys_all = [k for k in all_keys if k[0] == ds]
            disc_keys = [k for k in ds_keys_all if self.count_discarded.get(k, 0)]
            disc_keys_lim = self._limit_debug_channels(ds, disc_keys)
            if disc_keys_lim:
                FormatOutputDebugQA._save_grid_plots_static(
                    ds_dir / "discarded", self._discarded, disc_keys_lim, self.config,
                )

            excl_keys_all = sorted(k for k in self._excluded if k[0] == ds)
            excl_keys = self._limit_debug_channels(ds, excl_keys_all)
            if excl_keys:
                FormatOutputDebugQA._save_grid_plots_static(
                    ds_dir / "excluded_by_config", self._excluded, excl_keys, self.config,
                )

            self._save_summary(ds_dir / "summary.txt", ds_keys_all, excl_keys_all)

    def _limit_debug_channels(
        self, dataset: str, keys: list[tuple[str, int]],
    ) -> list[tuple[str, int]]:
        limit = self._per_dataset_channel_plot_limits.get(dataset)
        if limit is None and "::" in dataset:
            base = dataset.split("::", 1)[0]
            limit = self._per_dataset_channel_plot_limits.get(base)
        if limit is None:
            return keys
        if limit <= 0:
            return []
        chs = sorted({ch for _ds, ch in keys})
        if len(chs) <= limit:
            return keys
        seed = int(self.config.seed) + (abs(hash(dataset)) % 1_000_000)
        rng = np.random.default_rng(seed)
        chosen = set(rng.choice(chs, size=limit, replace=False).tolist())
        return [k for k in keys if k[1] in chosen]

    def _save_summary(
        self,
        path: Path,
        keys: list[tuple[str, int]],
        excluded_keys: list[tuple[str, int]],
    ) -> None:
        lines = []
        for key in keys:
            ds, ch = key
            n = self.count_kept.get(key, 0)
            nd = self.count_discarded.get(key, 0)
            lines.append(
                f"ch{ch:>3d}:  kept={n:>10,}  discarded={nd:>10,}",
            )
        for key in excluded_keys:
            ds, ch = key
            n_plotted = len(self._excluded.get(key, []))
            lines.append(
                f"ch{ch:>3d}:  EXCLUDED by config  "
                f"({n_plotted} sample plots saved for visual inspection)"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class FormatOutputDebugQA:
    """Per output format: grids from bytes actually written + shard_mixing from disk."""

    def __init__(
        self,
        config: ETLConfig,
        format_key: str,
        per_dataset_channel_plot_limits: dict[str, int] | None = None,
    ) -> None:
        if format_key not in FORMAT_OUTPUT_SUBDIR:
            raise ValueError(f"Unknown format_key: {format_key!r}")
        self.config = config
        self.format_key = format_key
        self.max_per_class = config.debug_samples_per_class
        self._per_dataset_channel_plot_limits = per_dataset_channel_plot_limits or {}
        sub = FORMAT_OUTPUT_SUBDIR[format_key]
        self._root = Path(config.debug_output_dir) / sub

        self._written: dict[tuple[str, int], list] = defaultdict(list)

    def add_written_sample(
        self,
        sample_id: str,
        source_dataset: str,
        channel_idx: int,
        signal_out: np.ndarray,
    ) -> None:
        key = (source_dataset, int(channel_idx))
        buf = self._written[key]
        if len(buf) < self.max_per_class:
            buf.append((sample_id, np.asarray(signal_out, dtype=np.float32).copy()))

    def write_shard_mixing_wds(self, split: str, format_output_root: str) -> None:
        wds_split = Path(format_output_root) / "wds" / split
        if not wds_split.is_dir():
            return
        shards = sorted(wds_split.glob("shard-*.tar"))[:_MAX_SHARDS_FOR_QA]
        base = self._root / "shard_mixing_wds" / split
        for si, tar_path in enumerate(shards):
            samples = _read_wds_samples_from_tar(tar_path, _MAX_TRACES_PER_SHARD)
            if not samples:
                continue
            _save_shard_mixing_figure(
                samples,
                base / f"shard_{si:06d}.png",
                f"Shard {si:06d} ({split}) — WDS on-disk — {self.format_key}",
            )

    def write_shard_mixing_hdf5(self, split: str, format_output_root: str) -> None:
        h5_path = Path(format_output_root) / "hdf5" / f"{split}.h5"
        if not h5_path.is_file():
            return
        with h5py.File(str(h5_path), "r") as f:
            n = int(f["offsets"].shape[0] - 1)
        if n <= 0:
            return
        max_shard = (n - 1) // self.config.samples_per_shard
        base = self._root / "shard_mixing_hdf5" / split
        for shard_idx in range(min(_MAX_SHARDS_FOR_QA, max_shard + 1)):
            samples = _read_hdf5_shard_traces(
                h5_path,
                self.config.samples_per_shard,
                shard_idx,
                _MAX_TRACES_PER_SHARD,
            )
            if not samples:
                continue
            _save_shard_mixing_figure(
                samples,
                base / f"shard_{shard_idx:06d}.png",
                f"Shard {shard_idx:06d} ({split}) — HDF5 on-disk — {self.format_key}",
            )

    def generate_written_reports(self) -> None:
        out = self._root
        out.mkdir(parents=True, exist_ok=True)
        log.info("Writing output-aligned QA for format %r under %s", self.format_key, out.resolve())

        all_keys = sorted(self._written.keys())
        datasets = sorted({k[0] for k in all_keys})

        for ds in datasets:
            ds_dir = out / ds
            ds_dir.mkdir(parents=True, exist_ok=True)
            ds_keys_all = [k for k in all_keys if k[0] == ds]
            ds_keys = self._limit_debug_channels(ds, ds_keys_all)
            self._save_grid_plots(ds_dir / "written", self._written, ds_keys)

    def _limit_debug_channels(
        self, dataset: str, keys: list[tuple[str, int]],
    ) -> list[tuple[str, int]]:
        limit = self._per_dataset_channel_plot_limits.get(dataset)
        if limit is None and "::" in dataset:
            base = dataset.split("::", 1)[0]
            limit = self._per_dataset_channel_plot_limits.get(base)
        if limit is None:
            return keys
        if limit <= 0:
            return []
        chs = sorted({ch for _ds, ch in keys})
        if len(chs) <= limit:
            return keys
        seed = (
            int(self.config.seed)
            + (abs(hash(dataset)) % 1_000_000)
            + (abs(hash(self.format_key)) % 10_000)
        )
        rng = np.random.default_rng(seed)
        chosen = set(rng.choice(chs, size=limit, replace=False).tolist())
        return [k for k in keys if k[1] in chosen]

    def _save_grid_plots(
        self,
        base_dir: Path,
        buffer: dict,
        keys: list[tuple[str, int]],
        cols: int = 5,
        rows: int = 5,
    ) -> None:
        self._save_grid_plots_static(base_dir, buffer, keys, self.config, cols, rows)

    @staticmethod
    def _save_grid_plots_static(
        base_dir: Path,
        buffer: dict,
        keys: list[tuple[str, int]],
        config: ETLConfig,
        cols: int = 5,
        rows: int = 5,
    ) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        per_page = cols * rows

        for key in keys:
            samples = buffer.get(key, [])
            if not samples:
                continue

            ds_name, ch = key
            for page_idx in range(0, len(samples), per_page):
                page = samples[page_idx : page_idx + per_page]
                n = len(page)
                actual_rows = (n + cols - 1) // cols
                fig, axes = plt.subplots(
                    actual_rows, cols, figsize=(3 * cols, 2.2 * actual_rows),
                    squeeze=False,
                )

                for i, (sid, sig) in enumerate(page):
                    r, c = divmod(i, cols)
                    ax = axes[r][c]
                    ax.plot(sig, linewidth=0.6, color="teal")
                    ax.set_title(sid, fontsize=5)
                    ax.tick_params(labelsize=5)

                for i in range(n, actual_rows * cols):
                    r, c = divmod(i, cols)
                    axes[r][c].axis("off")

                fig.suptitle(
                    f"{ds_name}  ch{ch}  ({base_dir.name})  page {page_idx // per_page}",
                    fontsize=9,
                )
                fig.tight_layout()
                fname = f"ch{ch}_page{page_idx // per_page:03d}.png"
                fig.savefig(base_dir / fname, dpi=120)
                plt.close(fig)


# Backwards compatibility alias (deprecated)
DebugQA = LoadStageDebug
