from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for HPC / SSH
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .config import ETLConfig
    from .processors import RawSample


log = logging.getLogger(__name__)


class DebugQA:
    """Collects sample plots during the ETL pass and generates QA reports."""

    def __init__(self, config: ETLConfig, per_dataset_channel_plot_limits: dict[str, int] | None = None) -> None:
        self.config = config
        self.max_per_class = config.debug_samples_per_class
        self._per_dataset_channel_plot_limits = per_dataset_channel_plot_limits or {}

        # Buffers: (dataset, channel_idx) → list of (sample_id, signal)
        self._kept: dict[tuple[str, int], list] = defaultdict(list)
        self._discarded: dict[tuple[str, int], list] = defaultdict(list)
        self._excluded: dict[tuple[str, int], list] = defaultdict(list)

        # Counters: (dataset, channel_idx) → int
        self.count_kept: dict[tuple[str, int], int] = defaultdict(int)
        self.count_discarded: dict[tuple[str, int], int] = defaultdict(int)

        # Per-shard sample tracking: split → list of (sample_id, dataset, signal)
        self._shard_samples: dict[str, list[list]] = defaultdict(
            lambda: [[] for _ in range(10)]
        )

    # ------------------------------------------------------------------
    # Collecting samples
    # ------------------------------------------------------------------

    def add_kept(self, sample: RawSample, signal: np.ndarray) -> None:
        key = (sample.source_dataset, sample.channel_idx)
        self.count_kept[key] += 1
        buf = self._kept[key]
        if len(buf) < self.max_per_class:
            buf.append((sample.sample_id, signal.copy()))

    def add_excluded(self, sample: RawSample, signal: np.ndarray) -> None:
        """Collect a sample from a channel that was excluded by config."""
        key = (sample.source_dataset, sample.channel_idx)
        buf = self._excluded[key]
        if len(buf) < self.max_per_class:
            buf.append((sample.sample_id, signal.copy()))

    def add_discarded(self, sample: RawSample, signal: np.ndarray) -> None:
        """Collect a dead or invalid sample for debug plots (not written to shards)."""
        key = (sample.source_dataset, sample.channel_idx)
        self.count_discarded[key] += 1
        buf = self._discarded[key]
        if len(buf) < self.max_per_class:
            buf.append((sample.sample_id, signal.copy()))

    def add_shard_sample(
        self,
        split: str,
        shard_idx: int,
        sample_id: str,
        source_dataset: str,
        signal: np.ndarray,
        samples_per_shard: int,
    ) -> None:
        """Track samples for per-shard mixing plots."""
        if shard_idx >= 10:
            return
        buf = self._shard_samples[split][shard_idx]
        if len(buf) < 10:
            buf.append((sample_id, source_dataset, signal.copy()))

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_reports(self) -> None:
        out = Path(self.config.debug_output_dir)
        out.mkdir(parents=True, exist_ok=True)
        log.info("Writing QA plots under %s", out.resolve())

        all_keys = sorted(
            set(self.count_kept.keys()) | set(self.count_discarded.keys()),
        )
        datasets = sorted({k[0] for k in all_keys})

        for ds in datasets:
            ds_dir = out / ds
            ds_dir.mkdir(parents=True, exist_ok=True)

            ds_keys_all = [k for k in all_keys if k[0] == ds]
            ds_keys = self._limit_debug_channels(ds, ds_keys_all)

            self._save_grid_plots(ds_dir / "kept", self._kept, ds_keys)
            disc_keys = [k for k in ds_keys if self.count_discarded.get(k, 0)]
            if disc_keys:
                self._save_grid_plots(
                    ds_dir / "discarded", self._discarded, disc_keys,
                )

            excl_keys_all = sorted(k for k in self._excluded if k[0] == ds)
            excl_keys = self._limit_debug_channels(ds, excl_keys_all)
            if excl_keys:
                self._save_grid_plots(
                    ds_dir / "excluded_by_config", self._excluded, excl_keys,
                )

            # Summary stays complete (all channels), even if plots are limited.
            self._save_summary(ds_dir / "summary.txt", ds_keys_all, excl_keys_all)

        self._save_shard_mixing_plots(out / "shard_mixing")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _limit_debug_channels(
        self, dataset: str, keys: list[tuple[str, int]]
    ) -> list[tuple[str, int]]:
        limit = self._per_dataset_channel_plot_limits.get(dataset)
        if limit is None and "::" in dataset:
            # Allow per-base-dataset limits when debug is split into sub-groups
            # (e.g. "{base}::{session}").
            base = dataset.split("::", 1)[0]
            limit = self._per_dataset_channel_plot_limits.get(base)
        if limit is None:
            return keys
        if limit <= 0:
            return []

        chs = sorted({ch for _ds, ch in keys})
        if len(chs) <= limit:
            return keys

        # Deterministic sampling per dataset and run.
        seed = int(self.config.seed) + (abs(hash(dataset)) % 1_000_000)
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

    def _save_shard_mixing_plots(self, base_dir: Path) -> None:
        """One figure per shard showing ~10 signals colored by source dataset."""
        base_dir.mkdir(parents=True, exist_ok=True)

        ds_colors = {}
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        ]

        for split, shard_bufs in self._shard_samples.items():
            split_dir = base_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for shard_idx, samples in enumerate(shard_bufs):
                if not samples:
                    continue

                n = len(samples)
                fig, axes = plt.subplots(
                    n, 1, figsize=(10, 1.8 * n), squeeze=False,
                )

                for i, (sid, ds_name, sig) in enumerate(samples):
                    if ds_name not in ds_colors:
                        ds_colors[ds_name] = palette[len(ds_colors) % len(palette)]
                    ax = axes[i][0]
                    ax.plot(sig, linewidth=0.6, color=ds_colors[ds_name])
                    ax.set_title(f"{sid}  [{ds_name}]", fontsize=7)
                    ax.tick_params(labelsize=5)

                fig.suptitle(
                    f"Shard {shard_idx:06d} ({split}) — data mixing check",
                    fontsize=10,
                )
                fig.tight_layout()
                fig.savefig(split_dir / f"shard_{shard_idx:06d}.png", dpi=120)
                plt.close(fig)

    # ------------------------------------------------------------------
    # Stats for manifest
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return per-dataset-per-channel statistics as a nested dict."""
        result: dict = {}
        all_ch_keys = sorted(
            set(self.count_kept.keys()) | set(self.count_discarded.keys()),
        )
        for key in all_ch_keys:
            ds, ch = key
            ds_dict = result.setdefault(ds, {})
            entry = {
                "samples": self.count_kept.get(key, 0),
                "discarded": self.count_discarded.get(key, 0),
            }
            ds_dict[f"ch{ch}"] = entry
        for key in sorted(self._excluded.keys()):
            ds, ch = key
            ds_dict = result.setdefault(ds, {})
            ds_dict[f"ch{ch}"] = {
                "samples": 0,
                "note": "excluded by config",
            }
        return result
