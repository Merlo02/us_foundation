"""Optional PNG diagnostics: RAW vs globally-normalized chunks (first epoch only).

Activated via YAML ``data.signal_trace_enabled``. Uses lazy matplotlib import.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

# PyTorch DataLoader worker id (set via ``worker_init_fn``).
_worker_id: int = 0


def trace_dataloader_worker_init(worker_id: int) -> None:
    """Register worker index for unique trace filenames under multiprocessing."""
    global _worker_id
    _worker_id = worker_id


class SignalTracer:
    """Plot at most one PNG per ``dataset_source`` per process during training epoch 0."""

    def __init__(
        self,
        enabled: bool,
        output_dir: str,
        train_epoch_mp: Any,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_dir = os.path.abspath(output_dir)
        self._epoch_mp = train_epoch_mp  # multiprocessing.Value('i')
        self._seen: set[str] = set()

    def _effective_epoch(self) -> int:
        return int(self._epoch_mp.value)

    def maybe_trace(
        self,
        raw_chunk: Any,
        normalized_chunk: Any,
        normalization_type: str,
        stats: dict[str, float],
        dataset_source: str,
    ) -> None:
        if not self.enabled:
            return
        if self._effective_epoch() > 0:
            return
        key = str(dataset_source)
        if key in self._seen:
            return
        self._seen.add(key)

        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            log.warning("signal_trace: matplotlib/numpy unavailable — skipping plot")
            return

        raw = np.asarray(raw_chunk, dtype=np.float32).ravel()
        norm = np.asarray(normalized_chunk, dtype=np.float32).ravel()
        os.makedirs(self.output_dir, exist_ok=True)
        safe = re.sub(r"[^\w.\-]+", "_", key)[:120]
        fname = f"{safe}_w{_worker_id}.png"
        path = os.path.join(self.output_dir, fname)

        n = np.arange(raw.size)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        axes[0].plot(n, raw, lw=0.5)
        axes[0].set_title("RAW chunk")
        axes[0].set_ylabel("amplitude")
        axes[1].plot(n, norm, lw=0.5, color="C1")
        axes[1].set_title(f"Normalized ({normalization_type})")
        axes[1].set_ylabel("scaled")
        for ax in axes:
            ax.set_xlabel("sample")
        sm = stats.get("signal_mean", float("nan"))
        ss = stats.get("signal_std", float("nan"))
        smn = stats.get("signal_min", float("nan"))
        smx = stats.get("signal_max", float("nan"))
        fig.suptitle(
            f"{key}\nETL globals: mean={sm:.6g} std={ss:.6g} min={smn:.6g} max={smx:.6g}",
            fontsize=9,
        )
        fig.tight_layout()
        try:
            fig.savefig(path, dpi=120)
        finally:
            plt.close(fig)