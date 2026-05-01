"""Optional PNG diagnostics: RAW vs globally-normalized chunks (first epoch only).

Activated via YAML ``data.signal_trace_enabled``. Uses lazy matplotlib import.

The current training epoch for gating traces is stored in a **process-wide**
``multiprocessing.Value`` created lazily at module scope. That value is **not**
held on :class:`SignalTracer` instances, so Lightning checkpoint pickling (which
may walk ``Trainer`` → ``datamodule`` → ``signal_tracer``) does not encounter
non-serializable ``Synchronized`` objects on those instances.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

log = logging.getLogger(__name__)

# PyTorch DataLoader worker id (set via ``worker_init_fn``).
_worker_id: int = 0

# Lazily allocated; shared across workers when DataLoader uses subprocesses.
_train_epoch_mp: Optional[Any] = None


def trace_dataloader_worker_init(worker_id: int) -> None:
    """Register worker index for unique trace filenames under multiprocessing."""
    global _worker_id
    _worker_id = worker_id


def _ensure_train_epoch_mp() -> Any:
    """Create the shared epoch counter once (fork-safe for worker visibility)."""
    global _train_epoch_mp
    if _train_epoch_mp is None:
        import multiprocessing as mp

        _train_epoch_mp = mp.Value("i", 0)
    return _train_epoch_mp


def set_signal_trace_epoch(epoch: int) -> None:
    """Update epoch seen by workers (Lightning ``on_*_epoch_start`` hooks)."""
    global _train_epoch_mp
    if _train_epoch_mp is None:
        return
    _train_epoch_mp.value = int(epoch)


class SignalTracer:
    """Plot at most one PNG per ``dataset_source`` per process during training epoch 0."""

    def __init__(
        self,
        enabled: bool,
        output_dir: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_dir = os.path.abspath(output_dir)
        self._seen: set[str] = set()
        if self.enabled:
            _ensure_train_epoch_mp()

    def _effective_epoch(self) -> int:
        global _train_epoch_mp
        if _train_epoch_mp is None:
            return 0
        return int(_train_epoch_mp.value)

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
