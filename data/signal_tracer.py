"""Optional PNG diagnostics for debugging the data pipeline (first epoch only).

Activated via YAML ``data.signal_trace_enabled: true``.  When disabled the
``SignalTracer`` instance is ``None`` in the DataModule, so every call-site
short-circuits with a single ``is not None`` check — zero overhead in normal
training.

Two plot layouts are produced:

**Variable-S** (``target_patches is None``)::

    | Raw (from ETL disk) | Post-processing + normalized |

**Fixed-S** (``target_patches`` set)::

    | Row 1: Raw (from ETL disk)   — full signal              |
    | Row 2: Post-preprocessed     — full signal (if ≠ raw)   |
    | Row 3: Chunk 0 | Chunk 1 | … Chunk N  (all normalized)  |

One PNG per ``dataset_source`` per worker, epoch 0 only.
Filename: ``{dataset_source_safe}_w{worker_id}.png``

The current training epoch is stored in a **process-wide**
``multiprocessing.Value`` so Lightning checkpoint pickling
(which walks ``Trainer → datamodule → signal_tracer``) never encounters
non-serialisable ``Synchronized`` objects on the instance.
"""
from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Optional

import numpy as np

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
    """Plot at most one PNG per ``dataset_source`` per worker during epoch 0.

    All actual work is gated behind :meth:`should_trace` so that when
    ``signal_trace_enabled: false`` (the DataModule stores ``None`` instead of
    an instance) no overhead is incurred at call-sites.
    """

    def __init__(self, enabled: bool, output_dir: str) -> None:
        self.enabled = bool(enabled)
        self.output_dir = os.path.abspath(output_dir)
        self._seen: set[str] = set()
        if self.enabled:
            _ensure_train_epoch_mp()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _effective_epoch(self) -> int:
        global _train_epoch_mp
        if _train_epoch_mp is None:
            return 0
        return int(_train_epoch_mp.value)

    def should_trace(self, dataset_source: str) -> bool:
        """True iff tracing is enabled, epoch == 0, and this source is new."""
        if not self.enabled:
            return False
        if self._effective_epoch() > 0:
            return False
        return str(dataset_source) not in self._seen

    def trace(
        self,
        raw_etl: np.ndarray,
        pp_full: np.ndarray,
        norm_chunks: list[np.ndarray],
        normalization_type: str,
        stats: dict[str, float],
        dataset_source: str,
        target_patches: Optional[int],
        window_size: int,
    ) -> None:
        """Emit the diagnostic PNG and mark *dataset_source* as seen.

        Parameters
        ----------
        raw_etl :
            Full signal as read from disk (before any online preprocessing).
        pp_full :
            Full signal after online preprocessing but before normalisation
            and chunking.  Equal to *raw_etl* when ``preprocessing_mode="raw"``.
        norm_chunks :
            List of post-normalisation arrays.  One element for variable-S;
            ``num_chunks`` elements for fixed-S.
        target_patches :
            ``None`` → variable-S layout; integer → fixed-S layout.
        window_size :
            ``W*`` in samples (used for labelling only).
        """
        key = str(dataset_source)
        self._seen.add(key)

        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            log.warning("signal_trace: matplotlib unavailable — skipping plot")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        safe = re.sub(r"[^\w.\-]+", "_", key)[:120]
        fname = f"{safe}_w{_worker_id}.png"
        path = os.path.join(self.output_dir, fname)

        raw = np.asarray(raw_etl, dtype=np.float32).ravel()
        pp = np.asarray(pp_full, dtype=np.float32).ravel()
        preprocessing_applied = not np.array_equal(raw, pp)

        sm = stats.get("signal_mean", float("nan"))
        ss = stats.get("signal_std", float("nan"))
        smn = stats.get("signal_min", float("nan"))
        smx = stats.get("signal_max", float("nan"))
        etl_stats_str = f"ETL globals: mean={sm:.4g} std={ss:.4g} min={smn:.4g} max={smx:.4g}"

        try:
            if target_patches is None:
                self._plot_variable_s(
                    plt, raw, pp, norm_chunks[0], normalization_type,
                    preprocessing_applied, key, etl_stats_str, window_size,
                )
            else:
                self._plot_fixed_s(
                    plt, raw, pp, norm_chunks, normalization_type,
                    preprocessing_applied, key, etl_stats_str,
                    target_patches, window_size,
                )
            plt.savefig(path, dpi=120, bbox_inches="tight")
            log.debug("signal_trace: saved %s", path)
        except Exception as exc:  # pragma: no cover
            log.warning("signal_trace: failed to save %s: %s", path, exc)
        finally:
            plt.close("all")

    # ------------------------------------------------------------------
    # Backward-compat wrapper (old signature kept for external callers)
    # ------------------------------------------------------------------

    def maybe_trace(
        self,
        raw_chunk: Any,
        normalized_chunk: Any,
        normalization_type: str,
        stats: dict[str, float],
        dataset_source: str,
    ) -> None:
        """Legacy entry point: ``(pre_norm_chunk, norm_chunk, …) → PNG``.

        Treated as variable-S with no online preprocessing (raw == pp).
        """
        if not self.should_trace(dataset_source):
            return
        raw = np.asarray(raw_chunk, dtype=np.float32).ravel()
        norm = np.asarray(normalized_chunk, dtype=np.float32).ravel()
        self.trace(
            raw_etl=raw,
            pp_full=raw,
            norm_chunks=[norm],
            normalization_type=normalization_type,
            stats=stats,
            dataset_source=dataset_source,
            target_patches=None,
            window_size=0,
        )

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _plot_variable_s(
        self,
        plt: Any,
        raw: np.ndarray,
        pp: np.ndarray,
        norm: np.ndarray,
        normalization_type: str,
        preprocessing_applied: bool,
        key: str,
        etl_stats_str: str,
        window_size: int,
    ) -> None:
        """Variable-S: 2 panels — raw | post-processed & normalized."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        axes[0].plot(np.arange(raw.size), raw, lw=0.6)
        axes[0].set_title("Raw (from ETL)")
        axes[0].set_ylabel("amplitude")
        axes[0].set_xlabel("sample")

        # Right panel: preprocessed (if different) + normalized overlaid,
        # or just normalized if no preprocessing was applied.
        ax1 = axes[1]
        n_plot = np.arange(norm.size)
        if preprocessing_applied and pp.size == norm.size:
            ax1_twin = ax1.twinx()
            ax1.plot(np.arange(pp.size), pp, lw=0.5, color="C2", label="preprocessed")
            ax1_twin.plot(n_plot, norm, lw=0.6, color="C1", label=f"normalized ({normalization_type})")
            ax1.set_ylabel("preprocessed amplitude", color="C2")
            ax1_twin.set_ylabel(f"normalized ({normalization_type})", color="C1")
            ax1.set_title("Post-processing & normalized")
        else:
            ax1.plot(n_plot, norm, lw=0.6, color="C1")
            ax1.set_title(f"Normalized ({normalization_type})")
            ax1.set_ylabel("scaled")
        ax1.set_xlabel("sample")

        pp_label = f"  |  preprocessing={'yes' if preprocessing_applied else 'none'}  |  W*={window_size}"
        fig.suptitle(f"{key}\n{etl_stats_str}{pp_label}", fontsize=9)
        fig.tight_layout()

    def _plot_fixed_s(
        self,
        plt: Any,
        raw: np.ndarray,
        pp: np.ndarray,
        norm_chunks: list[np.ndarray],
        normalization_type: str,
        preprocessing_applied: bool,
        key: str,
        etl_stats_str: str,
        target_patches: int,
        window_size: int,
    ) -> None:
        """Fixed-S: stacked layout — raw | [pp] | grid of N normalized chunks."""
        n_chunks = len(norm_chunks)
        chunks_per_row = min(n_chunks, 8)
        chunk_rows = math.ceil(n_chunks / chunks_per_row)

        n_header_rows = 2 if preprocessing_applied else 1
        total_rows = n_header_rows + chunk_rows

        fig_h = 3.0 * total_rows
        fig_w = max(14.0, 2.5 * chunks_per_row)
        fig = plt.figure(figsize=(fig_w, fig_h))

        # ── Row 1: raw ETL signal ──────────────────────────────────────
        ax_raw = fig.add_subplot(total_rows, 1, 1)
        ax_raw.plot(np.arange(raw.size), raw, lw=0.5)
        ax_raw.set_title("Raw (from ETL) — full signal")
        ax_raw.set_ylabel("amplitude")
        ax_raw.set_xlabel("sample")

        # ── Row 2 (optional): post-preprocessed full signal ───────────
        next_row = 2
        if preprocessing_applied:
            ax_pp = fig.add_subplot(total_rows, 1, 2)
            ax_pp.plot(np.arange(pp.size), pp, lw=0.5, color="C2")
            ax_pp.set_title("Post-preprocessed — full signal")
            ax_pp.set_ylabel("amplitude")
            ax_pp.set_xlabel("sample")
            next_row = 3

        # ── Remaining rows: normalized chunks ─────────────────────────
        for ci, chunk in enumerate(norm_chunks):
            row = next_row + ci // chunks_per_row
            col = ci % chunks_per_row
            ax = fig.add_subplot(total_rows, chunks_per_row, (row - 1) * chunks_per_row + col + 1)
            ax.plot(np.arange(chunk.size), np.asarray(chunk, dtype=np.float32), lw=0.6, color="C1")
            ax.set_title(f"Chunk {ci}", fontsize=8)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(f"norm ({normalization_type})", fontsize=7)

        pp_label = f"  |  preprocessing={'yes' if preprocessing_applied else 'none'}  |  target_patches={target_patches}  |  W*={window_size}"
        fig.suptitle(f"{key}\n{etl_stats_str}{pp_label}", fontsize=9)
        fig.tight_layout()
