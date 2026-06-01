"""Debug QA: one ``C×1`` matplotlib grid per sampled row.

Classification: rows are grouped by class and saved under
``<debug_output_dir>/class_<label>/<row_idx>.png`` (uniform sampling within
each class, up to ``samples_per_class`` rows kept).

Regression: continuous labels have no classes, so a flat random sample of
up to ``samples_per_class`` rows is saved under
``<debug_output_dir>/samples/<row_idx>.png`` with the label vector shown in
the title.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import h5py
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

log = logging.getLogger(__name__)


def _plot_frame(
    frame: np.ndarray,
    x: np.ndarray,
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    c = frame.shape[0]
    fig, axes = plt.subplots(c, 1, figsize=(8, 1.5 * c), sharex=True)
    if c == 1:
        axes = [axes]
    for ch_idx, ax in enumerate(axes):
        ax.plot(x, frame[ch_idx], linewidth=0.6)
        ax.set_ylabel(f"ch{ch_idx}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def write_class_grids(
    h5_path: str | Path,
    output_dir: str | Path,
    samples_per_class: int = 100,
    seed: int = 42,
) -> None:
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))

    with h5py.File(h5_path, "r") as f:
        signal_ds = f["signal"]
        n, c, t = signal_ds.shape
        fs = float(f.attrs.get("sampling_frequency_hz", 0.0))
        dataset_name = str(f.attrs.get("dataset_name", h5_path.stem))
        label_type = str(f.attrs.get("label_type", "label"))
        task_type = str(f.attrs.get("task_type", "classification"))

        if n == 0:
            log.warning("debug: %s is empty — no plots written", h5_path)
            return

        time_axis_us: Optional[np.ndarray] = None
        if fs > 0:
            time_axis_us = (np.arange(t) / fs * 1e6).astype(np.float32)
        x = time_axis_us if time_axis_us is not None else np.arange(t)
        xlabel = "time (µs)" if fs > 0 else "sample"

        if task_type == "regression":
            labels = np.asarray(f["labels"][:], dtype=np.float64)  # (N, K)
            label_names = [
                str(s) for s in f.attrs.get("label_names", [])
            ] or [f"y{i}" for i in range(labels.shape[1] if labels.ndim > 1 else 1)]
            k = min(int(samples_per_class), int(n))
            chosen = rng.choice(n, size=k, replace=False)
            chosen.sort()
            samples_dir = output_dir / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            for row_idx in chosen.tolist():
                frame = np.asarray(signal_ds[int(row_idx)], dtype=np.float32)
                vec = np.atleast_1d(labels[int(row_idx)])
                lbl = ", ".join(
                    f"{nm}={v:.3f}" for nm, v in zip(label_names, vec.tolist())
                )
                _plot_frame(
                    frame, x,
                    f"{dataset_name} — {label_type} — row {int(row_idx)}\n[{lbl}]",
                    xlabel, samples_dir / f"row_{int(row_idx):08d}.png",
                )
            log.info(
                "debug: regression — wrote %d/%d sample plots to %s",
                k, n, samples_dir,
            )
            return

        labels = np.asarray(f["labels"][:], dtype=np.int64)
        for class_id in sorted(np.unique(labels).tolist()):
            class_idx = np.where(labels == class_id)[0]
            if class_idx.size == 0:
                continue
            k = min(int(samples_per_class), int(class_idx.size))
            chosen = rng.choice(class_idx, size=k, replace=False)
            chosen.sort()

            class_dir = output_dir / f"class_{int(class_id)}"
            class_dir.mkdir(parents=True, exist_ok=True)

            for row_idx in chosen.tolist():
                frame = np.asarray(signal_ds[int(row_idx)], dtype=np.float32)
                _plot_frame(
                    frame, x,
                    f"{dataset_name} — {label_type}={int(class_id)} — row {int(row_idx)}",
                    xlabel, class_dir / f"row_{int(row_idx):08d}.png",
                )

            log.info(
                "debug: class %d — wrote %d/%d sample plots to %s",
                int(class_id), k, class_idx.size, class_dir,
            )


__all__ = ["write_class_grids"]
