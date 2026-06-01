"""Top-level orchestrator for the labeled (downstream) ETL.

Pipeline::

    cfg → resolve processor → discover files → for each file: load() → write
        → close writer → write_manifest → write_class_grids (if enabled)

Single dataset per run, single ``all.h5`` output, no splits. Works for both
classification processors (single-int labels, per-class counts) and
regression processors (``num_outputs``-vector labels, per-target stats).
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from .config import DownstreamETLConfig
from .debug import write_class_grids
from .manifest import write_manifest
from .processors import PROCESSOR_REGISTRY
from .writer import DownstreamHDF5Writer

log = logging.getLogger(__name__)


class _RegressionAccumulator:
    """Running per-target ``mean/std/min/max`` over the streamed labels."""

    def __init__(self, num_outputs: int) -> None:
        self.k = int(num_outputs)
        self.count = 0
        self._sum = np.zeros(self.k, dtype=np.float64)
        self._sqsum = np.zeros(self.k, dtype=np.float64)
        self._min = np.full(self.k, np.inf, dtype=np.float64)
        self._max = np.full(self.k, -np.inf, dtype=np.float64)

    def update(self, label) -> None:
        arr = np.asarray(label, dtype=np.float64).reshape(-1)
        self._sum += arr
        self._sqsum += arr * arr
        self._min = np.minimum(self._min, arr)
        self._max = np.maximum(self._max, arr)
        self.count += 1

    def stats(self) -> Optional[dict]:
        if self.count == 0:
            return None
        mean = self._sum / self.count
        var = np.maximum(self._sqsum / self.count - mean * mean, 0.0)
        std = np.sqrt(var)
        return {
            "count": int(self.count),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": self._min.tolist(),
            "max": self._max.tolist(),
        }


def run_downstream_etl(cfg: DownstreamETLConfig) -> Path:
    """Run the pipeline; return the path to the produced ``all.h5``."""
    ds_cfg = cfg.datasets[0]
    cls = PROCESSOR_REGISTRY.get(ds_cfg.processor)
    if cls is None:
        available = ", ".join(sorted(PROCESSOR_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown downstream processor {ds_cfg.processor!r} for dataset "
            f"{ds_cfg.name!r}. Available: {available}",
        )
    processor = cls(ds_cfg)
    is_regression = processor.task_type == "regression"

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / "all.h5"

    t_start = time.time()
    per_session: Counter = Counter()
    per_patient: Counter = Counter()
    per_label: Counter = Counter()
    reg_accum = (
        _RegressionAccumulator(processor.num_outputs) if is_regression else None
    )

    with DownstreamHDF5Writer(
        h5_path=h5_path,
        num_channels=processor.num_channels,
        samples_per_frame=processor.samples_per_frame,
        sampling_frequency_hz=processor.sampling_frequency_hz,
        dataset_name=processor.name,
        label_type=processor.label_type,
        num_classes=processor.num_classes,
        task_type=processor.task_type,
        num_outputs=processor.num_outputs,
        label_names=processor.label_names,
        flush_every=cfg.flush_every,
    ) as writer:
        files = list(processor.discover_files())
        log.info(
            "Downstream ETL: dataset=%s processor=%s task=%s files=%d output=%s",
            ds_cfg.name, ds_cfg.processor, processor.task_type, len(files), h5_path,
        )
        for fp in files:
            for row in processor.load(fp):
                writer.write(
                    signal=row["signal"],
                    label=row["label"],
                    session_id=row["session_id"],
                    patient_id=row["patient_id"],
                )
                per_session[int(row["session_id"])] += 1
                per_patient[int(row["patient_id"])] += 1
                if is_regression:
                    reg_accum.update(row["label"])
                else:
                    per_label[int(row["label"])] += 1
        n_written = writer.num_written

    elapsed = time.time() - t_start
    if is_regression:
        log.info(
            "Downstream ETL: wrote %d frames in %.1fs to %s "
            "(sessions=%d patients=%d num_outputs=%d label_names=%s)",
            n_written, elapsed, h5_path,
            len(per_session), len(per_patient), processor.num_outputs,
            list(processor.label_names) if processor.label_names else None,
        )
    else:
        log.info(
            "Downstream ETL: wrote %d frames in %.1fs to %s "
            "(sessions=%d patients=%d classes=%d)",
            n_written, elapsed, h5_path,
            len(per_session), len(per_patient), len(per_label),
        )

    write_manifest(
        output_dir=output_dir,
        processor=processor,
        num_samples=n_written,
        per_session=per_session,
        per_patient=per_patient,
        per_label=per_label,
        elapsed_seconds=elapsed,
        label_stats=reg_accum.stats() if reg_accum is not None else None,
    )

    if cfg.debug_enabled and n_written > 0:
        debug_dir = Path(cfg.debug_output_dir or (output_dir / "debug_qa"))
        write_class_grids(
            h5_path=h5_path,
            output_dir=debug_dir,
            samples_per_class=cfg.debug_samples_per_class,
            seed=cfg.debug_seed,
        )

    return h5_path


__all__ = ["run_downstream_etl"]
