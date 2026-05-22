"""Top-level orchestrator for the labeled (downstream) ETL.

Pipeline::

    cfg → resolve processor → discover files → for each file: load() → write
        → close writer → write_manifest → write_class_grids (if enabled)

Single dataset per run, single ``all.h5`` output, no splits.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path

from .config import DownstreamETLConfig
from .debug import write_class_grids
from .manifest import write_manifest
from .processors import PROCESSOR_REGISTRY
from .writer import DownstreamHDF5Writer

log = logging.getLogger(__name__)


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

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / "all.h5"

    t_start = time.time()
    per_session: Counter = Counter()
    per_patient: Counter = Counter()
    per_label: Counter = Counter()

    with DownstreamHDF5Writer(
        h5_path=h5_path,
        num_channels=processor.num_channels,
        samples_per_frame=processor.samples_per_frame,
        sampling_frequency_hz=processor.sampling_frequency_hz,
        dataset_name=processor.name,
        label_type=processor.label_type,
        num_classes=processor.num_classes,
        flush_every=cfg.flush_every,
    ) as writer:
        files = list(processor.discover_files())
        log.info(
            "Downstream ETL: dataset=%s processor=%s files=%d output=%s",
            ds_cfg.name, ds_cfg.processor, len(files), h5_path,
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
                per_label[int(row["label"])] += 1
        n_written = writer.num_written

    elapsed = time.time() - t_start
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
