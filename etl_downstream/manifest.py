"""Write a ``manifest.json`` describing the produced ``all.h5``."""
from __future__ import annotations

import datetime as _dt
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Optional

from .base_processor import DownstreamBaseProcessor


def write_manifest(
    output_dir: str | Path,
    processor: DownstreamBaseProcessor,
    num_samples: int,
    per_session: Mapping[int, int],
    per_patient: Mapping[int, int],
    per_label: Mapping[int, int],
    elapsed_seconds: float,
    label_stats: Optional[dict] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    payload = {
        "dataset_name": processor.name,
        "label_type": processor.label_type,
        "task_type": processor.task_type,
        "num_samples": int(num_samples),
        "num_channels": int(processor.num_channels),
        "samples_per_frame": int(processor.samples_per_frame),
        "sampling_frequency_hz": float(processor.sampling_frequency_hz),
        "num_classes": int(processor.num_classes),
        "per_session_counts": {str(k): int(v) for k, v in sorted(per_session.items())},
        "per_patient_counts": {str(k): int(v) for k, v in sorted(per_patient.items())},
        "elapsed_seconds": float(elapsed_seconds),
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "input_path": str(processor.config.input_path),
        "channels_to_exclude": list(processor.config.channels_to_exclude or []),
        "channels_to_keep": (
            list(processor.config.channels_to_keep)
            if processor.config.channels_to_keep is not None
            else None
        ),
    }

    if processor.task_type == "regression":
        # Continuous targets — per-class counts are meaningless; report the
        # number of targets, their names, and per-target distribution stats.
        payload["num_outputs"] = int(processor.num_outputs)
        payload["label_names"] = (
            list(processor.label_names) if processor.label_names is not None else None
        )
        payload["label_stats"] = label_stats
    else:
        payload["per_label_counts"] = {
            str(k): int(v) for k, v in sorted(per_label.items())
        }

    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def merge_counter(target: Counter, key: int) -> None:
    target[int(key)] += 1


__all__ = ["write_manifest", "merge_counter"]
