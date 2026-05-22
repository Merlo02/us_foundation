"""Configuration dataclass for the labeled (downstream) ETL pipeline.

The pretrain ETL's :class:`etl.config.DatasetConfig` is reused (same
``name`` / ``processor`` / ``input_path`` / ``channels_to_*`` / ``extra``
semantics), so YAML structures stay consistent across the two pipelines.

The downstream pipeline is much simpler than the pretrain one:

- exactly one dataset entry per run (one ``.h5`` out);
- no train/val/test split (the DataModule splits at training time);
- no multi-format branching (just the raw frames);
- no shard-based outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from etl.config import DatasetConfig  # re-exported below

__all__ = ["DatasetConfig", "DownstreamETLConfig"]


@dataclass
class DownstreamETLConfig:
    """Top-level configuration for ``run_downstream_etl``."""

    output_dir: str
    datasets: list[DatasetConfig] = field(default_factory=list)
    flush_every: int = 1024
    debug_enabled: bool = True
    debug_output_dir: Optional[str] = None
    debug_samples_per_class: int = 100
    debug_seed: int = 42

    def __post_init__(self) -> None:
        if len(self.datasets) != 1:
            raise ValueError(
                f"Downstream ETL expects exactly one dataset entry, "
                f"got {len(self.datasets)}.",
            )
        if not self.output_dir:
            raise ValueError("output_dir is required.")
        if self.debug_output_dir is None:
            self.debug_output_dir = str(Path(self.output_dir) / "debug_qa")
