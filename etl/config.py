from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DatasetConfig:
    """Configuration for a single source dataset."""

    name: str
    processor: str
    input_path: str
    channels_to_exclude: list[int] = field(default_factory=list)
    channels_to_keep: Optional[list[int]] = None
    extra: dict = field(default_factory=dict)


@dataclass
class ETLConfig:
    """Global ETL pipeline configuration."""

    datasets: list[DatasetConfig] = field(default_factory=list)
    # NOTE: The ETL pipeline is length-preserving by design: processors yield
    # native-length 1-D signals and the runner never truncates or pads.
    #
    # target_length/truncation_mode used to control truncation; they are kept
    # optional for backwards compatibility with older YAML configs but are
    # ignored by the runner.
    target_length: Optional[int] = None
    truncation_mode: Optional[str] = None
    output_dir: str = "./output"
    output_format: str = "both"  # "webdataset", "hdf5", "both"
    samples_per_shard: int = 1024
    batch_size: int = 64
    world_size: int = 16
    num_workers: int = 4
    split_ratios: dict = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    seed: int = 42
    hdf5_chunk_mb: int = 128
    debug_samples_per_class: int = 100
    debug_output_dir: str = ""
    dtype: str = "float32"
    min_signal_energy: float = 1e-6

    # ------------------------------------------------------------------
    # Preprocessing variants (Experiment D)
    # ------------------------------------------------------------------
    preprocessing_mode: str = "raw"  # "raw" | "envelope" | "bandpass"
    bandpass_low_hz: float = 1_000_000   # 1 MHz
    bandpass_high_hz: float = 10_000_000  # 10 MHz
    bandpass_order: int = 4

    # ------------------------------------------------------------------
    # Static subsampling (Experiment B2)
    # ------------------------------------------------------------------
    max_samples_per_dataset: dict = field(default_factory=dict)
    """Upper bound on the number of samples kept per base dataset name, e.g.
    ``{"lateral_gastrocnemius": 500_000}``. Applied after loading and before
    splitting; truncates the pool of a given dataset (shuffled first)."""

    # ------------------------------------------------------------------
    # WebDataset shard padding
    # ------------------------------------------------------------------
    pad_last_shard: bool = True
    """When ``True``, the ETL runner fills the last WebDataset shard with
    random duplicates (flagged ``is_filler=True`` in metadata) so every shard
    has exactly ``samples_per_shard`` samples. Prevents NCCL hangs during
    epoch-based DDP training."""

    def __post_init__(self) -> None:
        if not self.debug_output_dir:
            self.debug_output_dir = str(Path(self.output_dir) / "debug_qa")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def total_workers(self) -> int:
        return self.world_size * self.num_workers

    def validate(self) -> None:
        """Raise on invalid configuration."""
        assert self.samples_per_shard > 0, "samples_per_shard must be positive"
        assert self.samples_per_shard % self.batch_size == 0, (
            f"samples_per_shard ({self.samples_per_shard}) must be a multiple "
            f"of batch_size ({self.batch_size})"
        )
        assert self.output_format in ("webdataset", "hdf5", "both")
        assert self.preprocessing_mode in ("raw", "envelope", "bandpass"), (
            f"preprocessing_mode must be one of raw|envelope|bandpass, "
            f"got {self.preprocessing_mode!r}"
        )
        if self.preprocessing_mode == "bandpass":
            assert 0 < self.bandpass_low_hz < self.bandpass_high_hz, (
                "bandpass_low_hz must be positive and below bandpass_high_hz"
            )
        ratios = self.split_ratios
        assert abs(sum(ratios.values()) - 1.0) < 1e-6, "split_ratios must sum to 1"
