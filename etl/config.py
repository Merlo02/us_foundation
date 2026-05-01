from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Keys for multi-format ETL output (order preserved for stable writer iteration).
OUTPUT_FORMAT_KEYS: tuple[str, ...] = (
    "raw",
    "envelope",
    "bandpass",
    "interpolate",
)

# Subdirectory under ``output_dir`` for each logical format (``interpolate`` → ``interpolated``).
FORMAT_OUTPUT_SUBDIR: dict[str, str] = {
    "raw": "raw",
    "envelope": "envelope",
    "bandpass": "bandpass",
    "interpolate": "interpolated",
}


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
    # Used when ``output_formats["interpolate"]`` is True: fixed length after
    # linear resampling (short) or truncation (long).
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
    # Multi-format output (Experiment D): enable one or more concurrently.
    # ------------------------------------------------------------------
    output_formats: dict[str, bool] = field(default_factory=lambda: {
        "raw": True,
        "envelope": False,
        "bandpass": False,
        "interpolate": False,
    })
    bandpass_order: int = 4
    # Global default fractional bandwidth around ``extra.transmit_center_frequency_hz``
    # ``width = rf_bandwidth_fraction * tx_fc``. Override per dataset via
    # ``extra.rf_bandwidth_fraction``.
    rf_bandwidth_fraction: float = 0.8
    # Default for long-signal truncation in interpolate output when a dataset
    # omits ``extra.interpolation_truncate_mode`` (see per-dataset ``extra``).
    interpolation_truncate_mode: str = "left"

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

    def normalized_output_formats(self) -> dict[str, bool]:
        """All known keys → bool; missing keys ``False``; unknown keys raise."""
        src = dict(self.output_formats or {})
        unknown = set(src) - set(OUTPUT_FORMAT_KEYS)
        if unknown:
            raise AssertionError(
                f"Unknown output_formats keys: {sorted(unknown)}; "
                f"allowed: {list(OUTPUT_FORMAT_KEYS)}"
            )
        return {k: bool(src.get(k, False)) for k in OUTPUT_FORMAT_KEYS}

    def enabled_format_keys(self) -> list[str]:
        """Formats with ``True``, in canonical order."""
        d = self.normalized_output_formats()
        return [k for k in OUTPUT_FORMAT_KEYS if d[k]]

    def validate(self) -> None:
        """Raise on invalid configuration."""
        assert self.samples_per_shard > 0, "samples_per_shard must be positive"
        assert self.samples_per_shard % self.batch_size == 0, (
            f"samples_per_shard ({self.samples_per_shard}) must be a multiple "
            f"of batch_size ({self.batch_size})"
        )
        assert self.output_format in ("webdataset", "hdf5", "both")
        fmts = self.normalized_output_formats()
        enabled = [k for k in OUTPUT_FORMAT_KEYS if fmts[k]]
        assert enabled, "At least one output_formats entry must be True"
        if fmts["interpolate"]:
            assert self.target_length is not None and int(self.target_length) > 0, (
                "target_length must be a positive int when output_formats['interpolate'] is True"
            )
        if fmts["bandpass"] or fmts["envelope"]:
            assert 0.0 < float(self.rf_bandwidth_fraction) <= 1.0, (
                f"rf_bandwidth_fraction must be in (0, 1], "
                f"got {self.rf_bandwidth_fraction!r}"
            )
            for ds in self.datasets:
                ex = ds.extra or {}
                fc = ex.get("transmit_center_frequency_hz")
                if fc is None:
                    fc = ex.get("tx_fc_hz")
                assert fc is not None and float(fc) > 0, (
                    f"Dataset {ds.name!r}: extra.transmit_center_frequency_hz "
                    f"(or tx_fc_hz) is required and must be positive when "
                    f"envelope or bandpass output is enabled"
                )
                bw = ex.get("rf_bandwidth_fraction")
                if bw is not None:
                    assert 0.0 < float(bw) <= 1.0, (
                        f"Dataset {ds.name!r}: extra.rf_bandwidth_fraction "
                        f"must be in (0, 1], got {bw!r}"
                    )
        assert self.interpolation_truncate_mode in ("left", "right", "center"), (
            f"interpolation_truncate_mode must be left|right|center, "
            f"got {self.interpolation_truncate_mode!r}"
        )
        for ds in self.datasets:
            v = ds.extra.get("interpolation_truncate_mode")
            if v is None:
                continue
            m = str(v).lower()
            assert m in ("left", "right", "center"), (
                f"Dataset {ds.name!r}: extra.interpolation_truncate_mode must be "
                f"left|right|center, got {v!r}"
            )
        ratios = self.split_ratios
        assert abs(sum(ratios.values()) - 1.0) < 1e-6, "split_ratios must sum to 1"
