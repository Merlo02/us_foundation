from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

from ..config import DatasetConfig


@dataclass
class RawSample:
    """A single 1-D ultrasound signal extracted from a source file."""

    signal: np.ndarray
    sample_id: str
    source_dataset: str
    channel_idx: int
    sampling_frequency_hz: Optional[float] = None
    metadata: dict = field(default_factory=dict)


class BaseDatasetProcessor(ABC):
    """Interface that every source-dataset processor must implement.

    Subclass this, implement the three abstract members, and register the
    class in ``etl/__init__.py`` so that the YAML config can reference it
    by name.
    """

    def __init__(self, config: DatasetConfig, etl_target_length: int | None = None) -> None:
        self.config = config
        self.etl_target_length = etl_target_length

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def discover_files(self) -> list[str]:
        """Return a sorted list of raw file paths under *config.input_path*."""

    @abstractmethod
    def load_and_yield(self, filepath: str) -> Iterator[RawSample]:
        """Open *filepath*, iterate over channels (respecting exclusion lists),
        and ``yield`` one :class:`RawSample` per individual 1-D signal."""

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def dataset_name(self) -> str:
        return self.config.name

    def sampling_frequency_hz(self) -> Optional[float]:
        """Read the mandatory ``sampling_frequency_hz`` from the YAML ``extra`` block.

        Every processor should declare this in its ``DatasetConfig.extra`` so the
        downstream multi-tokenizer and CT-RoPE modules can route samples by
        acquisition frequency. Falling back to ``None`` is permitted for cases
        where the frequency is irrelevant (e.g. certain debug runs) but the
        runner will warn.
        """
        v = self.config.extra.get("sampling_frequency_hz")
        return float(v) if v is not None else None

    def should_keep_channel(self, ch_idx: int) -> bool:
        if self.config.channels_to_keep is not None:
            return ch_idx in self.config.channels_to_keep
        return ch_idx not in self.config.channels_to_exclude

    def load_all_channels(self, filepath: str) -> Iterator[RawSample]:
        """Like :meth:`load_and_yield` but temporarily disables channel
        filtering so that excluded channels are also yielded."""
        orig_exclude = self.config.channels_to_exclude
        orig_keep = self.config.channels_to_keep
        self.config.channels_to_exclude = []
        self.config.channels_to_keep = None
        try:
            yield from self.load_and_yield(filepath)
        finally:
            self.config.channels_to_exclude = orig_exclude
            self.config.channels_to_keep = orig_keep
