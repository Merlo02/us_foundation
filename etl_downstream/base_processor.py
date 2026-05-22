"""Abstract base for downstream (labeled, multi-channel) dataset processors.

A processor turns a raw labeled dataset into a stream of
``{"signal": (C, T) float32, "label": int, "session_id": int, "patient_id": int}``
dicts, one per kept frame. The runner consumes the stream and writes them
into a single ``all.h5`` file with the fixed schema documented in
:mod:`etl_downstream.writer`.

The pretrain :class:`etl.processors.base_processor.BaseDatasetProcessor`
yields single-channel unlabeled ``RawSample`` objects; the two bases stay
structurally independent on purpose (no ``Optional[label]`` everywhere).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from etl.config import DatasetConfig


class DownstreamBaseProcessor(ABC):
    """Interface every downstream-dataset processor must implement."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Dataset-level properties (all required; declared once per processor).
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.config.name

    @property
    @abstractmethod
    def sampling_frequency_hz(self) -> float: ...

    @property
    @abstractmethod
    def num_channels(self) -> int: ...

    @property
    @abstractmethod
    def samples_per_frame(self) -> int: ...

    @property
    @abstractmethod
    def num_classes(self) -> int: ...

    @property
    @abstractmethod
    def label_type(self) -> str: ...

    # ------------------------------------------------------------------
    # Iteration interface.
    # ------------------------------------------------------------------
    @abstractmethod
    def discover_files(self) -> list[Path]: ...

    @abstractmethod
    def load(self, filepath: Path) -> Iterator[dict]:
        """Yield ``{"signal": (C, T) float32, "label": int,
                     "session_id": int, "patient_id": int}`` per frame."""

    # ------------------------------------------------------------------
    # Channel filtering helper (shared with subclasses).
    # ------------------------------------------------------------------
    def should_keep_channel(self, ch_idx: int) -> bool:
        if self.config.channels_to_keep is not None:
            return ch_idx in self.config.channels_to_keep
        return ch_idx not in self.config.channels_to_exclude


__all__ = ["DownstreamBaseProcessor"]
