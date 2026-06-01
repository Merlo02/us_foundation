"""Abstract base for downstream (labeled, multi-channel) dataset processors.

A processor turns a raw labeled dataset into a stream of
``{"signal": (C, T) float32, "label": <int|float-vector>,
   "session_id": int, "patient_id": int}`` dicts, one per kept frame. The
runner consumes the stream and writes them into a single ``all.h5`` file
with the fixed schema documented in :mod:`etl_downstream.writer`.

A processor is either a **classifier** (``task_type='classification'`` —
the default; ``label`` is a single int class id) or a **regressor**
(``task_type='regression'`` — ``label`` is a length-``num_outputs`` vector
of continuous targets). The three regression hooks (:attr:`task_type`,
:attr:`num_outputs`, :attr:`label_names`) carry sensible classification
defaults so existing single-int processors need no changes.

The pretrain :class:`etl.processors.base_processor.BaseDatasetProcessor`
yields single-channel unlabeled ``RawSample`` objects; the two bases stay
structurally independent on purpose (no ``Optional[label]`` everywhere).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Sequence

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
    # Task kind (regression hooks). Classification is the default so
    # existing int-label processors stay untouched.
    # ------------------------------------------------------------------
    @property
    def task_type(self) -> str:
        """``"classification"`` (default) or ``"regression"``."""
        return "classification"

    @property
    def num_outputs(self) -> int:
        """Number of continuous targets per frame (regression only).

        ``0`` for classification. Regression processors must override this
        to a value ``>= 1``; the writer stores ``/labels`` as ``(N, K)``.
        """
        return 0

    @property
    def label_names(self) -> Optional[Sequence[str]]:
        """Optional names for the ``num_outputs`` regression targets."""
        return None

    # ------------------------------------------------------------------
    # Iteration interface.
    # ------------------------------------------------------------------
    @abstractmethod
    def discover_files(self) -> list[Path]: ...

    @abstractmethod
    def load(self, filepath: Path) -> Iterator[dict]:
        """Yield one dict per kept frame.

        Classification::

            {"signal": (C, T) float32, "label": int,
             "session_id": int, "patient_id": int}

        Regression::

            {"signal": (C, T) float32, "label": (num_outputs,) float,
             "session_id": int, "patient_id": int}
        """

    # ------------------------------------------------------------------
    # Channel filtering helper (shared with subclasses).
    # ------------------------------------------------------------------
    def should_keep_channel(self, ch_idx: int) -> bool:
        if self.config.channels_to_keep is not None:
            return ch_idx in self.config.channels_to_keep
        return ch_idx not in self.config.channels_to_exclude


__all__ = ["DownstreamBaseProcessor"]
