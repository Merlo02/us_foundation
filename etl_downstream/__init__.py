"""Labeled (downstream) ETL pipeline.

Produces a single ``all.h5`` file with a fixed schema (see
:mod:`etl_downstream.writer`). The DataModule in
:mod:`data.downstream_datamodule` is the sole consumer.
"""
from __future__ import annotations

from .base_processor import DownstreamBaseProcessor
from .config import DatasetConfig, DownstreamETLConfig
from .processors import PROCESSOR_REGISTRY
from .runner import run_downstream_etl
from .writer import DownstreamHDF5Writer

__all__ = [
    "DatasetConfig",
    "DownstreamBaseProcessor",
    "DownstreamETLConfig",
    "DownstreamHDF5Writer",
    "PROCESSOR_REGISTRY",
    "run_downstream_etl",
]
