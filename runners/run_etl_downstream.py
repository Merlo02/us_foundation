#!/usr/bin/env python3
"""CLI entrypoint for the labeled (downstream) ETL pipeline.

Usage (from ``us_foundation/``, in the numpy-2.x venv — see ``CLAUDE.md``)::

    ~/usf_etl_venv/bin/python -m runners.run_etl_downstream \\
        --config configs/etl/etl_downstream_spacone.yaml

Produces::

    <output_dir>/all.h5         # fixed-schema labeled HDF5
    <output_dir>/manifest.json
    <output_dir>/debug_qa/...   # optional per-class plots
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from etl_downstream import DatasetConfig, DownstreamETLConfig, run_downstream_etl


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Labeled (downstream) ETL pipeline")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    p.add_argument("--output_dir", type=str, default=None, help="Override output_dir")
    return p.parse_args()


def _load_config(args: argparse.Namespace) -> DownstreamETLConfig:
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if args.output_dir is not None:
        raw["output_dir"] = args.output_dir

    datasets = [DatasetConfig(**ds) for ds in (raw.pop("datasets", None) or [])]
    return DownstreamETLConfig(datasets=datasets, **raw)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    cfg = _load_config(args)
    run_downstream_etl(cfg)


if __name__ == "__main__":
    main()
