#!/usr/bin/env python3
"""CLI entrypoint for the ultrasound ETL pipeline.

Usage (from us_foundation/):
    python -m runners.run_etl --config configs/etl/etl_config_sassauna.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from etl import PROCESSOR_REGISTRY
from etl.config import DatasetConfig, ETLConfig
from etl.runner import run_etl


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ultrasound A-mode ETL pipeline")
    p.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--output_format", type=str, default=None)
    p.add_argument("--samples_per_shard", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def _load_config(args: argparse.Namespace) -> ETLConfig:
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}
    # Legacy keys (global Hz edges) are ignored; map old envelope fraction name.
    if raw.get("rf_bandwidth_fraction") is None and raw.get(
        "envelope_bandpass_fraction",
    ) is not None:
        raw["rf_bandwidth_fraction"] = raw.pop("envelope_bandpass_fraction")
    else:
        raw.pop("envelope_bandpass_fraction", None)
    raw.pop("bandpass_low_hz", None)
    raw.pop("bandpass_high_hz", None)

    datasets = [DatasetConfig(**ds) for ds in raw.pop("datasets", [])]

    config = ETLConfig(datasets=datasets, **raw)

    # Apply CLI overrides
    for key in (
        "output_dir", "output_format",
        "samples_per_shard", "seed",
    ):
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)

    return config


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()
    config = _load_config(args)

    processors = []
    for ds_cfg in config.datasets:
        cls = PROCESSOR_REGISTRY.get(ds_cfg.processor)
        if cls is None:
            available = ", ".join(sorted(PROCESSOR_REGISTRY.keys()))
            print(
                f"Unknown processor '{ds_cfg.processor}' for dataset '{ds_cfg.name}'. "
                f"Available: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        processors.append(cls(ds_cfg))

    run_etl(config, processors)


if __name__ == "__main__":
    main()
