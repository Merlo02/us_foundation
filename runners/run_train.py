#!/usr/bin/env python3
"""PyTorch Lightning training entry for the ultrasound foundation MAE.

Usage (from ``us_foundation/``)::

    python -m runners.run_train \\
        --config configs/model/experiments/exp_A_mode2_multi.yaml

Composition works via a lightweight ``defaults: [base]`` key inside each
experiment YAML. The base file is resolved relative to the experiment
file's directory parent (``configs/model/base.yaml``).

The script is DDP-ready: set ``train.devices`` and ``train.num_nodes`` in
the YAML (or override on the CLI) and submit through ``srun``/``sbatch``
on CINECA Leonardo.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    import lightning.pytorch as pl  # type: ignore[no-redef]

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

from data import HDF5DataModule, WebDatasetDataModule
from model import UltrasonicMAE

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# YAML composition (Hydra-like ``defaults: [base]``)
# ----------------------------------------------------------------------

def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_composed_yaml(path: Path) -> dict:
    """Load *path* and recursively merge any ``defaults: [<file>]`` entries.

    ``<file>`` is resolved relative to *path* 's parent's parent (so the
    base file lives at ``configs/model/base.yaml`` while experiment files
    live at ``configs/model/experiments/*.yaml``).
    """
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    defaults = raw.pop("defaults", []) or []
    merged: dict = {}
    for d in defaults:
        base_path = path.parent.parent / f"{d}.yaml"
        if not base_path.exists():
            base_path = path.parent / f"{d}.yaml"
        if not base_path.exists():
            raise FileNotFoundError(
                f"Cannot resolve defaults entry {d!r} from {path}"
            )
        _deep_update(merged, _load_composed_yaml(base_path))
    _deep_update(merged, raw)
    return merged


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the ultrasound MAE")
    p.add_argument("--config", type=str, required=True, help="Experiment YAML")
    p.add_argument(
        "--override", type=str, nargs="*", default=[],
        help="Dot-path overrides, e.g. train.max_epochs=50",
    )
    p.add_argument(
        "--ckpt-path", type=str, default=None,
        help="Resume from this checkpoint path",
    )
    return p.parse_args()


def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    for raw in overrides:
        if "=" not in raw:
            log.warning("Skipping malformed override %r", raw)
            continue
        key, value = raw.split("=", 1)
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        try:
            parsed: Any = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = value
        d[parts[-1]] = parsed


def _maybe_hide_cuda_for_cpu_training(train_cfg: dict) -> None:
    """If training on CPU, hide GPUs so Lightning never initializes CUDA.

    Otherwise ``Trainer.fit()`` → ``isolate_rng()`` may snapshot CUDA RNG states
    whenever ``torch.cuda.is_available()`` is true, which crashes on nodes with
    a broken or too-old NVIDIA driver vs the installed PyTorch CUDA build.
    """
    accel = str(train_cfg.get("accelerator", "gpu")).strip().lower()
    if accel != "cpu":
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    log.info(
        "train.accelerator=cpu: CUDA_VISIBLE_DEVICES cleared so CUDA is not initialized.",
    )


# ----------------------------------------------------------------------
# DataModule factory
# ----------------------------------------------------------------------

def _build_datamodule(cfg: dict) -> pl.LightningDataModule:
    data_cfg = cfg["data"]
    fmt = data_cfg.get("format", "hdf5")
    window_sizes = tuple(cfg["model"]["window_sizes"])
    target_patch_mm = float(cfg["model"]["target_patch_mm"])

    target_patches = data_cfg.get("target_patches", None)
    min_valid_patches = int(data_cfg.get("min_valid_patches", 1))

    if fmt == "hdf5":
        return HDF5DataModule(
            hdf5_dir=data_cfg["hdf5_dir"],
            batch_size=int(data_cfg["batch_size"]),
            num_workers=int(data_cfg["num_workers"]),
            window_sizes=window_sizes,
            target_patch_mm=target_patch_mm,
            target_patches=target_patches,
            min_valid_patches=min_valid_patches,
            sampling_strategy=data_cfg.get("sampling_strategy", "naive"),
            epoch_k=int(data_cfg.get("epoch_k", 500_000)),
            threshold_ratio=float(data_cfg.get("threshold_ratio", 0.1)),
            lg_dataset_name=data_cfg.get(
                "lg_dataset_name", "lateral_gastrocnemius_verasonics",
            ),
            seed=int(cfg.get("train", {}).get("seed", 42)),
            pin_memory=bool(data_cfg.get("pin_memory", True)),
            persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        )
    if fmt == "webdataset":
        return WebDatasetDataModule(
            shard_root=data_cfg["wds_root"],
            batch_size=int(data_cfg["batch_size"]),
            num_workers=int(data_cfg["num_workers"]),
            samples_per_shard=int(data_cfg.get("samples_per_shard", 1024)),
            window_sizes=window_sizes,
            target_patch_mm=target_patch_mm,
            target_patches=target_patches,
            min_valid_patches=min_valid_patches,
            pin_memory=bool(data_cfg.get("pin_memory", True)),
            persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        )
    raise ValueError(f"Unknown data.format {fmt!r}")


# ----------------------------------------------------------------------
# Model factory
# ----------------------------------------------------------------------

def _build_model(cfg: dict) -> UltrasonicMAE:
    m = cfg["model"]
    t = cfg["train"]
    # Fixed-S must match the DataModule setting so the tokenizer can emit a
    # coherent (B, target_patches, E) tensor.
    target_patches = cfg.get("data", {}).get("target_patches", None)
    dp = t.get("debug_pipeline") or {}
    return UltrasonicMAE(
        window_sizes=tuple(m["window_sizes"]),
        target_patch_mm=float(m["target_patch_mm"]),
        tokenizer_type=m.get("tokenizer_type", "mlp"),
        cnn_config=m.get("cnn_config"),
        target_patches=target_patches,
        embed_dim=int(m["embed_dim"]),
        encoder_depth=int(m["encoder_depth"]),
        encoder_heads=int(m["encoder_heads"]),
        encoder_mlp_ratio=float(m["encoder_mlp_ratio"]),
        decoder_dim=int(m["decoder_dim"]),
        decoder_depth=int(m["decoder_depth"]),
        decoder_heads=int(m["decoder_heads"]),
        decoder_mlp_ratio=float(m["decoder_mlp_ratio"]),
        masking_ratio=float(m["masking_ratio"]),
        use_ct_rope=bool(m.get("use_ct_rope", True)),
        ct_rope_base=float(m.get("ct_rope_base", 10_000.0)),
        dropout=float(m.get("dropout", 0.0)),
        lr=float(t["lr"]),
        weight_decay=float(t["weight_decay"]),
        betas=tuple(t.get("betas", (0.9, 0.95))),
        warmup_epochs=int(t["warmup_epochs"]),
        max_epochs=int(t["max_epochs"]),
        seed=int(t.get("seed", 42)),
        debug_pipeline_enabled=bool(dp.get("enabled", False)),
        debug_max_samples_per_base_dataset=int(dp.get("max_samples_per_base_dataset", 2)),
        debug_log_interval_batches=int(dp.get("log_interval_batches", 1)),
        debug_midpoint_log_k=int(dp.get("midpoint_log_k", 5)),
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = _load_composed_yaml(cfg_path)
    _apply_overrides(cfg, args.override)

    t = cfg["train"]
    _maybe_hide_cuda_for_cpu_training(t)
    pl.seed_everything(int(t.get("seed", 42)), workers=True)

    run_dir = Path(t["output_dir"]) / t["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    datamodule = _build_datamodule(cfg)
    model = _build_model(cfg)

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            filename="{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
    ]

    devices = int(t.get("devices", 1))
    num_nodes = int(t.get("num_nodes", 1))
    strategy: Any
    if devices > 1 or num_nodes > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = "auto"

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        max_epochs=int(t["max_epochs"]),
        devices=devices,
        num_nodes=num_nodes,
        accelerator=str(t.get("accelerator", "gpu")),
        strategy=strategy,
        precision=t.get("precision", "bf16-mixed"),
        gradient_clip_val=float(t.get("gradient_clip_val", 1.0)),
        accumulate_grad_batches=int(t.get("accumulate_grad_batches", 1)),
        log_every_n_steps=int(t.get("log_every_n_steps", 50)),
        val_check_interval=float(t.get("val_check_interval", 1.0)),
        callbacks=callbacks,
        logger=CSVLogger(str(run_dir), name="lightning_logs"),
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
