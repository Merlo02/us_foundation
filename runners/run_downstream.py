#!/usr/bin/env python3
"""PyTorch Lightning entry point for downstream training (classification / regression).

Mirrors :mod:`runners.run_train` but builds an
:class:`~model.UltrasonicDownstream` LightningModule on top of the
encoder, with a labeled DataModule
(:class:`~data.downstream_datamodule.DownstreamDataModule`) reading a
single ``all.h5`` produced by :mod:`etl_downstream`.

Usage (from ``us_foundation/``)::

    # Linear probe on a pretrained encoder
    python -m runners.run_downstream \\
        --config configs/model/experiments/cls_linear_probe.yaml

    # Fine-tune
    python -m runners.run_downstream \\
        --config configs/model/experiments/cls_finetune.yaml \\
        --override model.layerwise_lr_decay=0.75

    # From-scratch (same script, just a different config)
    python -m runners.run_downstream \\
        --config configs/model/experiments/cls_scratch.yaml

Config composition (``defaults: [base, base_downstream]`` etc.) is the
same hand-rolled merger used by :mod:`runners.run_train`.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, WandbLogger
    from pytorch_lightning.strategies import DDPStrategy
except ImportError:  # pragma: no cover
    import lightning.pytorch as pl  # type: ignore[no-redef]
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint  # type: ignore
    from lightning.pytorch.loggers import CSVLogger, WandbLogger  # type: ignore
    from lightning.pytorch.strategies import DDPStrategy  # type: ignore

from data.downstream_datamodule import DownstreamDataModule
from model import UltrasonicDownstream

log = logging.getLogger(__name__)


# Encoder hyper-parameters that MUST match the pretrained checkpoint.
# When ``model.pretrained_config`` is set, these fields are pulled from
# that file (overriding base.yaml). Downstream-only fields (head, channels,
# pooling, freeze flags, …) are not touched.
_ENCODER_FIELDS = (
    "window_sizes",
    "target_patch_mm",
    "tokenizer_type",
    "cnn_config",
    "embed_dim",
    "encoder_depth",
    "encoder_heads",
    "encoder_mlp_ratio",
    "use_ct_rope",
    "ct_rope_base",
    "rope_max_seq_len",
    "dropout",
)


def _sync_encoder_from_pretrained(cfg: dict) -> None:
    """If ``model.pretrained_config`` is set, copy encoder fields from it.

    Raises if the file is missing. Fields absent from the pretrained config
    are left untouched (so base.yaml defaults still apply for them).
    """
    m = cfg.get("model", {})
    pretrained_cfg_path = m.get("pretrained_config")
    if not pretrained_cfg_path:
        return
    p = Path(pretrained_cfg_path)
    if not p.exists():
        raise FileNotFoundError(
            f"model.pretrained_config does not exist: {p}",
        )
    with p.open(encoding="utf-8") as f:
        pre = yaml.safe_load(f) or {}
    pre_model = pre.get("model", {}) or {}
    copied: dict[str, Any] = {}
    for field in _ENCODER_FIELDS:
        if field in pre_model:
            copied[field] = pre_model[field]
    m.update(copied)
    log.info(
        "Loaded encoder hyper-parameters from pretrained config %s: %s",
        p, sorted(copied),
    )


# ----------------------------------------------------------------------
# YAML composition (Hydra-like ``defaults: [...]``) — identical to run_train.
# ----------------------------------------------------------------------

def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_composed_yaml(path: Path) -> dict:
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
    p = argparse.ArgumentParser(description="Train a downstream task on the US encoder")
    p.add_argument("--config", type=str, required=True, help="Experiment YAML")
    p.add_argument(
        "--override", type=str, nargs="*", default=[],
        help="Dot-path overrides, e.g. train.max_epochs=50",
    )
    p.add_argument(
        "--ckpt-path", type=str, default=None,
        help="Resume from this downstream-task checkpoint",
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
    accel = str(train_cfg.get("accelerator", "gpu")).strip().lower()
    if accel != "cpu":
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    log.info(
        "train.accelerator=cpu: CUDA_VISIBLE_DEVICES cleared so CUDA is not initialized.",
    )


# ----------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------

def _build_datamodule(cfg: dict) -> pl.LightningDataModule:
    data_cfg = cfg["data"]
    _test_id_raw = data_cfg.get("test_id")
    _val_id_raw = data_cfg.get("val_id")
    return DownstreamDataModule(
        h5_path=data_cfg["h5_path"],
        split_mode=str(data_cfg["split_mode"]),
        test_id=int(_test_id_raw) if _test_id_raw is not None else None,
        test_ratio=float(data_cfg.get("test_ratio", 0.2)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        val_id=int(_val_id_raw) if _val_id_raw is not None else None,
        grouped_val=bool(data_cfg.get("grouped_val", True)),
        seed=int(data_cfg.get("seed", cfg.get("train", {}).get("seed", 42))),
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg.get("num_workers", 4)),
        window_sizes=tuple(cfg["model"]["window_sizes"]),
        target_patch_mm=float(cfg["model"]["target_patch_mm"]),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        shuffle_train=bool(data_cfg.get("shuffle_train", True)),
        signal_trace_enabled=bool(data_cfg.get("signal_trace_enabled", False)),
        signal_trace_dir=data_cfg.get("signal_trace_dir", "signal_traces"),
        test_every_epoch=bool(cfg.get("train", {}).get("test_every_epoch", False)),
        apply_interpolate=bool(data_cfg.get("apply_interpolate", False)),
        target_length=data_cfg.get("target_length"),
        normalization_type=str(data_cfg.get("normalization_type", "none")),
        norm_eps_z=float(data_cfg.get("norm_eps_z", 1e-6)),
        norm_eps_mm=float(data_cfg.get("norm_eps_mm", 1e-10)),
    )


def _build_model(cfg: dict) -> UltrasonicDownstream:
    m = cfg["model"]
    t = cfg["train"]
    target_patches = cfg.get("data", {}).get("target_patches", None)
    return UltrasonicDownstream(
        window_sizes=tuple(m["window_sizes"]),
        target_patch_mm=float(m["target_patch_mm"]),
        tokenizer_type=m.get("tokenizer_type", "mlp"),
        cnn_config=m.get("cnn_config"),
        target_patches=target_patches,
        embed_dim=int(m["embed_dim"]),
        encoder_depth=int(m["encoder_depth"]),
        encoder_heads=int(m["encoder_heads"]),
        encoder_mlp_ratio=float(m["encoder_mlp_ratio"]),
        use_ct_rope=bool(m.get("use_ct_rope", True)),
        ct_rope_base=float(m.get("ct_rope_base", 10_000.0)),
        rope_max_seq_len=int(m.get("rope_max_seq_len", 512)),
        dropout=float(m.get("dropout", 0.0)),
        num_channels=int(m["num_channels"]),
        pooling_type=str(m.get("pooling_type", "mean")),
        head_type=str(m.get("head_type", "classification")),
        num_classes=m.get("num_classes"),
        num_outputs=m.get("num_outputs"),
        head_hidden_dim=m.get("head_hidden_dim"),
        head_dropout=float(m.get("head_dropout", 0.0)),
        head_num_layers=int(m.get("head_num_layers", 1)),
        pretrained_ckpt=m.get("pretrained_ckpt"),
        freeze_encoder=bool(m.get("freeze_encoder", False)),
        layerwise_lr_decay=m.get("layerwise_lr_decay"),
        lr=float(t["lr"]),
        weight_decay=float(t["weight_decay"]),
        betas=tuple(t.get("betas", (0.9, 0.95))),
        warmup_epochs=int(t["warmup_epochs"]),
        min_lr=float(t.get("min_lr", 1e-6)),
        warmup_lr_init=float(t.get("warmup_lr_init", 1e-6)),
        max_epochs=int(t["max_epochs"]),
        seed=int(t.get("seed", 42)),
    )


def _build_loggers(cfg: dict, run_dir: Path) -> list[Any]:
    t = cfg["train"]
    loggers: list[Any] = [CSVLogger(str(run_dir), name="lightning_logs")]
    wb_cfg = t.get("wandb") or {}
    if not wb_cfg.get("enabled", False):
        return loggers
    import importlib.util

    if importlib.util.find_spec("wandb") is None:
        raise ImportError(
            "train.wandb.enabled is true but `wandb` is not installed. "
            "Install with: pip install wandb",
        )

    offline = bool(wb_cfg.get("offline", True))
    wb_kwargs: dict[str, Any] = {
        "project": str(wb_cfg.get("project", "us_foundation_downstream")),
        "name": str(wb_cfg.get("name") or t.get("run_name", "run")),
        "offline": offline,
        "save_dir": str(run_dir),
        "log_model": False,
    }
    entity = wb_cfg.get("entity")
    if entity:
        wb_kwargs["entity"] = str(entity)
    loggers.append(WandbLogger(**wb_kwargs))
    return loggers


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
    _sync_encoder_from_pretrained(cfg)

    t = cfg["train"]
    _maybe_hide_cuda_for_cpu_training(t)
    pl.seed_everything(int(t.get("seed", 42)), workers=True)

    run_dir = Path(t["output_dir"]) / t["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    datamodule = _build_datamodule(cfg)
    model = _build_model(cfg)

    if bool(t.get("compile", False)):
        log.info(
            "Compiling model with torch.compile (mode=%s)",
            t.get("compile_mode", "default"),
        )
        model = torch.compile(model, mode=str(t.get("compile_mode", "default")))

    monitor = str(t.get("monitor", "val/acc"))
    monitor_mode = str(t.get("monitor_mode", "max"))
    ckpt_every_n = int(t.get("checkpoint_every_n_epochs", 0))

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            filename="{epoch:03d}-{" + monitor + ":.4f}",
            monitor=monitor,
            mode=monitor_mode,
            save_top_k=int(t.get("save_top_k", 3)),
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]
    if ckpt_every_n > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(run_dir / "checkpoints"),
                filename="periodic-{epoch:03d}",
                every_n_epochs=ckpt_every_n,
                save_top_k=-1,
                save_last=False,
            )
        )

    devices = int(t.get("devices", 1))
    num_nodes = int(t.get("num_nodes", 1))
    strategy: Any
    if devices > 1 or num_nodes > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = "auto"

    torch.set_float32_matmul_precision("high")

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
        logger=_build_loggers(cfg, run_dir),
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)

    # The single-corpus DataModule always builds a test split via its
    # splitter (LeaveOne*Out / RandomSplit). Run trainer.test() if the
    # test split is non-empty (test_dataloader returns a DataLoader).
    if datamodule.test_dataloader() is not None:
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
