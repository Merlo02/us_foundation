#!/usr/bin/env python3
"""Debug runner: train the supervisor's US_CNN_Class on the SAME dataloader
   produced by DownstreamDataModule, to compare against the transformer."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
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
from model.us_CNN import US_CNN_Class  # supervisor's CNN, untouched copy

log = logging.getLogger(__name__)

# Key for the integer class label in the batch produced by DownstreamDataModule.
# Change here if your DataModule uses another key (e.g. "target", "y").
LABEL_KEY = "label"


# ── YAML composition / overrides — identical to run_downstream.py ────────
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
            raise FileNotFoundError(f"Cannot resolve defaults entry {d!r} from {path}")
        _deep_update(merged, _load_composed_yaml(base_path))
    _deep_update(merged, raw)
    return merged


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train supervisor's US_CNN on my dataloader")
    p.add_argument("--config", type=str, required=True, help="Experiment YAML")
    p.add_argument("--override", type=str, nargs="*", default=[])
    p.add_argument("--ckpt-path", type=str, default=None)
    return p.parse_args()


def _maybe_hide_cuda_for_cpu_training(train_cfg: dict) -> None:
    if str(train_cfg.get("accelerator", "gpu")).strip().lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ── Datamodule: identical to run_downstream.py ───────────────────────────
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
        apply_interpolate=bool(data_cfg.get("apply_interpolate", False)),
        target_length=data_cfg.get("target_length"),
        normalization_type=str(data_cfg.get("normalization_type", "none")),
        norm_eps_z=float(data_cfg.get("norm_eps_z", 1e-6)),
        norm_eps_mm=float(data_cfg.get("norm_eps_mm", 1e-10)),
    )


# ── LightningModule wrapper around the UNTOUCHED supervisor's CNN ────────
class CNNLightning(pl.LightningModule):
    """Mirrors training/train_utils.py::train_loop in the supervisor's repo:
    CrossEntropyLoss + Adam, no scheduler. Only purpose is to feed
    US_CNN_Class with the batches produced by DownstreamDataModule."""

    def __init__(
        self,
        *,
        num_transducers: int,
        num_classes: int,
        us_window_size: int,
        filters,
        kernels,
        max_pools,
        dropout_rate: float,
        head_hidden_mult: float,
        lr: float,
        weight_decay: float,
        betas,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = US_CNN_Class(
            num_transducers=num_transducers,
            num_classes=num_classes,
            us_window_size=us_window_size,
            filters=tuple(filters),
            kernels=tuple(tuple(k) for k in kernels),
            max_pools=tuple(tuple(p) for p in max_pools),
            dropout_rate=float(dropout_rate),
            head_hidden_mult=float(head_hidden_mult),
        )
        self.criterion = nn.CrossEntropyLoss()
        from torchmetrics.classification import MulticlassAccuracy
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch, stage: str):
        x = batch["signal"]                  # (B, C, T)
        y = batch[LABEL_KEY].long()          # (B,)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        acc = {"train": self.train_acc, "val": self.val_acc, "test": self.test_acc}[stage]
        acc.update(logits.detach(), y.detach())
        self.log(f"{stage}/loss", loss, prog_bar=True,
                 on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        self.log(f"{stage}/acc", acc, prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _):       return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
            betas=tuple(self.hparams.betas),
        )


def _build_model(cfg: dict) -> CNNLightning:
    m = cfg["model"]
    t = cfg["train"]
    return CNNLightning(
        num_transducers=int(m["num_channels"]),
        num_classes=int(m["num_classes"]),
        us_window_size=int(m["us_window_size"]),
        filters=m["filters"],
        kernels=m["kernels"],
        max_pools=m["max_pools"],
        dropout_rate=float(m.get("dropout_rate", 0.05)),
        head_hidden_mult=float(m.get("head_hidden_mult", 0.5)),
        lr=float(t.get("lr", 1e-3)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        betas=tuple(t.get("betas", (0.9, 0.999))),
    )


def _build_loggers(cfg: dict, run_dir: Path) -> list[Any]:
    t = cfg["train"]
    loggers: list[Any] = [CSVLogger(str(run_dir), name="lightning_logs")]
    wb_cfg = t.get("wandb") or {}
    if not wb_cfg.get("enabled", False):
        return loggers
    import importlib.util
    if importlib.util.find_spec("wandb") is None:
        raise ImportError("train.wandb.enabled is true but wandb is not installed.")
    wb_kwargs: dict[str, Any] = {
        "project": str(wb_cfg.get("project", "us_foundation_downstream")),
        "name": str(wb_cfg.get("name") or t.get("run_name", "run")),
        "offline": bool(wb_cfg.get("offline", True)),
        "save_dir": str(run_dir),
        "log_model": False,
    }
    if wb_cfg.get("entity"):
        wb_kwargs["entity"] = str(wb_cfg["entity"])
    loggers.append(WandbLogger(**wb_kwargs))
    return loggers


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
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

    monitor = str(t.get("monitor", "val/acc"))
    monitor_mode = str(t.get("monitor_mode", "max"))
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            filename="{epoch:03d}-{" + monitor + ":.4f}",
            monitor=monitor, mode=monitor_mode,
            save_top_k=int(t.get("save_top_k", 1)),
            save_last=True, auto_insert_metric_name=False,
        ),
    ]

    devices = int(t.get("devices", 1))
    num_nodes = int(t.get("num_nodes", 1))
    strategy = DDPStrategy(find_unused_parameters=False) if (devices > 1 or num_nodes > 1) else "auto"
    torch.set_float32_matmul_precision("high")

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        max_epochs=int(t["max_epochs"]),
        devices=devices, num_nodes=num_nodes,
        accelerator=str(t.get("accelerator", "gpu")),
        strategy=strategy,
        precision=t.get("precision", 32),
        gradient_clip_val=float(t.get("gradient_clip_val", 0.0)),
        accumulate_grad_batches=int(t.get("accumulate_grad_batches", 1)),
        log_every_n_steps=int(t.get("log_every_n_steps", 50)),
        callbacks=callbacks,
        logger=_build_loggers(cfg, run_dir),
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    if datamodule.test_dataloader() is not None:
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()