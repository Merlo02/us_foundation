"""Downstream LightningModule: encoder wrapper + swappable head.

Supports the two evaluation protocols required for MAE-style transfer
learning:

- **Linear probing** (``freeze_encoder=True``): the encoder is set to
  ``.eval()`` and its parameters get ``requires_grad=False``; the forward
  through the encoder runs under ``torch.no_grad()`` so activations are
  not retained. Only the head is trained.
- **Fine-tuning** (``freeze_encoder=False``): everything is trainable.
  Optional layerwise LR decay (He et al. 2021 recipe) can be enabled by
  setting ``layerwise_lr_decay`` to e.g. ``0.65``-``0.75``.

The same code path covers *pre-trained vs from-scratch* — set
``pretrained_ckpt`` to a ``.ckpt`` path to load encoder weights from a
:class:`~model.us_mae.UltrasonicMAE` checkpoint (decoder + loss state is
dropped); set it to ``None`` for a random initialisation.
"""
from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    import lightning.pytorch as pl  # type: ignore[no-redef]

try:
    import torchmetrics
except ImportError:  # pragma: no cover
    torchmetrics = None  # type: ignore[assignment]

from schedulers import CosineLRSchedulerWrapper
from .encoder_wrapper import UltrasonicEncoderWrapper
from .heads import build_head

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Pretrained checkpoint loading
# ----------------------------------------------------------------------

_ENCODER_PREFIXES = ("tokenizer.", "encoder.")


def _filter_encoder_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep only encoder-side keys; drop decoder / loss / pretraining-only state."""
    return {
        k: v for k, v in state_dict.items()
        if any(k.startswith(p) for p in _ENCODER_PREFIXES)
    }


# ----------------------------------------------------------------------
# Param-group builder (shared between linear-probe / FT / FT+layerwise)
# ----------------------------------------------------------------------

def _split_decay_no_decay(
    named_params,
    weight_decay: float,
) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if (
            p.ndim <= 1
            or name.endswith(".bias")
            or "mask_token" in name
            or "pad_token" in name
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ----------------------------------------------------------------------
# LightningModule
# ----------------------------------------------------------------------

class UltrasonicDownstream(pl.LightningModule):
    """Downstream task (classification / regression) on top of the MAE encoder."""

    def __init__(
        self,
        # Tokenizer / encoder hyperparams (must match the pretrained ckpt if loading)
        window_sizes: tuple[int, ...] = (8, 16, 32),
        target_patch_mm: float = 0.6,
        tokenizer_type: str = "mlp",
        cnn_config: Optional[dict] = None,
        target_patches: Optional[int] = None,
        embed_dim: int = 256,
        encoder_depth: int = 8,
        encoder_heads: int = 8,
        encoder_mlp_ratio: float = 4.0,
        use_ct_rope: bool = True,
        ct_rope_base: float = 10_000.0,
        rope_max_seq_len: int = 512,
        dropout: float = 0.0,
        # Downstream-specific
        num_channels: int = 1,
        pooling_type: str = "mean",
        head_type: str = "classification",
        num_classes: Optional[int] = None,
        num_outputs: Optional[int] = None,
        head_hidden_dim: Optional[int] = None,
        head_dropout: float = 0.0,
        head_num_layers: int = 1,
        # Training mode
        pretrained_ckpt: Optional[str] = None,
        freeze_encoder: bool = False,
        layerwise_lr_decay: Optional[float] = None,
        # Optimiser / scheduler (same conventions as UltrasonicMAE)
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        warmup_lr_init: float = 1e-6,
        max_epochs: int = 100,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder_wrapper = UltrasonicEncoderWrapper(
            window_sizes=window_sizes,
            target_patch_mm=target_patch_mm,
            tokenizer_type=tokenizer_type,
            cnn_config=cnn_config,
            target_patches=target_patches,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            encoder_heads=encoder_heads,
            encoder_mlp_ratio=encoder_mlp_ratio,
            use_ct_rope=use_ct_rope,
            ct_rope_base=ct_rope_base,
            rope_max_seq_len=rope_max_seq_len,
            dropout=dropout,
            pooling_type=pooling_type,
        )

        head_in = self.encoder_wrapper.out_dim * int(num_channels)
        self.head = build_head(
            head_type=head_type,
            in_dim=head_in,
            num_classes=num_classes,
            num_outputs=num_outputs,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            num_layers=head_num_layers,
        )

        ht = str(head_type).lower()
        if ht == "classification":
            self.criterion: nn.Module = nn.CrossEntropyLoss()
        elif ht == "regression":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown head_type {head_type!r}")
        self._head_type = ht

        # Metrics — cloned per stage so train/val/test don't share state.
        self._build_metrics()

        if pretrained_ckpt is not None:
            self._load_pretrained_encoder(pretrained_ckpt)

        self._configure_freezing()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _build_metrics(self) -> None:
        if torchmetrics is None:
            log.warning("torchmetrics not available — metrics disabled.")
            self._metric_keys: list[str] = []
            return

        if self._head_type == "classification":
            nc = int(self.hparams.num_classes)
            task = "binary" if nc == 2 else "multiclass"
            kwargs = {"task": task}
            if task == "multiclass":
                kwargs["num_classes"] = nc
            base = torchmetrics.Accuracy(**kwargs)
            self.train_acc = base.clone()
            self.val_acc = base.clone()
            self.test_acc = base.clone()
            self._metric_keys = ["acc"]
        else:  # regression
            self.train_mae = torchmetrics.MeanAbsoluteError()
            self.val_mae = torchmetrics.MeanAbsoluteError()
            self.test_mae = torchmetrics.MeanAbsoluteError()
            self.train_mse = torchmetrics.MeanSquaredError()
            self.val_mse = torchmetrics.MeanSquaredError()
            self.test_mse = torchmetrics.MeanSquaredError()
            self._metric_keys = ["mae", "mse"]

    # ------------------------------------------------------------------
    # Pretrained-encoder loading
    # ------------------------------------------------------------------
    def _load_pretrained_encoder(self, ckpt_path: str) -> None:
        log.info("Loading pretrained encoder weights from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt  # raw state_dict
        encoder_sd = _filter_encoder_state(sd) #keep only encoder-prefixed keys, drop decoder keys
        if not encoder_sd:
            raise RuntimeError(
                f"No encoder-prefixed keys found in checkpoint {ckpt_path!r}. "
                f"Expected keys starting with {_ENCODER_PREFIXES}."
            )

        missing, unexpected = self.encoder_wrapper.load_state_dict(
            encoder_sd, strict=False,
        )
        log.info(
            "Pretrained load — kept %d encoder tensors; missing=%d, unexpected=%d.",
            len(encoder_sd), len(missing), len(unexpected),
        )
        if missing:
            log.info("  missing (first 10): %s", list(missing)[:10])
        if unexpected:
            log.info("  unexpected (first 10): %s", list(unexpected)[:10])

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------
    def _configure_freezing(self) -> None:
        if not self.hparams.freeze_encoder:
            return
        for p in self.encoder_wrapper.parameters():
            p.requires_grad = False
        # Forced eval-mode in linear probe is enforced by ``train()`` below.
        self.encoder_wrapper.eval()

    def train(self, mode: bool = True) -> "UltrasonicDownstream":
        """Override so the encoder stays in eval-mode under linear probe."""
        super().train(mode)
        if self.hparams.freeze_encoder:
            self.encoder_wrapper.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> torch.Tensor:
        ctx = torch.no_grad() if self.hparams.freeze_encoder else nullcontext()
        with ctx:
            feats = self.encoder_wrapper(batch)              # (B, C, E)
        feats = feats.flatten(1)                             # (B, C*E)
        return self.head(feats)                              # logits / values

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        logits = self(batch)
        target = batch["label"]

        if self._head_type == "classification":
            target = target.long()
            loss = self.criterion(logits, target)
        else:
            if logits.size(-1) == 1 and target.ndim == 1:
                target = target.float().unsqueeze(-1)
            else:
                target = target.float()
            loss = self.criterion(logits, target)

        on_step = stage == "train"
        sd = stage != "train"
        bs = (
            batch["signal"].size(0)
            if batch["signal"].ndim >= 1
            else 1
        )
        self.log(
            f"{stage}/loss", loss,
            prog_bar=True, sync_dist=sd, on_step=on_step, on_epoch=True,
            batch_size=bs,
        )

        # Metrics
        if torchmetrics is None:
            return loss
        if self._head_type == "classification":
            if logits.size(-1) == 1:
                preds = (logits.squeeze(-1) > 0).long()
            else:
                preds = logits.argmax(dim=-1)
            metric = getattr(self, f"{stage}_acc")
            metric(preds, target)
            self.log(
                f"{stage}/acc", metric,
                prog_bar=True, sync_dist=sd, on_step=on_step, on_epoch=True,
                batch_size=bs,
            )
        else:
            mae = getattr(self, f"{stage}_mae")
            mse = getattr(self, f"{stage}_mse")
            mae(logits, target)
            mse(logits, target)
            self.log(f"{stage}/mae", mae, prog_bar=True, sync_dist=sd, on_step=on_step, on_epoch=True, batch_size=bs)
            self.log(f"{stage}/mse", mse, prog_bar=False, sync_dist=sd, on_step=on_step, on_epoch=True, batch_size=bs)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    # ------------------------------------------------------------------
    # Optimiser + Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self) -> Any:
        wd = float(self.hparams.weight_decay)
        base_lr = float(self.hparams.lr)
        layerwise = self.hparams.layerwise_lr_decay
        layerwise = float(layerwise) if layerwise is not None else None

        if self.hparams.freeze_encoder:
            # Linear probe: head only.
            param_groups = _split_decay_no_decay(self.head.named_parameters(), wd)
            optim = torch.optim.AdamW(
                param_groups, lr=base_lr, betas=tuple(self.hparams.betas),
            )
        elif layerwise is None:
            # Fine-tune, uniform LR.
            param_groups = _split_decay_no_decay(self.named_parameters(), wd)
            optim = torch.optim.AdamW(
                param_groups, lr=base_lr, betas=tuple(self.hparams.betas),
            )
        else:
            # Fine-tune with layerwise LR decay (MAE recipe).
            optim = torch.optim.AdamW(
                self._build_layerwise_param_groups(base_lr, wd, layerwise),
                betas=tuple(self.hparams.betas),
            )

        total_steps = self.trainer.estimated_stepping_batches
        if not (
            isinstance(total_steps, (int, float))
            and total_steps > 0
            and total_steps != float("inf")
        ):
            total_steps = max(int(self.hparams.max_epochs), 1)

        scheduler = CosineLRSchedulerWrapper(
            optimizer=optim,
            total_training_opt_steps=int(total_steps),
            trainer=self.trainer,
            warmup_epochs=self.hparams.warmup_epochs,
            min_lr=self.hparams.min_lr,
            warmup_lr_init=self.hparams.warmup_lr_init,
            t_in_epochs=False,
        )

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def lr_scheduler_step(self, scheduler: Any, metric: Any) -> None:
        scheduler.step_update(num_updates=self.global_step)

    # ------------------------------------------------------------------
    # Layerwise LR decay (MAE He et al. 2021 fine-tune recipe)
    # ------------------------------------------------------------------
    def _layer_id_for(self, name: str, num_blocks: int) -> int:
        """Map parameter name -> layer index in ``[0, num_blocks + 1]``.

        - tokenizer / rotary / encoder.pad_token   -> 0 (most decayed)
        - encoder.blocks.{i}.*                     -> i + 1
        - encoder.norm.*                           -> num_blocks + 1
        - head.*                                   -> num_blocks + 1 (top)
        """
        if name.startswith("head."):
            return num_blocks + 1
        if name.startswith("encoder_wrapper.encoder.blocks."):
            idx = int(name.split(".")[3])
            return idx + 1
        if name.startswith("encoder_wrapper.encoder.norm."):
            return num_blocks + 1
        # tokenizer, rotary_encoder, encoder.pad_token, etc.
        return 0

    def _build_layerwise_param_groups(
        self, base_lr: float, weight_decay: float, decay: float,
    ) -> list[dict[str, Any]]:
        num_blocks = int(self.hparams.encoder_depth)
        # scale[layer] = decay ** (num_blocks + 1 - layer); top layer -> 1.0
        scales = [decay ** (num_blocks + 1 - lid) for lid in range(num_blocks + 2)]

        groups: dict[tuple[int, bool], dict[str, Any]] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            lid = self._layer_id_for(name, num_blocks)
            no_decay = (
                p.ndim <= 1
                or name.endswith(".bias")
                or "mask_token" in name
                or "pad_token" in name
            )
            key = (lid, no_decay)
            if key not in groups:
                groups[key] = {
                    "params": [],
                    "lr": base_lr * scales[lid],
                    "weight_decay": 0.0 if no_decay else weight_decay,
                    "layer_id": lid,
                    "no_decay": no_decay,
                }
            groups[key]["params"].append(p)
        return list(groups.values())
