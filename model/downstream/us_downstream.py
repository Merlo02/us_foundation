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

import numpy as np
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

# Upper bound on the number of embeddings fed to sklearn's TSNE per epoch.
# t-SNE is O(N log N) per iteration but still seconds-scale only for a few
# thousand points; above this cap we deterministically subsample (seeded by
# ``hparams.seed``) so the per-epoch plot stays cheap regardless of val/test
# split size.
_TSNE_MAX_POINTS = 2000


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
        # Per-test-epoch t-SNE of the pooled encoder embeddings (classification
        # only). Default True keeps the legacy behaviour; set False to skip the
        # sklearn projection + wandb.Image at test time (e.g. during sweeps).
        tsne_enabled: bool = True,
        # Training mode
        pretrained_ckpt: Optional[str] = None,
        freeze_encoder: bool = False,
        layerwise_lr_decay: Optional[float] = None,
        # LoRA — passed as a small dict so the YAML section maps 1:1.
        # ``None`` or ``{"enabled": false}`` keeps the legacy behaviour.
        lora: Optional[dict] = None,
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

        # LoRA must wrap the encoder AFTER the pretrained weights are
        # loaded (the original state-dict keys assume the un-wrapped
        # module tree) and BEFORE _configure_freezing so a user freeze
        # flag — if set — still applies on top of the wrapped module.
        self._lora_enabled = bool((lora or {}).get("enabled", False))
        if self._lora_enabled:
            self._wrap_encoder_with_lora(lora or {})

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
            base_kwargs: dict[str, Any] = {"task": task}
            if task == "multiclass":
                base_kwargs["num_classes"] = nc

            acc_base = torchmetrics.Accuracy(**base_kwargs)
            # Macro-averaged (scalar) Precision / Recall / F1 — logged via
            # self.log every step/epoch (DDP-safe). PER-CLASS values are NOT
            # produced by ``average=None`` vector metrics: those break
            # torchmetrics' DDP state sync (states of shape [1] vs [C] across
            # ranks → ``torch.stack`` size mismatch). Instead the per-class
            # precision/recall/F1 are derived from the confusion matrix at
            # epoch-end (see _log_per_class_scores_from_cm), which syncs
            # cleanly as a single (C, C) tensor.
            avg_kwargs = dict(base_kwargs)
            if task == "multiclass":
                avg_kwargs["average"] = "macro"
            prec_base = torchmetrics.Precision(**avg_kwargs)
            rec_base = torchmetrics.Recall(**avg_kwargs)
            f1_base = torchmetrics.F1Score(**avg_kwargs)

            self.train_acc = acc_base.clone()
            self.val_acc = acc_base.clone()
            self.test_acc = acc_base.clone()
            self.train_prec = prec_base.clone()
            self.val_prec = prec_base.clone()
            self.test_prec = prec_base.clone()
            self.train_rec = rec_base.clone()
            self.val_rec = rec_base.clone()
            self.test_rec = rec_base.clone()
            self.train_f1 = f1_base.clone()
            self.val_f1 = f1_base.clone()
            self.test_f1 = f1_base.clone()

            # Confusion matrices on val/test only — plotted at epoch-end and
            # logged as a wandb.Image (matrix figures cannot go through the
            # standard self.log() path because they are 2-D tensors).
            cm_kwargs = dict(base_kwargs)
            self.val_cm = torchmetrics.ConfusionMatrix(**cm_kwargs)
            self.test_cm = torchmetrics.ConfusionMatrix(**cm_kwargs)
            self._val_cm_dirty = False
            self._test_cm_dirty = False

            # t-SNE of the pooled encoder embeddings, logged as a wandb.Image
            # exactly like the confusion matrix — but ONLY during the real
            # test phase (trainer.test()), never per validation epoch: the
            # projection is too slow to run every epoch. The pooled
            # ``(B, C*E)`` test features + labels are accumulated per test
            # step in ``_step`` and flushed (gathered across DDP ranks,
            # projected with sklearn's TSNE, plotted on rank 0) in
            # ``on_test_epoch_end``.
            self._tsne_enabled = bool(self.hparams.tsne_enabled)
            self._test_emb: list[torch.Tensor] = []
            self._test_emb_labels: list[torch.Tensor] = []

            self._metric_keys = ["acc", "prec", "rec", "f1"]
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
    # LoRA wrap (PEFT)
    # ------------------------------------------------------------------
    def _wrap_encoder_with_lora(self, lora_cfg: dict) -> None:
        """Attach PEFT LoRA adapters to ``encoder_wrapper.encoder``.

        Only the encoder transformer stack is wrapped — the tokenizer and
        pooling stay untouched. PEFT freezes the base encoder parameters
        and adds small trainable ``lora_A`` / ``lora_B`` matrices to every
        ``nn.Linear`` whose suffix matches ``target_modules``. The default
        targets ``(qkv, proj, fc1, fc2)`` cover both attention and MLP in
        every block.
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "model.lora.enabled=true requires the `peft` package "
                "(install with: pip install peft).",
            ) from exc

        if self.hparams.freeze_encoder:
            log.warning(
                "model.lora.enabled=True together with freeze_encoder=True "
                "freezes the LoRA adapters too — only the head will train. "
                "Set freeze_encoder=False to let the LoRA adapters learn.",
            )

        target_modules = (
            lora_cfg.get("target_modules")
            or ["qkv", "proj", "fc1", "fc2"]
        )
        r = int(lora_cfg.get("r", 8))
        alpha = int(lora_cfg.get("lora_alpha", 16))
        dropout = float(lora_cfg.get("lora_dropout", 0.0))
        bias = str(lora_cfg.get("bias", "none"))

        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=bias,
            target_modules=list(target_modules),
        )
        self.encoder_wrapper.encoder = get_peft_model(
            self.encoder_wrapper.encoder, peft_config,
        )
        n_total = sum(
            p.numel() for p in self.encoder_wrapper.encoder.parameters()
        )
        n_train = sum(
            p.numel() for p in self.encoder_wrapper.encoder.parameters()
            if p.requires_grad
        )
        log.info(
            "LoRA wrap — r=%d alpha=%d dropout=%g targets=%s — encoder "
            "trainable params: %d / %d (%.4f%%)",
            r, alpha, dropout, list(target_modules),
            n_train, n_total, 100.0 * n_train / max(n_total, 1),
        )

    def on_fit_end(self) -> None:
        """Persist the LoRA adapter weights alongside the full checkpoint.

        Lightning's ``ModelCheckpoint`` already saves the entire LightningModule
        (base encoder + LoRA adapters + head) — this hook additionally writes
        the *adapter only* (``adapter_config.json`` + ``adapter_model.safetensors``)
        under ``<default_root_dir>/lora_adapter/`` so the small adapter file
        can be shipped, loaded with PEFT into a fresh base encoder, or
        compared across runs without round-tripping the multi-GB checkpoint.
        """
        if not getattr(self, "_lora_enabled", False):
            return
        if self.trainer is None or int(self.trainer.global_rank) != 0:
            return
        from pathlib import Path

        adapter_dir = Path(self.trainer.default_root_dir) / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.encoder_wrapper.encoder.save_pretrained(str(adapter_dir))
            log.info("Saved LoRA adapter to %s", adapter_dir)
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to save LoRA adapter to %s: %s", adapter_dir, exc)

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
        _, logits = self._forward_features(batch)
        return logits

    def _forward_features(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(feats, logits)``.

        ``feats`` are the pooled per-channel embeddings flattened to
        ``(B, C*E)`` — exactly what the head consumes, and what the
        per-epoch t-SNE plot projects. ``forward`` returns only the logits;
        ``_step`` additionally keeps ``feats`` to accumulate them for the
        embedding plot (classification only).
        """
        ctx = torch.no_grad() if self.hparams.freeze_encoder else nullcontext()
        with ctx:
            feats = self.encoder_wrapper(batch)              # (B, C, E)
        feats = feats.flatten(1)                             # (B, C*E)
        logits = self.head(feats)                            # logits / values
        return feats, logits

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        feats, logits = self._forward_features(batch)
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
            batch_size=bs, add_dataloader_idx=False,
        )

        # Metrics
        if torchmetrics is None:
            return loss
        if self._head_type == "classification":
            if logits.size(-1) == 1:
                preds = (logits.squeeze(-1) > 0).long()
            else:
                preds = logits.argmax(dim=-1)
            acc = getattr(self, f"{stage}_acc")
            prec = getattr(self, f"{stage}_prec")
            rec = getattr(self, f"{stage}_rec")
            f1 = getattr(self, f"{stage}_f1")
            acc(preds, target)
            prec(preds, target)
            rec(preds, target)
            f1(preds, target)
            common = dict(
                sync_dist=sd, on_step=on_step, on_epoch=True,
                batch_size=bs, add_dataloader_idx=False,
            )
            self.log(f"{stage}/acc", acc, prog_bar=True, **common)
            # Macro precision / recall / f1 (scalars). PER-CLASS values are
            # logged separately at epoch-end, derived from the confusion
            # matrix (see _log_per_class_scores_from_cm).
            self.log(f"{stage}/precision", prec, prog_bar=False, **common)
            self.log(f"{stage}/recall", rec, prog_bar=False, **common)
            self.log(f"{stage}/f1", f1, prog_bar=False, **common)
            # Confusion matrices are only meaningful on val / test (train CM
            # is essentially noise during early epochs of a moving target).
            if stage in ("val", "test"):
                cm = getattr(self, f"{stage}_cm")
                cm.update(preds, target)
                setattr(self, f"_{stage}_cm_dirty", True)
            # t-SNE embeddings are collected ONLY during the real test phase
            # (trainer.test()), never per validation epoch — the projection is
            # too slow to run every epoch. ``trainer.testing`` is False while
            # the test loader is routed through validation_step under
            # test_every_epoch, so those batches are skipped here too.
            if (
                getattr(self, "_tsne_enabled", False)
                and stage == "test"
                and self.trainer is not None
                and getattr(self.trainer, "testing", False)
            ):
                self._test_emb.append(feats.detach().float().cpu())
                self._test_emb_labels.append(target.detach().cpu())
        else:
            mae = getattr(self, f"{stage}_mae")
            mse = getattr(self, f"{stage}_mse")
            mae(logits, target)
            mse(logits, target)
            self.log(f"{stage}/mae", mae, prog_bar=True, sync_dist=sd, on_step=on_step, on_epoch=True, batch_size=bs, add_dataloader_idx=False)
            self.log(f"{stage}/mse", mse, prog_bar=False, sync_dist=sd, on_step=on_step, on_epoch=True, batch_size=bs, add_dataloader_idx=False)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """Dispatch to val/test based on the dataloader index.

        Lightning calls this once per dataloader returned by
        ``DownstreamDataModule.val_dataloader``. Index 0 is val, index 1
        (when ``test_every_epoch`` is set) is the test loader — we route
        it to ``_step(..., "test")`` so metrics land under ``test/...``
        and the existing ``test_*`` torchmetrics accumulators are reused.
        """
        stage = "test" if dataloader_idx == 1 else "val"
        return self._step(batch, stage)

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    # ------------------------------------------------------------------
    # Confusion-matrix + per-class score logging (classification only)
    # ------------------------------------------------------------------
    def on_validation_epoch_end(self) -> None:
        if self._head_type != "classification" or torchmetrics is None:
            return
        # Skip the sanity-check pass: its confusion matrix is untrained-model
        # noise, and logging it advances the wandb step ahead of
        # trainer.global_step (the drift that makes end-of-run images get
        # dropped). Reset the CM accumulators so the sanity batches don't leak
        # into epoch 0 (the macro prec/rec/f1 metrics are logged via self.log,
        # so Lightning resets those automatically).
        if self.trainer is not None and getattr(self.trainer, "sanity_checking", False):
            for st in ("val", "test"):
                cm = getattr(self, f"{st}_cm", None)
                if cm is not None:
                    cm.reset()
                setattr(self, f"_{st}_cm_dirty", False)
            return
        # Confusion matrix + per-class scores every validation epoch; flush the
        # test ones too when ``test_every_epoch`` has routed test batches
        # through validation_step (dirty flag tells us). Capture the test dirty
        # flag first since _flush_confusion_matrix("test") clears it. t-SNE is
        # NOT done here — see on_test_epoch_end.
        test_dirty = bool(getattr(self, "_test_cm_dirty", False))
        self._flush_confusion_matrix("val")
        if test_dirty:
            self._flush_confusion_matrix("test")

    def on_test_epoch_end(self) -> None:
        # Runs during ``trainer.test()`` (always invoked by run_downstream when
        # a test split exists), independent of ``test_every_epoch``. This is
        # where the test confusion matrix, per-class scores AND the (test-only)
        # t-SNE are logged.
        if self._head_type != "classification" or torchmetrics is None:
            return
        self._flush_confusion_matrix("test")
        self._flush_tsne("test")

    def _log_per_class_scores_from_cm(self, stage: str, cm: torch.Tensor) -> None:
        """Log per-class precision / recall / F1 derived from the (DDP-synced)
        confusion matrix.

        For class ``c`` (rows = true label, cols = predicted label)::

            precision_c = cm[c, c] / sum(cm[:, c])      # TP / (TP + FP)
            recall_c    = cm[c, c] / sum(cm[c, :])      # TP / (TP + FN)
            f1_c        = 2·P·R / (P + R)

        — identical to torchmetrics' per-class definitions with
        ``zero_division=0`` (a class with no predictions / no support scores
        0). Logged as ``{stage}/{metric}_class_{c}``; the macro mean is already
        logged under the unprefixed ``{stage}/{metric}`` from the scalar metric
        in ``_step``, so it is not re-logged here. Deriving from the CM avoids
        torchmetrics' broken DDP sync of ``average=None`` vector metrics.
        """
        eps = 1e-12
        cmf = cm.to(torch.float64)
        tp = torch.diag(cmf)
        col = cmf.sum(dim=0)  # predicted-positive per class (TP + FP)
        row = cmf.sum(dim=1)  # actual-positive per class    (TP + FN)
        precision = torch.where(col > 0, tp / col.clamp_min(eps), torch.zeros_like(tp))
        recall = torch.where(row > 0, tp / row.clamp_min(eps), torch.zeros_like(tp))
        denom = precision + recall
        f1 = torch.where(
            denom > 0, 2.0 * precision * recall / denom.clamp_min(eps),
            torch.zeros_like(tp),
        )
        for name, vals in (("precision", precision), ("recall", recall), ("f1", f1)):
            for c in range(int(vals.numel())):
                # cm is identical on every rank after compute(), so sync_dist
                # just re-broadcasts the same value (silences the epoch-level
                # logging recommendation without changing the result).
                self.log(
                    f"{stage}/{name}_class_{c}", float(vals[c]),
                    sync_dist=True, add_dataloader_idx=False,
                )

    def _flush_confusion_matrix(self, stage: str) -> None:
        """Compute the (DDP-synced) ``{stage}_cm``, log per-class scores derived
        from it (all ranks), then plot + log the matrix image (rank-0) and
        reset."""
        cm: Any = getattr(self, f"{stage}_cm", None)
        dirty_attr = f"_{stage}_cm_dirty"
        if cm is None or not getattr(self, dirty_attr, False):
            return
        # ``compute()`` syncs the matrix across DDP ranks automatically, so the
        # resulting (C, C) tensor is identical on every rank.
        cm_tensor = cm.compute().detach().cpu().to(torch.int64)
        cm.reset()
        setattr(self, dirty_attr, False)

        # Per-class precision/recall/f1 (logged on every rank — the CM is
        # already identical across ranks after the sync).
        self._log_per_class_scores_from_cm(stage, cm_tensor)

        cm_np = cm_tensor.numpy()
        # Only rank 0 plots + logs the matrix image to wandb (it is identical
        # on every rank after the sync, but writing the image once per rank
        # would spam wandb and waste rank-0 storage).
        if self.trainer is not None and int(self.trainer.global_rank) != 0:
            return
        wandb_logger = self._wandb_logger()
        if wandb_logger is None:
            return
        try:
            import wandb
        except ImportError:
            return

        import matplotlib
        matplotlib.use("Agg", force=False)  # safe to call repeatedly
        import matplotlib.pyplot as plt

        fig = self._plot_confusion_matrix(cm_np, stage=stage)
        run = wandb_logger.experiment
        run.log(
            {f"{stage}/conf_mat": wandb.Image(fig)},
            step=self._wandb_log_step(run),
        )
        plt.close(fig)

    def _wandb_log_step(self, run: Any) -> int:
        """A wandb step that is never behind the run's current step.

        wandb silently drops any history record whose step is < the run's
        current internal step. That step can run *ahead* of
        ``trainer.global_step`` — e.g. the sanity-check validation logs a
        confusion matrix before training advances global_step, so the wandb
        counter ends up a couple of steps ahead. Logging the end-of-run
        images (last-epoch val CM, test CM, test t-SNE) at ``self.global_step``
        then lands *behind* the wandb step and the record is dropped (the PNG
        is written to disk but never appears on the cloud). Taking the max of
        the two keeps the step monotonic so the record is always kept.
        """
        try:
            run_step = int(getattr(run, "step", 0) or 0)
        except (TypeError, ValueError):
            run_step = 0
        return max(int(self.global_step), run_step)

    def _wandb_logger(self) -> Optional[Any]:
        loggers: list = []
        if getattr(self, "loggers", None):
            loggers = list(self.loggers)
        elif getattr(self, "logger", None) is not None:
            loggers = [self.logger]
        for lg in loggers:
            if lg.__class__.__name__ == "WandbLogger":
                return lg
        return None

    def _plot_confusion_matrix(self, cm: np.ndarray, stage: str) -> Any:
        import matplotlib.pyplot as plt

        nc = cm.shape[0]
        fig, ax = plt.subplots(figsize=(max(4, 0.6 * nc + 2), max(4, 0.6 * nc + 2)))
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"{stage} confusion matrix (epoch {int(self.current_epoch)})")
        ax.set_xticks(range(nc))
        ax.set_yticks(range(nc))
        # Annotate each cell with the count; flip the text colour above the
        # midpoint of cmap so dark cells remain legible.
        threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.0
        for i in range(nc):
            for j in range(nc):
                ax.text(
                    j, i, str(int(cm[i, j])),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=8,
                )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # t-SNE embedding logging (classification only) — test phase only.
    # Embeddings are accumulated per test step, then flushed once in
    # on_test_epoch_end (too slow to project every validation epoch).
    # ------------------------------------------------------------------
    def _flush_tsne(self, stage: str) -> None:
        """Gather, project (sklearn TSNE), plot and log (rank-0) the pooled
        encoder embeddings accumulated for ``stage``, then reset the buffers.
        Mirrors :meth:`_flush_confusion_matrix`; only ever called for the
        ``test`` stage."""
        if not getattr(self, "_tsne_enabled", False):
            return
        emb_list: list = getattr(self, f"_{stage}_emb", [])
        lab_list: list = getattr(self, f"_{stage}_emb_labels", [])
        if not emb_list:
            return
        emb = torch.cat(emb_list, dim=0)
        labels = torch.cat(lab_list, dim=0)
        # Reset on every rank (the data is only used on rank 0, but the
        # buffers must be cleared everywhere for the next epoch).
        emb_list.clear()
        lab_list.clear()

        # Gather across DDP ranks (collective — must run on every rank, so it
        # precedes the rank-0 guard) so the projection sees the whole
        # val/test split, not just this rank's shard.
        emb, labels = self._gather_embeddings(emb, labels)

        if self.trainer is not None and int(self.trainer.global_rank) != 0:
            return
        wandb_logger = self._wandb_logger()
        if wandb_logger is None:
            return
        try:
            import wandb
        except ImportError:
            return

        import matplotlib
        matplotlib.use("Agg", force=False)  # safe to call repeatedly
        import matplotlib.pyplot as plt

        fig = self._compute_and_plot_tsne(emb, labels, stage=stage)
        if fig is None:
            return
        run = wandb_logger.experiment
        run.log(
            {f"{stage}/tsne": wandb.Image(fig)},
            step=self._wandb_log_step(run),
        )
        plt.close(fig)

    def _gather_embeddings(
        self, emb: torch.Tensor, labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-gather per-rank ``(emb, labels)`` into the full split on every
        rank. No-op outside DDP. Uses ``all_gather_object`` so the variable
        per-rank row counts (``drop_last=False`` on the eval loaders) need no
        manual padding."""
        trainer = self.trainer
        world_size = int(getattr(trainer, "world_size", 1)) if trainer is not None else 1
        if world_size <= 1:
            return emb, labels
        try:
            import torch.distributed as dist
        except ImportError:  # pragma: no cover
            return emb, labels
        if not dist.is_available() or not dist.is_initialized():
            return emb, labels
        emb_parts: list = [None] * world_size
        lab_parts: list = [None] * world_size
        dist.all_gather_object(emb_parts, emb)
        dist.all_gather_object(lab_parts, labels)
        emb = torch.cat([p for p in emb_parts if p is not None], dim=0)
        labels = torch.cat([p for p in lab_parts if p is not None], dim=0)
        return emb, labels

    def _compute_and_plot_tsne(
        self, emb: torch.Tensor, labels: torch.Tensor, stage: str,
    ) -> Optional[Any]:
        """Project ``emb`` to 2-D with sklearn's TSNE and scatter by class.

        Returns ``None`` (skip logging) when sklearn is missing or there are
        too few points for a meaningful embedding.
        """
        try:
            from sklearn.manifold import TSNE
        except ImportError:  # pragma: no cover
            log.warning("scikit-learn not available — skipping t-SNE plot.")
            return None

        emb_np = emb.numpy().astype(np.float32)
        labels_np = labels.numpy().astype(np.int64)
        n = emb_np.shape[0]
        if n < 5:
            return None  # t-SNE is meaningless with a handful of points

        # Bound the per-epoch cost: deterministically subsample above the cap.
        if n > _TSNE_MAX_POINTS:
            rng = np.random.RandomState(int(self.hparams.seed))
            sel = rng.choice(n, size=_TSNE_MAX_POINTS, replace=False)
            emb_np = emb_np[sel]
            labels_np = labels_np[sel]
            n = _TSNE_MAX_POINTS

        # sklearn requires perplexity < n_samples; scale it down for small n.
        perplexity = min(30.0, max(5.0, (n - 1) / 3.0))
        perplexity = min(perplexity, float(n - 1))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            random_state=int(self.hparams.seed),
        )
        emb_2d = tsne.fit_transform(emb_np)
        return self._plot_tsne(emb_2d, labels_np, stage=stage)

    def _plot_tsne(self, emb_2d: np.ndarray, labels: np.ndarray, stage: str) -> Any:
        import matplotlib.pyplot as plt

        nc = int(self.hparams.num_classes)
        cmap = plt.get_cmap("tab10" if nc <= 10 else "tab20")
        fig, ax = plt.subplots(figsize=(7, 6))
        for cls in range(nc):
            m = labels == cls
            if not np.any(m):
                continue
            ax.scatter(
                emb_2d[m, 0], emb_2d[m, 1],
                s=10, alpha=0.7, color=cmap(cls % cmap.N),
                label=f"class {cls}",
            )
        ax.set_title(f"{stage} t-SNE embeddings (epoch {int(self.current_epoch)})")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="best", fontsize=8, markerscale=1.5, framealpha=0.8)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Optimiser + Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self) -> Any:
        wd = float(self.hparams.weight_decay)
        base_lr = float(self.hparams.lr)
        # ``None`` and ``0.0`` are equivalent ("no layerwise decay"). The
        # alternative — uniformly multiplying every group's LR by zero —
        # would be silly.
        layerwise_raw = self.hparams.layerwise_lr_decay
        layerwise = float(layerwise_raw) if layerwise_raw is not None else None
        if layerwise is not None and layerwise <= 0.0:
            layerwise = None
        # LoRA + layerwise_lr_decay is rejected outright: PEFT renames the
        # encoder block paths to ``...encoder.base_model.model.blocks.{i}.*``
        # which would crash ``_layer_id_for`` regex on ``int("model")``,
        # and uniform LR is the LoRA-paper recipe anyway. Make it a hard
        # error rather than silently disabling, so a hyperparameter sweep
        # that lists both axes does not produce duplicate runs differing
        # only in an ignored field.
        if layerwise is not None and getattr(self, "_lora_enabled", False):
            raise ValueError(
                "model.lora.enabled=true requires model.layerwise_lr_decay "
                "in (null, 0.0). LoRA uses uniform LR for the adapters; any "
                "other value would be silently ignored and produce a duplicate "
                f"of an existing run (got layerwise_lr_decay={layerwise_raw!r}).",
            )
        # Same logic for linear probing: when ``freeze_encoder`` is on
        # only the head trains, so every value of ``layerwise_lr_decay``
        # collapses to the same optimiser. Refuse rather than silently
        # producing duplicate sweep rows.
        if layerwise is not None and bool(self.hparams.freeze_encoder):
            raise ValueError(
                "model.freeze_encoder=true requires model.layerwise_lr_decay "
                "in (null, 0.0). Linear probe trains only the head, so the "
                "encoder LR schedule is moot; any other value would be "
                "silently ignored and produce a duplicate of an existing run "
                f"(got layerwise_lr_decay={layerwise_raw!r}).",
            )

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
