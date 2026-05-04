"""Top-level LightningModule wiring tokenizer + CT-RoPE + encoder + decoder."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    import lightning.pytorch as pl  # type: ignore[no-redef]

from criterion import USReconstructionLoss
from schedulers import CosineLRSchedulerWrapper
from .training_debug import maybe_log_training_batch
from .backbone.us_decoder import USDecoder
from .backbone.us_encoder import USEncoder
from .positional.ct_rope import CTRoPE
from .tokenizer.multi_tokenizer import MultiTokenizer


class UltrasonicMAE(pl.LightningModule):
    """Masked Autoencoder for ultrasound A-mode signals.

    The full forward path:

    1. :class:`MultiTokenizer` routes each sample to the best-fitting
       branch (``W* ∈ {8, 16, 32}`` for the default config) and produces
       tokens ``(B, S, E)`` plus ``padding_mask``, ``window_size``, and
       ``patch_timestamps_us``.
    2. :class:`CTRoPE` is *wired into* each attention block of the
       encoder/decoder (shared instance) and rotates Q/K on the fly
       using the continuous patch midpoint timestamps.
    3. :class:`USEncoder` applies MAE masking (75 % by default) and
       transforms visible tokens.
    4. :class:`USDecoder` unshuffles the latent sequence, runs decoder
       transformer blocks, and the multi-size reconstruction head
       outputs the raw patch samples for every token.
    5. :func:`reconstruction_loss` computes per-patch-normalised MSE on
       the masked tokens only.

    Any of the steps above can be toggled via config: set ``use_ct_rope:
    false`` to disable CT-RoPE (used by Experiment A Mode 1, single
    tokenizer at 20 MHz).
    """

    def __init__(
        self,
        # Tokenizer
        window_sizes: tuple[int, ...] = (8, 16, 32),
        target_patch_mm: float = 0.6,
        tokenizer_type: str = "mlp",
        cnn_config: Optional[dict] = None,
        # Fixed-S batching (must match DataModule's ``target_patches``)
        target_patches: Optional[int] = None,
        # Model dims
        embed_dim: int = 256,
        encoder_depth: int = 8,
        encoder_heads: int = 8,
        encoder_mlp_ratio: float = 4.0,
        decoder_dim: int = 128,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        decoder_mlp_ratio: float = 4.0,
        masking_ratio: float = 0.75,
        # Positional encoding
        use_ct_rope: bool = True,
        ct_rope_base: float = 10_000.0,
        # Regularisation
        dropout: float = 0.0,
        # Criterion
        loss_type: str = "smooth_l1",
        loss_alpha: float = 0.0,
        norm_target: bool = True,
        # Optimiser
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        warmup_lr_init: float = 1e-6,
        max_epochs: int = 100,
        seed: int = 42,
        debug_pipeline_enabled: bool = False,
        debug_max_samples_per_base_dataset: int = 2,
        debug_log_interval_batches: int = 1,
        debug_midpoint_log_k: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = MultiTokenizer(
            window_sizes=window_sizes,
            embed_dim=embed_dim,
            target_patch_mm=target_patch_mm,
            tokenizer_type=tokenizer_type,
            cnn_config=cnn_config,
        )

        rotary_enc: Optional[nn.Module] = None
        rotary_dec: Optional[nn.Module] = None
        if use_ct_rope:
            head_dim_enc = embed_dim // encoder_heads
            head_dim_dec = decoder_dim // decoder_heads
            # One CTRoPE instance per stack because the head dim can differ
            # between encoder and decoder.
            rotary_enc = CTRoPE(dim=head_dim_enc, base=ct_rope_base)
            rotary_dec = CTRoPE(dim=head_dim_dec, base=ct_rope_base)

        self.encoder = USEncoder(
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=encoder_mlp_ratio,
            masking_ratio=masking_ratio,
            dropout=dropout,
            rotary=rotary_enc,
        )
        self.decoder = USDecoder(
            encoder_dim=embed_dim,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            mlp_ratio=decoder_mlp_ratio,
            dropout=dropout,
            window_sizes=window_sizes,
            rotary=rotary_dec,
        )

        self.criterion = USReconstructionLoss(
            loss_type=loss_type,
            alpha=loss_alpha,
            norm_target=norm_target,
        )

        self._debug_enabled = bool(debug_pipeline_enabled)
        self._debug_max_samples_per_base_dataset = int(debug_max_samples_per_base_dataset)
        self._debug_log_interval_batches = int(debug_log_interval_batches)
        self._debug_midpoint_log_k = int(debug_midpoint_log_k)
        self._debug_logged_counts: Optional[dict[str, int]] = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        signal = batch["signal"]                       # (B, T)
        signal_mask = batch["signal_mask"]             # (B, T) bool
        fs = batch["sampling_frequency_hz"]            # (B,)
        W_override = batch.get("window_size")          # (B,) optional

        tok = self.tokenizer(
            signal=signal,
            signal_mask=signal_mask,
            sampling_frequency_hz=fs,
            window_size_override=W_override,
            fixed_num_patches=self.hparams.target_patches,
        )
        if self._debug_enabled:
            self._tokenizer_output_for_debug = tok
        elif hasattr(self, "_tokenizer_output_for_debug"):
            self._tokenizer_output_for_debug = None

        enc = self.encoder(
            tokens=tok.tokens,
            padding_mask=tok.padding_mask,
            time_values_us=tok.patch_timestamps_us,
        )

        pred = self.decoder(
            latent=enc["latent"],
            ids_restore=enc["ids_restore"],
            len_keep=enc["len_keep"],
            window_size_per_sample=tok.window_size,
            time_values_us=tok.patch_timestamps_us,
        )

        out: dict = {
            "pred": pred,
            "mask": enc["mask"],
            "padding_mask": tok.padding_mask,
            "window_size": tok.window_size,
        }
        return out

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------
    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        out = self(batch)
        loss_dict = self.criterion(
            pred=out["pred"],
            signal=batch["signal"],
            mask=out["mask"],
            window_sizes=out["window_size"],
            padding_mask=out["padding_mask"],
            signal_lengths=batch["length"],
        )
        loss = loss_dict["loss"]
        on_step = stage == "train"
        self.log(f"{stage}/loss",         loss,                       prog_bar=True,  sync_dist=True, on_step=on_step, on_epoch=True, batch_size=batch["signal"].size(0))
        self.log(f"{stage}/masked_loss",  loss_dict["masked_loss"],   prog_bar=False, sync_dist=True, on_step=on_step, on_epoch=True, batch_size=batch["signal"].size(0))
        self.log(f"{stage}/visible_loss", loss_dict["visible_loss"],  prog_bar=False, sync_dist=True, on_step=on_step, on_epoch=True, batch_size=batch["signal"].size(0))
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch, "train")
        if self._debug_enabled:
            tok = getattr(self, "_tokenizer_output_for_debug", None)
            if tok is not None:
                maybe_log_training_batch(self, batch, tok, batch_idx)
        return loss

    def on_train_epoch_start(self) -> None:
        if not getattr(self, "_debug_enabled", False):
            return
        trainer = self.trainer
        if trainer is None:
            return
        if getattr(trainer, "global_rank", 0) != 0:
            return
        epoch = int(trainer.current_epoch)
        max_epochs = int(self.hparams.max_epochs)
        if epoch in (0, max_epochs - 1):
            self._debug_logged_counts = {}

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    # ------------------------------------------------------------------
    # Optimiser + Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self) -> Any:
        decay, no_decay = [], []
        for name, p in self.named_parameters():
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

        optim = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=tuple(self.hparams.betas),
        )

        total_steps = self.trainer.estimated_stepping_batches
        # PL 2.x returns max_steps (default: -1) for IterableDataset when
        # max_steps is not set → fall back to a manual estimate.
        if not (isinstance(total_steps, (int, float)) and total_steps > 0 and total_steps != float("inf")):
            dm = self.trainer.datamodule
            if (
                hasattr(dm, "_estimated_num_batches")
                and hasattr(dm, "_train_shards")
                and dm._train_shards
            ):
                steps_per_epoch = max(dm._estimated_num_batches(dm._train_shards), 1)
            else:
                steps_per_epoch = 1
            total_steps = steps_per_epoch * max(self.hparams.max_epochs, 1)

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
        # timm schedulers use step_update() instead of the standard step()
        scheduler.step_update(num_updates=self.global_step)
