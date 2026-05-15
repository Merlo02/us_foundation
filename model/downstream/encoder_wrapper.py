"""Decoder-less encoder wrapper for downstream tasks.

Owns the same encoder-side stack as :class:`~model.us_mae.UltrasonicMAE`
(``MultiTokenizer`` -> CT-RoPE -> ``USEncoder``) but:

- **No decoder.**
- **No MAE masking** at forward time: the encoder is called with
  ``bypass_masking=True`` so every token reaches the transformer blocks.
- **Multi-channel input.** Accepts ``signal: (B, C, T)`` (and the
  single-channel ``(B, T)`` case is auto-unsqueezed to ``C=1``). The
  channel axis is folded into the batch axis -> ``(B*C, T)`` -> encoded
  in one forward pass, then unfolded back to ``(B, C, E)``. **No Python
  loop over the channel axis.**
- **Sequence pooling.** A pluggable :class:`~model.downstream.pooling.Pooling`
  reduces the per-channel token sequence ``(B*C, S, E)`` to ``(B*C, E)``;
  the wrapper reshapes that to ``(B, C, E)`` for the head to fuse.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..backbone.us_encoder import USEncoder
from ..positional.ct_rope import CTRoPE
from ..positional.discrete_rope import DiscreteRoPE
from ..tokenizer.multi_tokenizer import MultiTokenizer
from .pooling import Pooling, build_pooling


class UltrasonicEncoderWrapper(nn.Module):
    """Tokenizer + (CT-)RoPE + Encoder + Pooling, multi-channel.

    Parameters mirror the encoder-side hyper-parameters of
    :class:`~model.us_mae.UltrasonicMAE` so a pretrained checkpoint can be
    loaded layer-for-layer (see
    :meth:`UltrasonicDownstream._load_pretrained_encoder`).
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
        # Encoder dims
        embed_dim: int = 256,
        encoder_depth: int = 8,
        encoder_heads: int = 8,
        encoder_mlp_ratio: float = 4.0,
        # Positional encoding
        use_ct_rope: bool = True,
        ct_rope_base: float = 10_000.0,
        rope_max_seq_len: int = 512,
        # Regularisation
        dropout: float = 0.0,
        # Pooling
        pooling_type: str = "mean",
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.target_patches = target_patches

        self.tokenizer = MultiTokenizer(
            window_sizes=window_sizes,
            embed_dim=embed_dim,
            target_patch_mm=target_patch_mm,
            tokenizer_type=tokenizer_type,
            cnn_config=cnn_config,
        )

        head_dim_enc = embed_dim // encoder_heads
        if use_ct_rope:
            rotary_enc: nn.Module = CTRoPE(dim=head_dim_enc, base=ct_rope_base)
        else:
            rotary_enc = DiscreteRoPE(
                dim=head_dim_enc, max_seq_len=rope_max_seq_len, base=ct_rope_base,
            )
        # The rotary module is stored only via ``USEncoder.rotary`` (the
        # encoder consumes it inside each TransformerBlock). The pretraining
        # ``UltrasonicMAE`` follows the same convention, so state-dict keys
        # line up under ``encoder.rotary.*``.
        self.encoder = USEncoder(
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=encoder_mlp_ratio,
            # masking_ratio is irrelevant in bypass_masking mode but kept
            # at the pretraining default so the saved hyper-param matches.
            masking_ratio=0.0,
            dropout=dropout,
            rotary=rotary_enc,
        )

        self.pooling: Pooling = build_pooling(pooling_type, embed_dim)

    # ------------------------------------------------------------------
    @property
    def out_dim(self) -> int:
        """Per-channel embedding dim (= ``embed_dim``)."""
        return self.embed_dim

    # ------------------------------------------------------------------
    def _to_bc_signal(self, signal: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Normalise input to ``(B*C, T)`` and return ``(B, C)`` for un-fold."""
        if signal.ndim == 2:                       # (B, T) single-channel
            signal = signal.unsqueeze(1)            # (B, 1, T)
        elif signal.ndim != 3:
            raise ValueError(
                f"signal must be (B, T) or (B, C, T), got shape {tuple(signal.shape)}"
            )
        B, C, T = signal.shape
        return signal.reshape(B * C, T), B, C

    @staticmethod
    def _broadcast_per_channel(
        x: Optional[torch.Tensor], C: int,
    ) -> Optional[torch.Tensor]:
        """Repeat a per-acquisition tensor along a new channel axis.

        ``x`` has leading dim ``B`` (e.g. ``(B,)`` or ``(B, T)``). The
        result has leading dim ``B * C`` with each acquisition repeated
        ``C`` times in order, matching the reshape used for ``signal``.
        """
        if x is None:
            return None
        return x.repeat_interleave(C, dim=0)

    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> torch.Tensor:
        """Encode a multi-channel batch.

        Required batch keys:

        - ``signal``: ``(B, C, T)`` or ``(B, T)`` (treated as ``C=1``).
        - ``signal_mask``: ``(B, T)`` bool. The validity mask is assumed
          to be the same across channels (validity is per-time-step, not
          per-channel). If you ever need per-channel masks, pass a
          ``(B, C, T)`` tensor — this method reshapes it the same way as
          ``signal``.
        - ``sampling_frequency_hz``: ``(B,)`` — shared across channels.

        Optional batch keys (used when present, recomputed otherwise):

        - ``window_size``: ``(B,)`` long — pre-routed ``W*`` per sample.
        - ``patch_timestamps_us``: ``(B, S)`` — loader-provided midpoint
          timestamps for CT-RoPE.

        Returns
        -------
        feats : ``(B, C, E)``
            Per-channel pooled embeddings. The task module flattens this
            to ``(B, C * E)`` before the head.
        """
        signal_bc, B, C = self._to_bc_signal(batch["signal"])

        sig_mask = batch["signal_mask"]
        if sig_mask.ndim == 2:                                   # (B, T)
            sig_mask_bc = sig_mask.repeat_interleave(C, dim=0)
        elif sig_mask.ndim == 3:                                 # (B, C, T)
            sig_mask_bc = sig_mask.reshape(B * C, -1)
        else:
            raise ValueError(
                f"signal_mask must be (B, T) or (B, C, T), got shape "
                f"{tuple(sig_mask.shape)}"
            )

        fs_bc = self._broadcast_per_channel(batch["sampling_frequency_hz"], C)
        ws_bc = self._broadcast_per_channel(batch.get("window_size"), C)
        tstamps_bc = self._broadcast_per_channel(batch.get("patch_timestamps_us"), C)

        tok = self.tokenizer(
            signal=signal_bc,
            signal_mask=sig_mask_bc,
            sampling_frequency_hz=fs_bc,
            window_size_override=ws_bc,
            fixed_num_patches=self.target_patches,
            patch_timestamps_us=tstamps_bc,
        )

        enc_out = self.encoder(
            tokens=tok.tokens,
            padding_mask=tok.padding_mask,
            time_values_us=tok.patch_timestamps_us,
            bypass_masking=True,
        )
        latent = enc_out["latent"]                # (B*C, S, E)
        valid_mask = enc_out["padding_mask"]      # (B*C, S)

        pooled = self.pooling(latent, valid_mask)  # (B*C, E)
        return pooled.reshape(B, C, self.embed_dim)
