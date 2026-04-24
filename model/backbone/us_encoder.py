"""MAE encoder for ultrasound A-mode signals.

Inspired by TimeFM's ``CerebroTransformer`` and MAE (He et al. 2021):

- Accepts **already-tokenized** inputs ``(B, S, E)`` from the
  :class:`MultiTokenizer`.
- Contains **no internal patch embedding**.
- Contains **no internal PE** — continuous-time RoPE is applied inside
  each attention block via the :class:`~model.positional.ct_rope.CTRoPE`
  module that the user wires in at construction time.
- Implements MAE masking: shuffles the tokens, selects the visible
  subset (``1 − masking_ratio``) and forwards only the visible tokens
  through the transformer blocks. Padded tokens (carried in via
  ``padding_mask``) are forced to be masked *first* so they never make
  it into the visible set.

The ``fix_init_weight`` re-scaling of attention-output and MLP-fc2
projections is taken verbatim from TimeFM's trick to keep the residual
stream well-conditioned at depth.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .attention import TransformerBlock


class USEncoder(nn.Module):
    """Masked Autoencoder encoder for ultrasound A-mode tokens.

    Parameters
    ----------
    embed_dim :
        Token embedding dimension ``E`` — must match the tokenizer's output.
    depth, num_heads, mlp_ratio :
        Standard transformer hyper-parameters.
    masking_ratio :
        Fraction of *valid* tokens that are masked during pre-training.
        Padded tokens are always masked out first so they never count as
        visible tokens.
    qkv_bias, dropout, norm_layer :
        Forwarded to :class:`TransformerBlock`.
    rotary :
        An instance of :class:`~model.positional.ct_rope.CTRoPE` (or any
        module with the same ``(q, k, time_values) → (q_rot, k_rot)``
        API). Optional — when ``None`` the encoder behaves as a plain
        transformer and expects the user to have added positional info
        externally.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        masking_ratio: float = 0.75,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        rotary: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.masking_ratio = float(masking_ratio)
        self.num_heads = int(num_heads)
        self.rotary = rotary

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pad_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self._initialize_weights()
        self._fix_init_weight()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _initialize_weights(self) -> None:
        trunc_normal_(self.mask_token, std=0.02, a=-0.04, b=0.04)
        trunc_normal_(self.pad_token, std=0.02, a=-0.04, b=0.04)
        self.apply(self._init_module)

    def _init_module(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _fix_init_weight(self) -> None:
        """Re-scale the residual projections (TimeFM trick)."""
        for i, blk in enumerate(self.blocks):
            layer_id = i + 1
            blk.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            blk.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))

    # ------------------------------------------------------------------
    # MAE masking
    # ------------------------------------------------------------------
    @staticmethod
    def _mae_shuffle_and_mask(
        padding_mask: torch.Tensor,   # (B, S) True on valid tokens
        masking_ratio: float,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample MAE-style shuffle.

        Returns
        -------
        ids_shuffle : (B, S) long
            Index mapping shuffled → original position, with padded tokens
            *forced* to the tail so they are always masked out.
        ids_restore : (B, S) long
            Inverse permutation of ``ids_shuffle`` — used by the decoder.
        mask : (B, S) float
            ``1`` on tokens that are masked (including padded), ``0`` on
            visible tokens. Used for the reconstruction loss.
        len_keep : (B,) long
            Per-sample number of visible tokens (= round((1 − r) · n_valid)).
        """
        B, S = padding_mask.shape
        device = padding_mask.device

        # Primary noise for valid tokens; padded tokens get a large noise
        # so the argsort places them last (= always masked).
        noise = torch.rand(B, S, device=device, generator=generator)
        noise = torch.where(padding_mask, noise, torch.full_like(noise, 2.0))

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        n_valid = padding_mask.sum(dim=1)                         # (B,)
        len_keep = ((1.0 - masking_ratio) * n_valid.float()).round().long()
        len_keep = torch.clamp(len_keep, min=1)  # at least one visible token

        # Build per-sample mask using broadcasting with a max across batch.
        idx = torch.arange(S, device=device).unsqueeze(0)         # (1, S)
        keep_mask_shuffled = idx < len_keep.unsqueeze(1)          # (B, S)
        # mask in shuffle space: 1 where masked
        mask_shuffled = (~keep_mask_shuffled).float()
        # also force padded positions (which are at the tail) to be masked
        # (already the case because they're after len_keep by construction)

        # Re-order mask back to original positions.
        mask = torch.gather(mask_shuffled, 1, ids_restore)
        return ids_shuffle, ids_restore, mask, len_keep

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        tokens: torch.Tensor,                    # (B, S, E)
        padding_mask: torch.Tensor,              # (B, S) bool
        time_values_us: Optional[torch.Tensor] = None,  # (B, S)
        generator: Optional[torch.Generator] = None,
    ) -> dict:
        """Run the MAE encoder.

        Returns a dict with:

        - ``latent``              ``(B, S_vis_max, E)``
        - ``ids_restore``         ``(B, S)``
        - ``mask``                ``(B, S)``  — 1 on masked, 0 on visible
        - ``padding_mask_visible`` ``(B, S_vis_max)``
        - ``time_values_visible`` ``(B, S_vis_max)`` (or ``None``)
        """
        B, S, E = tokens.shape
        # Replace padded rows with the pad_token so they carry a learned
        # representation rather than raw zeros.
        pad = self.pad_token.expand(B, S, E)
        tokens = torch.where(padding_mask.unsqueeze(-1), tokens, pad)

        ids_shuffle, ids_restore, mask, len_keep = self._mae_shuffle_and_mask(
            padding_mask, self.masking_ratio, generator=generator,
        )

        # Gather tokens in shuffle order so the first ``len_keep[b]`` entries
        # per sample are the visible ones. Because ``len_keep`` varies, we
        # gather the *full* S length and build a visibility mask instead.
        tokens_shuffled = torch.gather(
            tokens, 1, ids_shuffle.unsqueeze(-1).expand(-1, -1, E),
        )
        S_vis_max = int(len_keep.max().item())
        visible = tokens_shuffled[:, :S_vis_max, :]
        idx = torch.arange(S_vis_max, device=tokens.device).unsqueeze(0)
        visible_mask = idx < len_keep.unsqueeze(1)                # (B, S_vis_max)

        if time_values_us is not None:
            t_shuffled = torch.gather(time_values_us, 1, ids_shuffle)
            t_visible = t_shuffled[:, :S_vis_max]
        else:
            t_visible = None

        x = visible
        for blk in self.blocks:
            x = blk(
                x,
                padding_mask=visible_mask,
                rotary=self.rotary,
                time_values=t_visible,
            )
        x = self.norm(x)

        return {
            "latent": x,
            "ids_restore": ids_restore,
            "mask": mask,
            "len_keep": len_keep,
            "padding_mask_visible": visible_mask,
            "time_values_visible": t_visible,
        }
