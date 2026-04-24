"""MAE decoder for the ultrasound foundation model.

Inspired by TimeFM's ``MAEDecoder`` with the following adaptations for the
multi-tokenizer setting:

- **Multi-size reconstruction head.** One independent ``nn.Linear`` per
  tokenizer branch — ``head_{W}(decoder_dim → W)`` — with explicit
  per-sample routing. For the default ``window_sizes = (8, 16, 32)`` this
  is three linear layers; with so few branches a shared
  ``MultiOutSizeLinear`` does not add tangible value and hurts
  readability.
- **Continuous-time PE.** The decoder also uses :class:`CTRoPE` on its
  transformer blocks rather than learned absolute embeddings. Time
  values are unshuffled together with the tokens before the blocks run.
- **Re-insertion of mask tokens.** Standard MAE recipe: project the
  visible latents to the decoder dim, concatenate mask tokens for the
  masked positions, then unshuffle to restore the original sequence
  order before the transformer blocks.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .attention import TransformerBlock


class USDecoder(nn.Module):
    """Masked Autoencoder decoder with multi-size reconstruction head.

    Parameters
    ----------
    encoder_dim : dim of the encoder latent ``E_enc``.
    decoder_dim : dim of the decoder latent ``E_dec``.
    decoder_depth : number of decoder transformer blocks.
    decoder_heads : number of attention heads.
    mlp_ratio, qkv_bias, dropout, norm_layer : forwarded to
        :class:`TransformerBlock`.
    window_sizes : tuple ``(8, 16, 32)``. One reconstruction head is
        instantiated per entry.
    rotary : optional :class:`CTRoPE` applied on every block (shared with
        the encoder at caller discretion).
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int = 128,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        window_sizes: Sequence[int] = (8, 16, 32),
        rotary: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder_dim = int(encoder_dim)
        self.decoder_dim = int(decoder_dim)
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.rotary = rotary

        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_dim,
                num_heads=decoder_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                norm_layer=norm_layer,
            )
            for _ in range(decoder_depth)
        ])
        self.norm = norm_layer(decoder_dim)

        # One reconstruction head per window size — kept in a ModuleDict so
        # the per-sample routing below is explicit and debuggable.
        self.heads = nn.ModuleDict({
            str(w): nn.Linear(decoder_dim, w, bias=True)
            for w in self.window_sizes
        })

        self._initialize_weights()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _initialize_weights(self) -> None:
        trunc_normal_(self.mask_token, std=0.02, a=-0.04, b=0.04)
        self.apply(self._init_module)

    def _init_module(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        latent: torch.Tensor,              # (B, S_vis_max, E_enc)
        ids_restore: torch.Tensor,          # (B, S)
        len_keep: torch.Tensor,             # (B,)
        window_size_per_sample: torch.Tensor,  # (B,) int ∈ window_sizes
        time_values_us: Optional[torch.Tensor] = None,  # (B, S)
    ) -> torch.Tensor:
        """Reconstruct one raw patch per token.

        Returns a padded ``(B, S, W_max)`` tensor; downstream code should
        use ``window_size_per_sample`` to slice out the first ``W*``
        entries per sample.
        """
        B, S_vis_max, _ = latent.shape
        S = ids_restore.size(1)
        device = latent.device

        x = self.decoder_embed(latent)                   # (B, S_vis_max, E_dec)

        # Pad the visible latents back up to S by appending mask tokens.
        pad_len = S - S_vis_max
        if pad_len > 0:
            mask_pad = self.mask_token.expand(B, pad_len, self.decoder_dim)
            x = torch.cat([x, mask_pad], dim=1)
        # Now replace the *trailing* padding that lies beyond len_keep[b]
        # with mask tokens for all samples (already done by the cat above
        # for positions beyond S_vis_max; we also need to mask between
        # len_keep[b] and S_vis_max for samples with fewer visible tokens).
        idx = torch.arange(S, device=device).unsqueeze(0)
        visible_mask = idx < len_keep.unsqueeze(1)       # (B, S) in shuffle order
        x = torch.where(
            visible_mask.unsqueeze(-1),
            x,
            self.mask_token.expand(B, S, self.decoder_dim),
        )

        # Unshuffle back to the original token order.
        x = torch.gather(
            x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim),
        )

        # Transformer blocks with CT-RoPE on q/k (time_values already in
        # original token order because we unshuffled above).
        for blk in self.blocks:
            x = blk(x, rotary=self.rotary, time_values=time_values_us)
        x = self.norm(x)

        # Multi-size reconstruction head: gather per-sample outputs.
        W_max = max(self.window_sizes)
        out = x.new_zeros(B, S, W_max)
        for w in self.window_sizes:
            mask_w = (window_size_per_sample == w).view(B, 1, 1)   # (B, 1, 1)
            head_out = self.heads[str(w)](x)                        # (B, S, w)
            # Pad the head output to W_max so we can accumulate.
            if w < W_max:
                pad_tail = head_out.new_zeros(B, S, W_max - w)
                head_out = torch.cat([head_out, pad_tail], dim=-1)
            out = out + mask_w.to(out.dtype) * head_out
        return out


# ----------------------------------------------------------------------
# Reconstruction loss — MAE-style with per-patch normalisation
# ----------------------------------------------------------------------

def reconstruction_loss(
    pred: torch.Tensor,                    # (B, S, W_max)
    signal: torch.Tensor,                  # (B, T) padded
    signal_mask: torch.Tensor,             # (B, T) valid-sample mask
    mask: torch.Tensor,                    # (B, S) 1 = masked (target)
    padding_mask: torch.Tensor,            # (B, S) 1 = valid token
    window_size_per_sample: torch.Tensor,  # (B,)
    normalize_per_patch: bool = True,
) -> torch.Tensor:
    """MAE mean-squared error on *masked* tokens only.

    Builds the reconstruction target by chopping each sample's padded
    signal into ``W*`` non-overlapping patches, normalising each patch to
    zero-mean / unit-std (the trick from He et al. 2021) and computing
    MSE only on positions where both ``mask`` and ``padding_mask`` are
    true. Padding within the last (partial) patch is ignored.
    """
    B, S, W_max = pred.shape
    T = signal.size(1)
    device = pred.device

    target = pred.new_zeros(B, S, W_max)
    valid = torch.zeros(B, S, dtype=torch.bool, device=device)

    for b in range(B):
        w = int(window_size_per_sample[b].item())
        n_valid = int(padding_mask[b].sum().item())
        sig_len = int(signal_mask[b].sum().item())
        usable = min(n_valid, sig_len // w)
        if usable == 0:
            continue
        chunk = signal[b, : usable * w].view(usable, w).to(pred.dtype)
        if normalize_per_patch:
            mean = chunk.mean(dim=1, keepdim=True)
            std = chunk.std(dim=1, keepdim=True).clamp_min(1e-6)
            chunk = (chunk - mean) / std
        target[b, :usable, :w] = chunk
        valid[b, :usable] = True

    sq = (pred - target).pow(2)                            # (B, S, W_max)
    # Mean over the W_max axis first (only W* entries are meaningful; the
    # rest are zeros in both pred — because unused heads contribute zero —
    # and target, so they do not bias the mean beyond a constant factor
    # per sample).
    sq = sq.mean(dim=-1)                                   # (B, S)

    token_mask = mask.bool() & padding_mask & valid        # (B, S)
    n = token_mask.float().sum().clamp_min(1.0)
    return (sq * token_mask.float()).sum() / n
