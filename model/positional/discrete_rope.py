"""Discrete Rotary Positional Encoding (Discrete RoPE).

Classic integer-position RoPE as in LLaMA/Mistral, adapted to the
us_foundation attention layout ``(B, H, S, D)``.

Used as a drop-in replacement for :class:`~model.positional.ct_rope.CTRoPE`
when ``use_ct_rope: false`` — the forward signature is identical so the same
hook in :class:`~model.backbone.attention.MultiHeadSelfAttention` works for
both modes without modification.

Comparison with CT-RoPE
-----------------------
- CT-RoPE encodes **absolute timestamps** (µs) so inter-patch temporal
  distances are preserved across different sampling rates.
- Discrete RoPE encodes **token ordinals** (0, 1, 2, …) and treats all
  patches as equally spaced in "sequence space".

The ablation isolates whether temporal continuity in PE is beneficial
vs. mere ordering information.
"""
from __future__ import annotations

import torch
from torch import nn

from .ct_rope import rotate_half


class DiscreteRoPE(nn.Module):
    """Discrete (integer-position) rotary positional encoding.

    Parameters
    ----------
    dim :
        Head dimension ``D`` (must be even).
    max_seq_len :
        Number of positions to pre-compute at init.  If at runtime
        ``S > max_seq_len`` the cache is extended on-the-fly (no error).
    base :
        Frequency base for the exponential schedule (default 10 000,
        same as LLaMA / CT-RoPE so the two modes are directly comparable).
    """

    def __init__(
        self, dim: int, max_seq_len: int = 512, base: float = 10_000.0
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"DiscreteRoPE requires even head dim, got {dim}")
        self.dim = int(dim)
        self.base = float(base)
        self.max_seq_len = int(max_seq_len)

        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(self.max_seq_len)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _build_cache(self, length: int) -> None:
        """Pre-compute cos/sin tables of shape ``(length, D)``."""
        seq = torch.arange(length, dtype=torch.float32, device=self.inv_freq.device)
        # (length, D/2)
        freqs = torch.outer(seq, self.inv_freq)
        # duplicate along last axis → (length, D) — same convention as CTRoPE
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)
        self.max_seq_len = length

    def _ensure_cache(self, seq_len: int) -> None:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,                          # (B, H, S, D)
        k: torch.Tensor,                          # (B, H, S, D)
        time_values: torch.Tensor | None = None,  # ignored — kept for API parity
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply discrete RoPE to *q* and *k* using ordinal token positions.

        ``time_values`` is accepted but **ignored** — positions are always
        ``0, 1, …, S-1``.  This keeps the call-site in
        :class:`~model.backbone.attention.MultiHeadSelfAttention` identical
        for both CT-RoPE and discrete RoPE.
        """
        S = q.size(2)
        self._ensure_cache(S)

        # (1, 1, S, D) — broadcast over batch and head dims
        cos = self.cos_cache[:S].unsqueeze(0).unsqueeze(0).to(dtype=q.dtype)
        sin = self.sin_cache[:S].unsqueeze(0).unsqueeze(0).to(dtype=q.dtype)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot
