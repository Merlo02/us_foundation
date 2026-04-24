"""Continuous-Time Rotary Positional Encoding (CT-RoPE).

Direct port of :class:`ContinuousTimeRotaryEmbedding` from MIRA
(``MIRA-main/MIRA/models/modeling_mira.py`` lines 211–279), adapted for
standalone use (no dependency on HuggingFace internals).

The module takes continuous timestamps (in any consistent unit — for the
ultrasound foundation model we use microseconds so distinct sampling
rates land on the same absolute time axis) and rotates the query/key
channels according to

    θ_i(t) = base^(−2i/d) · t            for i = 0, …, d/2 − 1
    (q_rotated, k_rotated) = (q·cos θ + rotate_half(q)·sin θ,
                              k·cos θ + rotate_half(k)·sin θ)

compared to the discrete RoPE of LLaMA/Mistral, *t* can be an arbitrary
float (no integer position quantisation), which is exactly what we need
for the multi-frequency multi-patch-size setting.
"""
from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims — ``[a, b] → [-b, a]`` along the last axis."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class CTRoPE(nn.Module):
    """Continuous-time rotary positional encoding.

    Parameters
    ----------
    dim :
        Head dimension ``d`` (must be even). The inverse-frequency vector
        has size ``d/2``.
    base :
        Base of the exponential frequency schedule (default 10_000, same as
        LLaMA/Mistral/MIRA).
    """

    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"CTRoPE requires even head dim, got {dim}")
        self.dim = int(dim)
        self.base = float(base)

        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_cos_sin(
        self, time_values: torch.Tensor, device: torch.device, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` broadcastable to ``(B, num_heads, S, D)``.

        ``time_values`` has shape ``(B, S)`` and can carry arbitrary floats
        (including zero-padding on invalid positions). The angles are
        computed in float32 and cast back to *dtype* for the rotation.
        """
        t = time_values.to(device=device, dtype=torch.float32)
        # freqs: (B, S, D/2)
        freqs = torch.einsum("bs,d->bsd", t, self.inv_freq.to(device))
        # (B, S, D) by concatenation along the last axis
        emb = torch.cat((freqs, freqs), dim=-1)
        # Add head dimension for broadcasting: (B, 1, S, D)
        emb = emb.unsqueeze(1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(
        self,
        q: torch.Tensor,               # (B, num_heads, S, D)
        k: torch.Tensor,               # (B, num_heads, S, D)
        time_values: torch.Tensor,     # (B, S) — continuous timestamps
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply CT-RoPE to *q* and *k* using absolute ``time_values``."""
        B, H, S, D = q.shape
        if D != self.dim:
            raise ValueError(
                f"Input head dim {D} does not match CTRoPE dim {self.dim}"
            )
        if time_values.shape != (B, S):
            raise ValueError(
                f"time_values shape {tuple(time_values.shape)} does not match "
                f"(B={B}, S={S})"
            )

        cos, sin = self._compute_cos_sin(time_values, q.device, q.dtype)
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot
