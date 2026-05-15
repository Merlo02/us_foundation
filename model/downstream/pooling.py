"""Sequence pooling strategies for the downstream encoder wrapper.

Reduces an encoder output ``(B, S, E)`` together with a per-token validity
mask ``(B, S)`` to a fixed-size representation ``(B, E)``.

Currently only :class:`MeanPool` is implemented — the user-facing
requirement for the first iteration. The pluggable :class:`Pooling` base
class makes attentive-pool / CLS-token variants a single-class addition
later (see ``build_pooling``).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Pooling(nn.Module):
    """Abstract base: ``(B, S, E)`` + ``valid_mask (B, S)`` -> ``(B, E)``."""

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor,
    ) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class MeanPool(Pooling):
    """Masked mean over the token dimension.

    Padded tokens (``valid_mask == False``) contribute zero to both the
    numerator and the denominator, so the mean is computed over the
    *real* tokens only.
    """

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        m = valid_mask.unsqueeze(-1).to(x.dtype)              # (B, S, 1)
        denom = m.sum(dim=1).clamp(min=1.0)                   # (B, 1)
        return (x * m).sum(dim=1) / denom                     # (B, E)


def build_pooling(pooling_type: str, embed_dim: int) -> Pooling:
    """Factory for pooling modules.

    Parameters
    ----------
    pooling_type :
        ``"mean"`` for now. Future: ``"attentive"``, ``"cls"``.
    embed_dim :
        Token embedding dim (unused by mean-pool; kept in the signature so
        the registry can grow without breaking callers).
    """
    pt = str(pooling_type).lower()
    if pt == "mean":
        return MeanPool()
    raise ValueError(f"Unknown pooling_type {pooling_type!r}; expected 'mean'")
