"""Swappable prediction heads for downstream tasks.

Both heads share an identical structure (linear or shallow MLP). Only the
output dimension and the typical loss differ:

- :class:`ClassificationHead` -> logits of shape ``(B, num_classes)``
- :class:`RegressionHead`     -> values of shape ``(B, num_outputs)``

By default ``num_layers=1`` produces a single ``nn.Linear`` (the canonical
linear-probe head). ``num_layers > 1`` builds a small MLP with GELU and
dropout between layers.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: Optional[int],
    num_layers: int,
    dropout: float,
) -> nn.Module:
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if num_layers == 1:
        return nn.Linear(in_dim, out_dim)

    h = int(hidden_dim) if hidden_dim is not None else in_dim
    layers: list[nn.Module] = [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
    for _ in range(num_layers - 2):
        layers += [nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(h, out_dim))
    return nn.Sequential(*layers)


class ClassificationHead(nn.Module):
    """Classifier head producing un-normalised logits."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.net = _build_mlp(in_dim, num_classes, hidden_dim, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    """Regression head producing real-valued outputs."""

    def __init__(
        self,
        in_dim: int,
        num_outputs: int = 1,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_outputs = int(num_outputs)
        self.net = _build_mlp(in_dim, num_outputs, hidden_dim, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_head(
    head_type: str,
    in_dim: int,
    num_classes: Optional[int] = None,
    num_outputs: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.0,
    num_layers: int = 1,
) -> nn.Module:
    """Factory for downstream heads.

    Parameters
    ----------
    head_type :
        ``"classification"`` or ``"regression"``.
    in_dim :
        Input feature dimension (``C * E`` after channel-flatten fusion).
    num_classes :
        Required when ``head_type == "classification"``.
    num_outputs :
        Required when ``head_type == "regression"`` (default 1 if unset).
    hidden_dim, dropout, num_layers :
        MLP shape. ``num_layers == 1`` collapses to a single ``nn.Linear``.
    """
    ht = str(head_type).lower()
    if ht == "classification":
        if num_classes is None:
            raise ValueError("ClassificationHead requires num_classes")
        return ClassificationHead(
            in_dim=in_dim,
            num_classes=int(num_classes),
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
        )
    if ht == "regression":
        return RegressionHead(
            in_dim=in_dim,
            num_outputs=int(num_outputs) if num_outputs is not None else 1,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
        )
    raise ValueError(
        f"Unknown head_type {head_type!r}; expected 'classification' or 'regression'"
    )
