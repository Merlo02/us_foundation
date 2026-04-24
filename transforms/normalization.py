"""
On-the-fly signal normalisation transforms (GPU-compatible nn.Module).

Ported and extended from TimeFM ``transforms/normalization.py``.

These transforms are applied at training time (in the DataModule's
``on_before_batch_transfer`` or inside the LightningModule's step),
*not* during ETL.  ETL always writes raw float32 values.

Note on axis conventions
------------------------
All modules operate on the **last axis** (time axis) by default.
For a batch of 1-D ultrasound signals the tensor shape is ``(B, T)``,
so ``dim=-1`` normalises per-sample across time, which is the standard
choice for biomedical signals with large inter-subject amplitude variance.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ZScoreNormalization(nn.Module):
    """Per-sample (or dataset-wide) Z-score normalisation.

    Subtracts the mean and divides by the standard deviation along
    the specified axis.  Epsilon prevents division by zero for constant
    signals.

    Args:
        dim: Axis along which to compute statistics.  Defaults to ``-1``
            (per-sample normalisation along the time dimension).
        mean: Dataset-wide mean.  If ``None`` (default), computed from
            each sample independently.
        std: Dataset-wide std.  If ``None`` (default), computed from
            each sample independently.
        eps: Small value added to the std denominator.
    """

    def __init__(
        self,
        dim: int = -1,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self._mean = mean
        self._std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is not None:
            mean = torch.tensor(self._mean, dtype=x.dtype, device=x.device)
            std = torch.tensor(self._std, dtype=x.dtype, device=x.device)
        else:
            mean = x.mean(dim=self.dim, keepdim=True)
            std = x.std(dim=self.dim, keepdim=True)
        return (x - mean) / (std + self.eps)


class MinMaxNormalization(nn.Module):
    """Per-sample (or dataset-wide) min-max normalisation to [-1, 1].

    Args:
        dim: Axis along which to compute statistics (``-1`` by default).
            Pass ``None`` to compute the global min/max over the full tensor.
        min_val: Dataset-wide minimum.  ``None`` → per-sample.
        max_val: Dataset-wide maximum.  ``None`` → per-sample.
        eps: Small value added to the denominator.
    """

    def __init__(
        self,
        dim: Optional[int] = -1,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self._min_val = min_val
        self._max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._min_val is not None:
            min_x = torch.tensor(self._min_val, dtype=x.dtype, device=x.device)
            max_x = torch.tensor(self._max_val, dtype=x.dtype, device=x.device)
        elif self.dim is None:
            min_x = x.min()
            max_x = x.max()
        else:
            min_x = x.min(dim=self.dim, keepdim=True).values
            max_x = x.max(dim=self.dim, keepdim=True).values

        x = (x - min_x) / (max_x - min_x + self.eps)
        return (x - 0.5) * 2  # rescale to [-1, 1]
