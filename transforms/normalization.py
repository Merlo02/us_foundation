"""
On-the-fly signal normalisation transforms (GPU-compatible nn.Module).

Ported and extended from TimeFM ``transforms/normalization.py``.

Loader-side NumPy helpers apply **global per-signal** statistics computed at
ETL time (HDF5 sidecar datasets / WebDataset metadata). ETL still writes raw
sample payloads in ``data`` / ``signal.npy``; statistics are persisted alongside.

These Torch modules remain useful for GPU transforms elsewhere (per-batch /
per-sample). DataModules prefer :func:`normalize_signal_numpy` for chunked HDF5
reads with ETL-global stats.

Note on axis conventions
------------------------
All modules operate on the **last axis** (time axis) by default.
For a batch of 1-D ultrasound signals the tensor shape is ``(B, T)``,
so ``dim=-1`` normalises per-sample across time, which is the standard
choice for biomedical signals with large inter-subject amplitude variance.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
from torch import nn

NormalizationMode = Literal["none", "zscore", "minmax"]

NORMALIZATION_TYPES = frozenset(("none", "zscore", "minmax"))


def validate_normalization_type(mode: str) -> NormalizationMode:
    """Return ``mode`` if valid; raise ``ValueError`` otherwise."""
    if mode not in NORMALIZATION_TYPES:
        raise ValueError(
            f"normalization_type must be one of {sorted(NORMALIZATION_TYPES)}, got {mode!r}",
        )
    return mode  # type: ignore[return-value]


def normalize_signal_numpy(
    chunk: np.ndarray,
    mode: NormalizationMode,
    mean: float,
    std: float,
    vmin: float,
    vmax: float,
    eps_z: float = 1e-6,
    eps_mm: float = 1e-10,
) -> np.ndarray:
    """Apply global-stat normalization to a signal chunk (float32).

    - **zscore**: ``(chunk - mean) / (std + eps_z)``
    - **minmax**: ``(chunk - vmin) / (vmax - vmin + eps_mm)`` — no [-1, 1] remap
      (differs from :class:`MinMaxNormalization`).
    - **none**: return ``chunk`` as float32 view/array without copying when already float32.
    """
    x = np.asarray(chunk, dtype=np.float32)
    if mode == "none":
        return x
    if mode == "zscore":
        m = np.float32(mean)
        s = np.float32(std) + np.float32(eps_z)
        return ((x - m) / s).astype(np.float32)
    if mode == "minmax":
        lo = np.float32(vmin)
        denom = np.float32(vmax) - lo + np.float32(eps_mm)
        return ((x - lo) / denom).astype(np.float32)
    raise ValueError(f"Unknown normalization mode: {mode!r}")


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
