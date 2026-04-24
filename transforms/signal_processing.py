"""
On-the-fly signal processing transforms (nn.Module, runs on CPU via scipy).

Ported from TimeFM ``transforms/signal_processing.py`` with minor
extensions for ultrasound use cases.

IMPORTANT — GPU/CPU note
------------------------
Both ``ButterworthFilter`` and ``HilbertEnvelope`` use scipy under the
hood and therefore operate on **CPU tensors**.  If the batch is already
on GPU, call these transforms *before* ``batch.to(device)`` (i.e. in
``on_before_batch_transfer`` of the DataModule, which runs on CPU).

These transforms are the *nn.Module equivalents* of the numpy helpers in
``etl/standardize.py``.  They exist separately so that:
  1. ETL preprocessing is fixed at dataset creation time (offline).
  2. Training-time preprocessing can be applied on-the-fly without
     regenerating the dataset (useful for ablations in Experiment D).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import signal
from torch import nn


class ButterworthFilter(nn.Module):
    """Zero-phase Butterworth band-pass (or low/high-pass) filter.

    Wraps ``scipy.signal.butter`` + ``filtfilt`` as a PyTorch ``nn.Module``
    so it can be composed in a ``nn.Sequential`` transform pipeline.

    Args:
        order: Filter order.
        cutoff_freqs: Cutoff frequency (scalar for low/high pass) or
            ``[low, high]`` for band-pass.
        fs: Sampling frequency in Hz.
        btype: Filter type — ``"bandpass"`` (default), ``"lowpass"``,
            ``"highpass"``, ``"bandstop"``.
    """

    def __init__(
        self,
        order: int,
        cutoff_freqs: float | list[float] | tuple[float, ...],
        fs: float,
        btype: str = "bandpass",
    ) -> None:
        super().__init__()
        self.order = order
        self.cutoff_freqs = cutoff_freqs
        self.fs = fs
        self.btype = btype
        self.b, self.a = signal.butter(
            N=self.order, Wn=self.cutoff_freqs, btype=self.btype, fs=self.fs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the filter along the last axis.

        Args:
            x: Input tensor of shape ``(..., T)``.  Must be on CPU.

        Returns:
            Filtered tensor with the same shape, dtype float32.
        """
        filtered = signal.filtfilt(self.b, self.a, x.numpy(), axis=-1)
        return torch.tensor(filtered.copy(), dtype=torch.float32)


class HilbertEnvelope(nn.Module):
    """Amplitude envelope via Hilbert transform (|analytic signal|).

    Wraps ``scipy.signal.hilbert`` as a PyTorch ``nn.Module``.  Useful
    for converting raw RF ultrasound signals to their amplitude envelope
    without modifying the ETL dataset.

    Args:
        None — stateless transform.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the amplitude envelope along the last axis.

        Args:
            x: Input tensor of shape ``(..., T)``.  Must be on CPU.

        Returns:
            Amplitude envelope with the same shape, dtype float32.
        """
        envelope = np.abs(signal.hilbert(x.numpy(), axis=-1))
        return torch.tensor(envelope.copy(), dtype=torch.float32)
