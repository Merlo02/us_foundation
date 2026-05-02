"""
Signal processing transforms: numpy functions and nn.Module wrappers.

This module is the **canonical** home for the RF ultrasound preprocessing
primitives (bandpass filter, Hilbert envelope, interpolation/truncation).

Architecture
------------
* ``compute_*_numpy`` / ``bandpass_edges_from_center_frequency`` — pure numpy
  functions used by both the offline ETL pipeline (``etl/standardize.py``
  re-exports them under their original short names) and the online DataModule
  preprocessing path.
* ``ButterworthFilter``, ``HilbertEnvelope`` — ``nn.Module`` wrappers for
  composing transforms inside a PyTorch pipeline.

IMPORTANT — GPU/CPU note
------------------------
All scipy-based operations require **CPU arrays/tensors**.  Call transforms
*before* ``batch.to(device)`` (e.g. in ``on_before_batch_transfer``).

Online preprocessing notes
---------------------------
The numpy functions are applied per-sample inside ``__getitem__`` / stream
decode, *before* normalization and collation.  This is correct for the
variable-S path where the full signal is available.  For HDF5 fixed-S the
DataModule raises ``ValueError`` if a non-raw mode is requested (chunks are
pre-extracted; envelope on a chunk is physically wrong).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.signal import butter, filtfilt, hilbert
from scipy import signal as _scipy_signal
from torch import nn


# ===========================================================================
# Numpy preprocessing functions (canonical implementations)
# ===========================================================================

def bandpass_edges_from_center_frequency(
    tx_fc_hz: float,
    bandwidth_fraction: float,
    sampling_frequency_hz: float,
) -> tuple[float, float]:
    """Symmetric RF passband around carrier *tx_fc_hz*.

    Passband width is ``bandwidth_fraction * tx_fc_hz``.  Edges are clamped
    to ``(0, Nyquist)`` for ``filtfilt`` normalisation.
    """
    fc = float(tx_fc_hz)
    w = float(bandwidth_fraction) * fc
    nyq = 0.5 * float(sampling_frequency_hz)
    lo = max(1.0, fc - 0.5 * w)
    hi = min(fc + 0.5 * w, nyq * 0.999)
    if lo >= hi:
        span = min(0.01 * nyq, 0.5 * fc)
        lo = max(1.0, fc - span)
        hi = min(fc + span, nyq * 0.999)
    if lo >= hi:
        lo = max(1.0, hi * 0.5)
    return lo, hi


def compute_bandpass_numpy(
    signal: np.ndarray,
    sampling_frequency_hz: Optional[float],
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter via ``scipy.signal.filtfilt``.

    Returns the input unchanged (as float32) if ``sampling_frequency_hz`` is
    missing/non-positive or if the signal is too short for the filter.
    """
    if sampling_frequency_hz is None or sampling_frequency_hz <= 0:
        return np.asarray(signal, dtype=np.float32)

    nyq = 0.5 * sampling_frequency_hz
    low = max(1e-6, min(low_hz / nyq, 0.999))
    high = max(low + 1e-6, min(high_hz / nyq, 0.999))

    sig = np.asarray(signal, dtype=np.float64)
    if sig.size < 3 * order:
        return sig.astype(np.float32)

    b, a = butter(order, [low, high], btype="bandpass")
    filtered = filtfilt(b, a, sig, method="gust")
    return filtered.astype(np.float32)


def compute_envelope_numpy(signal: np.ndarray) -> np.ndarray:
    """Hilbert amplitude envelope ``|hilbert(x)|``, float32, same length.

    Should be applied on a band-limited (bandpass-filtered) signal for
    physically meaningful demodulation.
    """
    sig = np.asarray(signal, dtype=np.float64)
    return np.abs(hilbert(sig)).astype(np.float32)


def compute_interpolation_numpy(
    signal: np.ndarray,
    target_length: int,
    truncate_mode: str = "left",
) -> np.ndarray:
    """Resample *signal* to exactly *target_length* samples.

    Shorter signals are linearly interpolated; longer signals are truncated
    according to *truncate_mode* (``"left"`` keeps the first samples,
    ``"right"`` the last, ``"center"`` a centred window).
    """
    sig = np.asarray(signal, dtype=np.float32)
    tl = int(target_length)
    if tl <= 0:
        raise ValueError(f"target_length must be positive, got {tl}")
    n = sig.size
    if n > tl:
        if truncate_mode == "right":
            return sig[-tl:].copy()
        if truncate_mode == "center":
            start = (n - tl) // 2
            return sig[start: start + tl].copy()
        return sig[:tl].copy()  # "left"
    if n == tl:
        return sig.copy()
    if n == 0:
        return np.zeros(tl, dtype=np.float32)
    if n == 1:
        return np.full(tl, float(sig[0]), dtype=np.float32)
    xp = np.arange(n, dtype=np.float64)
    x = np.linspace(0.0, float(n - 1), num=tl, dtype=np.float64)
    out = np.interp(x, xp, sig.astype(np.float64))
    return out.astype(np.float32)


# ===========================================================================
# nn.Module wrappers (for composing transforms in a PyTorch pipeline)
# ===========================================================================

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
        self.b, self.a = _scipy_signal.butter(
            N=self.order, Wn=self.cutoff_freqs, btype=self.btype, fs=self.fs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the filter along the last axis.

        Args:
            x: Input tensor of shape ``(..., T)``.  Must be on CPU.

        Returns:
            Filtered tensor with the same shape, dtype float32.
        """
        filtered = _scipy_signal.filtfilt(self.b, self.a, x.numpy(), axis=-1)
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
        envelope = np.abs(_scipy_signal.hilbert(x.numpy(), axis=-1))
        return torch.tensor(envelope.copy(), dtype=torch.float32)
