from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def standardize_length(
    signal: np.ndarray,
    target_length: int,
    mode: str = "left",
) -> np.ndarray:
    """Enforce the truncation ceiling ``target_length`` on *signal*.

    Signals longer than ``target_length`` are truncated according to *mode*.
    Signals shorter than or equal to ``target_length`` are returned **at
    their native length** (no interpolation, no zero-padding). This preserves
    the native sampling rate so the downstream multi-tokenizer can pick the
    branch ``W*`` without destructive resampling.
    """
    n = len(signal)
    if n <= target_length:
        return np.asarray(signal, dtype=np.float32)
    return _truncate(signal, target_length, mode).astype(np.float32)


# ------------------------------------------------------------------
# Preprocessing variants (Experiment D)
# ------------------------------------------------------------------

def compute_envelope(signal: np.ndarray) -> np.ndarray:
    """Hilbert-transform envelope ``|hilbert(x)|``.

    Operates sample-wise on a 1-D float signal and returns a float32 array of
    the same length. Useful for amplitude-demodulation experiments where the
    RF carrier is not needed by the foundation model.
    """
    sig = np.asarray(signal, dtype=np.float64)
    analytic = hilbert(sig)
    return np.abs(analytic).astype(np.float32)


def compute_bandpass(
    signal: np.ndarray,
    sampling_frequency_hz: Optional[float],
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter using ``scipy.signal.filtfilt``.

    Requires a valid ``sampling_frequency_hz`` — if missing, the signal is
    returned unchanged and a warning is emitted by the caller.
    """
    if sampling_frequency_hz is None or sampling_frequency_hz <= 0:
        return np.asarray(signal, dtype=np.float32)

    nyq = 0.5 * sampling_frequency_hz
    low = low_hz / nyq
    high = high_hz / nyq
    # Guard against degenerate cases (e.g. low-frequency signals).
    low = max(1e-6, min(low, 0.999))
    high = max(low + 1e-6, min(high, 0.999))

    sig = np.asarray(signal, dtype=np.float64)
    if sig.size < 3 * order:
        return sig.astype(np.float32)

    b, a = butter(order, [low, high], btype="bandpass")
    filtered = filtfilt(b, a, sig, method="gust")
    return filtered.astype(np.float32)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _truncate(signal: np.ndarray, target: int, mode: str) -> np.ndarray:
    if mode == "left":
        return signal[:target]
    if mode == "right":
        return signal[-target:]
    start = (len(signal) - target) // 2
    return signal[start : start + target]


def sanitize_signal(signal: np.ndarray) -> np.ndarray:
    """Ensure finite float32 output (no discarding — replace NaN/Inf with 0)."""
    out = np.asarray(signal, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def is_dead_signal(signal: np.ndarray, min_energy: float) -> bool:
    """True if RMS energy is below *min_energy*, signal is constant, empty, or non-finite."""
    sig = np.asarray(signal, dtype=np.float32)
    if sig.size == 0:
        return True
    if not np.isfinite(sig).all():
        return True
    if np.ptp(sig) == 0:
        return True
    return float(np.mean(sig.astype(np.float64) ** 2)) < min_energy


def validate_sample(signal: np.ndarray, target_length: int) -> bool:
    """Shape/dtype check for one output sample.

    After the interpolation removal, signals have **variable** lengths; the
    only structural constraints are: 1-D, float32, finite, and no longer than
    the truncation ceiling ``target_length``.
    """
    sig = np.asarray(signal)
    if sig.ndim != 1:
        return False
    if sig.dtype != np.float32:
        return False
    if sig.size == 0 or sig.size > target_length:
        return False
    if not np.isfinite(sig).all():
        return False
    return True
