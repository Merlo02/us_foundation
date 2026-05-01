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
    """Hilbert envelope ``|hilbert(x)|``.

    Prefer **band-limited** input (e.g. after RF bandpass). Returns float32, same length.
    """
    sig = np.asarray(signal, dtype=np.float64)
    analytic = hilbert(sig)
    return np.abs(analytic).astype(np.float32)


def bandpass_edges_from_center_frequency(
    tx_fc_hz: float,
    bandwidth_fraction: float,
    sampling_frequency_hz: float,
) -> tuple[float, float]:
    """Symmetric RF passband around carrier ``tx_fc_hz``.

    Passband width is ``bandwidth_fraction * tx_fc_hz`` (narrower fractional
    width keeps more energy near the design frequency). Edges are clamped to
    ``(0, Nyquist)`` for ``filtfilt`` normalization.
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


def compute_interpolation(
    signal: np.ndarray,
    target_length: int,
    truncate_mode: str = "left",
) -> np.ndarray:
    """Resample to exactly ``target_length`` samples.

    **Shorter** than ``target_length``: linear interpolation along the sample
    axis (uniform abscissae from first to last index).

    **Longer** than ``target_length``: truncated with :func:`_truncate`:
    ``left`` keeps the *first* ``target_length`` samples (drop tail);
    ``right`` keeps the *last* ``target_length`` samples (drop head);
    ``center`` keeps a centered window.
    """
    sig = np.asarray(signal, dtype=np.float32)
    tl = int(target_length)
    if tl <= 0:
        raise ValueError("target_length must be positive")
    n = sig.size
    if n > tl:
        return _truncate(sig, tl, truncate_mode).astype(np.float32)
    if n == tl:
        return sig.copy()
    if n == 0:
        return np.zeros(tl, dtype=np.float32)
    if n == 1:
        return np.full(tl, float(sig[0]), dtype=np.float32)
    xp = np.arange(n, dtype=np.float64)
    x = np.linspace(0.0, float(n - 1), num=tl, dtype=np.float64)
    y = sig.astype(np.float64)
    out = np.interp(x, xp, y)
    return out.astype(np.float32)


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


def validate_sample(signal: np.ndarray) -> bool:
    """Shape/dtype check for one output sample.

    The ETL pipeline is length-preserving: signals have **variable** lengths.
    Structural constraints are: 1-D, float32, finite, and non-empty.
    """
    sig = np.asarray(signal)
    if sig.ndim != 1:
        return False
    if sig.dtype != np.float32:
        return False
    if sig.size == 0:
        return False
    if not np.isfinite(sig).all():
        return False
    return True
