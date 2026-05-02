from __future__ import annotations

from typing import Optional

import numpy as np

# Canonical implementations live in transforms/signal_processing.py.
# Re-exported here under their original short names for full backward
# compatibility with the ETL runner and any external callers.
from transforms.signal_processing import (
    bandpass_edges_from_center_frequency,
    compute_bandpass_numpy as compute_bandpass,
    compute_envelope_numpy as compute_envelope,
    compute_interpolation_numpy as compute_interpolation,
)

__all__ = [
    "bandpass_edges_from_center_frequency",
    "compute_bandpass",
    "compute_envelope",
    "compute_interpolation",
    "is_dead_signal",
    "sanitize_signal",
    "standardize_length",
    "validate_sample",
]


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
