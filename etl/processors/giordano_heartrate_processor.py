from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.signal import resample

from .base_processor import BaseDatasetProcessor, RawSample

DEFAULT_RESAMPLE_LENGTH = 500


class GiordanoHeartrateProcessor(BaseDatasetProcessor):
    """Giordano heartrate (2024): ``lvout*.txt`` CSV files under subject folders.

    Each file is read with comma separator and header row; columns that are
    all NaN are dropped. ``values.T`` matches the reference pipeline: each row
    is one A-mode trace (multiple acquisitions per file). The global ETL model
    uses one *logical* channel for this dataset, so every yielded sample has
    ``channel_idx=0``; the acquisition index is stored in metadata / sample_id.

    Signals are downsampled with Fourier resampling
    (``scipy.signal.resample``) along time to *resample_length* samples
    (default 500). Resampling to the global ``target_length`` is handled by
    :mod:`etl.standardize` (linear interpolation when shorter, truncation when
    longer).

    YAML ``extra``:

    - ``glob_pattern``: glob relative to ``input_path`` (default ``*/lvout*.txt``).
    - ``resample_length``: Fourier resample target length (default 500).
    """

    def discover_files(self) -> list[str]:
        root = Path(self.config.input_path)
        pattern = str(self.config.extra.get("glob_pattern", "*/lvout*.txt"))
        if not root.exists():
            return []
        if root.is_file():
            return [str(root.resolve())]
        files = sorted(root.glob(pattern))
        return [str(f.resolve()) for f in files]

    def load_and_yield(self, filepath: str) -> Iterator[RawSample]:
        n_out = int(self.config.extra.get("resample_length", DEFAULT_RESAMPLE_LENGTH))
        fp = Path(filepath)
        parent_dir = fp.parent.name

        df_raw = pd.read_csv(filepath, sep=",", header=0).dropna(axis=1, how="all")
        if df_raw.empty:
            return

        a_mode_signals = df_raw.values.astype(np.float64).T
        if a_mode_signals.size == 0:
            return

        if not self.should_keep_channel(0):
            return

        signals_out = resample(a_mode_signals, n_out, axis=1)
        signals_out = np.asarray(signals_out, dtype=np.float32)

        fname = fp.stem
        for a_mode_idx, row in enumerate(signals_out):
            yield RawSample(
                signal=row.ravel(),
                sample_id=(
                    f"{self.config.name}_{parent_dir}_{fname}_a{a_mode_idx}"
                ),
                source_dataset=self.config.name,
                channel_idx=0,
                sampling_frequency_hz=self.sampling_frequency_hz(),
                metadata={
                    "subject_dir": parent_dir,
                    "file": fp.name,
                    "a_mode_idx": a_mode_idx,
                },
            )
