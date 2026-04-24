from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

from .base_processor import BaseDatasetProcessor, RawSample

log = logging.getLogger(__name__)

# Default: only the four main experiment arrays (matches vostrikov_grawus analysis).
DEFAULT_EXPERIMENT_FILES = (
    "experiment1.npy",
    "experiment2.npy",
    "experiment3.npy",
    "experiment4.npy",
)


class GRAWUSProcessor(BaseDatasetProcessor):
    """Processor for the GRAWUS 2023 ultrasound dataset.

    Source files are ``.npy`` arrays with shape ``(metadata_rows + timepoints,
    n_channels * n_acquisitions)``.  The first ``metadata_rows`` rows (default
    4) are metadata and are stripped — same as ``arr[4:, :]`` in the reference
    script.  The remainder is reshaped to
    ``(n_acquisitions, n_channels, timepoints)``.

    By default only ``experiment1.npy`` … ``experiment4.npy`` under
    ``input_path`` are used (no recursive glob, so ``*_filtered``, ``old/``,
    etc. are ignored).  Override via YAML ``extra.experiment_files``.

    Channels excluded in config (e.g. 4 and 7) are dropped before yielding.
    """

    def discover_files(self) -> list[str]:
        root = Path(self.config.input_path)
        names = self.config.extra.get("experiment_files", list(DEFAULT_EXPERIMENT_FILES))
        files: list[str] = []
        for name in names:
            p = root / name
            if not p.is_file():
                log.warning("GRAWUS: expected file not found (skipped): %s", p)
                continue
            files.append(str(p.resolve()))
        files.sort()
        return files

    def load_and_yield(self, filepath: str) -> Iterator[RawSample]:
        n_meta = self.config.extra.get("metadata_rows", 4)
        n_ch = self.config.extra.get("n_channels", 8)

        arr = np.load(filepath, allow_pickle=True)
        signals = arr[n_meta:, :]
        n_tp = signals.shape[0]
        n_acq = signals.shape[1] // n_ch

        # (timepoints, n_acq * n_ch) → (n_acq, n_ch, timepoints)
        data = (
            signals.reshape(n_tp, n_acq, n_ch)
            .transpose(1, 2, 0)
            .astype(np.float32)
        )

        fname = Path(filepath).stem
        for acq_idx in range(data.shape[0]):
            for ch_idx in range(n_ch):
                if not self.should_keep_channel(ch_idx):
                    continue

                signal = data[acq_idx, ch_idx, :]
                yield RawSample(
                    signal=signal,
                    sample_id=f"grawus_{fname}_a{acq_idx}_ch{ch_idx}",
                    source_dataset=self.config.name,
                    channel_idx=ch_idx,
                    sampling_frequency_hz=self.sampling_frequency_hz(),
                    metadata={
                        "experiment": fname,
                        "acquisition": acq_idx,
                    },
                )
