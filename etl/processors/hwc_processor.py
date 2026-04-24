from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .base_processor import BaseDatasetProcessor, RawSample

CHANNEL_COLUMNS = ["TR_1", "TR_2", "TR_3", "TR_4"]


class HWCProcessor(BaseDatasetProcessor):
    """Processor for the HWC ultrasound dataset.

    Source files are ``.pkl`` pandas DataFrames with columns
    ``TR_1 .. TR_4`` (4 transducer channels) plus metadata columns
    like ``label``, ``subject_id``, etc.

    Each channel of each row becomes an independent 1-D sample.
    """

    def discover_files(self) -> list[str]:
        pattern = f"{self.config.input_path}/**/*.pkl"
        files = sorted(glob(pattern, recursive=True))
        return files

    def load_and_yield(self, filepath: str) -> Iterator[RawSample]:
        df = pd.read_pickle(filepath)
        fname = Path(filepath).stem

        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            for ch_idx, col in enumerate(CHANNEL_COLUMNS):
                if not self.should_keep_channel(ch_idx):
                    continue

                signal = np.asarray(row[col], dtype=np.float32).ravel()
                metadata: dict = {}
                if "label" in df.columns:
                    metadata["label"] = row["label"]
                if "subject_id" in df.columns:
                    metadata["subject"] = row["subject_id"]
                if "session_id" in df.columns:
                    metadata["session"] = row["session_id"]

                yield RawSample(
                    signal=signal,
                    sample_id=f"hwc_{fname}_r{row_idx}_ch{ch_idx}",
                    source_dataset=self.config.name,
                    channel_idx=ch_idx,
                    sampling_frequency_hz=self.sampling_frequency_hz(),
                    metadata=metadata,
                )
