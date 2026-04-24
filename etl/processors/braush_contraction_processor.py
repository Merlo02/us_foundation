from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .base_processor import BaseDatasetProcessor, RawSample


def _parse_arff_data_line(line: str) -> np.ndarray | None:
    """Parse one ARFF data line: comma-separated floats between ``[`` and ``]``.

    Falls back to the legacy slice ``line[2:bracket_end]`` if no ``[`` is found
    (matches the original variability script).
    """
    line = line.strip()
    if not line:
        return None
    try:
        bracket_end = line.rindex("]")
    except ValueError:
        return None
    if "[" in line:
        lo = line.index("[")
        ascan_str = line[lo + 1 : bracket_end]
    else:
        ascan_str = line[2:bracket_end]
    ascan_str = ascan_str.strip()
    if not ascan_str:
        return None
    return np.asarray(ascan_str.split(","), dtype=np.float32)


class BraushContractionProcessor(BaseDatasetProcessor):
    """Braush contraction dataset: single ``.arff`` with one A-scan per line.

    Each row after the header contains a bracketed comma-separated vector
    (shape ``(n_timepoints,)``).  There is **one channel** per acquisition
    (channel index 0).      Signals are long (e.g. 3000 samples). Set ``extra.interpolate_short: false``
    in YAML so the ETL pipeline **truncates** to ``target_length`` and never
    interpolates shorter segments (only zero-pad if ever shorter).

    Default input: ``completeDatabase.arff`` under ``input_path`` — override via
    ``extra.glob_pattern`` / ``extra.recursive``.
    """

    def discover_files(self) -> list[str]:
        root = Path(self.config.input_path)
        pattern = self.config.extra.get("glob_pattern", "*.arff")
        recursive = self.config.extra.get("recursive", False)
        if root.is_file():
            return [str(root.resolve())]
        if recursive:
            files = sorted(root.rglob(pattern))
        else:
            files = sorted(root.glob(pattern))
        return [str(f.resolve()) for f in files]

    def load_and_yield(self, filepath: str) -> Iterator[RawSample]:
        header_lines = int(self.config.extra.get("arff_header_lines", 16))
        fname = Path(filepath).stem
        row = 0

        with open(filepath, encoding="utf-8", errors="replace") as f:
            for line_idx, line in enumerate(f):
                if line_idx < header_lines:
                    continue
                signal = _parse_arff_data_line(line)
                if signal is None or signal.size == 0:
                    continue

                ch_idx = 0
                if not self.should_keep_channel(ch_idx):
                    continue

                yield RawSample(
                    signal=signal,
                    sample_id=f"{self.config.name}_{fname}_r{row}",
                    source_dataset=self.config.name,
                    channel_idx=ch_idx,
                    sampling_frequency_hz=self.sampling_frequency_hz(),
                    metadata={
                        "file": fname,
                        "row": row,
                        "source_line": line_idx,
                    },
                )
                row += 1
