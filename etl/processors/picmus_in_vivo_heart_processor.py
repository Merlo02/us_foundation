from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from .base_processor import BaseDatasetProcessor, RawSample


class PICMUSInVivoHeartProcessor(BaseDatasetProcessor):
    """PICMUS In_vivo_heart: UFF (HDF5) container with raw channel data.

    Expected dataset path: ``/channel_data/data``.

    The raw tensor may be:
    - 4D: (num_frames, num_waves, num_channels, num_samples) → reshaped to
      (num_frames * num_waves, num_channels, num_samples)
    - 3D: (num_acq, num_channels, num_samples)
    - 2D: (num_channels, num_samples) → reshaped to (1, num_channels, num_samples)

    Cropping: keep a depth window that ends at ``end_idx`` (default 2000) and
    starts at ``end_idx - etl_target_length``. This makes the discarded
    near-field proportional to the global ETL target length (e.g. 1250 → 750:2000,
    1400 → 600:2000).

    YAML ``extra``:
    - ``h5_dataset``: HDF5 dataset path (default ``/channel_data/data``)
    - ``end_idx``: window end index (exclusive, default 2000)
    - ``glob_pattern``: if input_path is a directory, glob pattern (default ``*.uff``)
    """

    def discover_files(self) -> list[str]:
        root = Path(self.config.input_path)
        if not root.exists():
            return []
        if root.is_file():
            return [str(root.resolve())]
        pattern = str(self.config.extra.get("glob_pattern", "*.uff"))
        files = sorted(root.glob(pattern))
        return [str(f.resolve()) for f in files]

    def load_and_yield(self, filepath: str) -> Iterator[RawSample]:
        if self.etl_target_length is None:
            raise RuntimeError(
                "PICMUSInVivoHeartProcessor requires etl_target_length from the runner."
            )

        h5_dataset = str(self.config.extra.get("h5_dataset", "/channel_data/data"))
        end_idx = int(self.config.extra.get("end_idx", 2000))
        start_idx = end_idx - int(self.etl_target_length)
        if start_idx < 0:
            raise ValueError(
                f"Computed start_idx={start_idx} is negative. "
                f"Decrease end_idx={end_idx} or target_length={self.etl_target_length}."
            )

        fp = Path(filepath)
        with h5py.File(filepath, "r") as f:
            raw = f[h5_dataset][:]

        dim = raw.shape
        if len(dim) == 4:
            num_frames, num_waves, num_channels, num_samples = dim
            tensor = raw.reshape((num_frames * num_waves, num_channels, num_samples))
            dims_meta = {
                "num_frames": int(num_frames),
                "num_waves": int(num_waves),
            }
        elif len(dim) == 3:
            tensor = raw
            dims_meta = {}
        elif len(dim) == 2:
            tensor = raw.reshape((1, dim[0], dim[1]))
            dims_meta = {}
        else:
            raise ValueError(f"Unexpected raw tensor shape: {dim}")

        tensor = np.asarray(tensor, dtype=np.float32)
        if tensor.shape[2] < end_idx:
            raise ValueError(
                f"Not enough samples for cropping: num_samples={tensor.shape[2]} < end_idx={end_idx}"
            )

        tensor = tensor[:, :, start_idx:end_idx]

        base_id = f"{self.config.name}_{fp.stem}"
        for acq_idx in range(tensor.shape[0]):
            for ch_idx in range(tensor.shape[1]):
                if not self.should_keep_channel(ch_idx):
                    continue
                sig = tensor[acq_idx, ch_idx]
                yield RawSample(
                    signal=sig.ravel(),
                    sample_id=f"{base_id}_a{acq_idx}_ch{ch_idx}",
                    source_dataset=self.config.name,
                    channel_idx=ch_idx,
                    sampling_frequency_hz=self.sampling_frequency_hz(),
                    metadata={
                        "file": fp.name,
                        "acq_idx": int(acq_idx),
                        "channel": int(ch_idx),
                        "crop_start_idx": int(start_idx),
                        "crop_end_idx": int(end_idx),
                        **dims_meta,
                    },
                )

