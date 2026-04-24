from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from .base_processor import BaseDatasetProcessor, RawSample


class PICMUSCarotidCrossProcessor(BaseDatasetProcessor):
    """PICMUS carotid cross: UFF (HDF5) container with channel data.

    This processor mirrors the reference extraction:
    - looks under ``/channel_data`` for ``data`` or ``data_real``
    - converts shapes to (num_acq, num_channels, num_samples)
    - yields one RawSample per (acquisition, channel)

    Signals are typically length 1536; the global ETL pipeline truncates to
    ``target_length`` (e.g. 1250). To enforce "no interpolation", set
    ``extra.interpolate_short: false`` in the YAML for this dataset.

    YAML ``extra``:
    - ``h5_group``: HDF5 group path (default ``/channel_data``)
    - ``data_key``: optional explicit dataset key (``data`` or ``data_real``)
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
        h5_group = str(self.config.extra.get("h5_group", "/channel_data"))
        explicit_key = self.config.extra.get("data_key")

        fp = Path(filepath)
        with h5py.File(filepath, "r") as f:
            if h5_group not in f:
                raise KeyError(f"Group '{h5_group}' not found in file.")

            grp = f[h5_group]
            if explicit_key is not None:
                if explicit_key not in grp:
                    raise KeyError(
                        f"Requested data_key='{explicit_key}' not found in {h5_group}."
                    )
                data_key = str(explicit_key)
            else:
                data_key = "data" if "data" in grp else "data_real" if "data_real" in grp else None
                if data_key is None:
                    raise KeyError(
                        f"No 'data' or 'data_real' found under {h5_group}."
                    )

            raw = grp[data_key][:]

        dim = raw.shape
        if len(dim) == 4:
            n_frames, n_waves, n_channels, n_samples = dim
            tensor = raw.reshape((n_frames * n_waves, n_channels, n_samples))
            dims_meta = {
                "num_frames": int(n_frames),
                "num_waves": int(n_waves),
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
                        "h5_group": h5_group,
                        "data_key": data_key,
                        "acq_idx": int(acq_idx),
                        "channel": int(ch_idx),
                        "num_samples_raw": int(sig.shape[0]),
                        **dims_meta,
                    },
                )

