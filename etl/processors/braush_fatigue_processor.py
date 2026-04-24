from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
import scipy.io as sio

from .base_processor import BaseDatasetProcessor, RawSample


class BraushFatigueProcessor(BaseDatasetProcessor):
    """Braush fatigue — stessa logica dello script di variabilità originale.

    Per ogni ``.mat`` nello zip: ``sio.loadmat(io.BytesIO(f.read()))``,
    poi ``mat[\"loadedData\"][:, :-1]`` (esclude ultima colonna timestamp).
    Ogni riga è un'acquisizione, un canale (0). La lunghezza viene portata a
    ``target_length`` dalla pipeline ETL (es. troncamento se
    ``interpolate_short: false`` nel YAML).
    """

    def discover_files(self) -> list[str]:
        root = Path(self.config.input_path)
        if root.is_file() and root.suffix.lower() == ".zip":
            return [str(root.resolve())]
        pattern = self.config.extra.get("glob_pattern", "*.zip")
        recursive = self.config.extra.get("recursive", False)
        if recursive:
            files = sorted(root.rglob(pattern))
        else:
            files = sorted(root.glob(pattern))
        return [str(f.resolve()) for f in files]

    def load_and_yield(self, filepath: str) -> Iterator[RawSample]:
        zip_stem = Path(filepath).stem
        row_global = 0

        with zipfile.ZipFile(filepath, "r") as z:
            names = sorted([n for n in z.namelist() if n.endswith(".mat")])
            for fname in names:
                with z.open(fname) as f:
                    mat = sio.loadmat(io.BytesIO(f.read()))

                arr = mat["loadedData"][:, :-1].astype(np.float64)
                mat_stem = Path(fname).stem

                for i in range(arr.shape[0]):
                    ch_idx = 0
                    if not self.should_keep_channel(ch_idx):
                        continue
                    signal = np.asarray(arr[i, :], dtype=np.float32).ravel()
                    yield RawSample(
                        signal=signal,
                        sample_id=f"{self.config.name}_{zip_stem}_{mat_stem}_r{row_global}",
                        source_dataset=self.config.name,
                        channel_idx=ch_idx,
                        sampling_frequency_hz=self.sampling_frequency_hz(),
                        metadata={
                            "zip": zip_stem,
                            "mat_file": mat_stem,
                            "row_in_zip_order": row_global,
                            "row_in_mat": i,
                        },
                    )
                    row_global += 1
