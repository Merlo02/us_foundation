from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np

try:
    import webdataset as wds
except ImportError:
    wds = None  # type: ignore[assignment]


def _make_json_safe(obj):
    """Convert numpy scalars to Python natives so json.dumps() works."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class WebDatasetWriter:
    """Writes standardized samples into WebDataset ``.tar`` shards.

    Each sample occupies two files per tar entry (``<key>.signal.npy`` and
    ``<key>.metadata.json``). The metadata dict is expected to already
    contain top-level ``sampling_frequency_hz`` and ``dataset_source`` fields
    (injected by the runner).
    """

    def __init__(
        self,
        output_dir: str,
        split: str,
        samples_per_shard: int,
    ) -> None:
        if wds is None:
            raise ImportError("webdataset is required for WebDatasetWriter")
        shard_dir = Path(output_dir) / "wds" / split
        shard_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(shard_dir / "shard-%06d.tar")
        self.sink = wds.ShardWriter(pattern, maxcount=samples_per_shard)
        self.count = 0

    def write(self, sample_id: str, signal: np.ndarray, metadata: dict) -> None:
        self.sink.write({
            "__key__": sample_id,
            "signal.npy": np.asarray(signal, dtype=np.float32),
            "metadata.json": _make_json_safe(metadata),
        })
        self.count += 1

    def close(self) -> None:
        self.sink.close()


class HDF5Writer:
    """CSR-style HDF5 writer for variable-length ultrasound signals.

    File layout (one file per split):

    - ``data``                 ``(M,) float32``  — all signals concatenated
    - ``offsets``              ``(N+1,) int64``  — ``signal[i] = data[offsets[i]:offsets[i+1]]``
    - ``sampling_frequencies`` ``(N,) float32``
    - ``dataset_sources``      ``(N,) vlen utf-8 str``

    This avoids the spurious zero-padding that would be necessary to store
    variable-length signals in a 2-D ``X`` dataset. ``offsets`` is kept in
    RAM by the ``HDF5Dataset`` at train time (N * 8 bytes), enabling O(1)
    random access to any signal.
    """

    _STR_DTYPE = h5py.string_dtype(encoding="utf-8")

    def __init__(
        self,
        output_dir: str,
        split: str,
        chunk_mb: int = 128,
    ) -> None:
        hdf5_dir = Path(output_dir) / "hdf5"
        hdf5_dir.mkdir(parents=True, exist_ok=True)
        path = hdf5_dir / f"{split}.h5"

        # Chunk sizing: pack roughly chunk_mb of float32 per chunk.
        samples_per_chunk = max(1, (chunk_mb * 1024 * 1024) // 4)

        self.f = h5py.File(str(path), "w")
        self.data = self.f.create_dataset(
            "data",
            shape=(0,),
            maxshape=(None,),
            dtype="float32",
            chunks=(samples_per_chunk,),
        )
        # offsets has N+1 entries: starts with 0, appended lazily on flush.
        self.offsets = self.f.create_dataset(
            "offsets",
            shape=(1,),
            maxshape=(None,),
            dtype="int64",
            chunks=(max(1, samples_per_chunk // 1024),),
        )
        self.offsets[0] = 0

        self.freqs = self.f.create_dataset(
            "sampling_frequencies",
            shape=(0,),
            maxshape=(None,),
            dtype="float32",
            chunks=(max(1, samples_per_chunk // 1024),),
        )
        self.sources = self.f.create_dataset(
            "dataset_sources",
            shape=(0,),
            maxshape=(None,),
            dtype=self._STR_DTYPE,
            chunks=(max(1, samples_per_chunk // 4096),),
        )

        # Batched buffers to reduce HDF5 resize overhead.
        self._buf_signals: list[np.ndarray] = []
        self._buf_freqs: list[float] = []
        self._buf_sources: list[str] = []
        self._flush_every = 1024

        self.count = 0
        self._total_samples = 0  # number of float32 values written to ``data``

    def write(
        self,
        signal: np.ndarray,
        sampling_frequency_hz: Optional[float],
        dataset_source: str,
    ) -> None:
        sig = np.ascontiguousarray(signal, dtype=np.float32)
        self._buf_signals.append(sig)
        self._buf_freqs.append(float(sampling_frequency_hz or 0.0))
        self._buf_sources.append(str(dataset_source))
        if len(self._buf_signals) >= self._flush_every:
            self._flush()

    def _flush(self) -> None:
        if not self._buf_signals:
            return

        lengths = np.fromiter(
            (s.shape[0] for s in self._buf_signals), dtype=np.int64,
            count=len(self._buf_signals),
        )
        flat = np.concatenate(self._buf_signals, axis=0)
        new_total = self._total_samples + int(flat.size)
        self.data.resize((new_total,))
        self.data[self._total_samples:new_total] = flat

        # Cumulative offsets: offsets[i+1] = offsets[i] + len(signal_i).
        new_offsets = np.cumsum(lengths) + self._total_samples
        old_n = self.count
        new_n = old_n + len(self._buf_signals)
        self.offsets.resize((new_n + 1,))
        self.offsets[old_n + 1 : new_n + 1] = new_offsets

        self.freqs.resize((new_n,))
        self.freqs[old_n:new_n] = np.asarray(self._buf_freqs, dtype=np.float32)

        self.sources.resize((new_n,))
        self.sources[old_n:new_n] = np.asarray(self._buf_sources, dtype=object)

        self.count = new_n
        self._total_samples = new_total
        self._buf_signals.clear()
        self._buf_freqs.clear()
        self._buf_sources.clear()

    def close(self) -> None:
        self._flush()
        self.f.attrs["num_samples"] = self.count
        self.f.attrs["total_data_length"] = self._total_samples
        self.f.attrs["layout"] = "csr"
        self.f.close()
