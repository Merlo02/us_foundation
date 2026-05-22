"""HDF5 writer for the labeled (downstream) ETL.

Produces a single ``all.h5`` with the fixed schema::

    /signal       (N, C, T) float32
    /labels       (N,)      int64
    /session_id   (N,)      int64
    /patient_id   (N,)      int64

    root attrs:
      sampling_frequency_hz : float32  (scalar)
      dataset_name          : utf8
      label_type            : utf8
      num_classes           : int64
      num_channels          : int64
      samples_per_frame     : int64

Rows are buffered in memory until ``flush_every`` are queued, then appended
to resizable datasets. ``close()`` trims and writes the root attributes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np


class DownstreamHDF5Writer:
    """Buffered row-at-a-time writer for the downstream schema."""

    def __init__(
        self,
        h5_path: str | Path,
        num_channels: int,
        samples_per_frame: int,
        sampling_frequency_hz: float,
        dataset_name: str,
        label_type: str,
        num_classes: int,
        flush_every: int = 1024,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.num_channels = int(num_channels)
        self.samples_per_frame = int(samples_per_frame)
        self.sampling_frequency_hz = float(sampling_frequency_hz)
        self.dataset_name = str(dataset_name)
        self.label_type = str(label_type)
        self.num_classes = int(num_classes)
        self.flush_every = max(1, int(flush_every))

        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self._f: Optional[h5py.File] = h5py.File(self.h5_path, "w", libver="latest")
        C, T = self.num_channels, self.samples_per_frame
        self._f.create_dataset(
            "signal", shape=(0, C, T), maxshape=(None, C, T),
            dtype="float32", chunks=(min(self.flush_every, 256), C, T),
        )
        for name in ("labels", "session_id", "patient_id"):
            self._f.create_dataset(
                name, shape=(0,), maxshape=(None,),
                dtype="int64", chunks=(min(self.flush_every, 1024),),
            )

        self._buf_signal: list[np.ndarray] = []
        self._buf_label: list[int] = []
        self._buf_session: list[int] = []
        self._buf_patient: list[int] = []
        self._n_written = 0

    # ------------------------------------------------------------------
    def write(
        self, signal: np.ndarray, label: int, session_id: int, patient_id: int,
    ) -> None:
        if signal.shape != (self.num_channels, self.samples_per_frame):
            raise ValueError(
                f"signal shape {signal.shape} does not match declared "
                f"({self.num_channels}, {self.samples_per_frame}).",
            )
        if signal.dtype != np.float32:
            signal = signal.astype(np.float32, copy=False)
        self._buf_signal.append(signal)
        self._buf_label.append(int(label))
        self._buf_session.append(int(session_id))
        self._buf_patient.append(int(patient_id))
        if len(self._buf_signal) >= self.flush_every:
            self._flush()

    # ------------------------------------------------------------------
    def _flush(self) -> None:
        if not self._buf_signal or self._f is None:
            return
        n_new = len(self._buf_signal)
        n0 = self._n_written
        n1 = n0 + n_new

        self._f["signal"].resize((n1, self.num_channels, self.samples_per_frame))
        self._f["signal"][n0:n1] = np.stack(self._buf_signal, axis=0)
        for name, buf in (
            ("labels", self._buf_label),
            ("session_id", self._buf_session),
            ("patient_id", self._buf_patient),
        ):
            self._f[name].resize((n1,))
            self._f[name][n0:n1] = np.asarray(buf, dtype=np.int64)

        self._buf_signal.clear()
        self._buf_label.clear()
        self._buf_session.clear()
        self._buf_patient.clear()
        self._n_written = n1

    # ------------------------------------------------------------------
    @property
    def num_written(self) -> int:
        return self._n_written + len(self._buf_signal)

    def close(self) -> None:
        if self._f is None:
            return
        self._flush()
        self._f.attrs["sampling_frequency_hz"] = np.float32(self.sampling_frequency_hz)
        self._f.attrs["dataset_name"] = self.dataset_name
        self._f.attrs["label_type"] = self.label_type
        self._f.attrs["num_classes"] = np.int64(self.num_classes)
        self._f.attrs["num_channels"] = np.int64(self.num_channels)
        self._f.attrs["samples_per_frame"] = np.int64(self.samples_per_frame)
        self._f.close()
        self._f = None

    def __enter__(self) -> "DownstreamHDF5Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["DownstreamHDF5Writer"]
