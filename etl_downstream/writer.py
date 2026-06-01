"""HDF5 writer for the labeled (downstream) ETL.

Produces a single ``all.h5`` with a fixed schema. The ``/labels`` dataset
shape/dtype depends on ``task_type``:

**Classification** (``task_type='classification'``)::

    /signal       (N, C, T) float32
    /labels       (N,)      int64        ← one class id per frame
    /session_id   (N,)      int64
    /patient_id   (N,)      int64

    root attrs:
      sampling_frequency_hz : float32  (scalar)
      dataset_name          : utf8
      label_type            : utf8
      task_type             : utf8   ("classification")
      num_classes           : int64
      num_channels          : int64
      samples_per_frame     : int64

**Regression** (``task_type='regression'``)::

    /signal       (N, C, T) float32
    /labels       (N, K)    float32      ← K continuous targets per frame
    /session_id   (N,)      int64
    /patient_id   (N,)      int64

    root attrs: same as above plus
      task_type   : utf8   ("regression")
      num_outputs : int64           (= K)
      label_names : utf8[K]         (one name per regression target)
      num_classes : int64           (= 0, kept for schema symmetry)

``task_type`` defaults to ``"classification"`` so existing single-int
processors (and any ``all.h5`` produced before this attr existed) keep
their exact behaviour — the downstream DataModule treats a missing
``task_type`` attr as classification.

Rows are buffered in memory until ``flush_every`` are queued, then appended
to resizable datasets. ``close()`` trims and writes the root attributes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import h5py
import numpy as np

_VALID_TASK_TYPES = ("classification", "regression")

# A label is either a single class id (classification) or a 1-D vector of
# continuous targets (regression).
LabelLike = Union[int, np.integer, float, Sequence[float], np.ndarray]


class DownstreamHDF5Writer:
    """Buffered row-at-a-time writer for the downstream schema.

    Parameters
    ----------
    num_classes :
        Number of classes (classification). Stored verbatim as a root attr;
        pass ``0`` for regression.
    task_type :
        ``"classification"`` (default) or ``"regression"``. Decides the
        ``/labels`` dataset shape/dtype and which extra attrs are written.
    num_outputs :
        Number of continuous targets per frame (regression only, ``K``).
        Required (``>= 1``) when ``task_type='regression'``; ignored
        otherwise.
    label_names :
        Optional ``K`` names for the regression targets (e.g. the DOF
        names). Stored as the ``label_names`` root attr when given.
    """

    def __init__(
        self,
        h5_path: str | Path,
        num_channels: int,
        samples_per_frame: int,
        sampling_frequency_hz: float,
        dataset_name: str,
        label_type: str,
        num_classes: int,
        task_type: str = "classification",
        num_outputs: int = 0,
        label_names: Optional[Sequence[str]] = None,
        flush_every: int = 1024,
    ) -> None:
        if task_type not in _VALID_TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {_VALID_TASK_TYPES}, got {task_type!r}.",
            )
        self.h5_path = Path(h5_path)
        self.num_channels = int(num_channels)
        self.samples_per_frame = int(samples_per_frame)
        self.sampling_frequency_hz = float(sampling_frequency_hz)
        self.dataset_name = str(dataset_name)
        self.label_type = str(label_type)
        self.num_classes = int(num_classes)
        self.task_type = str(task_type)
        self.is_regression = self.task_type == "regression"
        self.num_outputs = int(num_outputs)
        self.label_names = (
            [str(n) for n in label_names] if label_names is not None else None
        )
        self.flush_every = max(1, int(flush_every))

        if self.is_regression:
            if self.num_outputs < 1:
                raise ValueError(
                    "task_type='regression' requires num_outputs >= 1, got "
                    f"{self.num_outputs}.",
                )
            if self.label_names is not None and len(self.label_names) != self.num_outputs:
                raise ValueError(
                    f"label_names has {len(self.label_names)} entries but "
                    f"num_outputs={self.num_outputs}.",
                )

        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self._f: Optional[h5py.File] = h5py.File(self.h5_path, "w", libver="latest")
        C, T = self.num_channels, self.samples_per_frame
        self._f.create_dataset(
            "signal", shape=(0, C, T), maxshape=(None, C, T),
            dtype="float32", chunks=(min(self.flush_every, 256), C, T),
        )
        if self.is_regression:
            K = self.num_outputs
            self._f.create_dataset(
                "labels", shape=(0, K), maxshape=(None, K),
                dtype="float32", chunks=(min(self.flush_every, 1024), K),
            )
        else:
            self._f.create_dataset(
                "labels", shape=(0,), maxshape=(None,),
                dtype="int64", chunks=(min(self.flush_every, 1024),),
            )
        for name in ("session_id", "patient_id"):
            self._f.create_dataset(
                name, shape=(0,), maxshape=(None,),
                dtype="int64", chunks=(min(self.flush_every, 1024),),
            )

        self._buf_signal: list[np.ndarray] = []
        self._buf_label: list = []
        self._buf_session: list[int] = []
        self._buf_patient: list[int] = []
        self._n_written = 0

    # ------------------------------------------------------------------
    def _coerce_label(self, label: LabelLike):
        """Validate + coerce a label into the buffered representation.

        Classification → python ``int``. Regression → ``(K,)`` float32 array.
        """
        if self.is_regression:
            arr = np.asarray(label, dtype=np.float32).reshape(-1)
            if arr.shape != (self.num_outputs,):
                raise ValueError(
                    f"regression label has shape {arr.shape} after flatten, "
                    f"expected ({self.num_outputs},).",
                )
            return arr
        return int(label)

    def write(
        self,
        signal: np.ndarray,
        label: LabelLike,
        session_id: int,
        patient_id: int,
    ) -> None:
        if signal.shape != (self.num_channels, self.samples_per_frame):
            raise ValueError(
                f"signal shape {signal.shape} does not match declared "
                f"({self.num_channels}, {self.samples_per_frame}).",
            )
        if signal.dtype != np.float32:
            signal = signal.astype(np.float32, copy=False)
        self._buf_signal.append(signal)
        self._buf_label.append(self._coerce_label(label))
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

        if self.is_regression:
            self._f["labels"].resize((n1, self.num_outputs))
            self._f["labels"][n0:n1] = np.stack(self._buf_label, axis=0).astype(
                np.float32, copy=False,
            )
        else:
            self._f["labels"].resize((n1,))
            self._f["labels"][n0:n1] = np.asarray(self._buf_label, dtype=np.int64)

        for name, buf in (
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
        self._f.attrs["task_type"] = self.task_type
        self._f.attrs["num_classes"] = np.int64(self.num_classes)
        self._f.attrs["num_channels"] = np.int64(self.num_channels)
        self._f.attrs["samples_per_frame"] = np.int64(self.samples_per_frame)
        if self.is_regression:
            self._f.attrs["num_outputs"] = np.int64(self.num_outputs)
            if self.label_names is not None:
                self._f.attrs["label_names"] = list(self.label_names)
        self._f.close()
        self._f = None

    def __enter__(self) -> "DownstreamHDF5Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["DownstreamHDF5Writer"]
