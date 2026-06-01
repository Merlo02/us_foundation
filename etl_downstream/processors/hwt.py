"""HWT — hand/wrist kinematics ultrasound dataset (labeled, regression).

A-mode acquisitions from a wrist-worn array: 4 transducer channels, 400
samples per frame, with continuous kinematic targets (3 degrees of
freedom). The dataset ships **already in HDF5**, one file per subject
(``subject_{n}.h5``) under ``<input_path>/``, each holding a single
``HWT`` group::

    HWT/RF              (N, 4, 400) float64   raw RF frames
    HWT/combined_labels (N, 1, 3)   float64   [wr_fe, wr_rud, fg_fe]
    HWT/wr_fe           (N, 1, 1)   float64   wrist flexion/extension
    HWT/wr_rud          (N, 1, 1)   float64   wrist radial/ulnar deviation
    HWT/fg_fe           (N, 1, 1)   float64   finger group flexion/extension
    HWT/session         (N, 1, 1)   int64     per-subject session id (1..9)
    HWT/unique_session  (N, 1, 1)   int64     globally-unique session id
    HWT/subject         (N, 1, 1)   int64     subject id (constant per file)
    HWT/augmented       (N, 1, 1)   bool      True ⇒ synthetic augmentation
    HWT/timestamps      (N, 1, 1)   float64   frame epoch (ms)

This processor emits one **regression** row per kept frame::

    {"signal": (C, T) float32, "label": (num_outputs,) float32,
     "session_id": int, "patient_id": int}

with ``label`` the selected DOF values (``extra.label_dofs``, default all
three in the canonical order ``[wr_fe, wr_rud, fg_fe]``). ``patient_id`` is
the file's ``subject`` value; ``session_id`` is ``unique_session`` (default)
or ``session`` depending on ``extra.session_id_field``.

Unlike the spacone processor, HWT frames are stored as plain numeric HDF5
datasets (no pickled objects), so this runs fine on either venv. It is kept
under :mod:`etl_downstream` for consistency with the labeled-ETL pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional, Sequence

import h5py
import numpy as np

from ..base_processor import DownstreamBaseProcessor

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────

GROUP_KEY = "HWT"

# Per-DOF named datasets. Selecting by name (rather than slicing
# ``combined_labels`` by column) avoids relying on a column ordering that
# is not recorded in the file's attrs.
HWT_DOF_NAMES: tuple[str, ...] = ("wr_fe", "wr_rud", "fg_fe")

DEFAULT_NUM_CHANNELS = 4
DEFAULT_US_FRAME_LEN = 400
DEFAULT_READ_CHUNK = 4096

_VALID_SESSION_FIELDS = ("unique_session", "session")


class HWTProcessor(DownstreamBaseProcessor):
    """HWT hand/wrist kinematics — one regression row per multi-channel frame.

    Configuration via :attr:`DatasetConfig.extra`:

    - ``subjects``               — list of subject ids to include, e.g.
                                    ``[1, 2, 3, 4]`` (required). Each maps to
                                    ``<input_path>/subject_{id}.h5``.
    - ``sampling_frequency_hz``  — A-mode RF sampling frequency (Hz, required;
                                    not derivable from the files).
    - ``label_dofs``             — subset/order of ``HWT_DOF_NAMES`` to regress
                                    (default all three).
    - ``include_augmented``      — keep ``augmented==True`` frames (default
                                    ``False`` ⇒ real frames only).
    - ``session_id_field``       — ``"unique_session"`` (default) | ``"session"``.
    - ``num_channels_raw``       — number of transducer channels on disk
                                    (default 4); combined with
                                    ``channels_to_exclude`` / ``channels_to_keep``.
    - ``us_frame_len``           — expected samples per frame (default 400).
    - ``read_chunk``             — frames read from disk per block (default 4096).
    - ``label_type``             — descriptive tag stored in the h5/manifest
                                    (default = comma-joined ``label_dofs``).
    """

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def _extra(self) -> dict:
        return self.config.extra or {}

    @property
    def _subjects(self) -> list[int]:
        v = self._extra.get("subjects")
        if not v:
            raise ValueError(
                f"Dataset {self.config.name!r}: extra.subjects is required "
                f"(list of subject ids, e.g. [1, 2, 3, 4]).",
            )
        return [int(s) for s in v]

    @property
    def _label_dofs(self) -> list[str]:
        v = self._extra.get("label_dofs", list(HWT_DOF_NAMES))
        dofs = [str(d) for d in v]
        if not dofs:
            raise ValueError(
                f"Dataset {self.config.name!r}: extra.label_dofs is empty.",
            )
        unknown = [d for d in dofs if d not in HWT_DOF_NAMES]
        if unknown:
            raise ValueError(
                f"Dataset {self.config.name!r}: unknown label_dofs {unknown}. "
                f"Valid DOFs: {list(HWT_DOF_NAMES)}.",
            )
        return dofs

    @property
    def _include_augmented(self) -> bool:
        return bool(self._extra.get("include_augmented", False))

    @property
    def _session_id_field(self) -> str:
        v = str(self._extra.get("session_id_field", "unique_session"))
        if v not in _VALID_SESSION_FIELDS:
            raise ValueError(
                f"Dataset {self.config.name!r}: extra.session_id_field must be "
                f"one of {_VALID_SESSION_FIELDS}, got {v!r}.",
            )
        return v

    @property
    def _num_channels_raw(self) -> int:
        return int(self._extra.get("num_channels_raw", DEFAULT_NUM_CHANNELS))

    @property
    def _us_frame_len(self) -> int:
        return int(self._extra.get("us_frame_len", DEFAULT_US_FRAME_LEN))

    @property
    def _read_chunk(self) -> int:
        return max(1, int(self._extra.get("read_chunk", DEFAULT_READ_CHUNK)))

    @property
    def _kept_channel_indices(self) -> tuple[int, ...]:
        kept = tuple(
            i for i in range(self._num_channels_raw) if self.should_keep_channel(i)
        )
        if not kept:
            raise ValueError(
                f"Dataset {self.config.name!r}: channels_to_exclude / "
                f"channels_to_keep filtered out every transducer "
                f"(num_channels_raw={self._num_channels_raw}).",
            )
        return kept

    # ------------------------------------------------------------------
    # DownstreamBaseProcessor interface
    # ------------------------------------------------------------------
    @property
    def sampling_frequency_hz(self) -> float:
        v = self._extra.get("sampling_frequency_hz")
        if v is None or float(v) <= 0:
            raise ValueError(
                f"Dataset {self.config.name!r}: extra.sampling_frequency_hz is "
                f"required and must be positive — it is the A-mode RF sampling "
                f"frequency (Hz) and is NOT recorded in the HWT files. Set the "
                f"true value in the YAML.",
            )
        return float(v)

    @property
    def num_channels(self) -> int:
        return len(self._kept_channel_indices)

    @property
    def samples_per_frame(self) -> int:
        return self._us_frame_len

    @property
    def num_classes(self) -> int:
        return 0  # regression — kept at 0 for schema symmetry

    @property
    def label_type(self) -> str:
        return str(self._extra.get("label_type", ",".join(self._label_dofs)))

    @property
    def task_type(self) -> str:
        return "regression"

    @property
    def num_outputs(self) -> int:
        return len(self._label_dofs)

    @property
    def label_names(self) -> Optional[Sequence[str]]:
        return list(self._label_dofs)

    # ------------------------------------------------------------------
    # Iteration: one file per subject.
    # ------------------------------------------------------------------
    def discover_files(self) -> list[Path]:
        root = Path(self.config.input_path)
        if not root.exists():
            raise FileNotFoundError(
                f"Dataset {self.config.name!r}: input_path does not exist: {root}",
            )
        files: list[Path] = []
        for s in self._subjects:
            fp = root / f"subject_{s}.h5"
            if not fp.exists():
                raise FileNotFoundError(
                    f"Dataset {self.config.name!r}: subject file missing: {fp}",
                )
            files.append(fp)
        return files

    def load(self, filepath: Path) -> Iterator[dict]:
        kept_ch = list(self._kept_channel_indices)
        label_dofs = self._label_dofs
        contiguous_ch = kept_ch == list(range(self._num_channels_raw))

        with h5py.File(filepath, "r") as f:
            if GROUP_KEY not in f:
                raise KeyError(
                    f"{filepath}: expected a top-level group {GROUP_KEY!r}.",
                )
            g = f[GROUP_KEY]
            rf = g["RF"]
            n_total, c_raw, t_raw = rf.shape
            if c_raw != self._num_channels_raw:
                raise ValueError(
                    f"{filepath}: RF has {c_raw} channels but num_channels_raw="
                    f"{self._num_channels_raw}.",
                )
            if t_raw != self._us_frame_len:
                raise ValueError(
                    f"{filepath}: RF frame length {t_raw} != us_frame_len="
                    f"{self._us_frame_len}.",
                )

            patient_id = int(np.asarray(g["subject"][:]).reshape(-1)[0])
            session_arr = np.asarray(
                g[self._session_id_field][:], dtype=np.int64,
            ).reshape(-1)
            augmented = np.asarray(g["augmented"][:], dtype=bool).reshape(-1)

            # Build the (N, K) label matrix from the per-DOF named datasets,
            # so we never depend on combined_labels' (undocumented) column order.
            label_mat = np.stack(
                [
                    np.asarray(g[dof][:], dtype=np.float32).reshape(-1)
                    for dof in label_dofs
                ],
                axis=1,
            )  # (N, K)

            keep = np.ones(n_total, dtype=bool) if self._include_augmented else ~augmented
            kept_idx = np.where(keep)[0].astype(np.int64)  # sorted ascending

            log.info(
                "HWT subject_file=%s patient_id=%d frames=%d kept=%d "
                "(include_augmented=%s) sessions=%s dofs=%s channels=%s",
                filepath.name, patient_id, n_total, kept_idx.size,
                self._include_augmented,
                sorted(np.unique(session_arr[kept_idx]).tolist())
                if kept_idx.size else [],
                label_dofs, kept_ch,
            )
            if kept_idx.size == 0:
                log.warning("%s: no frames left after augmented filtering.", filepath.name)
                return

            chunk = self._read_chunk
            for c0 in range(0, kept_idx.size, chunk):
                block_idx = kept_idx[c0:c0 + chunk]
                lo, hi = int(block_idx[0]), int(block_idx[-1])
                is_contiguous = (hi - lo + 1) == block_idx.size
                if is_contiguous:
                    block = np.asarray(rf[lo:hi + 1], dtype=np.float32)
                else:
                    block = np.asarray(rf[block_idx.tolist()], dtype=np.float32)
                if not contiguous_ch:
                    block = block[:, kept_ch, :]
                for j in range(block.shape[0]):
                    gidx = int(block_idx[j])
                    yield {
                        "signal": np.ascontiguousarray(block[j]),  # (C_kept, T)
                        "label": label_mat[gidx],                  # (K,) float32
                        "session_id": int(session_arr[gidx]),
                        "patient_id": patient_id,
                    }


__all__ = [
    "DEFAULT_NUM_CHANNELS",
    "DEFAULT_READ_CHUNK",
    "DEFAULT_US_FRAME_LEN",
    "GROUP_KEY",
    "HWT_DOF_NAMES",
    "HWTProcessor",
]
