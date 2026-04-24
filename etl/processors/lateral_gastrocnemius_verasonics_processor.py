from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from .base_processor import BaseDatasetProcessor, RawSample


class LateralGastrocnemiusVerasonicsProcessor(BaseDatasetProcessor):
    """EPFL LTS5 Lateral Gastrocnemius (Dec 2021) — Verasonics MAT v7.3 (HDF5).

    Directory layout: ``session_*/exp_*.mat``.

    File structure:
    - ``RcvData`` is a (1,1) cell array stored as an HDF5 reference.
      Dereferencing returns a numeric dataset shaped (n_frames, n_channels, n_samples_total).
    - ``n_samples_total`` is plane waves concatenated: ``n_pw * n_samp_per_pw``.

    Separation into acquisitions follows the acquisition packing used in the dataset:
    - The stored trace length is divisible by ``pw_divisor_len`` (default 800). This
      divisor is used to infer the number of plane waves in that file:
      ``n_pw = n_samples_total // pw_divisor_len``.
    - The meaningful samples are the first ``samples_actual`` (default 768) values
      of each plane wave; any remaining tail is padding and is discarded.

    Steps:
    1) Infer ``n_pw`` from the divisor length
    2) Trim to ``n_pw * samples_actual`` (drop tail padding)
    3) Reshape (n_frames, n_channels, n_pw, samples_actual)
    4) Transpose to (n_frames, n_pw, n_channels, samples_actual)
    5) Flatten to (n_frames * n_pw, n_channels, samples_actual)

    Each (frame, plane-wave) is treated as one "acquisition" sample.

    Debugging: ``source_dataset`` includes the session as ``{name}::{session}``,
    so debug output is separated per session under ``debug_qa/``.

    YAML ``extra``:
    - ``glob_pattern``: default ``session_*/*.mat``
    - ``pw_divisor_len``: default 800 (used only to infer n_pw)
    - ``samples_actual``: default 768 (kept samples per plane wave)
    - ``exclude_sessions``: list[str] of session folder names to skip
    - ``exclude_files``: basenames to skip in *every* session (use sparingly: e.g.
      ``exp_2.mat`` here also drops ``session_2/exp_2.mat``).
    - ``exclude_files_by_session``: dict mapping session folder name → list of basenames
      to skip only in that session (e.g. only ``session_1`` drops ``exp_1.mat``).
    """

    def discover_files(self) -> list[str]:
        root = Path(self.config.input_path)
        if not root.exists():
            return []
        if root.is_file():
            return [str(root.resolve())]

        pattern = str(self.config.extra.get("glob_pattern", "session_*/*.mat"))
        files = sorted(root.glob(pattern))

        exclude_sessions = set(self.config.extra.get("exclude_sessions", []))
        exclude_files_global = set(self.config.extra.get("exclude_files", []))
        exclude_by_session: dict[str, set[str]] = {}
        raw_map = self.config.extra.get("exclude_files_by_session") or {}
        for sess, names in raw_map.items():
            exclude_by_session[str(sess)] = {str(n) for n in (names or [])}

        out: list[str] = []
        for f in files:
            session = f.parent.name
            if session in exclude_sessions:
                continue
            if f.name in exclude_files_global:
                continue
            if f.name in exclude_by_session.get(session, set()):
                continue
            out.append(str(f.resolve()))
        return out

    @staticmethod
    def _get_rcvdata_buffer(f: h5py.File) -> h5py.Dataset | None:
        try:
            rcv = f["RcvData"]
            refs = rcv[()]  # numpy array of HDF5 references
            return f[refs.flat[0]]
        except Exception:
            return None

    def load_and_yield(self, filepath: str) -> Iterator[RawSample]:
        fp = Path(filepath)
        session = fp.parent.name
        exp = fp.stem

        pw_divisor_len = int(self.config.extra.get("pw_divisor_len", 800))
        samples_actual = int(self.config.extra.get("samples_actual", 768))

        with h5py.File(filepath, "r") as f:
            buf = self._get_rcvdata_buffer(f)
            if buf is None:
                return
            data = buf[()]  # (n_frames, n_channels, n_samples_total)

        if data.ndim != 3:
            return

        n_frames, n_channels, n_samples_total = data.shape
        if n_frames == 0 or n_channels == 0 or n_samples_total == 0:
            return

        if pw_divisor_len <= 0:
            return

        n_pw = n_samples_total // pw_divisor_len
        if n_pw <= 0:
            return

        if samples_actual <= 0:
            return

        valid_len = n_pw * samples_actual
        if n_samples_total < valid_len:
            return
        data_valid = data[:, :, :valid_len]

        # Reshape to expose plane waves; any padding is only at the end and was trimmed above.
        reshaped = data_valid.reshape(n_frames, n_channels, n_pw, samples_actual)

        # (n_frames, n_pw, n_channels, samples_actual) → (n_frames*n_pw, n_channels, samples_actual)
        transposed = reshaped.transpose(0, 2, 1, 3)
        tensor = transposed.reshape(n_frames * n_pw, n_channels, samples_actual)

        tensor = np.asarray(tensor, dtype=np.float32)

        # Separate debug per session via dataset name.
        ds_for_debug = f"{self.config.name}::{session}"

        for acq_idx in range(tensor.shape[0]):
            for ch_idx in range(tensor.shape[1]):
                if not self.should_keep_channel(ch_idx):
                    continue
                sig = tensor[acq_idx, ch_idx]
                yield RawSample(
                    signal=sig.ravel(),
                    sample_id=f"{self.config.name}_{session}_{exp}_a{acq_idx}_ch{ch_idx}",
                    source_dataset=ds_for_debug,
                    channel_idx=ch_idx,
                    sampling_frequency_hz=self.sampling_frequency_hz(),
                    metadata={
                        "base_dataset": self.config.name,
                        "session": session,
                        "experiment": exp,
                        "file": fp.name,
                        "n_pw": int(n_pw),
                        "n_frames": int(n_frames),
                        "n_channels": int(n_channels),
                        "n_samples_total": int(n_samples_total),
                        "pw_divisor_len": int(pw_divisor_len),
                        "samples_actual": int(samples_actual),
                        "acq_idx": int(acq_idx),
                        "channel": int(ch_idx),
                    },
                )

