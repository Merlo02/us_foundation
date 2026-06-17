"""UltraPro — wearable A-mode ultrasound hand/wrist motions (labeled, classification).

A-mode acquisitions from a wrist-worn 8-element ultrasound array (the EPFL
"UltraPro" wearable; Xingchen Yang et al., *Simultaneous Prediction of Wrist
and Hand Motions via Wearable Ultrasound Sensing for Natural Control of Hand
Prostheses*, IEEE TNSRE). The raw corpus is laid out as one directory per
subject and one subdirectory per session::

    <input_path>/A1/exp2/session1/<ts>_1.bin   ← channel 1
    <input_path>/A1/exp2/session1/<ts>_2.bin   ← channel 2
    ...
    <input_path>/A1/exp2/session1/<ts>_8.bin   ← channel 8
    <input_path>/A1/exp2/session2/...
    <input_path>/A2/exp2/session1/...

Each ``.bin`` holds **one channel** as a flat ``uint8`` buffer of
``n_acq * bytes_per_acq`` bytes, reshaped to ``(n_acq, bytes_per_acq)``. The
first ``header_bytes`` bytes of every acquisition are a fixed device header
(empirically ``[0, 1]``) and are dropped, leaving ``samples_per_frame =
bytes_per_acq - header_bytes`` time-points per A-mode line. With the shipped
files: ``bytes_per_acq=1000``, ``header_bytes=2`` ⇒ ``T = 998`` and
``n_acq = 3000`` per session.

Labels are **progressive blocks**: the acquisitions within a session are
ordered by motion class, ``samples_per_class`` consecutive frames per class,
so ``label[i] = i // samples_per_class``. With the shipped files: 6 classes ×
500 frames = 3000 acquisitions, hence ``label`` cycles ``0,0,…,0,1,…,5``.

This processor stacks the per-channel ``.bin`` files of one session into one
``(C, T)`` frame per acquisition and emits a **classification** row::

    {"signal": (C, T) float32, "label": int,
     "session_id": int, "patient_id": int}

``patient_id`` is parsed from the subject directory (``A1`` → 1); ``session_id``
is, by default, a globally-unique encoding ``patient_id * 100 + session_num``
(so ``A1/session1`` → 101, ``A2/session3`` → 203) which makes
``split_mode='intra_session'`` leave-one-session-out CV meaningful across
subjects. Set ``extra.globally_unique_sessions: false`` to fall back to the
raw per-subject session number (1..3), which then collides across subjects.

The ``.bin`` files are plain ``uint8`` buffers (no pickled objects), so this
processor runs fine on either venv (unlike the spacone processor). It is kept
under :mod:`etl_downstream` for consistency with the labeled-ETL pipeline.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from ..base_processor import DownstreamBaseProcessor

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Constants (shipped-corpus defaults)
# ────────────────────────────────────────────────────────────────────

DEFAULT_EXP_SUBDIR = "exp2"
DEFAULT_NUM_CHANNELS = 8
DEFAULT_BYTES_PER_ACQ = 1000
DEFAULT_HEADER_BYTES = 2          # fixed device header dropped per acquisition
DEFAULT_NUM_CLASSES = 6
DEFAULT_SAMPLES_PER_CLASS = 500   # 6 classes × 500 = 3000 acquisitions / session
DEFAULT_LABEL_TYPE = "gesture"

_CHANNEL_SUFFIX_RE = re.compile(r"_(\d+)\.bin$")


def _patient_id_from_subject(subject: str) -> int:
    """Map ``A1`` → 1, ``A12`` → 12 (strip non-digits)."""
    digits = re.sub(r"\D", "", subject)
    if not digits:
        raise ValueError(
            f"Cannot parse a patient_id integer from subject directory {subject!r}.",
        )
    return int(digits)


def _session_num_from_name(name: str) -> int:
    """Map ``session1`` → 1, ``session03`` → 3 (strip non-digits)."""
    digits = re.sub(r"\D", "", name)
    if not digits:
        raise ValueError(
            f"Cannot parse a session number from session directory {name!r}.",
        )
    return int(digits)


def _channel_index_from_bin(path: Path) -> Optional[int]:
    """0-based channel index from a ``..._<N>.bin`` filename (``_1`` → 0)."""
    m = _CHANNEL_SUFFIX_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1)) - 1


class UltraProProcessor(DownstreamBaseProcessor):
    """UltraPro wearable A-mode — one classification row per multi-channel frame.

    Configuration via :attr:`DatasetConfig.extra`:

    - ``sampling_frequency_hz``    — A-mode RF sampling frequency (Hz, required;
                                     not recorded in the ``.bin`` files).
    - ``subjects``                 — subject directory names to include, e.g.
                                     ``["A1", "A2", "A3", "A4"]`` (default: every
                                     ``A*`` directory found under ``input_path``).
    - ``sessions_to_use``          — session directory names to include, e.g.
                                     ``["session1", "session2", "session3"]``
                                     (default: every ``session*`` found).
    - ``exp_subdir``               — experiment subdirectory between subject and
                                     session (default ``"exp2"``).
    - ``num_channels_raw``         — channels on disk (= number of ``.bin`` files
                                     per session, default 8); combined with
                                     ``channels_to_exclude`` / ``channels_to_keep``.
    - ``bytes_per_acq``            — bytes per acquisition on disk (default 1000).
    - ``header_bytes``             — leading header bytes dropped per acquisition
                                     (default 2 ⇒ ``samples_per_frame`` = 998).
    - ``num_classes``              — number of motion classes (default 6).
    - ``samples_per_class``        — consecutive frames per class (default 500).
    - ``globally_unique_sessions`` — ``True`` (default) ⇒ ``session_id =
                                     patient_id*100 + session_num``; ``False`` ⇒
                                     raw per-subject session number.
    - ``label_type``               — descriptive tag stored in the h5/manifest
                                     (default ``"gesture"``).
    """

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def _extra(self) -> dict:
        return self.config.extra or {}

    @property
    def _exp_subdir(self) -> str:
        return str(self._extra.get("exp_subdir", DEFAULT_EXP_SUBDIR))

    @property
    def _subjects(self) -> Optional[list[str]]:
        v = self._extra.get("subjects")
        return [str(s) for s in v] if v else None

    @property
    def _sessions_to_use(self) -> Optional[list[str]]:
        v = self._extra.get("sessions_to_use")
        return [str(s) for s in v] if v else None

    @property
    def _num_channels_raw(self) -> int:
        return int(self._extra.get("num_channels_raw", DEFAULT_NUM_CHANNELS))

    @property
    def _bytes_per_acq(self) -> int:
        return int(self._extra.get("bytes_per_acq", DEFAULT_BYTES_PER_ACQ))

    @property
    def _header_bytes(self) -> int:
        return int(self._extra.get("header_bytes", DEFAULT_HEADER_BYTES))

    @property
    def _samples_per_class(self) -> int:
        v = int(self._extra.get("samples_per_class", DEFAULT_SAMPLES_PER_CLASS))
        if v <= 0:
            raise ValueError(
                f"Dataset {self.config.name!r}: extra.samples_per_class must be "
                f"positive, got {v}.",
            )
        return v

    @property
    def _globally_unique_sessions(self) -> bool:
        return bool(self._extra.get("globally_unique_sessions", True))

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
                f"frequency (Hz) and is NOT recorded in the UltraPro .bin files. "
                f"Set the true value in the YAML.",
            )
        return float(v)

    @property
    def num_channels(self) -> int:
        return len(self._kept_channel_indices)

    @property
    def samples_per_frame(self) -> int:
        t = self._bytes_per_acq - self._header_bytes
        if t <= 0:
            raise ValueError(
                f"Dataset {self.config.name!r}: bytes_per_acq ({self._bytes_per_acq}) "
                f"must exceed header_bytes ({self._header_bytes}).",
            )
        return t

    @property
    def num_classes(self) -> int:
        return int(self._extra.get("num_classes", DEFAULT_NUM_CLASSES))

    @property
    def label_type(self) -> str:
        return str(self._extra.get("label_type", DEFAULT_LABEL_TYPE))

    # ------------------------------------------------------------------
    # Iteration: discover_files yields session directories; load() reads the
    # per-channel .bin files of one session and emits one row per acquisition.
    # ------------------------------------------------------------------
    def discover_files(self) -> list[Path]:
        root = Path(self.config.input_path)
        if not root.exists():
            raise FileNotFoundError(
                f"Dataset {self.config.name!r}: input_path does not exist: {root}",
            )

        if self._subjects is not None:
            subject_dirs = [root / s for s in self._subjects]
        else:
            subject_dirs = sorted(
                (d for d in root.iterdir() if d.is_dir() and d.name.startswith("A")),
                key=lambda d: _patient_id_from_subject(d.name),
            )

        sessions_filter = self._sessions_to_use
        session_dirs: list[Path] = []
        for subj in subject_dirs:
            exp_dir = subj / self._exp_subdir
            if not exp_dir.is_dir():
                log.warning(
                    "Dataset %s: experiment dir missing — skipping: %s",
                    self.config.name, exp_dir,
                )
                continue
            found = sorted(
                (d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("session")),
                key=lambda d: _session_num_from_name(d.name),
            )
            for sess in found:
                if sessions_filter is not None and sess.name not in sessions_filter:
                    continue
                session_dirs.append(sess)

        if not session_dirs:
            raise FileNotFoundError(
                f"Dataset {self.config.name!r}: no session directories found under "
                f"{root} (subjects={self._subjects or 'A*'}, "
                f"exp_subdir={self._exp_subdir!r}, "
                f"sessions_to_use={self._sessions_to_use or 'session*'}).",
            )
        return session_dirs

    def load(self, filepath: Path) -> Iterator[dict]:
        sess_dir = filepath
        if not sess_dir.is_dir():
            log.warning(
                "Dataset %s: session directory missing — skipping: %s",
                self.config.name, sess_dir,
            )
            return

        subject = sess_dir.parent.parent.name
        patient_id = _patient_id_from_subject(subject)
        session_num = _session_num_from_name(sess_dir.name)
        session_id = (
            patient_id * 100 + session_num
            if self._globally_unique_sessions else session_num
        )

        # Map 0-based channel index → .bin path from the ``_<N>.bin`` suffix.
        ch_to_path: dict[int, Path] = {}
        for b in sorted(sess_dir.glob("*.bin")):
            ch = _channel_index_from_bin(b)
            if ch is None:
                log.warning(
                    "Dataset %s: cannot parse channel index from %s — skipping.",
                    self.config.name, b.name,
                )
                continue
            ch_to_path[ch] = b

        kept_idx = self._kept_channel_indices
        missing = [i for i in kept_idx if i not in ch_to_path]
        if missing:
            log.warning(
                "Dataset %s: session %s missing .bin for channel(s) %s "
                "(found channels %s) — skipping session.",
                self.config.name, sess_dir, missing, sorted(ch_to_path),
            )
            return

        bytes_per_acq = self._bytes_per_acq
        header = self._header_bytes
        T = self.samples_per_frame

        # Read each kept channel into a (n_acq, T) float32 array.
        channels: list[np.ndarray] = []
        n_acq: Optional[int] = None
        for ci in kept_idx:
            raw = np.fromfile(ch_to_path[ci], dtype=np.uint8)
            if raw.size % bytes_per_acq != 0:
                raise ValueError(
                    f"{ch_to_path[ci]}: byte count {raw.size} is not a multiple of "
                    f"bytes_per_acq={bytes_per_acq}.",
                )
            mat = raw.reshape(-1, bytes_per_acq)[:, header:].astype(np.float32)
            if n_acq is None:
                n_acq = mat.shape[0]
            elif mat.shape[0] != n_acq:
                raise ValueError(
                    f"Session {sess_dir}: channel {ci} has {mat.shape[0]} "
                    f"acquisitions but a previous channel had {n_acq}; the 8 "
                    f"per-channel .bin files must be frame-aligned.",
                )
            channels.append(mat)

        assert n_acq is not None  # discover_files guarantees ≥ 1 kept channel file
        frames = np.stack(channels, axis=1)  # (n_acq, C_kept, T)

        # Progressive block labels: samples_per_class consecutive frames per class.
        samples_per_class = self._samples_per_class
        labels = (np.arange(n_acq, dtype=np.int64) // samples_per_class)
        expected = self.num_classes * samples_per_class
        if n_acq != expected:
            log.warning(
                "Dataset %s: session %s has %d acquisitions but num_classes(%d) × "
                "samples_per_class(%d) = %d; labels derived as floor(idx / %d) and "
                "will span classes 0..%d.",
                self.config.name, sess_dir, n_acq, self.num_classes,
                samples_per_class, expected, samples_per_class, int(labels.max()),
            )

        log.info(
            "UltraPro session=%s subject=%s patient_id=%d session_id=%d frames=%d "
            "channels=%s classes=%s",
            sess_dir.name, subject, patient_id, session_id, n_acq,
            list(kept_idx), sorted(np.unique(labels).tolist()),
        )

        for i in range(n_acq):
            yield {
                "signal": np.ascontiguousarray(frames[i]),  # (C_kept, T)
                "label": int(labels[i]),
                "session_id": int(session_id),
                "patient_id": int(patient_id),
            }


__all__ = [
    "DEFAULT_BYTES_PER_ACQ",
    "DEFAULT_EXP_SUBDIR",
    "DEFAULT_HEADER_BYTES",
    "DEFAULT_LABEL_TYPE",
    "DEFAULT_NUM_CHANNELS",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_SAMPLES_PER_CLASS",
    "UltraProProcessor",
]
