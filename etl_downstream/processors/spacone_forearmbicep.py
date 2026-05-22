"""Spacone forearm-bicep ultrasound dataset (labeled, multi-channel).

A-mode acquisitions from a forearm/bicep wearable: 6 transducer channels at
8 MHz, gesture + position labels, multiple subjects and sessions. Each
``{task}_session{N}.h5`` under ``<input_path>/<subject>/`` stores a pickled
pandas DataFrame under the key ``df_session``; one row per frame, with
``tx_0`` … ``tx_5`` cells holding 1-D 397-sample ultrasound waveforms.

This processor emits one row per kept frame::

    {"signal": (C, T) float32, "label": int,
     "session_id": int, "patient_id": int}

The ``label`` is either the gesture or the position int, controlled by
``extra.label_type`` in the dataset YAML (``"gesture"`` | ``"position"``).
``num_classes`` is derived from the corresponding class map.

The label-encoding helpers (``return_gesture_map``, ``POSITION_CLASS_MAPPING``,
``encode_labels``) and the transient-cutting logic are ported verbatim from
the supervisor's reference script — only the schema-driven indirection that
made the previous version unreadable was removed.

**Environment note.** Source ``.h5`` files were serialised under numpy >= 2.0;
their pickled ``tx_*`` cells crash numpy 1.x's C reconstruct path. Run this
processor only inside ``~/usf_etl_venv`` (see ``CLAUDE.md``).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from ..base_processor import DownstreamBaseProcessor

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Constants (ported from the supervisor's script)
# ────────────────────────────────────────────────────────────────────

INT32_MIN = -2147483648
INT64_MIN = -9223372036854775808
INVALID = -1

DEFAULT_TX_COLS: tuple[str, ...] = ("tx_0", "tx_1", "tx_2", "tx_3", "tx_4", "tx_5")
DEFAULT_US_FRAME_LEN = 397
DEFAULT_US_FS = 8_000_000.0
DEFAULT_US_PRR = 30.0  # frame rate (Hz), used by transient detection

TASK_GESTURES: dict[str, set[str]] = {
    "cilinder": {"rest", "rotopen", "rotclosed"},
    "pouring":  {"rest", "rotopen", "rotclosed", "pour"},
    "pinch":    {"rest", "open", "pinch"},
}

LEGACY_STRING_MAPPINGS: dict[str, dict[str, int]] = {
    "cilinder_pour_pinch": {
        "rest": 0, "open": 1, "rotopen": 2,
        "rotclosed": 3, "pinch": 4, "pour": 5,
    },
}

POSITION_CLASS_MAPPING: dict[str, int] = {"rest": 0, "front": 1, "side": 2}


# ────────────────────────────────────────────────────────────────────
# Label encoding
# ────────────────────────────────────────────────────────────────────

GestureMappingType = list[str] | str


def return_gesture_map(
    gesture_mapping: GestureMappingType,
) -> tuple[dict[str, int], dict[str, int]]:
    if isinstance(gesture_mapping, list):
        if len(gesture_mapping) == 0:
            raise ValueError("gesture_mapping list is empty.")
        unknown = [t for t in gesture_mapping if t not in TASK_GESTURES]
        if unknown:
            raise ValueError(
                f"Unknown task(s) {unknown}. Valid tasks: {sorted(TASK_GESTURES)}",
            )
        gestures: set[str] = set()
        for t in gesture_mapping:
            gestures |= TASK_GESTURES[t]
        ordered = sorted(gestures)
        return {g: i for i, g in enumerate(ordered)}, dict(POSITION_CLASS_MAPPING)

    if gesture_mapping not in LEGACY_STRING_MAPPINGS:
        raise ValueError(
            f"Unknown gesture_mapping {gesture_mapping!r}. "
            f"Valid: {sorted(LEGACY_STRING_MAPPINGS)} or a list of tasks from "
            f"{sorted(TASK_GESTURES)}.",
        )
    return LEGACY_STRING_MAPPINGS[gesture_mapping], dict(POSITION_CLASS_MAPPING)


def encode_labels(
    df: pd.DataFrame,
    gesture_mapping: GestureMappingType,
    *,
    drop_unknown_positions: bool = True,
) -> pd.DataFrame:
    required_cols = {"Label_gesture", "Label_position"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"encode_labels(): missing required columns: {sorted(missing)}")

    df = df.copy()
    df.loc[:, "Label_gesture"] = df["Label_gesture"].replace(
        {"rotopen2": "rotopen", "start": "rest"},
    )
    df.loc[:, "Label_position"] = df["Label_position"].replace(
        {"pour": "front", "start": "rest"},
    )

    gesture_map, position_map = return_gesture_map(gesture_mapping)
    df = df.loc[df["Label_gesture"].isin(gesture_map)].copy()
    df.loc[:, "Label_gesture_int"] = df["Label_gesture"].map(gesture_map).astype("Int64")

    if drop_unknown_positions:
        df = df.loc[df["Label_position"].isin(position_map)].copy()
    df.loc[:, "Label_position_int"] = df["Label_position"].map(position_map).astype("Int64")
    return df


# ────────────────────────────────────────────────────────────────────
# Waveform validity + transient detection
# ────────────────────────────────────────────────────────────────────

def waveform_has_nan_or_invalid(w, expected_len: Optional[int] = None) -> bool:
    if w is None:
        return True
    if isinstance(w, float) and np.isnan(w):
        return True
    try:
        a = np.asarray(w, dtype=float)
    except Exception:
        return True
    if a.ndim != 1:
        return True
    if expected_len is not None and a.size != expected_len:
        return True
    return bool(np.isnan(a).any())


def detect_transients(
    df: pd.DataFrame,
    label_col: str = "label",
    fs: float = DEFAULT_US_PRR,
    ms_to_cut: float = 500.0,
    invalid_label: int = INVALID,
    int32_min: int = INT32_MIN,
    drop_invalid_transitions: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    y = df[label_col].to_numpy()
    y = np.where(y == int32_min, invalid_label, y)
    cut_samples = max(int(round((ms_to_cut / 1000.0) * fs)), 1)
    valid = (y != invalid_label)
    y_ffill = pd.Series(np.where(valid, y, np.nan)).ffill().to_numpy()
    changes = (y_ffill != np.roll(y_ffill, 1))
    changes[0] = False
    transitions = np.where(changes & valid)[0].astype(int)
    if drop_invalid_transitions and len(transitions) > 0 and transitions[0] == 0:
        transitions = transitions[1:]
    n = len(df)
    windows = [(int(s), int(min(s + cut_samples, n))) for s in transitions]
    return transitions, windows, cut_samples


def cut_transients(
    df: pd.DataFrame,
    windows: list[tuple[int, int]],
    reset_index: bool = True,
) -> pd.DataFrame:
    n = len(df)
    keep_mask = np.ones(n, dtype=bool)
    for s, e in windows:
        s = max(0, int(s))
        e = min(n, int(e))
        if s < e:
            keep_mask[s:e] = False
    out = df.loc[keep_mask].copy()
    if reset_index:
        out = out.reset_index(drop=True)
    return out


def balance_dataset(
    df: pd.DataFrame,
    label_column: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Undersample majority classes to match the minority class count.

    Ported verbatim from the supervisor's ``balance_dataset`` helper.
    """
    if df.empty:
        return df
    class_counts = df[label_column].value_counts()
    min_count = int(class_counts.min())
    log.info(
        "balance_dataset(%s): min count = %d, per-class counts = %s",
        label_column, min_count, class_counts.to_dict(),
    )
    parts = [
        df.loc[df[label_column] == label].sample(n=min_count, random_state=random_state)
        for label in class_counts.index
    ]
    return pd.concat(parts, ignore_index=True)


# ────────────────────────────────────────────────────────────────────
# File / subject loader
# ────────────────────────────────────────────────────────────────────

def _session_token_from_stem(stem: str) -> Optional[tuple[str, int]]:
    m = re.search(r"session(\d+(?:_\d+)?)", stem)
    if not m:
        return None
    return "session" + m.group(1), int(m.group(1).replace("_", ""))


def _patient_id_from_subject(subject: str) -> int:
    """Map ``S01`` → 1, ``S12`` → 12. Falls back to a non-digit-stripped int parse."""
    digits = re.sub(r"\D", "", subject)
    if not digits:
        raise ValueError(
            f"Cannot parse a patient_id integer from subject {subject!r}.",
        )
    return int(digits)


def _load_subject_dataframe(
    subj_dir: Path,
    sessions_to_use: Sequence[str],
    tasks_to_consider: Sequence[str],
    tx_cols: Sequence[str],
    us_frame_len: int,
    remove_gesture_transients: bool,
    ms_to_cut: float,
    us_prr: float,
) -> pd.DataFrame:
    """Load every session file for one subject; apply transient cutting."""
    file_list: list[Path] = []
    for f in sorted(subj_dir.glob("*.h5")):
        task = f.stem.split("_")[0]
        if task not in tasks_to_consider:
            continue
        if not any(sess in f.stem for sess in sessions_to_use):
            continue
        file_list.append(f)

    log.info("Subject %s: found %d session files", subj_dir.name, len(file_list))

    rows: list[pd.DataFrame] = []
    for file in file_list:
        token = _session_token_from_stem(file.stem)
        if token is None:
            continue
        sess_token, session_id = token
        if sess_token not in sessions_to_use:
            continue

        df_curr = pd.read_hdf(file, key="df_session")
        df_curr["session_id"] = session_id

        invalid_cell = df_curr[list(tx_cols)].map(
            lambda w: waveform_has_nan_or_invalid(w, us_frame_len),
        )
        valid_mask = ~invalid_cell.any(axis=1)
        df_curr = df_curr.loc[valid_mask].copy()

        df_curr[df_curr["label"] == INT32_MIN] = INVALID
        df_curr[df_curr["label"] == INT64_MIN] = INVALID

        if remove_gesture_transients:
            _, windows, _ = detect_transients(
                df_curr, label_col="label", fs=us_prr,
                ms_to_cut=ms_to_cut, invalid_label=INVALID,
            )
            df_curr = cut_transients(df_curr, windows)
        else:
            df_curr = df_curr.reset_index(drop=True)

        rows.append(df_curr)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ────────────────────────────────────────────────────────────────────
# Processor
# ────────────────────────────────────────────────────────────────────

_VALID_LABEL_TYPES = ("gesture", "position")


class SpaconeForearmBicepProcessor(DownstreamBaseProcessor):
    """Spacone forearm/bicep — one row per multi-channel frame.

    Configuration via :attr:`DatasetConfig.extra`:

    - ``label_type``                 — ``"gesture"`` | ``"position"`` (required)
    - ``subjects``                   — e.g. ``["S01", …, "S05"]`` (required)
    - ``sessions_to_use``            — e.g. ``["session1", …, "session6"]``
    - ``tasks_to_consider``          — subset of ``("cilinder","pouring","pinch")``
    - ``tx_cols``                    — transducer column names (default 6)
    - ``us_frame_len``               — expected waveform length (default 397)
    - ``sampling_frequency_hz``      — A-mode US fs (default 8e6)
    - ``us_prr``                     — frame repetition rate (Hz) for transients
    - ``gesture_mapping``            — drives the gesture encoder; default = ``tasks_to_consider``
    - ``remove_gesture_transients``  — drop frames within ``ms_to_cut`` of transitions
    - ``ms_to_cut``                  — transient window (ms)
    """

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def _extra(self) -> dict:
        return self.config.extra or {}

    @property
    def _subjects(self) -> list[str]:
        v = self._extra.get("subjects")
        if not v:
            raise ValueError(
                f"Dataset {self.config.name!r}: extra.subjects is required.",
            )
        return [str(s) for s in v]

    @property
    def _sessions_to_use(self) -> list[str]:
        v = self._extra.get("sessions_to_use")
        if not v:
            raise ValueError(
                f"Dataset {self.config.name!r}: extra.sessions_to_use is required.",
            )
        return [str(s) for s in v]

    @property
    def _tasks_to_consider(self) -> list[str]:
        v = self._extra.get("tasks_to_consider", list(TASK_GESTURES.keys()))
        return [str(t) for t in v]

    @property
    def _tx_cols(self) -> list[str]:
        return [str(c) for c in self._extra.get("tx_cols", DEFAULT_TX_COLS)]

    @property
    def _us_frame_len(self) -> int:
        return int(self._extra.get("us_frame_len", DEFAULT_US_FRAME_LEN))

    @property
    def _us_prr(self) -> float:
        return float(self._extra.get("us_prr", DEFAULT_US_PRR))

    @property
    def _ms_to_cut(self) -> float:
        return float(self._extra.get("ms_to_cut", 0.0))

    @property
    def _remove_gesture_transients(self) -> bool:
        return bool(self._extra.get("remove_gesture_transients", True))

    @property
    def _balance_classes(self) -> bool:
        return bool(self._extra.get("balance_classes", False))

    @property
    def _gesture_mapping(self) -> GestureMappingType:
        v = self._extra.get("gesture_mapping")
        return list(self._tasks_to_consider) if v is None else v

    @property
    def _kept_channel_indices(self) -> tuple[int, ...]:
        kept = tuple(
            i for i in range(len(self._tx_cols)) if self.should_keep_channel(i)
        )
        if not kept:
            raise ValueError(
                f"Dataset {self.config.name!r}: channels_to_exclude / "
                f"channels_to_keep filtered out every transducer "
                f"(tx_cols={self._tx_cols}).",
            )
        return kept

    # ------------------------------------------------------------------
    # DownstreamBaseProcessor interface
    # ------------------------------------------------------------------
    @property
    def sampling_frequency_hz(self) -> float:
        return float(self._extra.get("sampling_frequency_hz", DEFAULT_US_FS))

    @property
    def num_channels(self) -> int:
        return len(self._kept_channel_indices)

    @property
    def samples_per_frame(self) -> int:
        return self._us_frame_len

    @property
    def label_type(self) -> str:
        v = self._extra.get("label_type")
        if v not in _VALID_LABEL_TYPES:
            raise ValueError(
                f"Dataset {self.config.name!r}: extra.label_type must be one of "
                f"{_VALID_LABEL_TYPES}, got {v!r}.",
            )
        return str(v)

    @property
    def num_classes(self) -> int:
        if self.label_type == "gesture":
            gmap, _ = return_gesture_map(self._gesture_mapping)
            return len(gmap)
        return len(POSITION_CLASS_MAPPING)

    # ------------------------------------------------------------------
    # Iteration: discover_files yields subject directories; load() walks
    # all sessions for that subject (the supervisor's transient-cut step
    # needs per-subject continuity, so we cannot split it per file).
    # ------------------------------------------------------------------
    def discover_files(self) -> list[Path]:
        root = Path(self.config.input_path)
        if not root.exists():
            raise FileNotFoundError(
                f"Dataset {self.config.name!r}: input_path does not exist: {root}",
            )
        return [root / s for s in self._subjects]

    def load(self, filepath: Path) -> Iterator[dict]:
        subj_dir = filepath
        if not subj_dir.is_dir():
            log.warning(
                "Dataset %s: subject directory missing — skipping: %s",
                self.config.name, subj_dir,
            )
            return
        subject = subj_dir.name
        patient_id = _patient_id_from_subject(subject)
        tx_cols = self._tx_cols
        kept_idx = self._kept_channel_indices
        kept_tx_cols = [tx_cols[i] for i in kept_idx]

        df = _load_subject_dataframe(
            subj_dir=subj_dir,
            sessions_to_use=self._sessions_to_use,
            tasks_to_consider=self._tasks_to_consider,
            tx_cols=tx_cols,
            us_frame_len=self._us_frame_len,
            remove_gesture_transients=self._remove_gesture_transients,
            ms_to_cut=self._ms_to_cut,
            us_prr=self._us_prr,
        )
        if df.empty:
            log.warning(
                "Subject %s: no data loaded (sessions=%s tasks=%s)",
                subject, self._sessions_to_use, self._tasks_to_consider,
            )
            return

        df = encode_labels(df, self._gesture_mapping, drop_unknown_positions=True)
        if df.empty:
            log.warning("Subject %s: empty after label encoding/filtering.", subject)
            return

        label_col_int = (
            "Label_gesture_int" if self.label_type == "gesture"
            else "Label_position_int"
        )

        if self._balance_classes:
            df = balance_dataset(df, label_column=label_col_int)
            if df.empty:
                log.warning("Subject %s: empty after class balancing.", subject)
                return

        log.info(
            "Subject %s (patient_id=%d): %d frames — sessions=%s "
            "label_type=%s classes=%s kept_channels=%s",
            subject, patient_id, len(df),
            sorted(df["session_id"].unique().tolist()),
            self.label_type,
            sorted(df[label_col_int].unique().tolist()),
            list(kept_idx),
        )

        for row in df.itertuples(index=False):
            row_d = row._asdict()
            signal = np.stack(
                [np.asarray(row_d[c], dtype=np.float32) for c in kept_tx_cols],
                axis=0,
            )  # (C_kept, T)
            yield {
                "signal": signal,
                "label": int(row_d[label_col_int]),
                "session_id": int(row_d["session_id"]),
                "patient_id": patient_id,
            }


__all__ = [
    "DEFAULT_TX_COLS",
    "DEFAULT_US_FRAME_LEN",
    "DEFAULT_US_FS",
    "DEFAULT_US_PRR",
    "INVALID",
    "LEGACY_STRING_MAPPINGS",
    "POSITION_CLASS_MAPPING",
    "SpaconeForearmBicepProcessor",
    "TASK_GESTURES",
    "balance_dataset",
    "cut_transients",
    "detect_transients",
    "encode_labels",
    "return_gesture_map",
    "waveform_has_nan_or_invalid",
]
