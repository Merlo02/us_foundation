#!/usr/bin/env python3
"""Summarise a downstream tuning sweep into a single ranked ``summary.csv``.

Layout expected (produced by ``runners.run_tuning_downstream``)::

    <sweep_root>/
        sweep_manifest.yaml
        group_0/
            run_0/
                config.yaml
                lightning_logs/version_0/metrics.csv
            run_1/…
            …                       # 6 held-out-session runs per group
        group_1/…
        …

For every ``group_*`` this script:

1. Reads the hyperparameters that are *shared across the whole group*
   (``model.lora.enabled``, ``model.pretrained_dir``,
   ``model.freeze_encoder``, ``model.layerwise_lr_decay``, ``train.lr``)
   from the **first** available run's ``config.yaml`` only — they are
   identical for every run of the group, so there is no need to read them
   per run.
2. Reads the test accuracy of every run from
   ``run_j/lightning_logs/version_0/metrics.csv``: among the rows that
   carry a non-empty ``test/acc`` it picks the one of the **last epoch**
   (Lightning logs ``test/acc`` once, at the final ``model.test()`` call).
3. Averages those per-run test accuracies into the group's mean
   ``test_accuracy``.

The groups are written to ``<sweep_root>/summary.csv`` sorted by
``test_accuracy`` descending, so the best hyperparameter combination is on
top. Runs whose ``metrics.csv`` is missing or has no ``test/acc`` row are
skipped (and reflected in ``n_runs``) rather than aborting the summary —
some runs crashed while logging to WandB, so the on-disk CSV is the source
of truth here.

Usage (from ``us_foundation/``, with a venv that has PyYAML, e.g. usf_venv)::

    python write_summary.py /leonardo_scratch/.../models_downstream/tuning_full/tuning_20260530_091739

    # Write the summary elsewhere:
    python write_summary.py <sweep_root> --output /tmp/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

log = logging.getLogger(__name__)

# Group-level hyperparameters surfaced in the summary, as
# ``(column_name, dotted_config_path)`` pairs. They are read once per group
# from the first available run's config.yaml (identical across the group).
_GROUP_PARAMS: list[tuple[str, str]] = [
    ("lora_enabled", "model.lora.enabled"),
    ("freeze_encoder", "model.freeze_encoder"),
    ("layerwise_lr_decay", "model.layerwise_lr_decay"),
    ("lr", "train.lr"),
    ("pretrained_dir", "model.pretrained_dir"),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Summarise a downstream tuning sweep into a single ranked "
            "summary.csv (one row per group, sorted by mean test accuracy)."
        ),
    )
    p.add_argument(
        "sweep_root", type=str,
        help="Path to the sweep root directory (contains group_* subdirs).",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Where to write the summary CSV "
             "(default: <sweep_root>/summary.csv).",
    )
    return p.parse_args()


def _natural_group_key(group_dir: Path) -> tuple[int, str]:
    """Sort ``group_2`` before ``group_10`` (numeric, not lexicographic)."""
    m = re.search(r"(\d+)$", group_dir.name)
    return (int(m.group(1)) if m else sys.maxsize, group_dir.name)


def _dig(cfg: dict, dotted: str) -> Any:
    """Return the nested value at ``dotted`` path, or ``None`` if absent."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _last_epoch_test_acc(metrics_csv: Path) -> Optional[float]:
    """Return ``test/acc`` from the last-epoch row of a run's metrics.csv.

    Lightning appends one row per logged step; ``test/acc`` is populated
    only on the final ``model.test()`` call. Among every row carrying a
    non-empty ``test/acc`` we keep the one with the largest ``epoch`` (the
    last epoch); ties / missing epoch values fall back to file order.
    """
    with metrics_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "test/acc" not in reader.fieldnames:
            return None
        best_epoch = -1.0
        best_acc: Optional[float] = None
        for row in reader:
            raw = (row.get("test/acc") or "").strip()
            if not raw:
                continue
            try:
                acc = float(raw)
            except ValueError:
                continue
            epoch_raw = (row.get("epoch") or "").strip()
            try:
                epoch = float(epoch_raw)
            except ValueError:
                epoch = best_epoch  # no epoch → keep file order (>= wins)
            if epoch >= best_epoch:
                best_epoch = epoch
                best_acc = acc
        return best_acc


def _summarise_group(group_dir: Path) -> Optional[dict[str, Any]]:
    """Build one summary row for a group, or ``None`` if it has no usable run."""
    run_dirs = sorted(
        (d for d in group_dir.iterdir()
         if d.is_dir() and d.name.startswith("run_")),
        key=lambda d: (
            int(m.group(1)) if (m := re.search(r"(\d+)$", d.name)) else sys.maxsize
        ),
    )
    if not run_dirs:
        log.warning("─ %s — no run_* subdirs, skipping", group_dir.name)
        return None

    # Group-level hyperparameters: read once from the first run whose
    # config.yaml exists (they are identical across the group).
    params: dict[str, Any] = {col: None for col, _ in _GROUP_PARAMS}
    for run_dir in run_dirs:
        cfg_path = run_dir / "config.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            for col, dotted in _GROUP_PARAMS:
                params[col] = _dig(cfg, dotted)
            break
    else:
        log.warning("─ %s — no config.yaml in any run, params left blank",
                    group_dir.name)

    accs: list[float] = []
    for run_dir in run_dirs:
        metrics_csv = run_dir / "lightning_logs" / "version_0" / "metrics.csv"
        if not metrics_csv.exists():
            log.warning("  %s/%s — no metrics.csv, skipping run",
                        group_dir.name, run_dir.name)
            continue
        acc = _last_epoch_test_acc(metrics_csv)
        if acc is None:
            log.warning("  %s/%s — no test/acc row in metrics.csv, skipping run",
                        group_dir.name, run_dir.name)
            continue
        accs.append(acc)

    if not accs:
        log.warning("─ %s — no run produced a test/acc, skipping group",
                    group_dir.name)
        return None

    mean = sum(accs) / len(accs)
    var = sum((a - mean) ** 2 for a in accs) / len(accs)
    log.info("── %s — test_accuracy=%.4f over %d run(s)",
             group_dir.name, mean, len(accs))

    row: dict[str, Any] = {
        "group": group_dir.name,
        "test_accuracy": round(mean, 6),
        "test_accuracy_std": round(var ** 0.5, 6),
        "n_runs": len(accs),
    }
    row.update(params)
    return row


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args()

    sweep_root = Path(args.sweep_root).expanduser().resolve()
    if not sweep_root.is_dir():
        print(f"Not a directory: {sweep_root}", file=sys.stderr)
        sys.exit(1)

    group_dirs = sorted(
        (d for d in sweep_root.iterdir()
         if d.is_dir() and d.name.startswith("group_")),
        key=_natural_group_key,
    )
    if not group_dirs:
        print(
            f"No ``group_*`` subdirectories found under {sweep_root}. "
            f"Is this actually a sweep root produced by run_tuning_downstream?",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = [r for d in group_dirs if (r := _summarise_group(d)) is not None]
    if not rows:
        print("No group produced a usable test accuracy — nothing to write.",
              file=sys.stderr)
        sys.exit(1)

    # Best combination on top.
    rows.sort(key=lambda r: r["test_accuracy"], reverse=True)

    fieldnames = (
        ["group", "test_accuracy", "test_accuracy_std", "n_runs"]
        + [col for col, _ in _GROUP_PARAMS]
    )
    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output else sweep_root / "summary.csv"
    )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote %d group(s) to %s (best: %s @ test_accuracy=%.4f)",
             len(rows), out_path, rows[0]["group"], rows[0]["test_accuracy"])


if __name__ == "__main__":
    main()
