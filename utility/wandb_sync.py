#!/usr/bin/env python3
"""Sync the offline WandB runs of a tuning sweep, one group at a time.

Layout expected (produced by ``runners.run_tuning_downstream``)::

    <sweep_root>/
        sweep_manifest.yaml
        group_0/
            wandb/
                offline-run-…/      # one per held-out session
                offline-run-…/
                …
            run_0/
                config.yaml
                checkpoints/last.ckpt
                lora_adapter/
                …
            run_1/…
        group_1/…
        …

For each ``group_*`` under ``<sweep_root>`` this script invokes ``wandb sync``
on every ``offline-run-*`` directory found under ``group_i/wandb/``. The
WandB ``run_group`` field is already baked into each offline run's protobuf
by ``run_downstream``'s ``WandbLogger(group="group_i", name="run_j")`` (you
can confirm with ``wandb.sdk.internal.datastore`` reading the ``.wandb``
file), so the synced runs carry the right group on the cloud automatically.

**Viewing the group on the WandB UI** — by default the workspace shows a
flat list of runs in the project. To see them grouped:

1. Open the project page.
2. In the *Runs* sidebar (or the *Runs* table), click *Group*.
3. Pick ``Group`` (the run-group field) as the grouping key.

After that, every sweep group becomes a collapsible section that contains
the N session runs of that hyperparameter combination.

**Idempotency** — ``wandb sync`` writes a ``run-<id>.wandb.synced`` marker
on success. This script skips runs that already carry that marker so
re-runs only upload the new ones. Pass ``--include-synced`` to force a
re-upload (rarely needed; a run that was deleted on the cloud cannot be
re-created with the same id — those produce a clean *Skipping* warning
instead of aborting the sweep sync).

Usage (from ``us_foundation/``)::

    # Activate the venv that has wandb installed first, then:
    python wandb_sync.py /leonardo_scratch/…/tuning/tuning_20260528_173834

    # Override the project / entity on the fly:
    python wandb_sync.py <sweep_root> --project myproj --entity myentity

    # Inspect what would be synced without actually uploading anything:
    python wandb_sync.py <sweep_root> --dry-run

    # Force re-attempt of already-synced offline dirs:
    python wandb_sync.py <sweep_root> --include-synced
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sync the offline WandB runs of a tuning sweep, one group at a time."
        ),
    )
    p.add_argument(
        "sweep_root", type=str,
        help="Path to the sweep root directory (contains group_* subdirs).",
    )
    p.add_argument(
        "--project", type=str, default=None,
        help="Override the WandB project on upload (default: leave as recorded).",
    )
    p.add_argument(
        "--entity", type=str, default=None,
        help="Override the WandB entity on upload (default: leave as recorded).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="List every wandb sync command that would run but do not execute.",
    )
    p.add_argument(
        "--stop-on-error", action="store_true",
        help="Abort on the first failed sync (default: continue, report at end).",
    )
    p.add_argument(
        "--include-synced", action="store_true",
        help="Re-attempt offline runs that already carry a "
             "``run-<id>.wandb.synced`` marker (default: skip them — "
             "wandb writes that marker on every successful upload, so the "
             "default behaviour makes this script idempotent).",
    )
    return p.parse_args()


def _wandb_executable() -> str:
    exe = shutil.which("wandb")
    if exe is None:
        print(
            "wandb CLI not found on PATH. Activate the env that has wandb "
            "installed (e.g. ``source ~/usf_venv/bin/activate``) and retry.",
            file=sys.stderr,
        )
        sys.exit(1)
    return exe


def _is_already_synced(offline_dir: Path) -> bool:
    """``wandb sync`` writes a ``run-<id>.wandb.synced`` marker on success."""
    return any(offline_dir.glob("run-*.wandb.synced"))


# Substring tokens that mean "the run on the cloud is gone for good" — the
# id was either deleted by hand or marked-as-deleted by the WandB backend,
# so re-uploading with the same id is impossible. We log + skip these so a
# single dead run doesn't abort the whole sweep sync.
_DEAD_RUN_HINTS = (
    "previously created and deleted",
    "409",
)


def _sync_one(
    wandb_exe: str, offline_dir: Path,
    project: str | None, entity: str | None,
    dry_run: bool,
    include_synced: bool = False,
) -> int:
    cmd = [wandb_exe, "sync"]
    if include_synced:
        # Tell the wandb CLI to upload even when the offline dir already
        # has a ``run-<id>.wandb.synced`` marker. Needed after the cloud
        # project is wiped — the local markers are stale but the CLI
        # would otherwise refuse to re-upload.
        cmd.append("--include-synced")
    if project:
        cmd += ["--project", project]
    if entity:
        cmd += ["--entity", entity]
    cmd.append(str(offline_dir))
    log.info("  %s", " ".join(cmd))
    if dry_run:
        return 0
    # Capture stderr so we can detect the dead-run case and turn it into
    # a skip rather than a fatal failure.
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0 and any(h in proc.stderr for h in _DEAD_RUN_HINTS):
        log.warning(
            "  Skipping %s: WandB rejected the upload because the run id "
            "was already created and deleted on the cloud. Delete the "
            "offline dir manually or re-run training to get a fresh id.",
            offline_dir.name,
        )
        return 0
    return proc.returncode


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
        d for d in sweep_root.iterdir()
        if d.is_dir() and d.name.startswith("group_")
    )
    if not group_dirs:
        print(
            f"No ``group_*`` subdirectories found under {sweep_root}. "
            f"Is this actually a sweep root produced by run_tuning_downstream?",
            file=sys.stderr,
        )
        sys.exit(1)

    wandb_exe = _wandb_executable()

    n_synced = 0
    n_skipped_synced = 0
    failures: list[Path] = []
    skipped_groups: list[Path] = []

    for group_dir in group_dirs:
        wandb_dir = group_dir / "wandb"
        if not wandb_dir.is_dir():
            log.warning("─ %s — no wandb/ subdir, skipping", group_dir.name)
            skipped_groups.append(group_dir)
            continue
        offline_runs = sorted(
            p for p in wandb_dir.iterdir()
            if p.is_dir() and p.name.startswith("offline-")
        )
        if not offline_runs:
            log.warning("─ %s — no offline-* runs under wandb/, skipping",
                        group_dir.name)
            skipped_groups.append(group_dir)
            continue
        log.info("── %s — %d offline run(s)", group_dir.name, len(offline_runs))
        for offline in offline_runs:
            if not args.include_synced and _is_already_synced(offline):
                log.info("  ↻ %s — already synced (skip)", offline.name)
                n_skipped_synced += 1
                continue
            rc = _sync_one(
                wandb_exe, offline, args.project, args.entity, args.dry_run,
                include_synced=args.include_synced,
            )
            if rc != 0:
                log.error("  FAILED (rc=%d): %s", rc, offline)
                failures.append(offline)
                if args.stop_on_error:
                    log.error("--stop-on-error set — aborting.")
                    sys.exit(1)
            else:
                n_synced += 1

    log.info(
        "Done — %d run(s) %s, %d already-synced skipped, %d failure(s)%s.",
        n_synced, "would be synced" if args.dry_run else "synced",
        n_skipped_synced, len(failures),
        f", {len(skipped_groups)} empty group(s)" if skipped_groups else "",
    )
    if failures:
        for f in failures:
            log.error("  failed: %s", f)
        sys.exit(2)


if __name__ == "__main__":
    main()
