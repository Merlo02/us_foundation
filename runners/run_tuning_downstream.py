#!/usr/bin/env python3
"""Grid-search hyperparameter tuning + leave-one-session-out CV.

Reads a tuning YAML (default location ``configs/model/base_tuning.yaml``)
that mirrors ``base_downstream.yaml`` field-for-field but with every
hyperparameter expressed as a *list of candidate values*. The orchestrator:

1. Loads the tuning YAML (with ``defaults: [...]`` composition) and any
   ``--override`` CLI flags.
2. Walks the merged config and treats every leaf-list under ``data`` /
   ``model`` / ``train`` as a grid axis (Cartesian product).
3. For each combination, enumerates the unique values of ``session_id``
   (or ``patient_id`` when ``split_mode == intra_patient``) from the
   downstream ``all.h5`` and rotates the held-out test group across
   them — full leave-one-session-out cross-validation. When
   ``grouped_val`` is true the validation group is rotated alongside
   (next session in the cycle) so val and test never coincide.
4. Materialises a temporary YAML per ``(combination, held-out session)``
   pair, with ``train.wandb.group = "group_i"`` /
   ``train.wandb.name = "run_j"`` and ``train.run_name = "run_j"`` under
   ``train.output_dir = <sweep_root>/group_i/``.
5. Schedules those runs across the available GPUs — **one run per GPU, no
   DDP** (each child is pinned via ``CUDA_VISIBLE_DEVICES`` and forced to
   ``devices=1``), up to ``--max-parallel`` at a time. Downstream runs are
   small, so N independent 1-GPU runs in parallel beat one N-GPU DDP run
   (no NCCL init, no per-step all-reduce) and use the GPUs fully.

Usage (from ``us_foundation/``)::

    # Default sweep (sweep_name = tuning_<UTC timestamp>):
    python -m runners.run_tuning_downstream --config configs/model/base_tuning.yaml

    # Override a grid axis on the CLI (yaml.safe_load on the value):
    python -m runners.run_tuning_downstream \\
        --config configs/model/base_tuning.yaml \\
        --override train.lr=[1.0e-4,5.0e-4] model.head_dropout=[0.0,0.5]

    # Dry-run: expand the grid + write per-run YAMLs but do NOT launch.
    python -m runners.run_tuning_downstream \\
        --config configs/model/base_tuning.yaml --dry-run

    # Schedule on 2 GPUs, 2 runs at a time (one run per GPU):
    python -m runners.run_tuning_downstream \\
        --config configs/model/base_tuning.yaml --gpus 0,1 --max-parallel 2

    # Wrap each run in SLURM:
    python -m runners.run_tuning_downstream \\
        --config configs/model/base_tuning.yaml \\
        --launch-cmd "srun python -m runners.run_downstream"

This script does **not** import torch / lightning / the model — every
training run is delegated to a fresh ``python -m runners.run_downstream``
subprocess. The orchestrator stays light and crash-isolated.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import h5py
import numpy as np
import yaml

log = logging.getLogger(__name__)


# Sections of the merged config that participate in the grid expansion.
_TUNED_SECTIONS = ("data", "model", "train")

# Per-run fields the orchestrator sets — never tune them via the grid.
_RESERVED_PATHS = frozenset({
    "data.test_id",
    "data.val_id",
    "train.output_dir",
    "train.run_name",
    "train.wandb.group",
    "train.wandb.name",
})


# ----------------------------------------------------------------------
# YAML composition + override application (duplicated from run_downstream
# to keep this orchestrator dependency-free — it must run on any node,
# even one without torch / lightning installed).
# ----------------------------------------------------------------------

def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_composed_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    defaults = raw.pop("defaults", []) or []
    merged: dict = {}
    for d in defaults:
        base_path = path.parent.parent / f"{d}.yaml"
        if not base_path.exists():
            base_path = path.parent / f"{d}.yaml"
        if not base_path.exists():
            raise FileNotFoundError(
                f"Cannot resolve defaults entry {d!r} from {path}"
            )
        _deep_update(merged, _load_composed_yaml(base_path))
    _deep_update(merged, raw)
    return merged


def _load_raw_yaml(path: Path) -> dict:
    """Load *only* what the user wrote in *path* (ignoring ``defaults``).

    Used to decide which leaves are grid axes. Inherited values stay at
    their composed defaults and never appear in the axes list — that
    avoids misinterpreting natural-list fields like ``betas: [0.9, 0.95]``
    or ``window_sizes: [8]`` as 2-option / 1-option grid axes when the
    user did not intend to tune them.
    """
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw.pop("defaults", None)
    return raw


def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    for raw in overrides:
        if "=" not in raw:
            log.warning("Skipping malformed override %r", raw)
            continue
        key, value = raw.split("=", 1)
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        try:
            parsed: Any = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = value
        d[parts[-1]] = parsed


# ----------------------------------------------------------------------
# Grid expansion
# ----------------------------------------------------------------------

def _walk_leaves(node: Any, prefix: str) -> Iterable[tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every non-dict leaf under *node*."""
    if isinstance(node, dict):
        for k, v in node.items():
            sub = f"{prefix}.{k}" if prefix else k
            yield from _walk_leaves(v, sub)
    else:
        yield prefix, node


def _discover_grid_axes(raw_cfg: dict) -> list[tuple[str, list[Any]]]:
    """Return ``(dotted_path, candidates)`` for every leaf-list the user wrote.

    A leaf is "user-written" iff it appears in the raw tuning YAML (NOT
    inherited via ``defaults: [...]``). This lets the user inherit
    natural-list fields like ``betas: [0.9, 0.95]`` from base_downstream
    without them being mistaken for grid axes.

    Scalar leaves are returned as single-candidate axes so they still
    flow into the materialised per-run config — they contribute 1 to the
    product, leaving the grid size unchanged.
    """
    axes: list[tuple[str, list[Any]]] = []
    for section in _TUNED_SECTIONS:
        sub = raw_cfg.get(section)
        if not isinstance(sub, dict):
            continue
        for path, value in _walk_leaves(sub, section):
            if path in _RESERVED_PATHS:
                continue
            if isinstance(value, list):
                axes.append((path, list(value)))
            else:
                axes.append((path, [value]))
    return axes


def _set_dot_path(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    d = cfg
    for p in parts[:-1]:
        if p not in d or not isinstance(d[p], dict):
            d[p] = {}
        d = d[p]
    d[parts[-1]] = value


def _materialise_combination(
    template: dict,
    axes: list[tuple[str, list[Any]]],
    combo: tuple[Any, ...],
) -> dict:
    out = copy.deepcopy(template)
    for (path, _candidates), value in zip(axes, combo):
        _set_dot_path(out, path, value)
    return out


def _incompatible_reason(combo_cfg: dict) -> Optional[str]:
    """Return a short reason if a materialised combination should be skipped.

    Rules:

    - ``gradual_unfreezeing.enabled=true`` + ``freeze_encoder=true``: linear
      probing never unfreezes, so gradual unfreezing is a no-op (duplicate of
      the ``freeze_encoder=true`` row).
    - ``gradual_unfreezeing.enabled=true`` + ``lora.enabled=true``: LoRA is its
      own freezing scheme; the two fine-tuning strategies are mutually
      exclusive.
    - ``lora.enabled=true`` + ``freeze_encoder=true``: PEFT already freezes
      the base encoder when LoRA is attached, and ``freeze_encoder=true``
      then freezes the LoRA adapters too — equivalent to a plain linear
      probe (already covered by ``lora.enabled=false``).
    - ``lora.enabled=true`` + ``layerwise_lr_decay not in (null, 0.0)``:
      LoRA uses uniform LR for the adapters by design; any non-trivial
      decay collapses to "ignored" inside the model (which raises
      :class:`ValueError` anyway) so the combination is a duplicate of
      the ``layerwise_lr_decay=0.0`` row of the sweep.
    - ``freeze_encoder=true`` + ``layerwise_lr_decay not in (null, 0.0)``:
      linear probe trains only the head, so the encoder LR schedule is
      moot; non-trivial decay would silently collapse to a duplicate.

    ``None`` ⇒ the combination is valid.
    """
    m = combo_cfg.get("model", {}) or {}
    lora_enabled = bool((m.get("lora") or {}).get("enabled", False))
    freeze_encoder = bool(m.get("freeze_encoder", False))
    gu_enabled = bool((m.get("gradual_unfreezeing") or {}).get("enabled", False))
    layerwise = m.get("layerwise_lr_decay")
    nontrivial_layerwise = (
        layerwise is not None and float(layerwise) > 0.0
    )

    if gu_enabled and freeze_encoder:
        return (
            "gradual_unfreezeing.enabled=true + freeze_encoder=true — linear "
            "probing never unfreezes the encoder, so gradual unfreezing is "
            "moot (duplicate of the freeze_encoder=true row of this sweep)"
        )
    if gu_enabled and lora_enabled:
        return (
            "gradual_unfreezeing.enabled=true + lora.enabled=true — LoRA "
            "already freezes the base encoder and trains only the adapters; "
            "the two fine-tuning strategies are mutually exclusive"
        )
    if lora_enabled and freeze_encoder:
        return (
            "lora.enabled=true + freeze_encoder=true would freeze the LoRA "
            "adapters too — equivalent to lora.enabled=false (linear probe)"
        )
    if lora_enabled and nontrivial_layerwise:
        return (
            f"lora.enabled=true requires layerwise_lr_decay in (null, 0.0); "
            f"got {layerwise!r} — duplicate of the lora.enabled=true + "
            f"layerwise_lr_decay=0.0 row of this sweep"
        )
    if freeze_encoder and nontrivial_layerwise:
        return (
            f"freeze_encoder=true requires layerwise_lr_decay in (null, 0.0); "
            f"got {layerwise!r} — linear probe trains only the head, so the "
            f"encoder LR schedule is moot (duplicate of the "
            f"freeze_encoder=true + layerwise_lr_decay=0.0 row of this sweep)"
        )
    return None


# ----------------------------------------------------------------------
# Session enumeration
# ----------------------------------------------------------------------

def _enumerate_test_ids(h5_path: str, split_mode: str) -> list[Any]:
    """Return the list of held-out values rotated by the outer loop.

    - ``intra_session`` → unique values of ``/session_id``
    - ``intra_patient`` → unique values of ``/patient_id``
    - ``random``        → ``[None]`` (single run per combination — no CV)
    """
    if split_mode == "random":
        return [None]
    if split_mode == "intra_session":
        key = "session_id"
    elif split_mode == "intra_patient":
        key = "patient_id"
    else:
        raise ValueError(
            f"split_mode={split_mode!r} not supported (expected "
            f"'intra_session', 'intra_patient', or 'random').",
        )
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            raise KeyError(
                f"Dataset {key!r} missing from {h5_path}; cannot enumerate "
                f"held-out groups.",
            )
        ids = np.unique(np.asarray(f[key][:], dtype=np.int64)).tolist()
    return [int(x) for x in ids]


def _rotate_val_id(test_ids: list[Any], run_idx: int) -> int:
    """Pick the next group in the cycle for validation."""
    n = len(test_ids)
    if n < 2:
        raise ValueError(
            "grouped_val=True requires at least 2 groups so val != test "
            f"(have {n}).",
        )
    return int(test_ids[(run_idx + 1) % n])


# ----------------------------------------------------------------------
# Per-run config + launch
# ----------------------------------------------------------------------

def _build_per_run_cfg(
    combo_cfg: dict,
    *,
    sweep_root: Path,
    group_idx: int,
    run_idx: int,
    test_id: Any,
    val_id: Any,
    group_prefix: str,
) -> dict:
    cfg = copy.deepcopy(combo_cfg)

    data = cfg.setdefault("data", {})
    # Always overwrite test_id / val_id so any inherited list (e.g.
    # ``test_id: [null]`` from base_tuning.yaml's "every leaf is a list"
    # convention) is replaced with a clean scalar that DownstreamDataModule
    # can int(...) without a TypeError.
    data["test_id"] = int(test_id) if test_id is not None else None
    data["val_id"] = int(val_id) if val_id is not None else None

    train = cfg.setdefault("train", {})
    # Group_i directory holds every run_j of one hyperparameter combo, so
    # checkpoints, csv logs, and per-run config snapshots stay co-located.
    group_dir = sweep_root / f"group_{group_idx}"
    train["output_dir"] = str(group_dir)
    train["run_name"] = f"run_{run_idx}"
    # One GPU per run, no DDP: the orchestrator schedules runs across GPUs
    # (CUDA_VISIBLE_DEVICES per child), so each child must train on a single
    # device. Forced here so it holds even if the grid left devices > 1.
    train["devices"] = 1
    train["num_nodes"] = 1

    # WandB grouping: only if wandb is enabled in the user's config.
    # ``group_prefix`` (from train.wandb.group in the tuning YAML, else the
    # sweep name) namespaces every combination's group so two different
    # sweeps synced into the same WandB project never collide on a generic
    # ``group_0`` — each combination becomes ``<prefix>_group_<i>`` and each
    # run ``<prefix>_group_<i>_run_<j>`` (the WandB UI groups by ``group``,
    # independent of on-disk location).
    #
    # ``save_dir`` is set to THIS run's own directory (<group>/run_j), NOT the
    # shared group dir: wandb's offline folder is ``offline-run-<ts>-<id>`` with
    # a random id, so a shared ``<group>/wandb/`` parent makes those dirs
    # impossible to map back to run_j — and, now that runs execute in parallel,
    # all processes also race on the shared ``latest-run`` / ``debug.log``
    # symlinks. One ``wandb/`` per run dir keeps each run's offline data
    # attributable and race-free. Sync with
    # ``wandb sync <sweep>/group_*/run_*/wandb/offline-run-*``.
    wandb_cfg = train.setdefault("wandb", {})
    if bool(wandb_cfg.get("enabled", False)):
        wandb_cfg["group"] = f"{group_prefix}_group_{group_idx}"
        wandb_cfg["name"] = f"{group_prefix}_group_{group_idx}_run_{run_idx}"
        wandb_cfg["save_dir"] = str(group_dir / f"run_{run_idx}")

    return cfg


def _write_run_yaml(cfg: dict, sweep_root: Path, group_idx: int, run_idx: int) -> Path:
    """Write the per-run launch YAML directly into the run directory.

    The launch YAML lives at ``<run_dir>/config.yaml`` so it is co-located
    with the run output (checkpoints, lora_adapter, wandb, lightning_logs).
    ``runners.run_downstream`` then overwrites the same path with the
    post-sync config — net effect: a single canonical ``config.yaml`` per
    run, and no redundant ``<sweep_root>/configs/`` mirror.
    """
    run_dir = sweep_root / f"group_{group_idx}" / f"run_{run_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _available_gpus(gpus_arg: Optional[str]) -> list[str]:
    """Resolve the list of GPU ids the scheduler may use.

    Priority: explicit ``--gpus`` → parent ``CUDA_VISIBLE_DEVICES`` (the SLURM
    allocation) → ``nvidia-smi -L`` count → ``["0"]``. Each scheduled run is
    pinned to exactly one of these ids via its own ``CUDA_VISIBLE_DEVICES``, so
    we only ever hand out ids that the parent could already see.
    """
    if gpus_arg:
        gpus = [g.strip() for g in gpus_arg.split(",") if g.strip()]
        if gpus:
            return gpus
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        gpus = [g.strip() for g in env.split(",") if g.strip()]
        if gpus:
            return gpus
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=True,
        )
        n = sum(1 for ln in out.stdout.splitlines() if ln.strip().startswith("GPU "))
        if n > 0:
            return [str(i) for i in range(n)]
    except (OSError, subprocess.SubprocessError):
        pass
    return ["0"]


def _run_pool(
    jobs: list[dict],
    launch_cmd: str,
    gpus: list[str],
    stop_on_error: bool,
    poll_interval: float = 2.0,
) -> tuple[list[tuple[dict, int]], int]:
    """Run *jobs* concurrently, one run per GPU (no DDP).

    Keeps at most ``len(gpus)`` children alive at once; each is pinned to a
    distinct GPU via ``CUDA_VISIBLE_DEVICES`` so the per-run config's
    ``devices=1`` lands on that physical device. Returns ``(failures,
    launched)`` where ``failures`` is a list of ``(job, returncode)``.

    With ``stop_on_error`` set, no further runs are launched after the first
    failure, but already-running children are allowed to finish.
    """
    from collections import deque

    queue: deque = deque(jobs)
    free_gpus = list(gpus)
    running: dict = {}                       # Popen -> (gpu, job)
    failures: list[tuple[dict, int]] = []
    launched = 0
    aborting = False

    while queue or running:
        while not aborting and free_gpus and queue:
            gpu = free_gpus.pop()
            job = queue.popleft()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = launch_cmd.split() + ["--config", str(job["run_yaml"])]
            log.info(
                "launching group_%d/run_%d on GPU %s: %s",
                job["combo_idx"], job["run_idx"], gpu, " ".join(cmd),
            )
            running[subprocess.Popen(cmd, env=env)] = (gpu, job)
            launched += 1

        if not running:
            break
        time.sleep(poll_interval)

        for proc in list(running):
            rc = proc.poll()
            if rc is None:
                continue
            gpu, job = running.pop(proc)
            free_gpus.append(gpu)
            if rc != 0:
                failures.append((job, int(rc)))
                log.error(
                    "group_%d/run_%d FAILED (rc=%d) — %s",
                    job["combo_idx"], job["run_idx"], rc, job["run_yaml"],
                )
                if stop_on_error and not aborting:
                    aborting = True
                    log.error(
                        "--stop-on-error set — not launching further runs; "
                        "waiting for %d in-flight run(s) to finish.",
                        len(running),
                    )

    return failures, launched


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Grid-search hyperparameters + leave-one-session-out CV for the "
            "downstream task by shelling out to runners.run_downstream once "
            "per (combination, held-out session) pair."
        ),
    )
    p.add_argument(
        "--config", type=str, required=True,
        help="Tuning YAML (e.g. configs/model/base_tuning.yaml)",
    )
    p.add_argument(
        "--override", type=str, nargs="*", default=[],
        help="Dot-path overrides; yaml.safe_load on the value. "
             "Example: ``train.lr=[1e-3,1e-4] model.head_dropout=[0.0,0.5]``",
    )
    p.add_argument(
        "--sweep-name", type=str, default=None,
        help="Subdirectory under train.output_dir for this sweep "
             "(default: ``tuning_<UTC timestamp>``).",
    )
    p.add_argument(
        "--launch-cmd", type=str,
        default="python -m runners.run_downstream",
        help="Command used to launch one downstream run. Defaults to "
             "``python -m runners.run_downstream``. Use e.g. "
             "``srun python -m runners.run_downstream`` under SLURM.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Expand the grid and write all per-run YAMLs, but do not "
             "actually launch run_downstream.",
    )
    p.add_argument(
        "--stop-on-error", action="store_true",
        help="Stop launching new runs on the first failure (in-flight runs "
             "still finish). Default: continue with the remaining runs.",
    )
    p.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU ids to schedule on (e.g. ``0,1,2,3``). "
             "Default: CUDA_VISIBLE_DEVICES if set, else all GPUs reported by "
             "nvidia-smi. Each run is pinned to exactly one of these.",
    )
    p.add_argument(
        "--max-parallel", type=int, default=None,
        help="Max concurrent runs (default: number of scheduled GPUs). Each "
             "run uses one GPU; capped at the number of GPUs.",
    )
    return p.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _enforce_single_orchestrator() -> None:
    """Guard against the orchestrator being launched as multiple SLURM tasks.

    This orchestrator manages its OWN GPU pool (one run_downstream child per
    GPU), so it must run exactly once. Under ``srun`` with
    ``--ntasks(-per-node) > 1`` it gets replicated, and every copy relaunches the
    whole sweep into the same run dirs → concurrent trainings colliding on
    checkpoints (``last-vN.ckpt`` …) with only ``SLURM_PROCID 0`` logging. So:
    let only PROCID 0 proceed, and scrub the SLURM task/rank vars from the
    environment so the plain-python children aren't mis-detected by Lightning as
    part of a multi-rank job.
    """
    ntasks = int(os.environ.get("SLURM_NTASKS", "1") or "1")
    procid = int(os.environ.get("SLURM_PROCID", "0") or "0")
    if ntasks > 1:
        if procid != 0:
            print(
                f"[run_tuning_downstream] SLURM_NTASKS={ntasks}: this "
                f"orchestrator must run as a single task — exiting on "
                f"SLURM_PROCID={procid}. Set --ntasks-per-node=1 in your sbatch.",
                file=sys.stderr,
            )
            sys.exit(0)
        log.warning(
            "SLURM_NTASKS=%d detected — this orchestrator manages its own GPU "
            "pool and must run as ONE task. Continuing only on SLURM_PROCID=0; "
            "set --ntasks-per-node=1 in your sbatch to avoid spawning duplicate "
            "orchestrators.", ntasks,
        )
    # Scrub SLURM task/rank vars so the launched children (inheriting this env
    # via os.environ.copy() in _run_pool) run as normal single-process jobs —
    # no false SLURM multi-rank detection in Lightning.
    for var in (
        "SLURM_NTASKS", "SLURM_NTASKS_PER_NODE", "SLURM_NPROCS",
        "SLURM_PROCID", "SLURM_LOCALID", "SLURM_STEP_NUM_TASKS",
        "SLURM_TASKS_PER_NODE",
    ):
        os.environ.pop(var, None)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _enforce_single_orchestrator()
    args = _parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Tuning config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    template_cfg = _load_composed_yaml(cfg_path)
    raw_cfg = _load_raw_yaml(cfg_path)
    _apply_overrides(template_cfg, args.override)
    # CLI overrides are also user-written → eligible to be grid axes.
    _apply_overrides(raw_cfg, args.override)

    axes = _discover_grid_axes(raw_cfg)
    if not axes:
        print(
            "No grid axes found in the tuning YAML (nothing to tune). "
            "Did you forget to wrap hyperparameters in lists?",
            file=sys.stderr,
        )
        sys.exit(1)

    real_axes = [(p, c) for p, c in axes if len(c) > 1]
    n_combinations = 1
    for _, c in axes:
        n_combinations *= len(c)
    log.info(
        "Discovered %d grid axes (%d non-trivial). Total combinations: %d",
        len(axes), len(real_axes), n_combinations,
    )
    for path, candidates in real_axes:
        log.info("  %-40s × %d  %s", path, len(candidates), candidates)

    # Sweep root directory layout:
    #   <base_output_dir>/<sweep_name>/
    #       sweep_manifest.yaml
    #       group_i/                         (one per kept combination)
    #           run_j/                       (one per held-out session)
    #               config.yaml              (launch + post-sync)
    #               checkpoints/             (best by monitor; +last if save_last)
    #               wandb/offline-run-…/     (this run's own offline data)
    #               lora_adapter/            (only if lora.enabled=true)
    #               lightning_logs/
    base_output = template_cfg.get("train", {}).get("output_dir")
    # The user may have wrapped output_dir as a 1-option list to keep the
    # "every hyperparameter is a list" rule — unwrap it before joining a
    # path. The orchestrator overrides train.output_dir per-run anyway.
    if isinstance(base_output, list):
        base_output = base_output[0] if base_output else None
    base_output_dir = Path(str(base_output) if base_output else "tuning_runs")
    sweep_name = args.sweep_name or f"tuning_{time.strftime('%Y%m%d_%H%M%S')}"
    sweep_root = base_output_dir / sweep_name
    sweep_root.mkdir(parents=True, exist_ok=True)

    # WandB group naming: prefix each combination's group with a name that is
    # unique to this sweep, so groups synced into a shared WandB project don't
    # all collapse onto a generic ``group_0`` / ``group_1`` across sweeps. The
    # prefix is taken from ``train.wandb.group`` in the tuning YAML when set
    # (unwrapping the 1-option list the "every leaf is a list" convention may
    # produce), otherwise it defaults to the sweep name.
    _wb_template = (template_cfg.get("train", {}) or {}).get("wandb", {}) or {}
    _gp = _wb_template.get("group")
    if isinstance(_gp, list):
        _gp = _gp[0] if _gp else None
    group_prefix = str(_gp) if _gp else sweep_name
    log.info("WandB group prefix: %s", group_prefix)

    (sweep_root / "sweep_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "source_config": str(cfg_path.resolve()),
                "sweep_name": sweep_name,
                "axes": [
                    {"path": p, "candidates": c, "n": len(c)}
                    for p, c in axes
                ],
                "n_combinations": n_combinations,
                "launch_cmd": args.launch_cmd,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # ── Phase 1: expand the grid and materialise every (combo, fold) run ──
    # All per-run YAMLs are written first; the GPU scheduler (phase 2) then
    # treats each as an independent 1-GPU job. This keeps the dry-run path
    # identical and decouples materialisation from launching.
    jobs: list[dict] = []
    skipped: list[tuple[int, str, dict]] = []  # (combo_idx, reason, combo_summary)

    for combo_idx, combo in enumerate(itertools.product(*[c for _, c in axes])):
        combo_cfg = _materialise_combination(template_cfg, axes, combo)

        # combo is aligned with the FULL axes list (incl. 1-option ones),
        # so index into it directly when summarising only the non-trivial
        # dimensions for the log line.
        combo_summary = {
            axes[i][0]: combo[i]
            for i in range(len(axes))
            if len(axes[i][1]) > 1
        }

        # Reject combinations that don't make sense to run (e.g. lora on
        # top of a frozen encoder — see _incompatible_reason). The group
        # index keeps incrementing so the per-combination numbering in
        # the sweep log still matches the Cartesian product order.
        reason = _incompatible_reason(combo_cfg)
        if reason is not None:
            log.info(
                "── group_%d / %d ── SKIP: %s | combo=%s",
                combo_idx, n_combinations, reason, combo_summary,
            )
            skipped.append((combo_idx, reason, combo_summary))
            continue

        # h5_path and split_mode can themselves be tuned — re-resolve them
        # per combination so the test_id rotation always matches.
        h5_path = str(combo_cfg["data"]["h5_path"])
        split_mode = str(combo_cfg["data"].get("split_mode", "intra_session"))
        grouped_val = bool(combo_cfg["data"].get("grouped_val", False))
        test_ids = _enumerate_test_ids(h5_path, split_mode)

        log.info(
            "── group_%d / %d ── split=%s | sessions=%d | combo=%s",
            combo_idx, n_combinations,
            split_mode, len(test_ids), combo_summary,
        )

        for run_idx, test_id in enumerate(test_ids):
            if (
                grouped_val and test_id is not None
                and split_mode != "random" and len(test_ids) > 1
            ):
                val_id = _rotate_val_id(test_ids, run_idx)
            else:
                val_id = None
            per_run_cfg = _build_per_run_cfg(
                combo_cfg,
                sweep_root=sweep_root,
                group_idx=combo_idx,
                run_idx=run_idx,
                test_id=test_id,
                val_id=val_id,
                group_prefix=group_prefix,
            )
            run_yaml = _write_run_yaml(
                per_run_cfg, sweep_root, combo_idx, run_idx,
            )
            log.info(
                "  group_%d/run_%d — test_id=%s val_id=%s → %s",
                combo_idx, run_idx, test_id, val_id, run_yaml,
            )
            jobs.append({
                "combo_idx": combo_idx,
                "run_idx": run_idx,
                "test_id": test_id,
                "val_id": val_id,
                "run_yaml": run_yaml,
            })

    n_kept = n_combinations - len(skipped)
    if skipped:
        log.info(
            "Skipped %d / %d combination(s) as incompatible:",
            len(skipped), n_combinations,
        )
        for g, reason, summary in skipped:
            log.info("  group_%d  reason: %s  combo=%s", g, reason, summary)

    if args.dry_run:
        log.info(
            "Dry-run complete — %d job(s) from %d kept combination(s) "
            "materialised under %s (skipped %d).",
            len(jobs), n_kept, sweep_root, len(skipped),
        )
        return

    # ── Phase 2: schedule the runs across GPUs, one run per GPU (no DDP) ──
    gpus = _available_gpus(args.gpus)
    max_parallel = args.max_parallel if args.max_parallel is not None else len(gpus)
    max_parallel = max(1, min(max_parallel, len(gpus)))
    pool_gpus = gpus[:max_parallel]
    log.info(
        "Scheduling %d run(s) across GPUs %s — 1 GPU/run, no DDP "
        "(max_parallel=%d).",
        len(jobs), pool_gpus, max_parallel,
    )

    failures, total_launched = _run_pool(
        jobs, args.launch_cmd, pool_gpus, args.stop_on_error,
    )

    if failures:
        log.warning(
            "Sweep finished with %d failed run(s) out of %d launched.",
            len(failures), total_launched,
        )
        for job, rc in failures:
            log.warning(
                "  group_%d / run_%d  rc=%d  cfg=%s",
                job["combo_idx"], job["run_idx"], rc, job["run_yaml"],
            )
        sys.exit(2)

    log.info(
        "Sweep finished cleanly — %d kept combination(s) (skipped %d) "
        "× sessions = %d total launches. Output under %s",
        n_kept, len(skipped), total_launched, sweep_root,
    )


if __name__ == "__main__":
    main()
