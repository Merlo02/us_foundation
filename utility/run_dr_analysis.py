#!/usr/bin/env python3
"""Dimensionality-reduction (t-SNE / UMAP) of encoder features at inference time.

Given a sweep GROUP directory (the kind produced by
:mod:`runners.run_tuning_downstream`), e.g.::

    /.../KFold_LinearProbing_interp_800/group_0
        run_0/  config.yaml  checkpoints/last.ckpt
        run_1/  config.yaml  checkpoints/last.ckpt
        ...

for **every** ``run_*`` under that directory this tool:

1. reads ``run_i/config.yaml`` — the post-sync flat config saved by
   :mod:`runners.run_downstream` — to recover ``data.h5_path``,
   ``data.test_id`` and every encoder / data hyper-parameter;
2. loads ``run_i/checkpoints/last.ckpt`` into an
   :class:`~model.UltrasonicDownstream`;
3. builds the held-out **test** split (rows whose ``session_id == test_id``
   for ``intra_session``; ``patient_id == test_id`` for ``intra_patient``)
   through :class:`~data.downstream_datamodule.DownstreamDataModule`, so the
   signals get the *exact same* interpolation + per-train-channel
   normalization used at evaluation time;
4. extracts the pooled encoder embeddings ``(N, C·E)`` — the very features
   the built-in test t-SNE projects and that feed the classification head;
5. projects them to 2-D with t-SNE and/or UMAP (``--mode``) and saves a
   scatter coloured by class label.

Plots land in a ``DR_plots`` directory placed **next to** the group
directory (``<parent-of-group>/DR_plots``), one PNG per ``(run, method)``
named ``<group>_<run>_test<id>_<method>.png``. Override with ``--output-dir``.

Single environment
-------------------
This runs end-to-end in one process. It needs both the model stack
(torch + Lightning + ``model``/``data``) **and** the projection libs
(``scikit-learn`` for t-SNE, ``umap-learn`` for UMAP). On Leonardo that is
the ``useless_venv`` environment (which has umap-learn installed alongside
torch/Lightning/timm/webdataset/h5py)::

    cd ~/us_foundation
    ~/useless_venv/bin/python -m utility.run_dr_analysis --mode both --path /.../KFold_LinearProbing_interp_800/group_0

CPU inference of a whole test session can be heavy; for big sessions run it
on a GPU compute node (``srun ... --gres=gpu:1``) or cap the work with
``--max-points`` / ``--max-batches``. UMAP is skipped (with a hint) if
``umap-learn`` is not importable, so ``--mode both`` still yields the t-SNE
plots.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting

import numpy as np  # safe everywhere; torch/model imported lazily in main()

# Make the project root importable so ``from model ...`` / ``from runners ...``
# resolve no matter how this script is launched (``python -m
# utility.run_dr_analysis`` or ``python utility/run_dr_analysis.py``).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

log = logging.getLogger(__name__)

# Display names for the 6-class spacone gesture task (alphabetical union,
# matches return_gesture_map). Any other corpus falls back to ``class {c}``.
_GESTURE_NAMES_6 = ["open", "pinch", "pour", "rest", "rotclosed", "rotopen"]

_CKPT_REL = Path("checkpoints") / "last.ckpt"
_CONFIG_REL = "config.yaml"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="t-SNE / UMAP of encoder features on the held-out test "
                    "session, for every run_* under a sweep group directory.",
    )
    p.add_argument(
        "--path", type=str, required=True,
        help="Group directory containing run_0/, run_1/, ... "
             "(e.g. .../KFold_LinearProbing_interp_800/group_0).",
    )
    p.add_argument(
        "--mode", type=str, default="both", choices=["tsne", "umap", "both"],
        help="Which projection(s) to compute (default: both).",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to write the PNGs. Default: <parent-of-group>/DR_plots "
             "(sibling of the group directory).",
    )
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Inference device (default: auto — cuda if available else cpu).",
    )
    p.add_argument(
        "--max-points", type=int, default=2000,
        help="Cap on points fed to t-SNE/UMAP; above this a seeded random "
             "subsample is taken (default: 2000, 0=all).",
    )
    p.add_argument(
        "--num-workers", type=int, default=None,
        help="Override data.num_workers from the run config (default: keep).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Seed for subsampling + projections (default: each run's "
             "train.seed from its config; 42 if absent).",
    )
    p.add_argument("--umap-n-neighbors", type=int, default=30,
                   help="UMAP n_neighbors (default: 30).")
    p.add_argument("--umap-min-dist", type=float, default=0.5,
                   help="UMAP min_dist (default: 0.1).")
    p.add_argument("--point-size", type=float, default=None,
                   help="Scatter marker size (default: auto — shrinks as the "
                        "point count grows).")
    p.add_argument("--alpha", type=float, default=0.7,
                   help="Scatter marker alpha/opacity (default: 0.7).")
    p.add_argument(
        "--max-batches", type=int, default=None,
        help="Process at most this many test batches per run "
             "(debug/smoke test; default: all).",
    )
    return p.parse_args()


# ----------------------------------------------------------------------
# Discovery + small helpers
# ----------------------------------------------------------------------

def _discover_runs(group_dir: Path) -> list[Path]:
    """Return ``run_*`` subdirs (numeric order) with a ckpt + config."""
    runs = []
    for d in group_dir.glob("run_*"):
        if not d.is_dir() or not (d / _CONFIG_REL).exists():
            continue
        if not (d / _CKPT_REL).exists():
            log.warning("Skipping %s — no %s", d.name, _CKPT_REL)
            continue
        runs.append(d)

    def _idx(d: Path) -> tuple[int, str]:
        m = re.search(r"run_(\d+)", d.name)
        return (int(m.group(1)) if m else 10**9, d.name)

    return sorted(runs, key=_idx)


def _seed_for(cfg: dict, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    return int(cfg.get("train", {}).get("seed", cfg.get("data", {}).get("seed", 42)))


def _label_names(label_type: Optional[str], num_classes: int) -> Optional[list[str]]:
    if str(label_type) == "gesture" and int(num_classes) == 6:
        return list(_GESTURE_NAMES_6)
    return None


def _maybe_subsample(
    X: np.ndarray, y: np.ndarray, max_points: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points and X.shape[0] > max_points:
        rng = np.random.RandomState(seed)
        sel = rng.choice(X.shape[0], size=max_points, replace=False)
        return X[sel], y[sel]
    return X, y


# ----------------------------------------------------------------------
# Feature extraction (encoder forward)
# ----------------------------------------------------------------------

def _extract_features(
    model: Any, loader: Any, device: Any, max_batches: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled encoder embeddings ``(N, C·E)`` + int labels for ``loader``.

    Uses ``encoder_wrapper(batch).flatten(1)`` — the same pooled
    representation that feeds the head and that the built-in test t-SNE
    projects.
    """
    import torch

    feats_all, labels_all = [], []
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            feats = model.encoder_wrapper(batch)  # (B, C, E)
            feats = feats.flatten(1)              # (B, C*E)
            feats_all.append(feats.float().cpu())
            labels_all.append(batch["label"].detach().cpu())
    if not feats_all:
        return np.empty((0, 0), np.float32), np.empty((0,), np.int64)
    X = torch.cat(feats_all, 0).numpy().astype(np.float32)
    y = torch.cat(labels_all, 0).numpy().reshape(-1).astype(np.int64)
    return X, y


# ----------------------------------------------------------------------
# Projections + plot
# ----------------------------------------------------------------------

def _project_tsne(X: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE
    n = X.shape[0]
    perplexity = min(30.0, max(5.0, (n - 1) / 3.0))
    perplexity = min(perplexity, float(n - 1))
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=seed).fit_transform(X)


def _project_umap(X: np.ndarray, seed: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    import umap  # optional dependency
    if not np.isfinite(X).all():
        raise ValueError("UMAP input contains NaN/Inf — check the features.")
    nn = int(min(n_neighbors, max(2, X.shape[0] - 1)))
    return umap.UMAP(n_components=2, n_neighbors=nn, min_dist=min_dist,
                     metric="euclidean", random_state=seed).fit_transform(X)


def _umap_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("umap") is not None


def _plot(emb2d, y, num_classes, method, title, out_path, label_names,
          point_size=None, alpha=0.7):
    import matplotlib.pyplot as plt
    n = int(len(y))
    # Auto-shrink markers as the point count grows so dense plots stay legible.
    s = point_size if point_size is not None else max(3.0, 12.0 * (2000.0 / max(n, 1)) ** 0.5)
    cmap = plt.get_cmap("tab10" if num_classes <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(num_classes):
        m = y == c
        if not np.any(m):
            continue
        lbl = (f"{c}: {label_names[c]}" if label_names and c < len(label_names)
               else f"class {c}")
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=s, alpha=alpha,
                   color=cmap(c % cmap.N), label=lbl)
    ax.set_title(title)
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.legend(loc="best", fontsize=8, markerscale=1.5, framealpha=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()

    group_dir = Path(args.path).resolve()
    if not group_dir.is_dir():
        print(f"--path is not a directory: {group_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = (Path(args.output_dir).resolve() if args.output_dir
               else group_dir.parent / "DR_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = {"tsne": ["tsne"], "umap": ["umap"], "both": ["tsne", "umap"]}[args.mode]
    if "umap" in methods and not _umap_available():
        log.warning(
            "UMAP requested but `umap-learn` is not importable in this env. "
            "Skipping UMAP (t-SNE still runs). `pip install umap-learn` to enable.",
        )
        methods = [m for m in methods if m != "umap"]
    if not methods:
        print("Nothing to do (only UMAP requested but unavailable).", file=sys.stderr)
        sys.exit(1)

    runs = _discover_runs(group_dir)
    if not runs:
        print(f"No run_* with {_CKPT_REL} + {_CONFIG_REL} under {group_dir}",
              file=sys.stderr)
        sys.exit(1)

    # Heavy, env-specific imports — only once we know there is work to do.
    import torch
    import yaml
    from model import UltrasonicDownstream
    from runners.run_downstream import _build_datamodule

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    log.info("group=%s | runs=%d | mode=%s | device=%s | out=%s",
             group_dir.name, len(runs), methods, device, out_dir)

    saved: list[Path] = []
    for run_dir in runs:
        cfg = yaml.safe_load((run_dir / _CONFIG_REL).read_text(encoding="utf-8"))
        if args.num_workers is not None:
            cfg.setdefault("data", {})["num_workers"] = int(args.num_workers)
        cfg.setdefault("data", {})["persistent_workers"] = False
        seed = _seed_for(cfg, args.seed)
        test_id = cfg.get("data", {}).get("test_id")
        log.info("── %s ── test_id=%s seed=%d", run_dir.name, test_id, seed)

        dm = _build_datamodule(cfg)
        dm.setup("test")
        loader = dm.test_dataloader()
        if loader is None:
            log.warning("  %s: empty test split (test_id=%s) — skipping.",
                        run_dir.name, test_id)
            continue

        model = UltrasonicDownstream.load_from_checkpoint(
            str(run_dir / _CKPT_REL), map_location=device, pretrained_ckpt=None,
        )
        model.eval().to(device)

        X, y = _extract_features(model, loader, device, args.max_batches)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if X.shape[0] < 5:
            log.warning("  %s: only %d test embeddings — too few to project.",
                        run_dir.name, X.shape[0])
            continue

        n_total = X.shape[0]
        Xp, yp = _maybe_subsample(X, y, args.max_points, seed)
        num_classes = int(dm.num_classes or (int(y.max()) + 1))
        label_names = _label_names(dm.label_type, num_classes)
        log.info("  features %s (using %d of %d) classes=%d",
                 X.shape, Xp.shape[0], n_total, num_classes)

        for method in methods:
            try:
                emb2d = (_project_tsne(Xp, seed) if method == "tsne"
                         else _project_umap(Xp, seed, args.umap_n_neighbors,
                                            args.umap_min_dist))
            except Exception as exc:  # pragma: no cover - robustness
                log.error("  %s/%s projection failed: %r", run_dir.name, method, exc)
                continue
            out_path = out_dir / f"{group_dir.name}_{run_dir.name}_test{test_id}_{method}.png"
            title = (f"{group_dir.name}/{run_dir.name} — {method.upper()} "
                     f"(test session {test_id}, n={Xp.shape[0]})")
            _plot(emb2d, yp, num_classes, method, title, out_path, label_names,
                  point_size=args.point_size, alpha=args.alpha)
            saved.append(out_path)
            log.info("  saved %s", out_path.name)

    log.info("Done — %d plot(s) written under %s", len(saved), out_dir)


if __name__ == "__main__":
    main()
