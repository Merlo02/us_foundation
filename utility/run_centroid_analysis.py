#!/usr/bin/env python3
"""Centroid-migration analysis — does each class cross a decision boundary
between the *train* sessions and the held-out *test* session?

Motivation
----------
On the downstream task the per-channel pooled embeddings ``(C·E)`` are
**concatenated** and fed to a *linear* head: ``logits = W · x + b``,
``pred = argmax_k(logits)``. A linear head partitions feature space
(``D = C·E ≈ thousands`` of dims) into ``K`` convex regions separated by
hyperplanes. The classic inter-session failure (val ≈ 100 %, test ≈ 50 %,
yet the test t-SNE looks clean) is **domain shift**: the encoder still
produces class-separable test features, but the *positions* of the class
clouds move across sessions, so the head's hyperplanes — fit on the train
sessions — land in the wrong place on the test session and whole classes
get dumped into a neighbour's region.

t-SNE cannot show this (non-linear, no inverse, per-embedding arbitrary).
The **centroid migration** test measures it exactly and cheaply.

The idea, precisely
-------------------
For class ``k`` the *centroid* is the mean feature vector over that class's
samples — a single point ``μ_k ∈ R^D`` (the "centre of mass" of the cloud).
The reference centroid is taken on the **validation** split (the data the
checkpoint was selected on, not directly fit), the moving one on the test:

    μ_k^val  = mean( X_val [y_val  == k] )      # reference (validation split)
    μ_k^test = mean( X_test[y_test == k] )      # held-out test session

The heart of the diagnostic is **not** the Euclidean distance between the two
centroids (in ~thousands of dims that is dominated by classification-
irrelevant directions and barely interpretable). It is **which decision
region each centroid falls in**, obtained by simply running the head:

    p_k^val  = argmax( head(μ_k^val ) )          # ~ k (val is in-distribution)
    p_k^test = argmax( head(μ_k^test) )

* ``p_k^val == k`` almost always — the head classifies validation
  (in-distribution) data correctly.
* If ``p_k^test = k' ≠ k`` the **whole** test cloud of class k has crossed a
  boundary into region k': the head will predict k' for (most of) it. That
  is exactly the "predicted-column k' full / column k empty" pattern you see
  in the confusion matrix.

Note: with ``grouped_val=false`` the val split is a random slice of the TRAIN
sessions, so ``μ_k^val`` ≈ the train centroid (same distribution); the val-vs-
test comparison is most informative with ``grouped_val=true`` (val = a separate
held-out session).

Why this is sound in ~thousands of dimensions: the logits depend *only* on
the projection of ``x`` onto ``rowspace(W)`` (dim ≤ K), because ``W·x_⊥ = 0``
for the orthogonal part. ``argmax(head(μ))`` implicitly reduces the high-dim
centroid to the only ≤K coordinates the classifier sees and reports the
region — no projection, no randomness, no approximation.

Reported per class (alongside ``p^val`` / ``p^test`` / ``shifted``):

* ``margin@val`` / ``margin@test`` — logit of the correct class minus the
  best competitor, evaluated at the centroid. Positive ⇒ on the correct side;
  it going **negative** at the test centroid is the continuous "how badly it
  crossed" signal.
* ``logit_shift`` — ``‖head(μ^test) − head(μ^val)‖``, the *decision-relevant*
  part of the drift (for a linear head this is ``‖W·(μ^test − μ^val)‖``).
* ``cloud→p^test`` — fraction of the test cloud whose own prediction matches
  the centroid's prediction. High ⇒ the centroid is representative; ~50/50 ⇒
  the class is splitting across two regions and the centroid alone would
  mislead (cross-check with the confusion matrix in that case).
* ``acc@test`` — per-class test accuracy, for context.

Usage
-----
Same entry points / layout as :mod:`utility.run_dr_analysis` — point ``--path``
at a sweep GROUP directory and it loops over every ``run_*`` (each
``run_i/config.yaml`` + ``run_i/checkpoints/last.ckpt``)::

    cd ~/us_foundation
    ~/useless_venv/bin/python -m utility.run_centroid_analysis \\
        --path /.../KFold_LinearProbing_interp_800/group_0

For each run it: (1) reads the post-sync ``config.yaml``; (2) loads
``last.ckpt`` into :class:`~model.UltrasonicDownstream`; (3) builds the
DataModule so signals get the exact same interpolation + per-train-channel
normalization used at eval; (4) extracts pooled embeddings on the
**validation split** (``--val-max-batches``) and on the **whole held-out
test session**; (5) computes per-class centroids, runs the head on
them, and reports the migration table; (6) writes a combined CSV and a
per-run margin bar plot under ``<parent-of-group>/centroid_analysis`` (or
``--output-dir``).

Modes (``--mode``)
------------------
* ``centroid`` (default) — the per-class centroid-migration table + margin
  plots described above.
* ``lda`` — the **linear analog** of the t-SNE view: a supervised LDA on the
  **joint** val+test features with **12 labels** (val-c and test-c as distinct
  groups), projected to its top-2 discriminant axes. Same data + encoding as
  ``tsne`` (colour=true class, val=dots, test=×, per-class val→test drift
  arrows), so the linear and non-linear views are directly comparable.
* ``tsne`` — joint t-SNE of val+test in one shared embedding (same encoding as
  ``lda``). Both show the per-class **distribution shift** between val and
  test; neither shows the head's decision boundaries (use the centroid/logit
  numbers for the head's verdict).
* ``all`` — centroid + lda + tsne.

Classification runs only (a regression head has no decision regions).
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting

import numpy as np

# Make the project root importable regardless of how this is launched.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Reuse discovery / feature-extraction helpers from the DR analysis tool.
from utility.run_dr_analysis import (  # noqa: E402
    _CKPT_REL,
    _CONFIG_REL,
    _discover_runs,
    _extract_features,
    _label_names,
    _maybe_subsample,
    _project_tsne,
    _seed_for,
)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-class centroid-migration analysis (val vs held-out "
                    "test session) for every run_* under a sweep group dir.",
    )
    p.add_argument(
        "--path", type=str, required=True,
        help="Group directory containing run_0/, run_1/, ... "
             "(e.g. .../KFold_LinearProbing_interp_800/group_0).",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to write the CSV + PNGs. Default: "
             "<parent-of-group>/centroid_analysis (sibling of the group dir).",
    )
    p.add_argument(
        "--mode", type=str.lower, default="centroid",
        choices=["centroid", "lda", "tsne", "all"],
        help="centroid (default): per-class centroid-migration table + margin "
             "plots. lda: joint val+test LDA (12-label) 2-D scatter. tsne: joint "
             "val+test t-SNE 2-D scatter. lda and tsne share the same data + "
             "encoding (aligned: linear vs non-linear). all: centroid+lda+tsne.",
    )
    p.add_argument(
        "--max-points", type=int, default=1500,
        help="Per-split cap on points fed to the joint LDA / t-SNE views; above "
             "this a seeded random subsample is taken (0 = all; default: 1500). "
             "The same subsample is shared by lda and tsne so they're aligned.",
    )
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Inference device (default: auto — cuda if available else cpu).",
    )
    p.add_argument(
        "--num-workers", type=int, default=None,
        help="Override data.num_workers from the run config (default: keep).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Seed passed to torch (default: each run's train.seed; 42 if absent).",
    )
    p.add_argument(
        "--val-max-batches", type=int, default=0,
        help="Cap on VALIDATION batches used to estimate the reference "
             "centroids (the val loader is NOT shuffled). 0 = use the whole "
             "val split (default: 0).",
    )
    p.add_argument(
        "--test-max-batches", type=int, default=0,
        help="Cap on TEST batches (0 = whole held-out session; default: 0).",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Only write the CSV / log the tables, skip the per-run margin plot.",
    )
    return p.parse_args()


# ----------------------------------------------------------------------
# Head logits + small numeric helpers
# ----------------------------------------------------------------------

def _head_logits(model: Any, X: np.ndarray, device: Any, chunk: int = 8192) -> np.ndarray:
    """Run the classification head on features ``X (N, D)`` → logits ``(N, K)``.

    Works for any head (linear or MLP); for the linear head this is exactly
    ``X @ W.T + b``. Chunked + on-device for memory safety.
    """
    import torch

    if X.shape[0] == 0:
        return np.zeros((0, 0), np.float32)
    dtype = next(model.head.parameters()).dtype
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            xb = torch.from_numpy(X[i:i + chunk]).to(device=device, dtype=dtype)
            out.append(model.head(xb).float().cpu().numpy())
    return np.concatenate(out, 0)


def _to_kcol(logits: np.ndarray, num_classes: int) -> np.ndarray:
    """Normalise head output to ``K`` columns.

    A binary head emits a single logit (decision ``logit > 0``); expand it to
    two columns ``[-logit, +logit]`` so argmax / margin logic is uniform.
    """
    if logits.ndim == 2 and logits.shape[1] == 1 and num_classes == 2:
        return np.concatenate([-logits, logits], axis=1)
    return logits


def _margin(logit_row: np.ndarray, k: int) -> float:
    """Logit of class ``k`` minus the best competitor (signed margin)."""
    others = np.delete(logit_row, k)
    best_other = float(others.max()) if others.size else float("-inf")
    return float(logit_row[k] - best_other)


def _cname(label_names: Optional[list[str]], c: int) -> str:
    if label_names is not None and 0 <= c < len(label_names):
        return label_names[c]
    return f"class{c}"


# ----------------------------------------------------------------------
# Per-run analysis
# ----------------------------------------------------------------------

def _analyse_run(
    model: Any,
    Xva: np.ndarray, yva: np.ndarray,
    Xte: np.ndarray, yte: np.ndarray,
    num_classes: int,
    device: Any,
    label_names: Optional[list[str]],
) -> tuple[list[dict], float]:
    """Return ``(per_class_rows, overall_test_accuracy)``.

    ``per_class_rows`` has one dict per class with the centroid-migration
    fields documented in the module docstring. The reference centroid is the
    validation split; the moving one is the held-out test session.
    """
    # Head predictions on the FULL clouds (used for cloud-agreement + acc).
    Lte_all = _to_kcol(_head_logits(model, Xte, device), num_classes)
    pte_all = Lte_all.argmax(1).astype(np.int64)
    overall_acc = float(np.mean(pte_all == yte)) if yte.size else float("nan")

    rows: list[dict] = []
    for k in range(num_classes):
        vam = yva == k
        tem = yte == k
        n_va, n_te = int(vam.sum()), int(tem.sum())
        row: dict[str, Any] = {
            "class": k, "class_name": _cname(label_names, k),
            "n_val": n_va, "n_test": n_te,
            "pred_val_centroid": None, "pred_test_centroid": None,
            "shifted": None, "margin_val": None, "margin_test": None,
            "logit_shift": None, "cloud_frac_to_pte": None, "acc_test": None,
        }
        if n_va == 0:
            rows.append(row)  # cannot anchor the val (reference) centroid
            continue

        mu_va = Xva[vam].mean(0)
        Lc_va = _to_kcol(_head_logits(model, mu_va[None], device), num_classes)[0]
        row["pred_val_centroid"] = int(Lc_va.argmax())
        row["margin_val"] = round(_margin(Lc_va, k), 4)

        if n_te == 0:
            rows.append(row)  # class absent in the test session
            continue

        mu_te = Xte[tem].mean(0)
        Lc_te = _to_kcol(_head_logits(model, mu_te[None], device), num_classes)[0]
        cpte = int(Lc_te.argmax())
        row["pred_test_centroid"] = cpte
        row["shifted"] = bool(cpte != k)
        row["margin_test"] = round(_margin(Lc_te, k), 4)
        row["logit_shift"] = round(float(np.linalg.norm(Lc_te - Lc_va)), 4)
        row["cloud_frac_to_pte"] = round(float(np.mean(pte_all[tem] == cpte)), 4)
        row["acc_test"] = round(float(np.mean(pte_all[tem] == k)), 4)
        rows.append(row)

    return rows, overall_acc


def _log_table(run_name: str, test_id: Any, rows: list[dict], overall_acc: float,
               label_names: Optional[list[str]]) -> None:
    n_shift = sum(1 for r in rows if r["shifted"] is True)
    n_eval = sum(1 for r in rows if r["shifted"] is not None)
    log.info(
        "── %s (test_id=%s) — overall test acc=%.3f — shifted %d/%d classes",
        run_name, test_id, overall_acc, n_shift, n_eval,
    )
    hdr = (f"   {'class':<14} {'n_va':>6} {'n_te':>6} {'val→':>7} "
           f"{'test→':>7} {'shift':>6} {'mg_va':>7} {'mg_te':>7} "
           f"{'lg_shift':>9} {'cloud→':>7} {'acc':>6}")
    log.info(hdr)
    for r in rows:
        def f(x, fmt="{:.2f}"):
            return "—" if x is None else (fmt.format(x) if isinstance(x, float) else str(x))
        shift = ("—" if r["shifted"] is None
                 else ("CROSS" if r["shifted"] else "ok"))
        ctest = ("—" if r["pred_test_centroid"] is None
                 else _cname(label_names, r["pred_test_centroid"]))
        cval = ("—" if r["pred_val_centroid"] is None
                else _cname(label_names, r["pred_val_centroid"]))
        log.info(
            "   %-14s %6d %6d %7s %7s %6s %7s %7s %9s %7s %6s",
            r["class_name"], r["n_val"], r["n_test"], cval, ctest, shift,
            f(r["margin_val"]), f(r["margin_test"]), f(r["logit_shift"]),
            f(r["cloud_frac_to_pte"]), f(r["acc_test"]),
        )


def _plot_margins(run_name: str, test_id: Any, rows: list[dict],
                  overall_acc: float, label_names: Optional[list[str]],
                  out_path: Path) -> None:
    """Grouped bar chart: per-class signed margin at the val vs test centroid.

    A class whose *test* margin goes negative (red) has crossed a boundary —
    its whole cloud is on the wrong side of the head's hyperplane.
    """
    import matplotlib.pyplot as plt

    ev = [r for r in rows if r["margin_test"] is not None]
    if not ev:
        return
    labels = [r["class_name"] for r in ev]
    mva = [float(r["margin_val"]) for r in ev]
    mte = [float(r["margin_test"]) for r in ev]
    x = np.arange(len(ev))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(ev) + 2), 5))
    ax.bar(x - w / 2, mva, w, label="margin @ val centroid", color="0.6")
    te_colors = ["tab:green" if m >= 0 else "tab:red" for m in mte]
    ax.bar(x + w / 2, mte, w, label="margin @ test centroid", color=te_colors)
    ax.axhline(0.0, color="k", lw=1.0)
    for xi, r in zip(x, ev):
        if r["shifted"]:
            ax.annotate(f"→ {_cname(label_names, r['pred_test_centroid'])}",
                        (xi + w / 2, mte[ev.index(r)]),
                        ha="center", va="top" if mte[ev.index(r)] < 0 else "bottom",
                        fontsize=8, color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("signed logit margin (correct class − best competitor)")
    n_shift = sum(1 for r in ev if r["shifted"])
    ax.set_title(f"{run_name} — centroid margins (test_id={test_id}, "
                 f"acc={overall_acc:.3f}, {n_shift}/{len(ev)} classes crossed)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Joint val+test 2-D views (LDA / t-SNE) — shared scatter, aligned encoding
# ----------------------------------------------------------------------

def _scatter_joint_2d(
    Zv: np.ndarray, yv: np.ndarray, Zt: np.ndarray, yt: np.ndarray,
    num_classes: int, label_names: Optional[list[str]],
    title: str, xlabel: str, ylabel: str, out_path: Path,
    draw_arrows: bool = True,
) -> bool:
    """Scatter a joint val+test 2-D embedding: colour=true class, val=dots,
    test=×, per-class val→test centroid arrow. Shared by the LDA and t-SNE
    views so the two are visually aligned (same encoding, same points)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cmap = plt.get_cmap("tab10" if num_classes <= 10 else "tab20")
    colors = [cmap(c % cmap.N) for c in range(num_classes)]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    for c in range(num_classes):
        mv, mt = yv == c, yt == c
        if np.any(mv):
            ax.scatter(Zv[mv, 0], Zv[mv, 1], s=8, alpha=0.45,
                       color=colors[c], linewidths=0)
        if np.any(mt):
            ax.scatter(Zt[mt, 0], Zt[mt, 1], s=30, alpha=0.85,
                       color=colors[c], marker="x", linewidths=1.1)
        if draw_arrows and np.any(mv) and np.any(mt):
            v, t = Zv[mv].mean(0), Zt[mt].mean(0)
            ax.annotate("", xy=(t[0], t[1]), xytext=(v[0], v[1]),
                        arrowprops=dict(arrowstyle="->", color=colors[c],
                                        lw=1.6, alpha=0.9))
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[c],
                      markersize=7, label=f"{c}: {_cname(label_names, c)}")
               for c in range(num_classes)]
    handles += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4",
               markersize=7, label="val (dots)"),
        Line2D([0], [0], marker="x", color="0.2", markersize=8,
               linestyle="None", label="test (×)"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=7, framealpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _embed_lda12(
    Xv: np.ndarray, yv: np.ndarray, Xt: np.ndarray, yt: np.ndarray,
    num_classes: int,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Supervised LDA on the joint val+test set with **12 labels**.

    Val gets labels ``0..K-1`` and test ``K..2K-1`` (val-c and test-c are
    distinct groups), so Fisher finds the linear directions that best separate
    all 2K class×split groups — the linear analog of the joint t-SNE. Returns
    the top-2 projection split back into ``(Zv, Zt)``, or ``None`` if it can't
    produce 2 axes.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    X12 = np.concatenate([Xv, Xt], 0)
    y12 = np.concatenate([yv, yt + num_classes], 0)
    if np.unique(y12).size < 3:
        return None
    Z = LinearDiscriminantAnalysis(n_components=2).fit(X12, y12).transform(X12)
    if Z.shape[1] < 2:
        return None
    n_v = Xv.shape[0]
    return Z[:n_v, :2], Z[n_v:, :2]


def _plot_lda_joint(
    run_name: str, test_id: Any,
    Xv: np.ndarray, yv: np.ndarray, Xt: np.ndarray, yt: np.ndarray,
    num_classes: int, label_names: Optional[list[str]], out_path: Path,
) -> bool:
    """Linear (LDA) analog of the joint t-SNE: 12-label LDA of the joint
    val+test set, projected to 2-D, scattered with the shared encoding. Inputs
    are already subsampled (same points as the t-SNE view → aligned)."""
    emb = _embed_lda12(Xv, yv, Xt, yt, num_classes)
    if emb is None:
        log.warning("  %s: LDA needs >=3 class×split groups — skipping.", run_name)
        return False
    Zv, Zt = emb
    return _scatter_joint_2d(
        Zv, yv, Zt, yt, num_classes, label_names,
        title=(f"{run_name} — joint val+test LDA (12-label, test_id={test_id}); "
               f"dots=val, ×=test, arrows=val→test drift, colour=true class"),
        xlabel="LDA-1", ylabel="LDA-2", out_path=out_path,
    )


def _plot_tsne_joint(
    run_name: str, test_id: Any,
    Xv: np.ndarray, yv: np.ndarray, Xt: np.ndarray, yt: np.ndarray,
    num_classes: int, label_names: Optional[list[str]], out_path: Path,
    seed: int = 42,
) -> bool:
    """Joint t-SNE of val + test in ONE shared embedding (non-linear analog of
    :func:`_plot_lda_joint`). Embeds the concatenated features together (no
    asymmetric "fit on A, apply to B" projection) and scatters with the shared
    encoding. Inputs are already subsampled (same points as the LDA view).

    Shows the per-class **distribution shift**; it CANNOT show the head's
    decision boundaries (t-SNE is non-linear / has no inverse), so read it as a
    qualitative drift map, complementary to the centroid/logit analysis."""
    n_v = Xv.shape[0]
    if n_v + Xt.shape[0] < 10:
        log.warning("  %s: too few points for joint t-SNE — skipping.", run_name)
        return False
    emb = _project_tsne(np.concatenate([Xv, Xt], 0), seed)
    return _scatter_joint_2d(
        emb[:n_v], yv, emb[n_v:], yt, num_classes, label_names,
        title=(f"{run_name} — joint val+test t-SNE (test_id={test_id}); "
               f"dots=val, ×=test, arrows=val→test drift, colour=true class"),
        xlabel="t-SNE 1", ylabel="t-SNE 2", out_path=out_path,
    )


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
               else group_dir.parent / "centroid_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(group_dir)
    if not runs:
        print(f"No run_* with {_CKPT_REL} + {_CONFIG_REL} under {group_dir}",
              file=sys.stderr)
        sys.exit(1)

    # Heavy imports only once there is work to do.
    import torch
    import yaml
    from model import UltrasonicDownstream
    from runners.run_downstream import _build_datamodule

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    val_cap = args.val_max_batches or None  # 0 → all
    test_cap = args.test_max_batches or None
    log.info("group=%s | runs=%d | mode=%s | device=%s | val_batches=%s | out=%s",
             group_dir.name, len(runs), args.mode, device,
             val_cap if val_cap is not None else "all", out_dir)

    csv_path = out_dir / f"{group_dir.name}_centroid_shift.csv"
    csv_cols = ["run", "test_id", "class", "class_name", "n_val", "n_test",
                "pred_val_centroid", "pred_test_centroid", "shifted",
                "margin_val", "margin_test", "logit_shift",
                "cloud_frac_to_pte", "acc_test", "overall_test_acc"]
    csv_rows: list[dict] = []
    n_plots = 0
    analysed = 0

    for run_dir in runs:
        cfg = yaml.safe_load((run_dir / _CONFIG_REL).read_text(encoding="utf-8"))
        if str(cfg.get("model", {}).get("head_type", "classification")) != "classification":
            log.info("  %s: head_type != classification — skipping (no decision regions).",
                     run_dir.name)
            continue
        data_cfg = cfg.setdefault("data", {})
        if args.num_workers is not None:
            data_cfg["num_workers"] = int(args.num_workers)
        data_cfg["persistent_workers"] = False
        seed = _seed_for(cfg, args.seed)
        test_id = cfg.get("data", {}).get("test_id")
        torch.manual_seed(seed)

        dm = _build_datamodule(cfg)
        dm.setup("fit")
        num_classes = int(dm.num_classes or 0)
        if num_classes < 2:
            log.warning("  %s: num_classes=%s — skipping.", run_dir.name, num_classes)
            continue
        label_names = _label_names(dm.label_type, num_classes)

        test_loader = dm.test_dataloader()
        if test_loader is None:
            log.warning("  %s: empty test split (test_id=%s) — skipping.",
                        run_dir.name, test_id)
            continue
        # val_dataloader() returns ``[val, test]`` when test_every_epoch is set;
        # the reference centroid only needs the val loader (index 0).
        val_loader = dm.val_dataloader()
        if isinstance(val_loader, (list, tuple)):
            val_loader = val_loader[0] if val_loader else None
        if val_loader is None or getattr(dm, "val_ds", None) is None or len(dm.val_ds) == 0:
            log.warning("  %s: empty val split — skipping (val is the reference "
                        "set; set data.val_ratio>0 / grouped_val to populate it).",
                        run_dir.name)
            continue

        model = UltrasonicDownstream.load_from_checkpoint(
            str(run_dir / _CKPT_REL), map_location=device, pretrained_ckpt=None,
        )
        model.eval().to(device)

        Xva, yva = _extract_features(model, val_loader, device, val_cap)
        Xte, yte = _extract_features(model, test_loader, device, test_cap)
        if Xva.shape[0] == 0 or Xte.shape[0] == 0:
            log.warning("  %s: empty val (%d) or test (%d) features — skipping.",
                        run_dir.name, Xva.shape[0], Xte.shape[0])
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        log.info("  %s: val feats %s, test feats %s",
                 run_dir.name, Xva.shape, Xte.shape)
        analysed += 1

        if args.mode in ("centroid", "all"):
            rows, overall_acc = _analyse_run(
                model, Xva, yva, Xte, yte, num_classes, device, label_names,
            )
            _log_table(run_dir.name, test_id, rows, overall_acc, label_names)
            for r in rows:
                csv_rows.append({"run": run_dir.name, "test_id": test_id,
                                 "overall_test_acc": round(overall_acc, 4), **r})
            if not args.no_plot:
                out_png = out_dir / f"{group_dir.name}_{run_dir.name}_test{test_id}_centroids.png"
                try:
                    _plot_margins(run_dir.name, test_id, rows, overall_acc,
                                  label_names, out_png)
                    n_plots += 1
                    log.info("  saved %s", out_png.name)
                except Exception as exc:  # pragma: no cover - robustness
                    log.error("  %s: margin plot failed: %r", run_dir.name, exc)

        # LDA and t-SNE share one subsample so the two views show the SAME
        # points (aligned): subsample val/test once here.
        if args.mode in ("lda", "tsne", "all") and not args.no_plot:
            cap = args.max_points or None  # 0 → all
            Xv, yv = _maybe_subsample(Xva, yva, cap, seed)
            Xt, yt = _maybe_subsample(Xte, yte, cap, seed + 1)

            if args.mode in ("lda", "all"):
                out_png = out_dir / f"{group_dir.name}_{run_dir.name}_test{test_id}_lda.png"
                try:
                    if _plot_lda_joint(run_dir.name, test_id, Xv, yv, Xt, yt,
                                       num_classes, label_names, out_png):
                        n_plots += 1
                        log.info("  saved %s", out_png.name)
                except Exception as exc:  # pragma: no cover - robustness
                    log.error("  %s: LDA plot failed: %r", run_dir.name, exc)

            if args.mode in ("tsne", "all"):
                out_png = out_dir / f"{group_dir.name}_{run_dir.name}_test{test_id}_tsne.png"
                try:
                    if _plot_tsne_joint(run_dir.name, test_id, Xv, yv, Xt, yt,
                                        num_classes, label_names, out_png, seed=seed):
                        n_plots += 1
                        log.info("  saved %s", out_png.name)
                except Exception as exc:  # pragma: no cover - robustness
                    log.error("  %s: t-SNE plot failed: %r", run_dir.name, exc)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_cols)
            w.writeheader()
            w.writerows(csv_rows)
        log.info("Wrote %s (%d rows)", csv_path, len(csv_rows))
    elif args.mode in ("centroid", "all"):
        log.warning("No classification runs analysed — nothing written.")

    log.info("Done — %d run(s) analysed, %d plot(s) under %s",
             analysed, n_plots, out_dir)


if __name__ == "__main__":
    main()
