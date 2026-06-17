"""Periodic t-SNE diagnostics of the encoder embeddings during pretraining.

Shared by :class:`~model.us_mae.UltrasonicMAE` and
:class:`~model.us_jepa.UltrasonicJEPA` — any LightningModule exposing
``self.tokenizer`` (:class:`MultiTokenizer`) and ``self.encoder``
(:class:`USEncoder`, called here with ``bypass_masking=True``).

The MAE has reconstruction plots (``runners/run_test.py``) as an intermediate
quality signal; LeJEPA has no decoder, so the only way to *see* the encoder
improving is to look at the embedding space. Two independent projections:

- ``pretrain`` — embeds the first **validation** batch of this run
  (mean-pooled encoder output), coloured by ``dataset_source``. Needs no
  extra data; shows whether the encoder is dataset-invariant.
- ``labeled``  — embeds an external downstream classification ``all.h5``
  (``(N, C, T)`` frames), restricted to a single session. The scheme is
  identical to the downstream t-SNE
  (:meth:`model.downstream.us_downstream.UltrasonicDownstream._forward_features`):
  the backbone is called once per channel (the channel axis is folded into
  the batch axis, same vectorised pattern as
  :class:`~model.downstream.encoder_wrapper.UltrasonicEncoderWrapper`), each
  channel is mean-pooled over tokens to ``(B, C, E)`` and the channels are
  then **concatenated** (``flatten(1)``) into one ``(C·E,)`` vector per
  frame; the t-SNE runs on these concatenated vectors, coloured by class
  label. Shows how well the *as-is* backbone clusters downstream classes
  while pretraining progresses.

Configured from YAML via ``train.debug_embeddings`` (the runner passes the
section as a plain dict and injects the pretraining input regime under the
underscore-prefixed keys so the labeled frames go through the exact pipeline
the encoder is trained on)::

    train:
      debug_embeddings:
        tsne_embeddings: true        # MASTER switch for ANY t-SNE
        tsne_mode: [pretrain, labeled]
        every_n_epochs: 10           # cadence of the "pretrain" mode
        tsne_max_steps: 1000         # sklearn optimisation budget
        tsne_labeled_h5: /path/to/downstream/all.h5
        tsne_labeled_session_id: 5
        tsne_labeled_every_n_epochs: 40
        tsne_labeled_max_samples: 1000

Everything runs on rank 0 only, outside the loss graph, and is wrapped in
try/except — a failing diagnostic must never kill a multi-day run. Outputs:
PNG + ``.npz`` under ``{run_dir}/tsne/`` plus a wandb image when a
WandbLogger is attached (offline-safe).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

_VALID_MODES = ("pretrain", "labeled")

# Upper bound on the points fed to sklearn's TSNE (same rationale as
# ``model/downstream/us_downstream.py::_TSNE_MAX_POINTS``).
_TSNE_MAX_POINTS = 2000


# ----------------------------------------------------------------------
# Encoder helper
# ----------------------------------------------------------------------

@torch.no_grad()
def encode_batch_embeddings(
    module: Any,
    signal: torch.Tensor,                          # (N, T)
    signal_mask: torch.Tensor,                     # (N, T) bool
    sampling_frequency_hz: torch.Tensor,           # (N,)
    window_size: Optional[torch.Tensor],           # (N,) or None
    patch_timestamps_us: Optional[torch.Tensor],   # (N, S) or None
    fixed_num_patches: Optional[int],
) -> torch.Tensor:
    """Tokenizer → encoder (``bypass_masking=True``) → masked mean → ``(N, E)``."""
    tok = module.tokenizer(
        signal=signal,
        signal_mask=signal_mask,
        sampling_frequency_hz=sampling_frequency_hz,
        window_size_override=window_size,
        fixed_num_patches=fixed_num_patches,
        patch_timestamps_us=patch_timestamps_us,
    )
    enc = module.encoder(
        tokens=tok.tokens,
        padding_mask=tok.padding_mask,
        time_values_us=tok.patch_timestamps_us,
        bypass_masking=True,
    )
    latent = enc["latent"]                   # (N, S, E)
    valid = enc["padding_mask"]              # (N, S)
    m = valid.unsqueeze(-1).to(latent.dtype)
    denom = m.sum(dim=1).clamp(min=1.0)
    return (latent * m).sum(dim=1) / denom   # (N, E)


# ----------------------------------------------------------------------
# t-SNE + plotting helpers
# (recipe copied from us_downstream._compute_and_plot_tsne / _plot_tsne)
# ----------------------------------------------------------------------

def _tsne_project(
    emb_np: np.ndarray,
    labels_np: np.ndarray,
    seed: int,
    max_steps: int,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Project to 2-D with sklearn TSNE; returns ``(points_2d, labels)``.

    Subsamples above ``_TSNE_MAX_POINTS`` and scales the perplexity for
    small N. Returns ``None`` when sklearn is missing or N is too small.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:  # pragma: no cover
        log.warning("debug_embeddings: scikit-learn unavailable — skipping t-SNE.")
        return None

    n = emb_np.shape[0]
    if n < 5:
        return None  # t-SNE is meaningless with a handful of points
    if n > _TSNE_MAX_POINTS:
        rng = np.random.RandomState(int(seed))
        sel = rng.choice(n, size=_TSNE_MAX_POINTS, replace=False)
        emb_np = emb_np[sel]
        labels_np = labels_np[sel]
        n = _TSNE_MAX_POINTS

    # sklearn requires perplexity < n_samples; scale it down for small n.
    perplexity = min(30.0, max(5.0, (n - 1) / 3.0))
    perplexity = min(perplexity, float(n - 1))

    # The kwarg was renamed n_iter -> max_iter in sklearn 1.5; pick whichever
    # the installed version exposes (same trick as the downstream module).
    import inspect

    iter_kw = (
        "max_iter"
        if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    )
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=int(seed),
        **{iter_kw: max(250, int(max_steps))},
    )
    return tsne.fit_transform(emb_np.astype(np.float32)), labels_np


def _scatter_by_label(
    points: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    title: str,
) -> Optional[Any]:
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        log.warning("debug_embeddings: matplotlib unavailable — skipping plot.")
        return None

    n_labels = len(label_names)
    cmap = plt.get_cmap("tab10" if n_labels <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(7, 6))
    for li, name in enumerate(label_names):
        m = labels == li
        if not np.any(m):
            continue
        ax.scatter(
            points[m, 0], points[m, 1],
            s=10, alpha=0.7, color=cmap(li % cmap.N), label=str(name),
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=8, markerscale=1.5, framealpha=0.8)
    fig.tight_layout()
    return fig


def _wandb_experiment(module: Any) -> Optional[Any]:
    """Return the wandb run of an attached WandbLogger, or ``None``."""
    trainer = getattr(module, "trainer", None)
    for lg in (getattr(trainer, "loggers", None) or []):
        if type(lg).__name__ == "WandbLogger":
            return lg.experiment
    return None


# ----------------------------------------------------------------------
# Debugger
# ----------------------------------------------------------------------

class EmbeddingTSNEDebugger:
    """Rank-0 periodic t-SNE of the pretraining encoder embeddings."""

    def __init__(self, cfg: Optional[dict]) -> None:
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("tsne_embeddings", False))
        modes = cfg.get("tsne_mode") or []
        if isinstance(modes, str):
            modes = [modes]
        self.modes = tuple(str(m) for m in modes)
        unknown = [m for m in self.modes if m not in _VALID_MODES]
        if unknown:
            raise ValueError(
                f"train.debug_embeddings.tsne_mode entries must be in "
                f"{_VALID_MODES}, got {unknown}.",
            )
        self.every_n_epochs = int(cfg.get("every_n_epochs", 10))
        self.tsne_max_steps = int(cfg.get("tsne_max_steps", 1000))

        self.labeled_h5 = cfg.get("tsne_labeled_h5") or None
        _sid = cfg.get("tsne_labeled_session_id")
        self.labeled_session_id = int(_sid) if _sid is not None else None
        self.labeled_every_n_epochs = int(cfg.get("tsne_labeled_every_n_epochs", 40))
        self.labeled_max_samples = int(cfg.get("tsne_labeled_max_samples", 1000))

        # Pretraining input regime, injected by runners.run_train so the
        # labeled frames see the exact pipeline the encoder is trained on.
        self.normalization_type = str(cfg.get("_normalization_type", "none"))
        self.apply_interpolate = bool(cfg.get("_apply_interpolate", False))
        _tl = cfg.get("_target_length")
        self.target_length = int(_tl) if _tl is not None else None

        if self.enabled and "labeled" in self.modes and not self.labeled_h5:
            log.warning(
                "debug_embeddings: 'labeled' listed in tsne_mode but "
                "tsne_labeled_h5 is null — the labeled t-SNE is skipped.",
            )
            self.modes = tuple(m for m in self.modes if m != "labeled")

        # Lazily built labeled corpus: None = not built yet, False = unusable.
        self._labeled: Any = None

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------
    @staticmethod
    def _rank0_epoch(module: Any) -> Optional[int]:
        """Current epoch when on rank 0 outside sanity-check, else ``None``."""
        trainer = getattr(module, "trainer", None)
        if trainer is None or getattr(trainer, "sanity_checking", False):
            return None
        if not bool(getattr(trainer, "is_global_zero", True)):
            return None
        return int(trainer.current_epoch)

    # ------------------------------------------------------------------
    # "pretrain" mode — first validation batch of the epoch
    # ------------------------------------------------------------------
    def maybe_run_pretrain(self, module: Any, batch: dict, batch_idx: int) -> None:
        if not self.enabled or "pretrain" not in self.modes or batch_idx != 0:
            return
        epoch = self._rank0_epoch(module)
        if epoch is None or self.every_n_epochs <= 0 or epoch % self.every_n_epochs != 0:
            return
        try:
            self._run_pretrain(module, batch, epoch)
        except Exception as exc:  # noqa: BLE001 — diagnostics must not kill the run
            log.warning("debug_embeddings: pretrain t-SNE failed: %s", exc)

    def _run_pretrain(self, module: Any, batch: dict, epoch: int) -> None:
        emb = encode_batch_embeddings(
            module,
            signal=batch["signal"],
            signal_mask=batch["signal_mask"],
            sampling_frequency_hz=batch["sampling_frequency_hz"],
            window_size=batch.get("window_size"),
            patch_timestamps_us=batch.get("patch_timestamps_us"),
            fixed_num_patches=module.hparams.target_patches,
        ).float().cpu().numpy()

        bases = [str(s).split("::", 1)[0] for s in batch["dataset_source"]]
        names = sorted(set(bases))
        name_to_idx = {n: i for i, n in enumerate(names)}
        labels = np.asarray([name_to_idx[b] for b in bases], dtype=np.int64)

        self._project_and_save(
            module, emb, labels, names, epoch,
            stem=f"tsne_pretrain_epoch_{epoch:03d}",
            title=f"val-batch embeddings by dataset (epoch {epoch})",
            wandb_tag="debug/tsne_pretrain",
        )

    # ------------------------------------------------------------------
    # "labeled" mode — external downstream classification corpus
    # ------------------------------------------------------------------
    def maybe_run_labeled(self, module: Any) -> None:
        if not self.enabled or "labeled" not in self.modes:
            return
        epoch = self._rank0_epoch(module)
        if (
            epoch is None
            or self.labeled_every_n_epochs <= 0
            or epoch % self.labeled_every_n_epochs != 0
        ):
            return
        try:
            self._run_labeled(module, epoch)
        except Exception as exc:  # noqa: BLE001
            log.warning("debug_embeddings: labeled t-SNE failed: %s", exc)

    def _ensure_labeled(self, module: Any) -> Any:
        """Build ``(DownstreamDataset, row_indices)`` once; ``False`` if unusable."""
        if self._labeled is not None:
            return self._labeled
        from data.downstream_datamodule import DownstreamDataset

        ds = DownstreamDataset(
            self.labeled_h5,
            window_sizes=tuple(module.hparams.window_sizes),
            target_patch_mm=float(module.hparams.target_patch_mm),
            apply_interpolate=self.apply_interpolate,
            target_length=self.target_length,
            normalization_type=self.normalization_type,
        )
        if ds.is_regression:
            log.warning(
                "debug_embeddings: %s is a regression corpus — the labeled "
                "t-SNE needs class labels; skipping.", self.labeled_h5,
            )
            self._labeled = False
            return False
        if self.labeled_session_id is None:
            log.warning(
                "debug_embeddings: tsne_labeled_session_id is null — "
                "the labeled t-SNE is skipped.",
            )
            self._labeled = False
            return False
        sel = np.where(ds.session_id == self.labeled_session_id)[0].astype(np.int64)
        if sel.size == 0:
            log.warning(
                "debug_embeddings: session_id=%d matches no rows in %s "
                "(available: %s) — skipping labeled t-SNE.",
                self.labeled_session_id, self.labeled_h5,
                sorted(np.unique(ds.session_id).tolist()),
            )
            self._labeled = False
            return False
        if sel.size > self.labeled_max_samples:
            rng = np.random.default_rng(int(module.hparams.seed))
            sel = np.sort(rng.choice(sel, size=self.labeled_max_samples, replace=False))
        # Per-channel stats from the embedded session itself (no train split
        # exists at pretraining time; the session is the whole population).
        if ds.normalization_type != "none":
            ds.set_channel_stats(*ds.compute_channel_stats(sel.tolist()))
        log.info(
            "debug_embeddings: labeled t-SNE corpus ready — %s session_id=%d "
            "(%d frames, C=%d, num_classes=%d)",
            ds.dataset_name, self.labeled_session_id, sel.size,
            ds.num_channels, ds.num_classes,
        )
        self._labeled = (ds, sel)
        return self._labeled

    @torch.no_grad()
    def _run_labeled(self, module: Any, epoch: int) -> None:
        ready = self._ensure_labeled(module)
        if not ready:
            return
        ds, indices = ready
        device = module.device
        fs = float(ds.sampling_frequency_hz)
        W = int(ds.window_size)
        ts_row = ds.patch_timestamps_us.to(device)   # (S,)

        embs: list[torch.Tensor] = []
        labels: list[int] = []
        chunk = 32
        for start in range(0, int(indices.size), chunk):
            sel = indices[start:start + chunk]
            frames = []
            for i in sel:
                item = ds[int(i)]
                frames.append(item["signal"])         # (C, T)
                labels.append(int(item["label"]))
            sig = torch.stack(frames, dim=0).to(device)   # (b, C, T)
            b, C, T = sig.shape
            # Backbone once per channel: fold the channel axis into the batch
            # axis (same vectorised pattern as UltrasonicEncoderWrapper).
            sig_bc = sig.reshape(b * C, T)
            mask_bc = torch.ones(b * C, T, dtype=torch.bool, device=device)
            fs_bc = torch.full((b * C,), fs, dtype=torch.float32, device=device)
            w_bc = torch.full((b * C,), W, dtype=torch.long, device=device)
            ts_bc = ts_row.unsqueeze(0).expand(b * C, -1)
            emb_bc = encode_batch_embeddings(
                module, sig_bc, mask_bc, fs_bc, w_bc, ts_bc,
                fixed_num_patches=module.hparams.target_patches,
            )                                             # (b*C, E)
            # Concatenate the per-channel embeddings — exactly the (B, C·E)
            # representation the downstream t-SNE projects
            # (us_downstream._forward_features: feats.flatten(1)).
            embs.append(emb_bc.view(b, -1).float().cpu())

        emb_np = torch.cat(embs, dim=0).numpy()
        labels_np = np.asarray(labels, dtype=np.int64)
        names = [f"class {c}" for c in range(int(ds.num_classes))]

        self._project_and_save(
            module, emb_np, labels_np, names, epoch,
            stem=f"tsne_labeled_epoch_{epoch:03d}",
            title=(
                f"{ds.dataset_name} session {self.labeled_session_id} "
                f"by class (epoch {epoch})"
            ),
            wandb_tag="debug/tsne_labeled",
        )

    # ------------------------------------------------------------------
    # Shared output path
    # ------------------------------------------------------------------
    def _project_and_save(
        self,
        module: Any,
        emb_np: np.ndarray,
        labels_np: np.ndarray,
        label_names: list[str],
        epoch: int,
        stem: str,
        title: str,
        wandb_tag: str,
    ) -> None:
        projected = _tsne_project(
            emb_np, labels_np, seed=int(module.hparams.seed),
            max_steps=self.tsne_max_steps,
        )
        if projected is None:
            return
        points, labels_np = projected
        fig = _scatter_by_label(points, labels_np, label_names, title)
        if fig is None:
            return

        out_dir = Path(str(module.trainer.default_root_dir)) / "tsne"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
        np.savez(
            out_dir / f"{stem}.npz",
            points_2d=points.astype(np.float32),
            labels=labels_np.astype(np.int64),
            label_names=np.asarray(label_names),
            epoch=epoch,
        )
        log.info("debug_embeddings: saved %s.{png,npz} in %s", stem, out_dir)

        exp = _wandb_experiment(module)
        if exp is not None:
            try:
                import wandb
                exp.log({wandb_tag: wandb.Image(fig), "epoch": epoch})
            except Exception as exc:  # pragma: no cover
                log.warning("debug_embeddings: wandb image log failed: %s", exc)

        import matplotlib.pyplot as plt
        plt.close(fig)
