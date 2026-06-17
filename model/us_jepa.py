"""Top-level LightningModule for LeJEPA pretraining (encoder-only).

Faithful port of BioFoundation's LeJEPA path
(``BioFoundation-main/tasks/pretrain_task_LuMamba.py`` — methods
``generate_lejepa_views``, ``SIGReg``, ``LeJEPA``, ``sigreg_2d_projection``),
adapted to this codebase's variable-length, multi-frequency batches:

- **Views as a fraction of valid patches.** BioFoundation crops views of a
  fixed absolute width (``patch_size · num_patches_local/global`` out of a
  constant 1280-step window) — meaningful only when every signal has the
  same length. Here each sample has its own native length and its own
  ``W*``, so a view is defined as ``n_view = round(ratio · n_valid)``
  patches of that sample (``local_view_ratio`` / ``global_view_ratio``;
  the defaults 0.125 / 0.5 reproduce BioFoundation's 4/32 and 16/32
  proportions). Because the multi-tokenizer picks ``W*`` so that one patch
  ≈ ``target_patch_mm`` of tissue depth, a patch fraction is also a depth
  fraction — consistent across datasets and sampling rates.
- **Patch-aligned crops.** Views start at a multiple of ``W*`` so the
  loader-provided ``patch_timestamps_us`` (chunk-offset / resampling aware)
  can be sliced per view. CT-RoPE only sees relative time inside a
  sequence, so this is equivalent to BioFoundation's arbitrary-offset crops.
- **Masked mean pooling.** Views in a batch have different patch counts, so
  the view embedding is a masked mean over valid tokens where BioFoundation
  uses a plain ``mean(dim=1)``.
- **No decoder, no MAE masking.** The encoder is always called with
  ``bypass_masking=True``; the loss is purely
  ``(1−λ)·prediction + λ·SIGReg`` (BioFoundation's ``use_lejepa_only``).

The submodule attribute names (``tokenizer``, ``encoder``) match
:class:`~model.us_mae.UltrasonicMAE`, so JEPA checkpoints load into the
downstream :class:`~model.downstream.encoder_wrapper.UltrasonicEncoderWrapper`
through the existing ``pretrained_ckpt`` mechanism with no changes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    import lightning.pytorch as pl  # type: ignore[no-redef]

from schedulers import CosineLRSchedulerWrapper
from .backbone.us_encoder import USEncoder
from .embedding_debug import EmbeddingTSNEDebugger, _wandb_experiment
from .positional.ct_rope import CTRoPE
from .positional.discrete_rope import DiscreteRoPE
from .tokenizer.multi_tokenizer import MultiTokenizer

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# DDP helpers (copied from BioFoundation pretrain_task_LuMamba.py)
# ----------------------------------------------------------------------

def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


class UltrasonicJEPA(pl.LightningModule):
    """LeJEPA pretraining for ultrasound A-mode signals.

    The full forward path per step:

    1. :meth:`generate_lejepa_views` crops ``num_global_views`` global and
       ``num_local_views`` local views per sample (random patch-aligned
       start, size = ``ratio · n_valid`` patches).
    2. Each view group is packed into a single ``(B·V, T)`` batch (the
       analogue of BioFoundation's ``rearrange('V B C T -> (B V) C T')``)
       and encoded once: :class:`MultiTokenizer` → :class:`USEncoder` with
       ``bypass_masking=True`` → masked mean over valid tokens → ``(V, B, D)``.
    3. :meth:`LeJEPA` computes the prediction loss between each local-view
       embedding and the mean of the global-view embeddings, plus the
       :meth:`SIGReg` sliced Epps–Pulley isotropic-Gaussian penalty on the
       local-view embeddings: ``loss = (1−λ)·sim + λ·sigreg``.
    """

    def __init__(
        self,
        # Tokenizer (identical to UltrasonicMAE)
        window_sizes: tuple[int, ...] = (8, 16, 32),
        target_patch_mm: float = 0.6,
        tokenizer_type: str = "mlp",
        cnn_config: Optional[dict] = None,
        # Fixed-S batching (must match DataModule's ``target_patches``).
        # Only used by the embedding-debug full-signal forward; the views
        # themselves are always variable-S.
        target_patches: Optional[int] = None,
        # Encoder dims
        embed_dim: int = 256,
        encoder_depth: int = 8,
        encoder_heads: int = 8,
        encoder_mlp_ratio: float = 4.0,
        # Positional encoding
        use_ct_rope: bool = True,
        ct_rope_base: float = 10_000.0,
        rope_max_seq_len: int = 512,
        # Regularisation
        dropout: float = 0.0,
        # LeJEPA (defaults = BioFoundation LuMamba_pretrain.yaml; the two
        # ratios reproduce its num_patches_local/global out of 32 patches)
        num_global_views: int = 2,
        num_local_views: int = 4,
        global_view_ratio: float = 0.5,
        local_view_ratio: float = 0.125,
        lambd_lejepa: float = 0.002,
        num_slices: int = 128,
        sigreg_log_every_n_epochs: int = 5,
        # Optimiser (identical to UltrasonicMAE)
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        warmup_lr_init: float = 1e-6,
        max_epochs: int = 100,
        seed: int = 42,
        # Shared pretraining diagnostics (see model/embedding_debug.py)
        debug_embeddings: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        for name, ratio in (
            ("local_view_ratio", local_view_ratio),
            ("global_view_ratio", global_view_ratio),
        ):
            if not 0.0 < float(ratio) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {ratio!r}")
        if num_global_views < 1 or num_local_views < 1:
            raise ValueError(
                f"num_global_views/num_local_views must be >= 1, got "
                f"{num_global_views}/{num_local_views}",
            )
        if local_view_ratio > global_view_ratio:
            log.warning(
                "local_view_ratio (%.3f) > global_view_ratio (%.3f) — local "
                "views are supposed to be the smaller crops.",
                local_view_ratio, global_view_ratio,
            )

        self.tokenizer = MultiTokenizer(
            window_sizes=window_sizes,
            embed_dim=embed_dim,
            target_patch_mm=target_patch_mm,
            tokenizer_type=tokenizer_type,
            cnn_config=cnn_config,
        )

        head_dim_enc = embed_dim // encoder_heads
        if use_ct_rope:
            rotary_enc: nn.Module = CTRoPE(dim=head_dim_enc, base=ct_rope_base)
        else:
            rotary_enc = DiscreteRoPE(
                dim=head_dim_enc, max_seq_len=rope_max_seq_len, base=ct_rope_base,
            )

        self.encoder = USEncoder(
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=encoder_mlp_ratio,
            # Irrelevant in bypass_masking mode (same convention as the
            # downstream UltrasonicEncoderWrapper).
            masking_ratio=0.0,
            dropout=dropout,
            rotary=rotary_enc,
        )

        self._emb_debug = EmbeddingTSNEDebugger(debug_embeddings)

    # ------------------------------------------------------------------
    # View generation (BioFoundation generate_lejepa_views, variable-length)
    # ------------------------------------------------------------------
    def _crop_views(
        self,
        signal: torch.Tensor,    # (B, T) padded
        ts: torch.Tensor,        # (B, S) patch timestamps
        W: torch.Tensor,         # (B,) window size per sample
        n_valid: torch.Tensor,   # (B,) valid patches per sample
        ratio: float,
    ) -> dict:
        """One random patch-aligned crop per sample, sized ``ratio · n_valid``.

        Fully vectorised (no Python loop over B). Returns the cropped
        signals padded to the in-view maximum length, with masks, the
        sliced patch timestamps, and the crop positions for diagnostics.
        """
        B, T = signal.shape
        device = signal.device

        n_view = (float(ratio) * n_valid.float()).round().long().clamp(min=1)
        n_view = torch.minimum(n_view, n_valid)
        max_start = n_valid - n_view                               # (B,) >= 0
        u = torch.rand(B, device=device)
        start_patch = torch.minimum(
            (u * (max_start + 1).float()).floor().long(), max_start,
        )

        # Signal-space crop via gather.
        view_len = n_view * W                                      # (B,)
        start_sample = start_patch * W
        T_v = int(view_len.max().item())
        pos = torch.arange(T_v, device=device).unsqueeze(0)        # (1, T_v)
        idx = (start_sample.unsqueeze(1) + pos).clamp(max=max(T - 1, 0))
        view_signal = torch.gather(signal, 1, idx)
        view_mask = pos < view_len.unsqueeze(1)                    # (B, T_v)
        view_signal = view_signal * view_mask.to(view_signal.dtype)

        # Token-space slice of the loader timestamps.
        S = ts.size(1)
        S_v = int(n_view.max().item())
        tpos = torch.arange(S_v, device=device).unsqueeze(0)       # (1, S_v)
        tidx = (start_patch.unsqueeze(1) + tpos).clamp(max=max(S - 1, 0))
        view_ts = torch.gather(ts, 1, tidx)
        token_mask = tpos < n_view.unsqueeze(1)
        view_ts = view_ts * token_mask.to(view_ts.dtype)

        return {
            "signal": view_signal,             # (B, T_v)
            "signal_mask": view_mask,          # (B, T_v) bool
            "patch_timestamps_us": view_ts,    # (B, S_v)
            "start_patch": start_patch,        # (B,)
            "start_sample": start_sample,      # (B,)
            "n_patches": n_view,               # (B,)
        }

    def generate_lejepa_views(
        self,
        batch: dict,
        num_global_views: Optional[int] = None,
        num_local_views: Optional[int] = None,
    ) -> tuple[list[dict], list[dict], list[torch.Tensor], list[torch.Tensor]]:
        """Generate global and local views for LeJEPA training.

        Mirrors BioFoundation's ``generate_lejepa_views`` — same return
        quadruple ``(global_views, all_views, start_times_global,
        start_times_local)``, with ``all_views`` being the local views as in
        the original — except each view is a dict of per-sample crops (see
        :meth:`_crop_views`) instead of an equal-width tensor, and the start
        times are ``(B,)`` tensors of sample offsets.
        """
        if num_global_views is None:
            num_global_views = int(self.hparams.num_global_views)
        if num_local_views is None:
            num_local_views = int(self.hparams.num_local_views)

        signal = batch["signal"]
        device = signal.device
        length = batch["length"].to(device)
        W = batch["window_size"].to(device)
        ts = batch["patch_timestamps_us"].to(device)
        n_valid = torch.div(length, W, rounding_mode="trunc").clamp(min=1)

        all_views: list[dict] = []
        start_times_local: list[torch.Tensor] = []
        for _ in range(num_local_views):
            v = self._crop_views(
                signal, ts, W, n_valid, float(self.hparams.local_view_ratio),
            )
            all_views.append(v)
            start_times_local.append(v["start_sample"])

        global_views: list[dict] = []
        start_times_global: list[torch.Tensor] = []
        for _ in range(num_global_views):
            v = self._crop_views(
                signal, ts, W, n_valid, float(self.hparams.global_view_ratio),
            )
            global_views.append(v)
            start_times_global.append(v["start_sample"])

        return global_views, all_views, start_times_global, start_times_local

    # ------------------------------------------------------------------
    # View encoding
    # ------------------------------------------------------------------
    def _encode_views(self, views: list[dict], batch: dict) -> torch.Tensor:
        """Pack ``V`` views into one ``(B·V, ·)`` batch and encode → ``(V, B, D)``.

        Equivalent to BioFoundation's ``rearrange('V B C T -> (B V) C T')``
        + ``self.model.encode(...)`` + sequence mean: one tokenizer+encoder
        forward per view group; the sequence mean is masked because views
        have different patch counts.
        """
        V = len(views)
        B = views[0]["signal"].size(0)
        device = views[0]["signal"].device
        T_max = max(v["signal"].size(1) for v in views)
        S_max = max(v["patch_timestamps_us"].size(1) for v in views)

        def _pack(key: str, total: int, fill) -> torch.Tensor:
            rows = []
            for v in views:
                x = v[key]
                if x.size(1) < total:
                    pad = x.new_full((B, total - x.size(1)), fill)
                    x = torch.cat([x, pad], dim=1)
                rows.append(x)
            # (B, V, total) → (B·V, total): index = b·V + v — same '(B V)'
            # ordering as BioFoundation's rearrange, inverted below.
            return torch.stack(rows, dim=1).reshape(B * V, total)

        signal = _pack("signal", T_max, 0.0)
        signal_mask = _pack("signal_mask", T_max, False)
        ts = _pack("patch_timestamps_us", S_max, 0.0)

        fs = batch["sampling_frequency_hz"].to(device)
        W = batch["window_size"].to(device)
        fs_bv = fs.unsqueeze(1).expand(B, V).reshape(B * V)
        w_bv = W.unsqueeze(1).expand(B, V).reshape(B * V)

        tok = self.tokenizer(
            signal=signal,
            signal_mask=signal_mask,
            sampling_frequency_hz=fs_bv,
            window_size_override=w_bv,
            fixed_num_patches=None,   # views are variable-S by construction
            patch_timestamps_us=ts,
        )
        enc = self.encoder(
            tokens=tok.tokens,
            padding_mask=tok.padding_mask,
            time_values_us=tok.patch_timestamps_us,
            bypass_masking=True,
        )
        latent = enc["latent"]                    # (B·V, S, E)
        valid = enc["padding_mask"]               # (B·V, S)
        m = valid.unsqueeze(-1).to(latent.dtype)
        emb = (latent * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # (B·V, E)
        return emb.view(B, V, -1).transpose(0, 1)                    # (V, B, E)

    # ------------------------------------------------------------------
    # SIGReg (copied from BioFoundation pretrain_task_LuMamba.py)
    # ------------------------------------------------------------------
    def SIGReg(
        self, x: torch.Tensor, global_step: int, num_slices: int = 128,
    ) -> torch.Tensor:
        """Computes the Sliced Epps–Pulley statistic in a DDP-safe manner.

        ``x``: ``(N, M)`` real tensor. Copied from BioFoundation; the only
        local adaptation is running with autocast disabled in float32 — the
        empirical characteristic function is complex-valued and autocast
        (bf16-mixed) does not support complex tensors.
        """
        device = x.device
        with torch.autocast(device_type=device.type, enabled=False):
            x = x.float()

            # -------- 1. Random projections (synchronized via global_step) ----
            g = torch.Generator(device=device)
            g.manual_seed(int(global_step))

            M = x.size(1)
            A = torch.randn((M, num_slices), generator=g, device=device)
            A /= (A.norm(dim=0) + 1e-12)

            # Project sample: (N, num_slices)
            x_proj = x @ A

            # -------- 2. Empirical characteristic function (complex) ----------
            t = torch.linspace(-5, 5, 17, device=device)
            exp_f = torch.exp(-0.5 * t**2)

            # x_proj: (N, S) → (N, S, 17)
            x_t = x_proj.unsqueeze(2) * t

            # (N, S, 17) → (S, 17) complex
            ecf_local = (1j * x_t).exp().mean(dim=0)

            # -------- 3. All-reduce complex ECF -------------------------------
            ecf_real = torch.view_as_real(ecf_local)        # (S, 17, 2)
            if is_dist_initialized():
                dist.all_reduce(ecf_real, op=dist.ReduceOp.SUM)
                ecf_real /= get_world_size()
            ecf = torch.view_as_complex(ecf_real)

            # -------- 4. Epps–Pulley error ------------------------------------
            err = (ecf - exp_f).abs().square() * exp_f

            # -------- 5. Compute true global N --------------------------------
            local_N = torch.tensor([x.size(0)], device=device, dtype=torch.long)
            if is_dist_initialized():
                dist.all_reduce(local_N, op=dist.ReduceOp.SUM)
            total_N = float(local_N.item())

            # -------- 6. Epps–Pulley statistic --------------------------------
            T_stat = torch.trapz(err, t, dim=1) * total_N
            return T_stat.mean()

    # ------------------------------------------------------------------
    # LeJEPA loss (copied from BioFoundation pretrain_task_LuMamba.py;
    # the encoding part lives in _encode_views)
    # ------------------------------------------------------------------
    def LeJEPA(
        self,
        g_emb: torch.Tensor,    # (V_global, B, D)
        a_emb: torch.Tensor,    # (V_local, B, D) — BioFoundation's all_views
        lambd: float,
        global_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(1−λ)·sim + λ·sigreg`` from the view embeddings.

        ``sim`` is the squared distance between each local-view embedding
        and the mean of the global-view embeddings; ``sigreg`` is the mean
        :meth:`SIGReg` statistic over the local views (as in the original).
        Returns ``(loss, sim, sigreg)``.
        """
        # compute mean of global views and similarity loss
        centers = g_emb.mean(0)                      # (B, D)
        sim = (centers - a_emb).square().mean()      # scalar

        # each emb is (B, D), each sigreg is a scalar
        sigreg = torch.stack([
            self.SIGReg(emb, global_step, num_slices=int(self.hparams.num_slices))
            for emb in a_emb
        ]).mean()

        loss = (1.0 - lambd) * sim + lambd * sigreg
        return loss, sim, sigreg

    # ------------------------------------------------------------------
    # SIGReg 2-D projection diagnostics
    # (copied from BioFoundation pretrain_task_LuMamba.py)
    # ------------------------------------------------------------------
    def sigreg_2d_projection(self, x: torch.Tensor, global_step: int) -> torch.Tensor:
        """Returns 2D projected samples used for logging visualization."""
        device = x.device
        g = torch.Generator(device=device)
        g.manual_seed(int(global_step))

        M = x.size(1)
        A = torch.randn((M, 2), generator=g, device=device)
        A /= (A.norm(dim=0) + 1e-12)

        with torch.no_grad():
            proj2d = (x.float() @ A).detach().cpu()
        return proj2d  # (N, 2)

    def log_scatter_2D_SigREG(self, proj2d: torch.Tensor) -> None:
        """Scatter of the 2-D SIGReg projection (adapted from BioFoundation):
        PNG + ``.npz`` under ``{run_dir}/sigreg_2d/`` + wandb image when a
        WandbLogger is attached."""
        try:
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            return

        pts = proj2d.to(torch.float32).cpu().numpy()
        epoch = int(self.current_epoch)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.7)
        ax.set_title(f"SIGReg 2D Projection (Epoch {epoch})")
        ax.set_xlabel("slice dim 1")
        ax.set_ylabel("slice dim 2")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()

        out_dir = Path(str(self.trainer.default_root_dir)) / "sigreg_2d"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"SIGReg_2D_epoch_{epoch:03d}.png", bbox_inches="tight")
        np.savez(
            out_dir / f"SIGReg_2D_epoch_{epoch:03d}.npz",
            points_2d=pts.astype(np.float32),
            epoch=epoch,
            xlabel="slice dim 1",
            ylabel="slice dim 2",
            title=f"SIGReg 2D Projection (Epoch {epoch})",
        )

        exp = _wandb_experiment(self)
        if exp is not None:
            try:
                import wandb
                exp.log({"LeJEPA_2D_batch_0": wandb.Image(fig), "epoch": epoch})
            except Exception as exc:  # pragma: no cover
                log.warning("sigreg_2d: wandb image log failed: %s", exc)

        plt.close(fig)

    def _maybe_log_sigreg_2d(self, a_emb: torch.Tensor) -> None:
        """Rank-0, batch-0, every ``sigreg_log_every_n_epochs`` epochs (the
        ``(epoch+1) % 5`` cadence of BioFoundation's training_step)."""
        n = int(self.hparams.sigreg_log_every_n_epochs)
        if n <= 0:
            return
        trainer = self.trainer
        if trainer is None or not bool(getattr(trainer, "is_global_zero", True)):
            return
        if (int(self.current_epoch) + 1) % n != 0:
            return
        try:
            # Pick the first local view for visualization, shape (B, D).
            proj2d = self.sigreg_2d_projection(
                a_emb[0].detach(), int(self.global_step),
            )
            self.log_scatter_2D_SigREG(proj2d)
        except Exception as exc:  # noqa: BLE001 — diagnostics must not kill the run
            log.warning("sigreg_2d projection logging failed: %s", exc)

    # ------------------------------------------------------------------
    # View tracing (epoch 0 / batch 0 / rank 0, via the shared SignalTracer)
    # ------------------------------------------------------------------
    def _maybe_trace_views(
        self,
        batch: dict,
        global_views: list[dict],
        local_views: list[dict],
    ) -> None:
        """Plot a few samples' original signal + extracted views.

        Uses ``SignalTracer.trace_jepa_views`` from the DataModule (only
        present when ``data.signal_trace_enabled=true``): one PNG per base
        dataset in the first training batch of epoch 0, max 4 datasets.
        """
        trainer = self.trainer
        if (
            trainer is None
            or not bool(getattr(trainer, "is_global_zero", True))
            or int(self.current_epoch) != 0
        ):
            return
        tracer = getattr(getattr(trainer, "datamodule", None), "signal_tracer", None)
        if tracer is None:
            return
        try:
            seen: set[str] = set()
            picks: list[tuple[str, int]] = []
            for i, src in enumerate(batch["dataset_source"]):
                base = str(src).split("::", 1)[0]
                if base in seen:
                    continue
                seen.add(base)
                picks.append((base, i))
                if len(picks) >= 4:
                    break

            for base, i in picks:
                L = int(batch["length"][i].item())
                original = batch["signal"][i, :L].detach().float().cpu().numpy()
                W = int(batch["window_size"][i].item())

                def _extract(views: list[dict], i: int = i) -> list:
                    out = []
                    for v in views:
                        n_val = int(v["signal_mask"][i].sum().item())
                        sig = v["signal"][i, :n_val].detach().float().cpu().numpy()
                        out.append((sig, int(v["start_sample"][i].item())))
                    return out

                tracer.trace_jepa_views(
                    original=original,
                    global_views=_extract(global_views),
                    local_views=_extract(local_views),
                    dataset_source=base,
                    window_size=W,
                    epoch=int(self.current_epoch),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("jepa view tracing failed: %s", exc)

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------
    def _step(self, batch: dict, stage: str, batch_idx: int = 0) -> torch.Tensor:
        global_views, all_views, _, _ = self.generate_lejepa_views(batch)

        g_emb = self._encode_views(global_views, batch)   # (V_g, B, D)
        a_emb = self._encode_views(all_views, batch)      # (V_l, B, D)

        loss, sim, sigreg = self.LeJEPA(
            g_emb, a_emb,
            lambd=float(self.hparams.lambd_lejepa),
            global_step=int(self.global_step),
        )

        on_step = stage == "train"
        # sync_dist only for val/test; training metrics are per-rank (DDP
        # already synchronises gradients — extra all-reduces waste bandwidth).
        sd = stage != "train"
        bs = batch["signal"].size(0)
        self.log(f"{stage}/loss",        loss,   prog_bar=True,  sync_dist=sd, on_step=on_step, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/sim_loss",    sim,    prog_bar=False, sync_dist=sd, on_step=on_step, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/sigreg_loss", sigreg, prog_bar=False, sync_dist=sd, on_step=on_step, on_epoch=True, batch_size=bs)

        if stage == "train" and batch_idx == 0:
            self._maybe_log_sigreg_2d(a_emb)
            self._maybe_trace_views(batch, global_views, all_views)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train", batch_idx)

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch, "val", batch_idx)
        self._emb_debug.maybe_run_pretrain(self, batch, batch_idx)
        return loss

    def on_validation_epoch_end(self) -> None:
        self._emb_debug.maybe_run_labeled(self)

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test", batch_idx)

    # ------------------------------------------------------------------
    # Optimiser + Scheduler (identical to UltrasonicMAE)
    # ------------------------------------------------------------------
    def configure_optimizers(self) -> Any:
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if (
                p.ndim <= 1
                or name.endswith(".bias")
                or "mask_token" in name
                or "pad_token" in name
            ):
                no_decay.append(p)
            else:
                decay.append(p)

        optim = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=tuple(self.hparams.betas),
        )

        total_steps = self.trainer.estimated_stepping_batches
        # PL 2.x returns max_steps (default: -1) for IterableDataset when
        # max_steps is not set → fall back to a manual estimate.
        if not (isinstance(total_steps, (int, float)) and total_steps > 0 and total_steps != float("inf")):
            dm = self.trainer.datamodule
            if (
                hasattr(dm, "_estimated_num_batches")
                and hasattr(dm, "_train_shards")
                and dm._train_shards
            ):
                steps_per_epoch = max(dm._estimated_num_batches(dm._train_shards), 1)
            else:
                steps_per_epoch = 1
            total_steps = steps_per_epoch * max(self.hparams.max_epochs, 1)

        scheduler = CosineLRSchedulerWrapper(
            optimizer=optim,
            total_training_opt_steps=int(total_steps),
            trainer=self.trainer,
            warmup_epochs=self.hparams.warmup_epochs,
            min_lr=self.hparams.min_lr,
            warmup_lr_init=self.hparams.warmup_lr_init,
            t_in_epochs=False,
        )

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def lr_scheduler_step(self, scheduler: Any, metric: Any) -> None:
        # timm schedulers use step_update() instead of the standard step()
        scheduler.step_update(num_updates=self.global_step)
