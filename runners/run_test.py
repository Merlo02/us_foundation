#!/usr/bin/env python3
"""Inference & evaluation script for a trained Ultrasound MAE.

Usage (from ``us_foundation/``)::

    python -m runners.run_test \
        --test-data /path/to/test.h5 \
        --config /path/to/config.yaml \
        --checkpoint /path/to/model.ckpt \
        --num-samples 3 \
        --output-dir /path/to/output

Produces:
  - Total test loss printed to stdout.
  - Per-dataset reconstruction plots saved as PNGs.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data import HDF5DataModule
from model import UltrasonicMAE

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test/evaluate a trained Ultrasound MAE")
    p.add_argument(
        "--test-data", type=str, required=True,
        help="Path to the test HDF5 file (e.g. /path/to/test.h5)",
    )
    p.add_argument(
        "--config", type=str, required=True,
        help="Path to the flat YAML config (as saved at end of training)",
    )
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained model .ckpt file",
    )
    p.add_argument(
        "--num-samples", type=int, default=3,
        help="Number of samples to plot per dataset type (default: 3)",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save plots (default: same directory as checkpoint)",
    )
    p.add_argument(
        "--device", type=int, default=0,
        help="GPU index to use (default: 0)",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size from config (useful to reduce memory usage)",
    )
    return p.parse_args()


# ----------------------------------------------------------------------
# DataModule factory (mirrors run_train._build_datamodule for test)
# ----------------------------------------------------------------------

def _build_test_datamodule(cfg: dict, test_h5_path: Path, batch_size_override: Optional[int] = None) -> HDF5DataModule:
    data_cfg = cfg["data"]
    window_sizes = tuple(cfg["model"]["window_sizes"])
    target_patch_mm = float(cfg["model"]["target_patch_mm"])

    target_patches = data_cfg.get("target_patches", None)
    min_valid_patches = int(data_cfg.get("min_valid_patches", 1))
    normalization_type = str(data_cfg.get("normalization_type", "none"))
    norm_eps_z = float(data_cfg.get("norm_eps_z", 1e-6))
    norm_eps_mm = float(data_cfg.get("norm_eps_mm", 1e-10))
    preprocessing_mode = str(data_cfg.get("preprocessing_mode", "raw"))
    apply_interpolate = bool(data_cfg.get("apply_interpolate", False))
    interpolation_from_sassauna = bool(data_cfg.get("interpolation_from_sassauna", True))
    strict_target_length = data_cfg.get("strict_target_length", None)
    etl_config_path = data_cfg.get("etl_config_path") or None

    batch_size = batch_size_override or int(data_cfg["batch_size"])

    hdf5_dir = test_h5_path.parent

    return HDF5DataModule(
        hdf5_dir=str(hdf5_dir),
        batch_size=batch_size,
        num_workers=int(data_cfg.get("num_workers", 4)),
        window_sizes=window_sizes,
        target_patch_mm=target_patch_mm,
        target_patches=target_patches,
        min_valid_patches=min_valid_patches,
        sampling_strategy=data_cfg.get("sampling_strategy", "naive"),
        epoch_k=int(data_cfg.get("epoch_k", 500_000)),
        threshold_ratio=float(data_cfg.get("threshold_ratio", 0.1)),
        lg_dataset_name=data_cfg.get("lg_dataset_name", "lateral_gastrocnemius_verasonics"),
        lg_budget_split_ratios=tuple(
            data_cfg.get("lg_budget_split_ratios", (0.8, 0.1, 0.1)),
        ),
        seed=int(cfg.get("train", {}).get("seed", 42)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=False,
        normalization_type=normalization_type,
        norm_eps_z=norm_eps_z,
        norm_eps_mm=norm_eps_mm,
        signal_trace_enabled=False,
        signal_trace_dir="unused",
        preprocessing_mode=preprocessing_mode,
        apply_interpolate=apply_interpolate,
        interpolation_from_sassauna=interpolation_from_sassauna,
        strict_target_length=int(strict_target_length) if strict_target_length is not None else None,
        etl_config_path=etl_config_path,
    )


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def _load_model(cfg: dict, ckpt_path: str, device: torch.device) -> UltrasonicMAE:
    m = cfg["model"]
    t = cfg.get("train", {})
    target_patches = cfg.get("data", {}).get("target_patches", None)

    model = UltrasonicMAE.load_from_checkpoint(
        ckpt_path,
        window_sizes=tuple(m["window_sizes"]),
        target_patch_mm=float(m["target_patch_mm"]),
        tokenizer_type=m.get("tokenizer_type", "mlp"),
        cnn_config=m.get("cnn_config"),
        target_patches=target_patches,
        embed_dim=int(m["embed_dim"]),
        encoder_depth=int(m["encoder_depth"]),
        encoder_heads=int(m["encoder_heads"]),
        encoder_mlp_ratio=float(m["encoder_mlp_ratio"]),
        decoder_dim=int(m["decoder_dim"]),
        decoder_depth=int(m["decoder_depth"]),
        decoder_heads=int(m["decoder_heads"]),
        decoder_mlp_ratio=float(m["decoder_mlp_ratio"]),
        masking_ratio=float(m["masking_ratio"]),
        use_ct_rope=bool(m.get("use_ct_rope", True)),
        ct_rope_base=float(m.get("ct_rope_base", 10_000.0)),
        rope_max_seq_len=int(m.get("rope_max_seq_len", 512)),
        dropout=float(m.get("dropout", 0.0)),
        lr=float(t.get("lr", 1e-4)),
        weight_decay=float(t.get("weight_decay", 0.05)),
        betas=tuple(t.get("betas", (0.9, 0.95))),
        min_lr=float(t.get("min_lr", 1e-6)),
        loss_alpha=float(t.get("loss_alpha", 0.0)),
        norm_target=bool(t.get("norm_target", True)),
        warmup_epochs=int(t.get("warmup_epochs", 1)),
        max_epochs=int(t.get("max_epochs", 200)),
        seed=int(t.get("seed", 42)),
        map_location=device,
    )
    model = model.to(device)
    model.eval()
    return model


# ----------------------------------------------------------------------
# Sample collection data structures
# ----------------------------------------------------------------------

@dataclass
class ChunkResult:
    """Stores inference results for a single chunk."""
    chunk_index: int
    num_chunks: int
    pred: np.ndarray       # (n_patches, W) un-padded
    mask: np.ndarray       # (n_patches,) bool, 1=masked
    signal: np.ndarray     # (length,) original normalized signal
    window_size: int
    fs_hz: float


@dataclass
class AcquisitionCollector:
    """Accumulates chunks for a single acquisition (fixedS mode)."""
    num_chunks: int
    chunks: dict[int, ChunkResult] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return len(self.chunks) == self.num_chunks

    def add(self, chunk: ChunkResult) -> None:
        self.chunks[chunk.chunk_index] = chunk

    def stitch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
        """Returns (pred_full, mask_full, signal_full, window_size, fs_hz)."""
        sorted_chunks = [self.chunks[i] for i in sorted(self.chunks.keys())]
        pred_parts = [c.pred.reshape(-1) for c in sorted_chunks]
        mask_parts = [c.mask for c in sorted_chunks]
        signal_parts = [c.signal for c in sorted_chunks]
        return (
            np.concatenate(pred_parts),
            np.concatenate(mask_parts),
            np.concatenate(signal_parts),
            sorted_chunks[0].window_size,
            sorted_chunks[0].fs_hz,
        )


def _base_dataset_name(src: str) -> str:
    return src.split("::", 1)[0]


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
    window_size: int,
    fs_hz: float,
    title: str,
    save_path: Path,
) -> None:
    """Generate and save a reconstruction plot matching the reference style."""
    n_samples = original.shape[0]
    n_patches = mask.shape[0]
    usable_samples = n_patches * window_size

    orig = original[:usable_samples]
    recon = reconstructed[:usable_samples]

    time_ms = np.arange(usable_samples) / fs_hz * 1000.0
    residual = orig - recon

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))

    ax.plot(time_ms, orig, color="#7B2D8B", linewidth=1.0, label="Original")
    ax.plot(time_ms, recon, color="#E91E7B", linewidth=1.0, label="Reconstructed")
    ax.plot(time_ms, residual, color="#F5A623", linewidth=0.8, label="Residual")

    for p_idx in range(n_patches):
        if mask[p_idx]:
            t_start = p_idx * window_size / fs_hz * 1000.0
            t_end = (p_idx + 1) * window_size / fs_hz * 1000.0
            ax.axvspan(t_start, t_end, alpha=0.25, color="gray", label="Mask" if p_idx == int(np.where(mask)[0][0]) else None)

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_xlim(time_ms[0], time_ms[-1])

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot: %s", save_path)


# ----------------------------------------------------------------------
# Test loop
# ----------------------------------------------------------------------

def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move tensor values in batch to device, leave non-tensors as-is."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _denormalize_pred(pred_patches: np.ndarray, signal: np.ndarray, window_size: int) -> np.ndarray:
    """De-normalize predictions from per-patch normalized space back to signal space.

    When ``train.norm_target`` is true, targets are normalized per patch during training;
    decoder outputs live in that normalized space. To compare with the DataModule-normalised
    signal, invert per patch: ``pred_denorm = pred * std(orig_patch) + mean(orig_patch)``.
    """
    n_patches = pred_patches.shape[0]
    W = window_size
    denormed = np.empty_like(pred_patches)
    for i in range(n_patches):
        orig_patch = signal[i * W: (i + 1) * W]
        mu = orig_patch.mean()
        std = orig_patch.std()
        if std < 1e-6:
            std = 1e-6
        denormed[i] = pred_patches[i] * std + mu
    return denormed


_STAT_KEYS = ("n", "sum_t", "sum_p", "sum_tt", "sum_pp", "sum_tp", "sse")


@torch.no_grad()
def _batch_recon_stats(
    pred: torch.Tensor,
    signal: torch.Tensor,
    mask: torch.Tensor,
    window_sizes: torch.Tensor,
    padding_mask: torch.Tensor,
    signal_lengths: torch.Tensor,
    norm_target: bool,
    norm_eps: float = 1e-6,
) -> dict[str, float]:
    """Per-batch (target, pred) sufficient statistics on masked patches.

    Mirrors :class:`USReconstructionLoss` masking exactly (``m & padding & valid``)
    and computes in per-patch normalised space when ``norm_target`` — so the
    aggregated RMSE / R / R² live in the same frame as the reported loss.
    """
    n = sum_t = sum_p = sum_tt = sum_pp = sum_tp = sse = 0.0
    for W_val in window_sizes.unique().tolist():
        W = int(W_val)
        sel = window_sizes == W
        if not sel.any():
            continue
        sub_pred = pred[sel]
        sub_signal = signal[sel]
        sub_mask = mask[sel]
        sub_pad = padding_mask[sel]
        sub_lens = signal_lengths[sel]

        Bw = sub_pred.size(0)
        n_patches = sub_lens.div(W, rounding_mode="trunc")
        max_p = int(n_patches.max().item())
        if max_p == 0:
            continue
        target = sub_signal[:, : max_p * W].reshape(Bw, max_p, W)
        pidx = torch.arange(max_p, device=pred.device).unsqueeze(0)
        valid = pidx < n_patches.unsqueeze(1)
        if norm_target:
            mu = target.mean(dim=-1, keepdim=True)
            std = target.std(dim=-1, keepdim=True).clamp(min=norm_eps)
            target = (target - mu) / std
        sub_pred_w = sub_pred[:, :max_p, :W]
        rm = (sub_mask[:, :max_p].bool() & sub_pad[:, :max_p].bool() & valid).unsqueeze(-1)
        t_sel = target[rm.expand_as(target)]
        p_sel = sub_pred_w[rm.expand_as(sub_pred_w)]
        if t_sel.numel() == 0:
            continue
        n      += float(t_sel.numel())
        sum_t  += float(t_sel.sum().item())
        sum_p  += float(p_sel.sum().item())
        sum_tt += float((t_sel * t_sel).sum().item())
        sum_pp += float((p_sel * p_sel).sum().item())
        sum_tp += float((t_sel * p_sel).sum().item())
        sse    += float(((t_sel - p_sel) ** 2).sum().item())
    return {
        "n": n, "sum_t": sum_t, "sum_p": sum_p,
        "sum_tt": sum_tt, "sum_pp": sum_pp, "sum_tp": sum_tp, "sse": sse,
    }


def _metrics_from_stats(stats: dict[str, float]) -> Optional[dict[str, float]]:
    """Compute RMSE / R / R² from accumulated sufficient statistics.

    Returns ``None`` when no masked elements were collected.
    """
    n = stats["n"]
    if n <= 0:
        return None
    mu_t = stats["sum_t"] / n
    mu_p = stats["sum_p"] / n
    var_t = max(stats["sum_tt"] / n - mu_t * mu_t, 0.0)
    var_p = max(stats["sum_pp"] / n - mu_p * mu_p, 0.0)
    cov_tp = stats["sum_tp"] / n - mu_t * mu_p
    rmse = (stats["sse"] / n) ** 0.5
    denom = (var_t * var_p) ** 0.5
    r = cov_tp / denom if denom > 1e-12 else float("nan")
    tss = stats["sum_tt"] - n * mu_t * mu_t
    r2 = 1.0 - stats["sse"] / tss if tss > 1e-12 else float("nan")
    return {"n": n, "rmse": rmse, "r": r, "r2": r2}


def _extract_chunk_result(
    batch: dict,
    model_out: dict,
    idx: int,
    norm_target: bool = True,
) -> ChunkResult:
    """Extract inference results for a single item in the batch."""
    W = int(model_out["window_size"][idx].item())
    length = int(batch["length"][idx].item())
    n_patches = length // W

    pred_raw = model_out["pred"][idx, :n_patches, :W].cpu().numpy()
    mask = model_out["mask"][idx, :n_patches].cpu().numpy().astype(bool)
    signal = batch["signal"][idx, :length].cpu().numpy()

    pred = _denormalize_pred(pred_raw, signal, W) if norm_target else pred_raw
    fs_hz = float(batch["sampling_frequency_hz"][idx].item())

    chunk_index = int(batch["chunk_index"][idx].item()) if "chunk_index" in batch else -1
    num_chunks = int(batch["num_chunks"][idx].item()) if "num_chunks" in batch else -1

    return ChunkResult(
        chunk_index=chunk_index,
        num_chunks=num_chunks,
        pred=pred,
        mask=mask,
        signal=signal,
        window_size=W,
        fs_hz=fs_hz,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()

    test_h5 = Path(args.test_data)
    if not test_h5.exists():
        print(f"Test data not found: {test_h5}", file=sys.stderr)
        sys.exit(1)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    plot_dir = output_dir / "reconstruction_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    # Build DataModule and get test DataLoader
    datamodule = _build_test_datamodule(cfg, test_h5, batch_size_override=args.batch_size)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    if test_loader is None:
        print("No test data available (test.h5 not found in hdf5_dir).", file=sys.stderr)
        sys.exit(1)

    # Load model
    model = _load_model(cfg, str(ckpt_path), device)
    log.info("Model loaded from %s", ckpt_path)

    target_patches = cfg.get("data", {}).get("target_patches", None)
    is_fixed_s = target_patches is not None
    num_samples_to_plot = args.num_samples
    norm_target = bool(cfg.get("train", {}).get("norm_target", True))
    log.info(
        "train.norm_target=%s — prediction denormalisation for plots: %s",
        norm_target,
        norm_target,
    )

    # Collection structures
    # variableS: dict[base_name, list[ChunkResult]]  — each item is a full signal
    # fixedS:    dict[unique_acq_key, AcquisitionCollector]
    #            unique_acq_key = f"{src}::{acq_counter}" where acq_counter increments
    #            every time chunk_index == 0 (start of a new acquisition).
    if is_fixed_s:
        # Maps unique acquisition key → collector
        acquisition_collectors: dict[str, AcquisitionCollector] = {}
        # Per-source counter: incremented at each chunk_index == 0
        acq_id_counter: dict[str, int] = defaultdict(int)
        # Current active acquisition key for each source
        active_acq_key: dict[str, str] = {}
        completed_by_base: dict[str, list[tuple[str, AcquisitionCollector]]] = defaultdict(list)
    else:
        collected_by_base: dict[str, list[ChunkResult]] = defaultdict(list)

    # Track which base datasets are already fully collected
    fully_collected_bases: set[str] = set()

    total_loss = 0.0
    total_samples = 0
    recon_stats = {k: 0.0 for k in _STAT_KEYS}

    # Per-dataset accumulators (keyed by base dataset name)
    recon_stats_by_base: dict[str, dict[str, float]] = defaultdict(
        lambda: {k: 0.0 for k in _STAT_KEYS}
    )
    loss_sum_by_base: dict[str, float] = defaultdict(float)
    samples_by_base: dict[str, int] = defaultdict(int)

    log.info("Starting test loop (%d batches)...", len(test_loader))

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = _move_batch_to_device(batch, device)
            B = batch["signal"].size(0)

            model_out = model(batch)

            loss_dict = model.criterion(
                pred=model_out["pred"],
                signal=batch["signal"],
                mask=model_out["mask"],
                window_sizes=model_out["window_size"],
                padding_mask=model_out["padding_mask"],
                signal_lengths=batch["length"],
            )
            total_loss += loss_dict["loss"].item() * B
            total_samples += B

            batch_stats = _batch_recon_stats(
                pred=model_out["pred"],
                signal=batch["signal"],
                mask=model_out["mask"],
                window_sizes=model_out["window_size"],
                padding_mask=model_out["padding_mask"],
                signal_lengths=batch["length"],
                norm_target=norm_target,
            )
            for k in _STAT_KEYS:
                recon_stats[k] += batch_stats[k]

            # Per-dataset breakdown: group sample indices by base dataset name,
            # then re-run loss + recon stats on each subset.
            base_to_indices: dict[str, list[int]] = defaultdict(list)
            for i in range(B):
                base_to_indices[_base_dataset_name(batch["dataset_source"][i])].append(i)
            for base, indices in base_to_indices.items():
                idx = torch.tensor(indices, device=device, dtype=torch.long)
                nb = idx.numel()
                sub_loss = model.criterion(
                    pred=model_out["pred"][idx],
                    signal=batch["signal"][idx],
                    mask=model_out["mask"][idx],
                    window_sizes=model_out["window_size"][idx],
                    padding_mask=model_out["padding_mask"][idx],
                    signal_lengths=batch["length"][idx],
                )
                loss_sum_by_base[base] += sub_loss["loss"].item() * nb
                samples_by_base[base] += nb
                sub_stats = _batch_recon_stats(
                    pred=model_out["pred"][idx],
                    signal=batch["signal"][idx],
                    mask=model_out["mask"][idx],
                    window_sizes=model_out["window_size"][idx],
                    padding_mask=model_out["padding_mask"][idx],
                    signal_lengths=batch["length"][idx],
                    norm_target=norm_target,
                )
                for k in _STAT_KEYS:
                    recon_stats_by_base[base][k] += sub_stats[k]

            # Collect samples for plotting
            for i in range(B):
                src = batch["dataset_source"][i]
                base = _base_dataset_name(src)

                if base in fully_collected_bases:
                    continue

                chunk = _extract_chunk_result(batch, model_out, i, norm_target=norm_target)

                if is_fixed_s:
                    # chunk_index == 0 marks the start of a new acquisition.
                    # Assign a fresh unique key so different acquisitions from the
                    # same dataset_source never share a collector.
                    if chunk.chunk_index == 0:
                        acq_id = acq_id_counter[src]
                        acq_id_counter[src] += 1
                        acq_key = f"{src}::{acq_id}"
                        active_acq_key[src] = acq_key
                        acquisition_collectors[acq_key] = AcquisitionCollector(
                            num_chunks=chunk.num_chunks,
                        )

                    acq_key = active_acq_key.get(src)
                    if acq_key is None:
                        # Chunk mid-acquisition arrived before chunk 0 was seen
                        # (e.g. first batch starts mid-acquisition). Skip it —
                        # we can't reconstruct a partial acquisition.
                        continue

                    collector = acquisition_collectors[acq_key]
                    collector.add(chunk)

                    if collector.is_complete:
                        completed_by_base[base].append((acq_key, collector))
                        if len(completed_by_base[base]) >= num_samples_to_plot:
                            fully_collected_bases.add(base)
                else:
                    collected_by_base[base].append(chunk)
                    if len(collected_by_base[base]) >= num_samples_to_plot:
                        fully_collected_bases.add(base)

            if (batch_idx + 1) % 50 == 0:
                log.info(
                    "  Batch %d/%d — running loss: %.6f",
                    batch_idx + 1, len(test_loader), total_loss / max(total_samples, 1),
                )

    avg_loss = total_loss / max(total_samples, 1)
    space = "normalized" if norm_target else "signal"
    print(f"\n{'='*60}")
    print(f"  Test Loss: {avg_loss:.6f}  (over {total_samples} samples)")

    global_metrics = _metrics_from_stats(recon_stats)
    if global_metrics is not None:
        print(f"  Test RMSE: {global_metrics['rmse']:.6f}   R: {global_metrics['r']:.6f}   "
              f"R²: {global_metrics['r2']:.6f}   "
              f"(masked elements: {int(global_metrics['n']):,}, space: {space})")
    print(f"{'='*60}\n")
    if global_metrics is not None:
        log.info(
            "Test metrics — loss=%.6f  rmse=%.6f  R=%.6f  R²=%.6f  "
            "(masked elements: %d, %s space)",
            avg_loss, global_metrics["rmse"], global_metrics["r"], global_metrics["r2"],
            int(global_metrics["n"]), space,
        )

    # Per-dataset metrics
    if recon_stats_by_base:
        print(f"  Per-dataset metrics (space: {space}):")
        for base in sorted(recon_stats_by_base):
            nb = samples_by_base[base]
            base_loss = loss_sum_by_base[base] / max(nb, 1)
            m = _metrics_from_stats(recon_stats_by_base[base])
            if m is not None:
                print(f"    {base:<40s} loss={base_loss:.6f}  rmse={m['rmse']:.6f}  "
                      f"R={m['r']:.6f}  R²={m['r2']:.6f}  "
                      f"(masked: {int(m['n']):,}, {nb} samples)")
                log.info(
                    "Per-dataset [%s] — loss=%.6f  rmse=%.6f  R=%.6f  R²=%.6f  "
                    "(masked: %d, %d samples)",
                    base, base_loss, m["rmse"], m["r"], m["r2"], int(m["n"]), nb,
                )
            else:
                print(f"    {base:<40s} loss={base_loss:.6f}  "
                      f"(no masked elements, {nb} samples)")
        print(f"{'='*60}\n")

    # Generate plots
    log.info("Generating reconstruction plots...")

    if is_fixed_s:
        for base, acquisitions in completed_by_base.items():
            for sample_idx, (src, collector) in enumerate(acquisitions[:num_samples_to_plot]):
                pred_full, mask_full, signal_full, W, fs_hz = collector.stitch()
                save_path = plot_dir / f"{base}_{sample_idx}.png"
                _plot_reconstruction(
                    original=signal_full,
                    reconstructed=pred_full,
                    mask=mask_full,
                    window_size=W,
                    fs_hz=fs_hz,
                    title=f"{base} (sample {sample_idx})",
                    save_path=save_path,
                )
    else:
        for base, chunks in collected_by_base.items():
            for sample_idx, chunk in enumerate(chunks[:num_samples_to_plot]):
                pred_signal = chunk.pred.reshape(-1)
                save_path = plot_dir / f"{base}_{sample_idx}.png"
                _plot_reconstruction(
                    original=chunk.signal,
                    reconstructed=pred_signal,
                    mask=chunk.mask,
                    window_size=chunk.window_size,
                    fs_hz=chunk.fs_hz,
                    title=f"{base} (sample {sample_idx})",
                    save_path=save_path,
                )

    log.info("Done. Plots saved to %s", plot_dir)


if __name__ == "__main__":
    main()
