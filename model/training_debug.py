"""Optional training-time debug logs for tokenizer routing + CT-RoPE midpoints.

Activated via ``train.debug_pipeline.enabled`` on :class:`~model.us_mae.UltrasonicMAE`.
Only rank 0 logs; only first and last training epochs (by ``max_epochs``); bounded
samples per base dataset name.

On epoch 0 / batch 0 the same lines (plus the epoch sample count) are also written
to ``debug_pipeline_epoch0.log`` in the same directory as Lightning's ``metrics.csv``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from .tokenizer.multi_tokenizer import TokenizerOutput
    from .us_mae import UltrasonicMAE


log = logging.getLogger(__name__)


def _find_csv_log_dir(trainer) -> Optional[Path]:
    """Return the directory where Lightning's CSVLogger writes ``metrics.csv``."""
    try:
        for logger in getattr(trainer, "loggers", []):
            ld = getattr(logger, "log_dir", None)
            if ld is not None:
                return Path(ld)
    except Exception:
        pass
    return None


def _epoch_train_samples_str(trainer) -> str:
    """Read ``_epoch_train_samples`` from the DataModule if available."""
    try:
        dm = getattr(trainer, "datamodule", None)
        if dm is not None:
            n = getattr(dm, "_epoch_train_samples", None)
            if n is not None:
                return str(int(n))
    except Exception:
        pass
    return "unknown"


def _valid_midpoints_us(
    patch_ts: torch.Tensor,
    pad_mask: torch.Tensor,
    k: int,
) -> tuple[list[float], list[float]]:
    """First ``k`` midpoint times (µs) and consecutive deltas (µs), valid tokens only."""
    vm = pad_mask.bool()
    vals = patch_ts[vm].detach().float().cpu().tolist()[: max(k, 0)]
    deltas = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    return vals, deltas


def analytic_valid_patches_per_chunk(
    full_len_samples: int,
    window_size: int,
    target_patches: int,
) -> tuple[int, list[int]]:
    """Mirror HDF5/WebDataset chunking: valid patch counts per contiguous chunk."""
    target_T = target_patches * window_size
    if target_T <= 0:
        return 0, []
    num_chunks = max(1, (full_len_samples + target_T - 1) // target_T)
    out: list[int] = []
    for c in range(num_chunks):
        chunk_start = c * target_T
        remaining = full_len_samples - chunk_start
        valid = min(remaining, target_T) // window_size
        out.append(int(valid))
    return num_chunks, out


def maybe_log_training_batch(
    module: "UltrasonicMAE",
    batch: dict,
    tok_out: "TokenizerOutput",
    batch_idx: int,
) -> None:
    """Log a bounded debug trace for qualifying batches.

    On epoch 0 / batch 0 the same lines (plus epoch sample count) are also
    written to ``debug_pipeline_epoch0.log`` next to ``metrics.csv``.
    """
    trainer = module.trainer
    if trainer is None:
        return
    if getattr(trainer, "global_rank", 0) != 0:
        return
    if not getattr(module, "_debug_enabled", False):
        return

    interval = getattr(module, "_debug_log_interval_batches", 1)
    if interval > 1 and (batch_idx % interval) != 0:
        return

    epoch = int(getattr(trainer, "current_epoch", 0))
    max_epochs = int(module.hparams.max_epochs)
    last_ep = max_epochs - 1
    if epoch not in (0, last_ep):
        return

    counts = getattr(module, "_debug_logged_counts", None)
    if counts is None:
        return

    max_per_ds = getattr(module, "_debug_max_samples_per_base_dataset", 2)
    k_mid = getattr(module, "_debug_midpoint_log_k", 5)
    write_file = (epoch == 0 and batch_idx == 0)

    B = batch["signal"].size(0)
    rng = np.random.default_rng(
        int(getattr(module.hparams, "seed", 42))
        + epoch * 100_003
        + batch_idx * 17
        + max_epochs,
    )
    order = rng.permutation(B)

    tokenizer_type = getattr(module.tokenizer, "tokenizer_type", "mlp")
    window_sizes = tuple(module.tokenizer.window_sizes)
    tp = module.hparams.target_patches

    dataset_sources = batch["dataset_source"]
    full_lens = batch.get("full_length_samples")
    chunk_ix = batch.get("chunk_index")
    num_chunks_t = batch.get("num_chunks")

    file_blocks: list[str] = []

    for pos in order.tolist():
        ds_full = dataset_sources[pos]
        base = str(ds_full).split("::", 1)[0]
        if counts.get(base, 0) >= max_per_ds:
            continue

        W_star = int(tok_out.window_size[pos].item())
        fs_hz = float(tok_out.sampling_frequency_hz[pos].item())
        length_sig = int(batch["length"][pos].item())
        padded_t = int(batch["signal"].size(1))

        branch_idx = window_sizes.index(W_star) if W_star in window_sizes else -1
        branch_desc = f"{tokenizer_type}_branch idx={branch_idx} W*={W_star}"

        pad_m = tok_out.padding_mask[pos]
        S_tok = int(tok_out.tokens.size(1))
        n_valid = int(pad_m.sum().item())
        n_pad_slot = S_tok - n_valid

        vals_us, deltas_us = _valid_midpoints_us(
            tok_out.patch_timestamps_us[pos],
            pad_m,
            k_mid,
        )
        vals_s = [v / 1e6 for v in vals_us]
        deltas_s = [d / 1e6 for d in deltas_us]

        lines = [
            f"[debug_pipeline] epoch={epoch}/{last_ep} batch={batch_idx} sample={pos}",
            f"  dataset_source={ds_full}",
            f"  input: length_samples={length_sig} padded_T={padded_t}",
            f"  routing: fs_hz={fs_hz:g} → {branch_desc}",
            f"  tokenizer_out: S={S_tok} n_valid_tokens={n_valid} n_pad_slots={n_pad_slot}",
            f"  CT-RoPE midpoints_first_{len(vals_us)} (µs): {[round(v, 6) for v in vals_us]}",
            f"  CT-RoPE midpoints_first_{len(vals_s)} (s):   {[round(v, 9) for v in vals_s]}",
            f"  CT-RoPE delta_midpoint_us (consecutive): {[round(d, 6) for d in deltas_us]}",
            f"  CT-RoPE delta_midpoint_s (consecutive):   {[round(d, 9) for d in deltas_s]}",
        ]

        if full_lens is not None and chunk_ix is not None and num_chunks_t is not None:
            full_len = int(full_lens[pos].item())
            ci = int(chunk_ix[pos].item())
            nc = int(num_chunks_t[pos].item())
            lines.append(
                f"  chunk_meta: full_acquisition_samples={full_len} "
                f"chunk_index={ci} num_chunks={nc}",
            )
            if tp is not None and nc > 1:
                _, per_chunk = analytic_valid_patches_per_chunk(full_len, W_star, int(tp))
                detail = ", ".join(f"c{cid}:valid_patches={vp}" for cid, vp in enumerate(per_chunk))
                lines.append(f"  chunk_analytic_valid_patches: {detail}")

        msg = "\n".join(lines)
        log.info(msg)
        if write_file:
            file_blocks.append(msg)
        counts[base] = counts.get(base, 0) + 1

    # Write the file on the very first batch of epoch 0 only.
    if write_file and file_blocks:
        csv_dir = _find_csv_log_dir(trainer)
        if csv_dir is not None:
            try:
                csv_dir.mkdir(parents=True, exist_ok=True)
                out_path = csv_dir / "debug_pipeline_epoch0.log"
                epoch_samples_str = _epoch_train_samples_str(trainer)
                header = (
                    "=== debug_pipeline — epoch=0 batch=0 ===\n"
                    f"epoch_train_samples (total all ranks): {epoch_samples_str}\n"
                    + "=" * 50
                )
                out_path.write_text(
                    header + "\n\n" + "\n\n".join(file_blocks) + "\n",
                    encoding="utf-8",
                )
                log.info("debug_pipeline: epoch-0 log written to %s", out_path)
            except Exception as exc:
                log.warning("debug_pipeline: failed to write epoch-0 log: %s", exc)
