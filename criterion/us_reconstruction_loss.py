"""
Masked reconstruction criterion for the Ultrasound MAE.

Inspired by:
  - BioFoundation PretrainCriterion  (criterion/pretrain_criterion.py)
  - TimeFM MaskedReconstructionLoss  (criterion/masked_reconstruction_loss.py)

Key differences from the reference repos:
  - Operates on variable-length, variable-window-size signals: the decoder
    produces `pred (B, S, W_max)` and each sample has its own `window_size`
    and `valid_len` (number of non-padding tokens).
  - Supports per-patch normalisation in the target (MAE He et al. 2021 style).
  - alpha > 0 also penalises visible-patch reconstruction (like TimeFM).
  - Separates the criterion from the decoder so it can be swapped independently.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class USReconstructionLoss(nn.Module):
    """Masked-patch reconstruction loss for the Ultrasound MAE.

    The decoder outputs ``pred (B, S, W_max)`` where each sample uses only
    its own ``window_size`` W* columns of the last dimension.  Targets are
    the original patchified signal values, per-patch normalised (zero mean,
    unit std) to match the MAE training objective.

    Args:
        loss_type: Base element-wise loss. One of ``"l1"``, ``"l2"``,
            ``"smooth_l1"`` (Huber).
        alpha: Weight for the *visible*-patch auxiliary loss.  When ``alpha=0``
            only masked patches contribute to the gradient (pure MAE).  A
            small positive value (e.g. ``0.1``) regularises the encoder to
            maintain visible-patch reconstruction, as in TimeFM.
        norm_target: If ``True`` (default), normalise each target patch to
            zero mean / unit std before computing the loss (MAE He et al.
            2021).  Helps with signals whose amplitude varies across datasets.
        norm_eps: Epsilon added to std denominator.
    """

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        alpha: float = 0.0,
        norm_target: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if loss_type not in ("l1", "l2", "smooth_l1"):
            raise ValueError(
                f"Invalid loss_type '{loss_type}'. Choose 'l1', 'l2', or 'smooth_l1'."
            )
        self.loss_type = loss_type
        self.alpha = alpha
        self.norm_target = norm_target
        self.norm_eps = norm_eps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _elem_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Element-wise loss without reduction, shape == pred.shape."""
        if self.loss_type == "l1":
            return F.l1_loss(pred, target, reduction="none")
        elif self.loss_type == "l2":
            return F.mse_loss(pred, target, reduction="none")
        else:  # smooth_l1
            return F.smooth_l1_loss(pred, target, reduction="none")

    @staticmethod
    def _patchify(signal: torch.Tensor, window_size: int) -> torch.Tensor:
        """Split a 1-D signal (T,) into non-overlapping patches (S, W).

        If ``T`` is not divisible by ``W``, the tail is discarded.
        """
        T = signal.shape[-1]
        n_patches = T // window_size
        return signal[..., : n_patches * window_size].reshape(
            *signal.shape[:-1], n_patches, window_size
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        signal: torch.Tensor,
        mask: torch.Tensor,
        window_sizes: torch.Tensor,
        padding_mask: torch.Tensor,
        signal_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute masked (and optionally visible) reconstruction loss.

        Args:
            pred: ``(B, S, W_max)`` — decoder output (padded to the maximum
                window size in the batch).
            signal: ``(B, T_max)`` — padded input signals (float32).
            mask: ``(B, S)`` — binary token mask; **1 = masked**, 0 = visible.
            window_sizes: ``(B,)`` — per-sample window size W*.
            padding_mask: ``(B, S)`` — **1 = real token**, 0 = padding.
            signal_lengths: ``(B,)`` — number of real samples in each signal.

        Returns:
            Dict with keys:
              - ``"loss"`` — scalar training loss.
              - ``"masked_loss"`` — loss on masked (reconstructed) patches.
              - ``"visible_loss"`` — loss on visible patches (0 if alpha=0).
        """
        B, S, W_max = pred.shape
        device = pred.device

        total_masked = pred.new_zeros(1)
        total_visible = pred.new_zeros(1)
        n_masked = pred.new_zeros(1)
        n_visible = pred.new_zeros(1)

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

            pidx = torch.arange(max_p, device=device).unsqueeze(0)
            valid = pidx < n_patches.unsqueeze(1)

            if self.norm_target:
                mu = target.mean(dim=-1, keepdim=True)
                std = target.std(dim=-1, keepdim=True).clamp(min=self.norm_eps)
                normed = (target - mu) / std
                target = torch.where(valid.unsqueeze(-1), normed, target.new_zeros(1))

            sub_pred_w = sub_pred[:, :max_p, :W]
            ploss = self._elem_loss(sub_pred_w, target).mean(dim=-1)

            m = sub_mask[:, :max_p].bool()
            p = sub_pad[:, :max_p].bool()

            rm = m & p & valid
            rv = (~m) & p & valid

            if rm.any():
                total_masked = total_masked + ploss[rm].sum()
                n_masked = n_masked + rm.float().sum()

            if rv.any():
                total_visible = total_visible + ploss[rv].sum()
                n_visible = n_visible + rv.float().sum()

        masked_loss = total_masked / n_masked.clamp(min=1)
        visible_loss = total_visible / n_visible.clamp(min=1)
        loss = masked_loss + self.alpha * visible_loss

        return {
            "loss": loss,
            "masked_loss": masked_loss.detach(),
            "visible_loss": visible_loss.detach(),
        }
