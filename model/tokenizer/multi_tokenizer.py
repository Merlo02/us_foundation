"""Multi-frequency tokenizer for ultrasound A-mode signals.

Two configurable branch types behind a common ``TokenizerBranch`` API:

- :class:`MLPBranch` — MOIRAI-style shared-weight ``MultiInSizeLinear``
  (a single ``(num_W, E, max_W)`` tensor masked per branch). Works on
  **flattened non-overlapping patches** of the input signal.
- :class:`CNNBranch` — three independent ``Conv1d`` modules (one per
  window size) with configurable ``kernel_size``, ``stride``, ``groups``,
  ``padding`` and ``bias``. Defaults to non-overlapping patches
  (``kernel_size = stride = W``) so the spatial interpretation matches
  the MLP branch exactly.

Routing between branches is determined by :func:`select_branch`, which
minimises ``|W·c/(2·fs) − target_mm|`` over the configured window sizes
so every sample lands on the branch whose patch depth is closest to the
chosen physical target (default 0.6 mm).

Output is a :class:`TokenizerOutput` dataclass containing:

- ``tokens``          ``(B, S_max, E)``
- ``padding_mask``    ``(B, S_max)``  — ``True`` on valid tokens
- ``window_size``     ``(B,)``        — ``W*`` picked for each sample
- ``patch_timestamps_us`` ``(B, S_max)`` — midpoint timestamps of each patch
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SPEED_OF_SOUND_M_S = 1_540.0  # axial sound speed in tissue, m/s
SPEED_OF_SOUND_MM_S = 1_540_000.0  # mm/s (= 1.54 mm/µs)


# ======================================================================
# Public routing helper
# ======================================================================

def select_branch(
    fs_hz: float, window_sizes: Sequence[int], target_mm: float = 0.6
) -> int:
    """Select ``W*`` from *window_sizes* that best approximates *target_mm*.

    Physical patch depth for a signal with sampling frequency ``fs`` is::

        depth(W) = W · c / (2 · fs)          [mm, with c = 1540 m/s]

    Returns the ``W`` minimising ``|depth(W) − target_mm|``.
    """
    if fs_hz is None or fs_hz <= 0 or not window_sizes:
        return int(window_sizes[0])
    return int(min(
        window_sizes,
        key=lambda W: abs(W * SPEED_OF_SOUND_MM_S / (2.0 * float(fs_hz)) - target_mm),
    ))


# ======================================================================
# Output container
# ======================================================================

@dataclass
class TokenizerOutput:
    tokens: torch.Tensor                # (B, S_max, E)
    padding_mask: torch.Tensor          # (B, S_max) — True on valid tokens
    window_size: torch.Tensor           # (B,)       — W* per sample
    patch_timestamps_us: torch.Tensor   # (B, S_max) — midpoint timestamps
    sampling_frequency_hz: torch.Tensor # (B,)


# ======================================================================
# MLPBranch — MOIRAI-style MultiInSizeLinear
# ======================================================================

class MLPBranch(nn.Module):
    """Shared-weight multi-input-size linear projection.

    Implements MOIRAI's :class:`MultiInSizeLinear` inlined (no external
    dependency on ``uni2ts``). The weight tensor has shape
    ``(num_W, E, max_W)`` and each branch ``i`` uses only the first
    ``window_sizes[i]`` columns; remaining columns are masked out.

    Forward expects flattened non-overlapping patches
    ``x: (B, S, max_W)`` together with a ``branch_idx: (B,)`` tensor
    indicating which ``W`` each sample uses. The masking makes unused
    positions contribute zero regardless of their numerical value.
    """

    def __init__(
        self,
        window_sizes: Sequence[int],
        embed_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.embed_dim = int(embed_dim)
        self.max_w = max(self.window_sizes)
        n = len(self.window_sizes)

        self.weight = nn.Parameter(torch.empty(n, embed_dim, self.max_w))
        self.bias = nn.Parameter(torch.empty(n, embed_dim)) if bias else None

        # mask[i, 0, j] = 1 iff j < window_sizes[i]
        mask = torch.zeros(n, 1, self.max_w)
        for i, w in enumerate(self.window_sizes):
            mask[i, 0, :w] = 1.0
        self.register_buffer("mask", mask, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i, w in enumerate(self.window_sizes):
            nn.init.kaiming_uniform_(self.weight[i, :, :w], a=math.sqrt(5))
            nn.init.zeros_(self.weight[i, :, w:])
            if self.bias is not None:
                fan_in = w
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(
        self,
        patches: torch.Tensor,   # (B, S, max_W)
        branch_idx: torch.Tensor,  # (B,) long
    ) -> torch.Tensor:
        """Return ``(B, S, E)`` — sum over branches gated by one-hot ``branch_idx``."""
        B = patches.size(0)
        out = patches.new_zeros(B, patches.size(1), self.embed_dim)
        for i in range(len(self.window_sizes)):
            w_i = self.weight[i] * self.mask[i]           # (E, max_W)
            proj = F.linear(
                patches, w_i,
                self.bias[i] if self.bias is not None else None,
            )  # (B, S, E)
            sel = (branch_idx == i).view(B, 1, 1).to(proj.dtype)
            out = out + sel * proj
        return out


# ======================================================================
# CNNBranch — one Conv1d per window size
# ======================================================================

class CNNBranch(nn.Module):
    """Configurable Conv1d bank, one module per window size.

    The common case is non-overlapping patches
    (``kernel_size = stride = window_size``), which is exactly equivalent
    to the MLP branch up to the choice of weight initialisation. Arbitrary
    strides, paddings and ``groups`` are supported for experimentation.
    """

    def __init__(
        self,
        window_sizes: Sequence[int],
        embed_dim: int,
        kernel_size: Optional[int] = None,
        stride: Optional[int] = None,
        groups: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.embed_dim = int(embed_dim)

        convs = []
        for w in self.window_sizes:
            k = int(kernel_size) if kernel_size is not None else int(w)
            s = int(stride) if stride is not None else int(w)
            convs.append(nn.Conv1d(
                in_channels=1,
                out_channels=embed_dim,
                kernel_size=k,
                stride=s,
                padding=int(padding),
                groups=int(groups),
                bias=bool(bias),
            ))
        self.convs = nn.ModuleList(convs)

    def forward(
        self,
        signal: torch.Tensor,       # (B, T) — already padded to common T
        signal_mask: torch.Tensor,  # (B, T) — True on valid signal samples
        branch_idx: torch.Tensor,   # (B,)   — which W*
        s_max_override: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(tokens (B, S_max, E), padding_mask (B, S_max))``.

        If ``s_max_override`` is given, the output sequence length is fixed
        to that value (fixed-S mode). Per-sample valid patches are clipped
        to it, and samples with fewer valid patches remain zero-padded in
        the leftover positions.
        """
        B, T = signal.shape
        device = signal.device

        # Run every branch on the entire batch and then gather the right slice
        # per sample. This keeps the forward graph differentiable and avoids
        # per-sample Python loops over the conv weights.
        branch_tokens = []
        branch_S = []
        for conv in self.convs:
            x = conv(signal.unsqueeze(1))      # (B, E, S_i)
            branch_tokens.append(x.transpose(1, 2))
            branch_S.append(x.size(-1))

        S_max = int(s_max_override) if s_max_override is not None else max(branch_S)
        tokens = signal.new_zeros(B, S_max, self.embed_dim)
        padding_mask = torch.zeros(B, S_max, dtype=torch.bool, device=device)

        lengths = signal_mask.sum(dim=1).long()  # (B,)
        for b in range(B):
            i = int(branch_idx[b].item())
            w = self.window_sizes[i]
            # Number of *valid* non-overlapping patches before padding.
            n_valid = int(lengths[b].item()) // w
            n_valid = min(n_valid, branch_S[i], S_max)
            if n_valid > 0:
                tokens[b, :n_valid] = branch_tokens[i][b, :n_valid]
                padding_mask[b, :n_valid] = True
        return tokens, padding_mask


# ======================================================================
# MultiTokenizer
# ======================================================================

class MultiTokenizer(nn.Module):
    """Front-end tokenizer with per-sample frequency routing.

    Parameters
    ----------
    window_sizes :
        Non-overlapping patch sizes, e.g. ``(8, 16, 32)``. The best match for
        each sample is picked by :func:`select_branch`.
    embed_dim :
        Output token embedding dimension ``E``.
    target_patch_mm :
        Target physical patch depth in millimetres. Default 0.6 mm.
    tokenizer_type :
        ``"mlp"`` → :class:`MLPBranch`; ``"cnn"`` → :class:`CNNBranch`.
    cnn_config :
        Forwarded to :class:`CNNBranch` when ``tokenizer_type="cnn"``.
    """

    def __init__(
        self,
        window_sizes: Sequence[int] = (8, 16, 32),
        embed_dim: int = 256,
        target_patch_mm: float = 0.6,
        tokenizer_type: str = "mlp",
        cnn_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if len(window_sizes) == 0:
            raise ValueError("window_sizes must have at least one entry")
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.embed_dim = int(embed_dim)
        self.target_patch_mm = float(target_patch_mm)
        self.tokenizer_type = tokenizer_type.lower()

        if self.tokenizer_type == "mlp":
            self.branch: nn.Module = MLPBranch(self.window_sizes, embed_dim)
        elif self.tokenizer_type == "cnn":
            cfg = dict(cnn_config or {})
            self.branch = CNNBranch(
                self.window_sizes,
                embed_dim,
                kernel_size=cfg.get("kernel_size"),
                stride=cfg.get("stride"),
                groups=cfg.get("groups", 1),
                padding=cfg.get("padding", 0),
                bias=cfg.get("bias", True),
            )
        else:
            raise ValueError(
                f"Unknown tokenizer_type {tokenizer_type!r}; expected 'mlp' or 'cnn'"
            )

        # Pre-compute window-sizes-as-tensor for fast routing lookup.
        self.register_buffer(
            "_window_sizes_buffer",
            torch.tensor(self.window_sizes, dtype=torch.long),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Routing (CPU, differentiable-independent)
    # ------------------------------------------------------------------
    def route_batch(self, fs_hz: torch.Tensor) -> torch.Tensor:
        """Return ``branch_idx: (B,) long`` for a batch of sampling frequencies."""
        if fs_hz.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=fs_hz.device)

        fs = fs_hz.clamp_min(1.0)  # avoid div-by-zero
        # depth_mm[b, i] = W_i * c / (2 * fs[b])
        W = self._window_sizes_buffer.to(fs.device).float()  # (num_W,)
        depths = W.unsqueeze(0) * SPEED_OF_SOUND_MM_S / (2.0 * fs.unsqueeze(1))
        return torch.argmin((depths - self.target_patch_mm).abs(), dim=1).long()

    # ------------------------------------------------------------------
    # MLP path: flatten valid non-overlapping patches into (B, S_max, max_W)
    # ------------------------------------------------------------------
    def _flatten_for_mlp(
        self,
        signal: torch.Tensor,      # (B, T)
        signal_mask: torch.Tensor,  # (B, T) bool
        branch_idx: torch.Tensor,   # (B,) long
        s_max_override: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the ``(B, S_max, max_W)`` patch tensor expected by MLPBranch.

        In variable-S mode ``S_max = max_b(length_b // W_b)``.
        In fixed-S mode (``s_max_override`` given) ``S_max`` is clamped to
        the requested value; per-sample valid patches are bounded by
        ``min(length_b // W_b, s_max_override)`` so short or partial
        chunks contribute only the real patches and the rest stays zero.
        """
        B, T = signal.shape
        max_W = max(self.window_sizes)
        lengths = signal_mask.sum(dim=1).long()  # (B,)

        # Per-sample valid patches, optionally clipped to the fixed S cap.
        per_sample_S = []
        for b in range(B):
            W = self.window_sizes[int(branch_idx[b].item())]
            s_native = int(lengths[b].item()) // W
            if s_max_override is not None:
                s_native = min(s_native, int(s_max_override))
            per_sample_S.append(s_native)
        if s_max_override is not None:
            S_max = int(s_max_override)
        else:
            S_max = max(per_sample_S) if per_sample_S else 0

        patches = signal.new_zeros(B, S_max, max_W)
        padding_mask = torch.zeros(
            B, S_max, dtype=torch.bool, device=signal.device,
        )
        for b in range(B):
            W = self.window_sizes[int(branch_idx[b].item())]
            S_b = per_sample_S[b]
            if S_b == 0:
                continue
            chunk = signal[b, : S_b * W].view(S_b, W)
            patches[b, :S_b, :W] = chunk
            padding_mask[b, :S_b] = True
        return patches, padding_mask

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------
    def _compute_timestamps(
        self,
        fs_hz: torch.Tensor,       # (B,)
        branch_idx: torch.Tensor,   # (B,)
        s_max: int,
    ) -> torch.Tensor:
        """Midpoint timestamps (µs) ``t_i = (i·W + W/2) / fs * 1e6``."""
        B = fs_hz.size(0)
        device = fs_hz.device
        Ws = self._window_sizes_buffer.to(device)
        W_per_sample = Ws[branch_idx].float()           # (B,)
        i_idx = torch.arange(s_max, device=device).float()  # (S_max,)
        # (B, S_max)
        fs_safe = fs_hz.clamp_min(1.0).float()
        ts_us = (i_idx.unsqueeze(0) * W_per_sample.unsqueeze(1)
                 + W_per_sample.unsqueeze(1) * 0.5) / fs_safe.unsqueeze(1) * 1e6
        return ts_us

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        signal: torch.Tensor,        # (B, T) padded
        signal_mask: torch.Tensor,   # (B, T) bool
        sampling_frequency_hz: torch.Tensor,  # (B,)
        window_size_override: Optional[torch.Tensor] = None,  # (B,) optional W*
        fixed_num_patches: Optional[int] = None,
    ) -> TokenizerOutput:
        """Tokenize a batch of variable-length signals.

        ``window_size_override`` lets the DataModule pass the pre-computed
        ``W*`` used to build ``patch_timestamps_us`` so routing is fully
        consistent with the loader (recommended). If ``None``, the tokenizer
        re-routes from the sampling frequencies.

        ``fixed_num_patches`` activates the fixed-S mode: the output tokens
        tensor has shape ``(B, fixed_num_patches, E)`` regardless of the
        per-sample signal length. This requires the loader to have already
        sliced each signal to at most ``fixed_num_patches · W*`` samples
        (see ``HDF5Dataset`` / ``WebDatasetDataModule`` with
        ``target_patches`` set).
        """
        device = signal.device
        fs = sampling_frequency_hz.to(device).float()

        if window_size_override is not None:
            Ws = self._window_sizes_buffer.to(device)
            branch_idx = (window_size_override.to(device).unsqueeze(1) == Ws).long().argmax(dim=1)
        else:
            branch_idx = self.route_batch(fs)

        if isinstance(self.branch, MLPBranch):
            patches, padding_mask = self._flatten_for_mlp(
                signal, signal_mask, branch_idx,
                s_max_override=fixed_num_patches,
            )
            tokens = self.branch(patches, branch_idx)
        else:
            tokens, padding_mask = self.branch(
                signal, signal_mask, branch_idx,
                s_max_override=fixed_num_patches,
            )

        s_max = tokens.size(1)
        timestamps = self._compute_timestamps(fs, branch_idx, s_max)
        # Zero-out timestamps on padded positions.
        timestamps = timestamps * padding_mask.to(timestamps.dtype)

        W_per_sample = self._window_sizes_buffer.to(device)[branch_idx]
        return TokenizerOutput(
            tokens=tokens,
            padding_mask=padding_mask,
            window_size=W_per_sample,
            patch_timestamps_us=timestamps,
            sampling_frequency_hz=fs,
        )
