"""Sequence pooling strategies for the downstream encoder wrapper.

Reduces an encoder output ``(B, S, E)`` together with a per-token validity
mask ``(B, S)`` to a fixed-size per-channel representation ``(B, out_dim)``.
The width ``out_dim`` depends on the strategy:

- ``mean`` / ``max`` / ``attentive`` -> ``out_dim = E`` (one vector per channel)
- ``flatten`` (binned)               -> ``out_dim = num_bins * E``
- ``flatten_full`` (plain flatten)   -> ``out_dim = num_patches * E``
- ``mlpflatten`` (MLP + flatten)     -> ``out_dim = num_patches * E_out``

``mean``/``max``/``attentive`` collapse the whole depth (token) axis to a single
vector. ``flatten`` instead keeps coarse depth resolution by pooling the
sequence into ``num_bins`` contiguous bins and flattening them — the same trick
the CNN baseline uses (max-pool to a fixed number of depth bins, then flatten),
which preserves *where* in depth a feature occurred instead of averaging it
away. Crucially ``num_bins`` is fixed, so ``out_dim`` does **not** grow with the
runtime sequence length ``S`` (no ``S*E`` blow-up).

``flatten_full`` and ``mlpflatten`` keep the *full* depth resolution by
concatenating **all** tokens (a true flatten, not a binned one). ``flatten_full``
is parameter-free — it concatenates the raw tokens at full width ``E``
(``out_dim = num_patches * E``). ``mlpflatten`` first projects every token
``E -> E_out`` with a shared MLP (``out_dim = num_patches * E_out``), so ``E_out``
keeps the flattened width affordable. Because both make ``out_dim`` scale with the
token count, they require a fixed ``num_patches`` known up front (supplied from
``target_patches`` under fixed-S batching, or set explicitly via
``pooling_config.num_patches`` on a constant-length corpus).

Every pooler respects the validity mask so padded tokens never contribute. The
downstream wrapper reads ``pooling.out_dim`` to size the head, so a strategy that
changes the dimensionality (``flatten``) wires up automatically.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class Pooling(nn.Module):
    """Abstract base: ``(B, S, E)`` + ``valid_mask (B, S)`` -> ``(B, out_dim)``."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)

    @property
    def out_dim(self) -> int:
        """Width of the pooled vector returned per channel (head reads this)."""
        return self.embed_dim

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor,
    ) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class MeanPool(Pooling):
    """Masked mean over the token dimension.

    Padded tokens (``valid_mask == False``) contribute zero to both the
    numerator and the denominator, so the mean is over the *real* tokens only.
    """

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        m = valid_mask.unsqueeze(-1).to(x.dtype)              # (B, S, 1)
        denom = m.sum(dim=1).clamp(min=1.0)                   # (B, 1)
        return (x * m).sum(dim=1) / denom                     # (B, E)


class MaxPool(Pooling):
    """Masked max over the token dimension (control / baseline).

    Padded positions are set to the dtype minimum before the max so they can
    never win. A fully-padded row (never happens in the fixed-T downstream
    loader) would return that minimum; we clamp it back to 0 to stay finite.
    """

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        neg = torch.finfo(x.dtype).min
        xm = x.masked_fill(~valid_mask.unsqueeze(-1), neg)    # (B, S, E)
        out = xm.max(dim=1).values                            # (B, E)
        return out.masked_fill(out == neg, 0.0)


class AttentivePool(Pooling):
    """Attention pooling: one learnable query attends over the tokens.

    A single learnable query ``q`` scores every token and the output is the
    attention-weighted sum of the (projected) token values::

        z = sum_s softmax_s(q . k_s / sqrt(d)) * v_s

    This is a strict generalisation of :class:`MeanPool` (uniform attention
    weights recover the mean): it lets the model concentrate on the few depth
    tokens that carry the discriminative echo and down-weight the rest, instead
    of diluting them in a flat average. Padded tokens are excluded via the
    ``key_padding_mask`` of :class:`torch.nn.MultiheadAttention`.

    ``out_dim = E`` — a drop-in replacement for mean/max pooling.
    """

    def __init__(
        self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0,
    ) -> None:
        super().__init__(embed_dim)
        num_heads = int(num_heads)
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"AttentivePool: embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads}).",
            )
        self.num_heads = num_heads
        # The seed query (ViT-style learnable token); broadcast over the batch.
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        # Pre-norm the keys/values (the encoder latent is already LayerNorm'd,
        # but a dedicated norm keeps the pooling head numerically well-behaved
        # and is the standard recipe for attention-pooling heads).
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        q = self.query.expand(B, -1, -1)                      # (B, 1, E)
        kv = self.kv_norm(x)                                  # (B, S, E)
        key_padding_mask = ~valid_mask                        # (B, S) True = ignore
        # Guard against an all-padded row (softmax over all -inf -> NaN): let it
        # attend to everything. Cannot happen in the fixed-T downstream loader
        # (mask all-True), but keeps the module safe under padding.
        all_masked = key_padding_mask.all(dim=1, keepdim=True)
        key_padding_mask = key_padding_mask & ~all_masked
        out, _ = self.attn(
            q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False,
        )
        return out.squeeze(1)                                 # (B, E)


class BinFlattenPool(Pooling):
    """Adaptive bin pooling + flatten — keeps coarse depth resolution.

    Splits the *valid* token sequence into ``num_bins`` contiguous bins, pools
    each bin (``mean`` or ``max``), and flattens to ``(B, num_bins * E)``. This
    mirrors what the CNN baseline does — pool to a fixed number of depth bins,
    then flatten — instead of collapsing the whole depth axis to one vector,
    so the head still sees *where* in depth each feature lives.

    Because ``num_bins`` is a fixed hyper-parameter, ``out_dim = num_bins * E``
    is **independent of the runtime sequence length ``S``** — no ``S*E`` blow-up
    (e.g. S=100, E=768, C=6 raw-flattened would be 460 800 features; with
    ``num_bins=8`` it is 8*768 = 6 144 per channel). ``num_bins=1`` reduces
    exactly to :class:`MeanPool` (a useful sanity check).

    The bin index of valid token with rank ``r`` (0-based among the valid
    tokens of its sample, total ``n``) is ``floor(r * num_bins / n)``, so the
    binning adapts to each sample's valid length under padding.
    """

    def __init__(self, embed_dim: int, num_bins: int = 8, mode: str = "mean") -> None:
        super().__init__(embed_dim)
        num_bins = int(num_bins)
        if num_bins < 1:
            raise ValueError(f"BinFlattenPool: num_bins must be >= 1, got {num_bins}.")
        if mode not in ("mean", "max"):
            raise ValueError(f"BinFlattenPool: mode must be 'mean'|'max', got {mode!r}.")
        self.num_bins = num_bins
        self.mode = mode

    @property
    def out_dim(self) -> int:
        return self.num_bins * self.embed_dim

    def _bin_id(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """Per-token contiguous bin index in ``[0, num_bins)`` (mask-aware)."""
        m = valid_mask.to(torch.float32)                      # (B, S)
        n = m.sum(dim=1).clamp(min=1.0)                       # (B,)
        rank = torch.cumsum(m, dim=1) - m                     # valid tokens before s
        bin_id = torch.floor(rank * self.num_bins / n.unsqueeze(1))
        bin_id = bin_id.clamp(max=self.num_bins - 1).long()
        return bin_id.masked_fill(~valid_mask, 0)             # park padded in bin 0

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        nb = self.num_bins
        bin_id = self._bin_id(valid_mask)                     # (B, S)
        if self.mode == "mean":
            m = valid_mask.unsqueeze(-1).to(x.dtype)          # (B, S, 1)
            idx = bin_id.unsqueeze(-1).expand(-1, -1, E)      # (B, S, E)
            out = x.new_zeros(B, nb, E).scatter_add_(1, idx, x * m)
            cnt = x.new_zeros(B, nb).scatter_add_(1, bin_id, m.squeeze(-1))
            out = out / cnt.unsqueeze(-1).clamp(min=1.0)
        else:  # max — nb is small, a per-bin loop is clean and dtype-safe
            neg = torch.finfo(x.dtype).min
            xm = x.masked_fill(~valid_mask.unsqueeze(-1), neg)
            bins = []
            for b in range(nb):
                sel = (bin_id == b).unsqueeze(-1)             # (B, S, 1)
                vals = xm.masked_fill(~sel, neg).max(dim=1).values  # (B, E)
                bins.append(vals.masked_fill(vals == neg, 0.0))
            out = torch.stack(bins, dim=1)                    # (B, nb, E)
        return out.reshape(B, nb * E)


# ----------------------------------------------------------------------
# True-flatten helpers (shared by FlattenPool / MLPFlattenPool)
# ----------------------------------------------------------------------

def _require_num_patches(num_patches: Optional[int], pooler: str) -> int:
    """Validate the fixed token count a true-flatten pooler needs up front."""
    if num_patches is None:
        raise ValueError(
            f"{pooler} needs a fixed token count, but none is available: set "
            "data.target_patches (fixed-S batching emits exactly that many "
            "tokens) or pass pooling_config.num_patches explicitly. A true "
            "flatten makes out_dim = num_patches * <width>, which cannot be "
            "sized from a runtime-varying sequence length S.",
        )
    num_patches = int(num_patches)
    if num_patches < 1:
        raise ValueError(f"{pooler}: num_patches must be >= 1, got {num_patches}.")
    return num_patches


def _pad_or_slice_tokens(z: torch.Tensor, num_patches: int) -> torch.Tensor:
    """Normalise the token axis of ``z (B, S, W)`` to exactly ``num_patches``.

    No-op when ``S == num_patches`` (the fixed-S guarantee, and the constant-S
    case of a fixed-length interpolated corpus). Zero-pads a shorter sequence
    and slices a longer one so a true-flatten ``out_dim`` stays static.
    """
    B, S, W = z.shape
    if S == num_patches:
        return z
    if S > num_patches:
        return z[:, :num_patches]
    pad = z.new_zeros(B, num_patches - S, W)
    return torch.cat([z, pad], dim=1)


class FlattenPool(Pooling):
    """Plain, parameter-free **true** flatten over the depth axis.

    Concatenates ALL encoder tokens ``(B, S, E)`` into a single ``(B, S * E)``
    vector — no per-token projection (cf. :class:`MLPFlattenPool`) and no binning
    (cf. :class:`BinFlattenPool`). The raw encoder features at full width ``E``
    go straight into the head, keeping the full per-token depth resolution.

    Like :class:`MLPFlattenPool`, a true flatten makes ``out_dim = num_patches *
    E`` scale with the token count, so it needs a FIXED ``num_patches`` known at
    construction time (injected from ``target_patches`` under fixed-S batching;
    or set ``pooling_config.num_patches`` to run variable-S on a constant-length
    corpus). The incoming sequence is normalised to exactly ``num_patches``
    tokens (zero-padded / sliced) so ``out_dim`` stays static. Padded tokens are
    zeroed so each output slot maps to a fixed token position.
    """

    def __init__(self, embed_dim: int, num_patches: Optional[int]) -> None:
        super().__init__(embed_dim)
        self.num_patches = _require_num_patches(num_patches, "FlattenPool")

    @property
    def out_dim(self) -> int:
        return self.num_patches * self.embed_dim

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        z = x * valid_mask.unsqueeze(-1).to(x.dtype)          # zero padded tokens
        z = _pad_or_slice_tokens(z, self.num_patches)         # (B, num_patches, E)
        return z.reshape(z.size(0), self.num_patches * self.embed_dim)


class MLPFlattenPool(Pooling):
    """Token-wise shared MLP reduction + **true** flatten over the depth axis.

    Each of the ``S`` encoder tokens ``(B, S, E)`` is projected independently by
    a *shared* MLP ``E -> E_out`` (the SAME weights at every token position and
    across the batch), then the whole token axis is concatenated into a single
    ``(B, S * E_out)`` vector. Unlike :class:`BinFlattenPool` — which first pools
    the sequence into a fixed number of coarse bins — this keeps the FULL
    per-token depth resolution; ``E_out < E`` is what keeps the flattened width
    affordable (e.g. S=50, E=768 would flatten to 38 400 per channel, but with
    ``E_out=32`` it is 50*32 = 1 600).

    Because a true flatten makes ``out_dim = num_patches * E_out`` scale with the
    token count, this pooler needs a FIXED ``num_patches`` known at construction
    time so the downstream head can size itself. It is supplied by the wrapper
    from ``target_patches`` (fixed-S batching emits exactly ``target_patches``
    tokens per sample); set ``pooling_config.num_patches`` explicitly to run it
    variable-S on a constant-length corpus. At forward time the incoming
    sequence is normalised to exactly ``num_patches`` tokens (zero-padded if
    shorter, sliced if longer) so ``out_dim`` stays static regardless of the
    batch's runtime ``S``.

    Padded tokens (``valid_mask == False``) are zeroed *after* the MLP, so each
    output slot maps to a fixed token position and padding never injects the
    encoder's learned ``pad_token`` value into the flattened vector.
    """

    def __init__(
        self,
        embed_dim: int,
        num_patches: Optional[int],
        e_out: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(embed_dim)
        num_patches = _require_num_patches(num_patches, "MLPFlattenPool")
        e_out = int(e_out)
        if e_out < 1:
            raise ValueError(f"MLPFlattenPool: e_out must be >= 1, got {e_out}.")
        if num_layers < 1:
            raise ValueError(
                f"MLPFlattenPool: num_layers must be >= 1, got {num_layers}.",
            )
        self.num_patches = num_patches
        self.e_out = e_out

        # Shared token MLP E -> E_out. nn.Linear / nn.Sequential operate on the
        # last dim, so the same weights are broadcast over every (batch, token).
        if num_layers == 1:
            self.mlp: nn.Module = nn.Linear(embed_dim, e_out)
        else:
            h = int(hidden_dim) if hidden_dim is not None else embed_dim
            layers: list[nn.Module] = [
                nn.Linear(embed_dim, h), nn.GELU(), nn.Dropout(dropout),
            ]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout)]
            layers.append(nn.Linear(h, e_out))
            self.mlp = nn.Sequential(*layers)

    @property
    def out_dim(self) -> int:
        return self.num_patches * self.e_out

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        z = self.mlp(x)                                       # (B, S, E_out)
        z = z * valid_mask.unsqueeze(-1).to(z.dtype)          # zero padded tokens
        z = _pad_or_slice_tokens(z, self.num_patches)         # (B, num_patches, E_out)
        return z.reshape(z.size(0), self.num_patches * self.e_out)


def build_pooling(
    pooling_type: str, embed_dim: int, config: Optional[dict] = None,
) -> Pooling:
    """Factory for pooling modules.

    Parameters
    ----------
    pooling_type :
        ``"mean"`` (default), ``"max"``, ``"attentive"``, ``"flatten"``
        (binned), ``"flatten_full"`` (plain true flatten), or ``"mlpflatten"``.
    embed_dim :
        Token embedding dim ``E``.
    config :
        Optional per-strategy knobs (ignored by mean/max):
        - attentive:    ``num_heads`` (default 1), ``dropout`` (default 0.0)
        - flatten:      ``num_bins`` (default 8), ``mode`` ('mean'|'max', default 'mean')
        - flatten_full: ``num_patches`` (token count to flatten to — injected from
          ``target_patches`` by the encoder wrapper; required). No other knobs:
          a parameter-free concat of all tokens, ``out_dim = num_patches * E``.
        - mlpflatten:   ``e_out`` (required, per-token reduced dim), ``num_patches``
          (as for flatten_full; required), ``hidden_dim`` / ``num_layers`` (default
          1 => single Linear) / ``dropout`` for a deeper shared MLP.
    """
    cfg = dict(config or {})
    pt = str(pooling_type).lower()
    if pt == "mean":
        return MeanPool(embed_dim)
    if pt == "max":
        return MaxPool(embed_dim)
    if pt == "attentive":
        return AttentivePool(
            embed_dim,
            num_heads=int(cfg.get("num_heads", 1)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if pt in ("flatten", "binflatten", "bins"):
        return BinFlattenPool(
            embed_dim,
            num_bins=int(cfg.get("num_bins", 8)),
            mode=str(cfg.get("mode", "mean")),
        )
    if pt in ("flatten_full", "rawflatten", "trueflatten", "flattenfull"):
        return FlattenPool(embed_dim, num_patches=cfg.get("num_patches"))
    if pt in ("mlpflatten", "mlp_flatten", "mlpflat"):
        e_out = cfg.get("e_out")
        if e_out is None:
            raise ValueError(
                "pooling_type='mlpflatten' requires pooling_config.e_out "
                "(the per-token reduced dim; out_dim = num_patches * e_out).",
            )
        return MLPFlattenPool(
            embed_dim,
            num_patches=cfg.get("num_patches"),
            e_out=int(e_out),
            hidden_dim=cfg.get("hidden_dim"),
            num_layers=int(cfg.get("num_layers", 1)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    raise ValueError(
        f"Unknown pooling_type {pooling_type!r}; expected one of "
        f"'mean', 'max', 'attentive', 'flatten', 'flatten_full', 'mlpflatten'.",
    )


__all__ = [
    "Pooling",
    "MeanPool",
    "MaxPool",
    "AttentivePool",
    "BinFlattenPool",
    "FlattenPool",
    "MLPFlattenPool",
    "build_pooling",
]
