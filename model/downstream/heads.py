"""Swappable prediction heads for downstream tasks.

Every head receives the per-channel pooled embeddings ``(B, C, E)`` produced by
:class:`~model.downstream.encoder_wrapper.UltrasonicEncoderWrapper` and returns
logits / values of shape ``(B, out_dim)``. They differ only in how the ``C``
channel embeddings are *fused* before the final projection MLP:

- ``concat`` — flatten ``(B, C, E) -> (B, C*E)`` then MLP (the legacy behaviour).
  Realised by :class:`ClassificationHead` / :class:`RegressionHead`, which also
  carry the task semantics (output dim + typical loss).
- ``posenc`` (:class:`PosEncConcatHead`) — add a *learnable* per-channel
  positional encoding ``P_i`` to each ``E_i``, then concat ``(E_i + P_i)`` and MLP.
- ``crossattention`` (:class:`ChannelCrossAttentionHead`) — add the same ``P_i``,
  treat the channels as tokens and let a single learnable query cross-attend over
  them, yielding one ``(B, E)`` vector that feeds the MLP.

The output dimension and typical loss are selected by the *task*:

- classification -> logits ``(B, num_classes)`` (CrossEntropy)
- regression     -> values ``(B, num_outputs)`` (MSE)

By default ``num_layers=1`` produces a single ``nn.Linear`` (the canonical
linear-probe MLP). ``num_layers > 1`` builds a small MLP with GELU and dropout
between layers.

``head_type`` (see :func:`normalize_head_type`) selects fusion + task: either a
legacy string (``"classification"`` / ``"regression"`` => ``concat`` fusion) or a
dict ``{"type": <fusion>, "task": <task>, ...fusion kwargs...}``.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: Optional[int],
    num_layers: int,
    dropout: float,
) -> nn.Module:
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if num_layers == 1:
        return nn.Linear(in_dim, out_dim)

    h = int(hidden_dim) if hidden_dim is not None else in_dim
    layers: list[nn.Module] = [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
    for _ in range(num_layers - 2):
        layers += [nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(h, out_dim))
    return nn.Sequential(*layers)


class ClassificationHead(nn.Module):
    """Classifier head producing un-normalised logits."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.net = _build_mlp(in_dim, num_classes, hidden_dim, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either the fused flat vector ``(B, C*E)`` or the raw
        # per-channel embeddings ``(B, C, E)`` — the concat fusion flattens.
        if x.ndim == 3:
            x = x.flatten(1)
        return self.net(x)


class RegressionHead(nn.Module):
    """Regression head producing real-valued outputs."""

    def __init__(
        self,
        in_dim: int,
        num_outputs: int = 1,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_outputs = int(num_outputs)
        self.net = _build_mlp(in_dim, num_outputs, hidden_dim, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either the fused flat vector ``(B, C*E)`` or the raw
        # per-channel embeddings ``(B, C, E)`` — the concat fusion flattens.
        if x.ndim == 3:
            x = x.flatten(1)
        return self.net(x)


class ChannelPosEnc(nn.Module):
    """Learnable per-channel positional encoding.

    Holds one ``E``-dim vector per channel (``nn.Parameter`` of shape
    ``(1, C, E)``) and adds it to the per-channel embeddings ``(B, C, E)``. The
    channels of A-mode ultrasound correspond to fixed physical sensor positions,
    so a small learnable table (ViT-style) is the natural choice — a
    sinusoidal/NeRF scheme would need real spatial coordinates we do not have.
    """

    def __init__(self, num_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, int(num_channels), int(embed_dim)))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos


class PosEncConcatHead(nn.Module):
    """Per-channel learnable PE + flatten + MLP.

    Adds :class:`ChannelPosEnc` to the per-channel embeddings ``(B, C, E)``,
    then flattens to ``(B, C*E)`` and applies the task MLP — i.e. the legacy
    ``concat`` fusion with channel-position information injected first.
    """

    def __init__(
        self,
        num_channels: int,
        embed_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.pos_enc = ChannelPosEnc(num_channels, embed_dim)
        in_dim = int(num_channels) * int(embed_dim)
        self.net = _build_mlp(in_dim, int(out_dim), hidden_dim, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)                 # (B, C, E)
        return self.net(x.flatten(1))       # (B, out_dim)


class ChannelCrossAttentionHead(nn.Module):
    """Single-query cross-attention over the channel tokens, then MLP.

    Adds the learnable per-channel PE, then a single (by default) learnable
    query cross-attends over the ``C`` channel tokens ``(B, C, E)`` to produce
    one ``(B, E)`` aggregate which feeds the task MLP. A lightweight,
    pre-norm cross-attention block (residual attention + residual FFN, no query
    self-attention) translated from LUNA's ``CrossAttentionBlock`` /
    ``ClassificationHeadWithQueries`` to the downstream setting.
    """

    def __init__(
        self,
        num_channels: int,
        embed_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        num_attention_heads: int = 4,
        num_queries: int = 1,
        ff_ratio: float = 2.0,
        attn_dropout: float = 0.0,
        use_pos_enc: bool = True,
    ) -> None:
        super().__init__()
        embed_dim = int(embed_dim)
        num_attention_heads = int(num_attention_heads)
        if embed_dim % num_attention_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_attention_heads ({num_attention_heads}).",
            )
        self.num_queries = int(num_queries)
        self.use_pos_enc = bool(use_pos_enc)
        if self.use_pos_enc:
            self.pos_enc = ChannelPosEnc(num_channels, embed_dim)

        self.query = nn.Parameter(torch.zeros(1, self.num_queries, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_attention_heads, dropout=attn_dropout, batch_first=True,
        )

        ff_hidden = max(1, int(embed_dim * float(ff_ratio)))
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim),
        )

        self.net = _build_mlp(
            embed_dim * self.num_queries, int(out_dim), hidden_dim, num_layers, dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, E)
        B = x.size(0)
        if self.use_pos_enc:
            x = self.pos_enc(x)
        q = self.query.expand(B, -1, -1)                       # (B, Q, E)
        attn_out, _ = self.attn(self.q_norm(q), self.kv_norm(x), self.kv_norm(x))
        q = q + attn_out                                       # residual attention
        q = q + self.ffn(self.ff_norm(q))                      # residual FFN
        pooled = q.reshape(B, -1)                              # (B, Q*E); Q=1 => (B, E)
        return self.net(pooled)                                # (B, out_dim)


# ----------------------------------------------------------------------
# head_type parsing + factory
# ----------------------------------------------------------------------

_FUSION_TYPES = ("concat", "posenc", "crossattention")
_TASK_TYPES = ("classification", "regression")


def normalize_head_type(head_type: Union[str, dict]) -> dict:
    """Resolve ``head_type`` into ``{"fusion", "task", "kwargs"}``.

    Accepts either:

    - a **string** ``"classification"`` / ``"regression"`` — the legacy form,
      mapped to the ``concat`` fusion with that task; or
    - a **dict** ``{"type": <fusion>, "task": <task>, ...fusion kwargs...}`` where
      ``type`` defaults to ``"concat"`` and ``task`` to ``"classification"``. Any
      remaining keys are returned in ``kwargs`` for the selected fusion head.
    """
    if isinstance(head_type, str):
        task = head_type.lower()
        if task not in _TASK_TYPES:
            raise ValueError(
                f"Unknown head_type {head_type!r}; expected one of {_TASK_TYPES} "
                f"(string form) or a dict with a 'type' field.",
            )
        return {"fusion": "concat", "task": task, "kwargs": {}}
    if isinstance(head_type, dict):
        d = dict(head_type)
        fusion = str(d.pop("type", "concat")).lower()
        task = str(d.pop("task", "classification")).lower()
        if fusion not in _FUSION_TYPES:
            raise ValueError(
                f"Unknown head_type.type {fusion!r}; expected one of {_FUSION_TYPES}.",
            )
        if task not in _TASK_TYPES:
            raise ValueError(
                f"Unknown head_type.task {task!r}; expected one of {_TASK_TYPES}.",
            )
        return {"fusion": fusion, "task": task, "kwargs": d}
    raise TypeError(
        f"head_type must be a str or dict, got {type(head_type).__name__}.",
    )


def build_head(
    head_type: Union[str, dict],
    embed_dim: int,
    num_channels: int,
    num_classes: Optional[int] = None,
    num_outputs: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.0,
    num_layers: int = 1,
) -> nn.Module:
    """Factory for downstream heads.

    Parameters
    ----------
    head_type :
        Fusion + task selector — a legacy string (``"classification"`` /
        ``"regression"`` => ``concat`` fusion) or a dict
        ``{"type": concat|posenc|crossattention, "task": classification|regression,
        ...fusion kwargs...}``. See :func:`normalize_head_type`.
    embed_dim :
        Per-channel embedding dim ``E`` (encoder ``out_dim``).
    num_channels :
        Channel count ``C``. The ``concat`` / ``posenc`` MLP sees ``C * E``;
        ``crossattention`` sees ``E`` (the aggregated query output).
    num_classes :
        Required when the task is ``classification``.
    num_outputs :
        Required when the task is ``regression`` (default 1 if unset).
    hidden_dim, dropout, num_layers :
        Final-MLP shape, shared by every fusion. ``num_layers == 1`` collapses
        to a single ``nn.Linear``.
    """
    spec = normalize_head_type(head_type)
    fusion, task, kwargs = spec["fusion"], spec["task"], spec["kwargs"]

    embed_dim = int(embed_dim)
    num_channels = int(num_channels)

    if task == "classification":
        if num_classes is None:
            raise ValueError("classification head requires num_classes")
        out_dim = int(num_classes)
    else:  # regression
        out_dim = int(num_outputs) if num_outputs is not None else 1

    if fusion == "concat":
        in_dim = num_channels * embed_dim
        if task == "classification":
            return ClassificationHead(
                in_dim=in_dim, num_classes=out_dim,
                hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers,
            )
        return RegressionHead(
            in_dim=in_dim, num_outputs=out_dim,
            hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers,
        )

    if fusion == "posenc":
        return PosEncConcatHead(
            num_channels=num_channels, embed_dim=embed_dim, out_dim=out_dim,
            hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers,
        )

    # fusion == "crossattention"
    return ChannelCrossAttentionHead(
        num_channels=num_channels, embed_dim=embed_dim, out_dim=out_dim,
        hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers,
        num_attention_heads=int(kwargs.get("num_attention_heads", 4)),
        num_queries=int(kwargs.get("num_queries", 1)),
        ff_ratio=float(kwargs.get("ff_ratio", 2.0)),
        attn_dropout=float(kwargs.get("attn_dropout", 0.0)),
        use_pos_enc=bool(kwargs.get("use_pos_enc", True)),
    )
