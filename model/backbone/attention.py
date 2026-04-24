"""Transformer building blocks with optional CT-RoPE on query/key.

The blocks are intentionally minimal — no drop-path, no stochastic depth
gymnastics — to keep the MAE backbone auditable. They are inspired by the
``CustomAttentionBlock`` found in TimeFM's ``models/modules/attention.py``
but adapted for standalone use and for the continuous-time positional
encoding of :mod:`model.positional.ct_rope`.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0,
        act: Callable[..., nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class MultiHeadSelfAttention(nn.Module):
    """Standard MHSA with optional CT-RoPE rotation of Q and K.

    Parameters
    ----------
    dim : embedding dimension ``E``
    num_heads : number of attention heads
    qkv_bias : whether to include bias in the fused QKV projection
    dropout : attention + projection dropout
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                       # (B, S, E)
        padding_mask: Optional[torch.Tensor] = None,  # (B, S) True on valid
        rotary: Optional[nn.Module] = None,   # CTRoPE instance
        time_values: Optional[torch.Tensor] = None,  # (B, S)
    ) -> torch.Tensor:
        B, S, E = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, d)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rotary is not None:
            if time_values is None:
                raise ValueError("CT-RoPE requires time_values")
            q, k = rotary(q, k, time_values)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, S, S)
        if padding_mask is not None:
            # Block attention *to* padded positions (keys).
            mask = padding_mask[:, None, None, :]  # (B, 1, 1, S)
            attn = attn.masked_fill(~mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, S, E)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional CT-RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads, qkv_bias=qkv_bias, dropout=dropout,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        rotary: Optional[nn.Module] = None,
        time_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            padding_mask=padding_mask,
            rotary=rotary,
            time_values=time_values,
        )
        x = x + self.mlp(self.norm2(x))
        return x
