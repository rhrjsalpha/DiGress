# Graphormer/G P3/modules_base_attention/multihead_attention.py
# 재작성 2025-07-22  ★ 변경 범위 전체

from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

# ----------(★) quant_noise 불필요 시 graceful fallback ----------
try:
    from Graphormer.GP3.modules.quant_noise import quant_noise
except ModuleNotFoundError:              #
    def quant_noise(layer, *_, **__) -> nn.Module:  #
        return layer                                #


class MultiheadAttention(nn.Module):
    r"""PyTorch-native multi-head attention (+ optional bias-kv)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        self_attention: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ) -> None:
        super().__init__()

        # ---------- 기본 하이퍼파라미터 ----------
        self.embed_dim = embed_dim
        self.kdim = kdim or embed_dim
        self.vdim = vdim or embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.self_attention = self_attention

        # ---------- 차원 체크 ----------
        assert embed_dim % num_heads == 0, "`embed_dim` must be divisible by `num_heads`"
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5            #

        # ---------- Q/K/V/Out 프로젝션 ----------
        self.q_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.k_proj = quant_noise(nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()          #

    # ------------------------------------------------------------------ #
    # (★) Xavier 초기화 – 4개 프로젝션 모두
    # ------------------------------------------------------------------ #
    def reset_parameters(self) -> None:
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(proj.weight, gain=1 / math.sqrt(2))
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        query: Tensor,                          # (T, B, E)
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        *,
        attn_bias: Optional[Tensor] = None,     # (B, H, T, S)
        key_padding_mask: Optional[Tensor] = None,  # (B, S)
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,     # (T, S)
    ) -> Tuple[Tensor, Optional[Tensor]]:

        # ---- Q / K / V 준비 ------------------------------------------------
        if self.self_attention or key is None:
            key = value = query
        T, B, _ = query.size()
        S = key.size(0)

        q = self.q_proj(query) * self.scale       # scaling 먼저
        k = self.k_proj(key)
        v = self.v_proj(value)

        # ---- (B*H, len, head_dim) 로 reshape ------------------------------
        def _reshape(x: Tensor) -> Tensor:        # helper 함수
            return x.contiguous().view(-1, B * self.num_heads, self.head_dim).transpose(0, 1)

        q, k, v = map(_reshape, (q, k, v))        # (B*H, T/S, head_dim)

        # ---- Scaled Dot-Product Attention ---------------------------------
        attn_weights = torch.bmm(q, k.transpose(1, 2))   # (B*H, T, S)

        if attn_bias is not None:                         # add-bias
            attn_weights += attn_bias.view(B * self.num_heads, T, S)

        if attn_mask is not None:                         # attn_mask
            attn_weights += attn_mask.unsqueeze(0)

        if key_padding_mask is not None:                  # padding mask
            pad = key_padding_mask[:, None, :].expand(-1, self.num_heads, -1)
            attn_weights = attn_weights.masked_fill(
                pad.reshape(B * self.num_heads, 1, S),
                float("-inf"),
            )

        attn_probs = self.dropout(torch.softmax(attn_weights, dim=-1, dtype=torch.float32))
        attn_output = torch.bmm(attn_probs, v)            # (B*H, T, head_dim)

        # ---- Merge heads ---------------------------------------------------
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(T, B, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        if need_weights:
            attn_probs = attn_probs.view(B, self.num_heads, T, S)  # ★ 반환용 reshape
            return attn_output, attn_probs
        return attn_output, None

