import math
import torch
import torch.nn as nn

from Graphormer.GP3.modules.quant_noise import quant_noise


class MultiheadAttention(nn.Module):
    """
    Multi-headed attention implementation.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention

        assert self.qkv_same_dim, "Self-attention requires query, key and value to be of the same size"

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.0)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.0)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key=None,
        value=None,
        attn_bias=None,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
    ):
        """
        Forward pass for multi-head attention.

        Args:
            query: Tensor of shape (T, B, C)
            key: Tensor of shape (S, B, C)
            value: Tensor of shape (S, B, C)
            attn_bias: Optional bias to add to the attention scores
            key_padding_mask: Optional mask to exclude certain keys
            need_weights: Whether to return attention weights
            attn_mask: Optional attention mask

        Returns:
            attn_output: Attention output
            attn_weights: Attention weights if need_weights is True
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q = self.q_proj(query)
        k = self.k_proj(key if key is not None else query)
        v = self.v_proj(value if value is not None else query)
        q *= self.scaling

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, -1)

        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)

        attn_weights_float = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights_float)

        attn_output = torch.bmm(attn_weights, v)
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(tgt_len, bsz, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        if need_weights:
            return attn_output, attn_weights_float
        return attn_output, None
