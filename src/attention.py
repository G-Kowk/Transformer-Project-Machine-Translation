# src/attention.py
# 手工实现 MHA，接口兼容 encoder/decoder 两类注意力（自注意力与 encoder-decoder attention）。
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

# 缩放点积注意力
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)     # 在attention权重上做dropout

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (b, h, q_len, kv_len)，计算点积相似度：scores = Q @ K^T，每个查询位置对每个键位置的相似度分数。
        if attn_mask is not None:
            # attn_mask expected to be broadcastable to scores and contain -inf in blocked positions
            scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)     # 对 kv_len 维度做 softmax，把 scores 变为概率分布（注意力权重）。
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (b, h, q_len, head_dim)
        return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0     # 保证每个 head 的维度是整数。
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 内部使用上面定义的缩放点积注意力，并在最终输出上再做一次 dropout
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query: (batch, q_len, d_model)
        key:   (batch, kv_len, d_model)
        value: (batch, kv_len, d_model)
        attn_mask: (q_len, kv_len) or broadcastable; contains -inf for masked positions
        returns:
            out: (batch, q_len, d_model)
            attn: (batch, n_heads, q_len, kv_len)
        """
        b, q_len, _ = query.size()
        _, kv_len, _ = key.size()

        Q = self.w_q(query).view(b, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(key).view(b, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(value).view(b, kv_len, self.n_heads, self.head_dim).transpose(1, 2)

        if attn_mask is not None:
            # If attn_mask shape is (q_len, kv_len), expand to (1,1,q_len,kv_len)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

        out, attn = self.attention(Q, K, V, attn_mask)
        out = out.transpose(1,2).contiguous().view(b, q_len, self.d_model)
        out = self.w_o(out)
        out = self.dropout(out)
        return out, attn
