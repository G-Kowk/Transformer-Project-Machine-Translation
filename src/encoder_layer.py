# src/encoder_layer.py
# 单个编码层，包括多头自注意力层和前馈全连接网络，并在每个子层后加入残差连接和层归一化
import torch.nn as nn
from src.attention import MultiHeadAttention
from src.ffn import PositionwiseFFN

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)  # 实现自注意力机制，让每个位置可与序列中所有位置交互
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)  # 对每个位置单独的向量进行非线性变换
        self.norm1 = nn.LayerNorm(d_model)   # 层归一化，稳定训练、加速收敛
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 自注意力层
        attn_out, attn = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout(attn_out)   # 残差连接
        src = self.norm1(src)    # 层归一化

        # 前馈网络层
        ffn_out = self.ffn(src)
        src = src + self.dropout(ffn_out)    # 残差连接
        src = self.norm2(src)    # 层归一化
        return src, attn
