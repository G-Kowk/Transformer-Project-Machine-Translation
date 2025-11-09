# src/decoder_layer.py
# 单个解码层
import torch.nn as nn
from src.attention import MultiHeadAttention
from src.ffn import PositionwiseFFN

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)    # Masked Multi-Head Self-Attention，注意力，有mask，防止看到未来内容
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads, dropout)   # Encoder-Decoder Attention，让解码器根据编码器的输出（memory）获取源语句上下文
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 输入目标序列，学习目标序列内部的依赖关系
        attn_out1, a1 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(attn_out1)
        tgt = self.norm1(tgt)

        # 建立源语句与目标语句之间的对应关系
        attn_out2, a2 = self.enc_dec_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout(attn_out2)
        tgt = self.norm2(tgt)

        # 前馈网络，对每个时间步的向量独立地经过两层线性变换 + 激活函数（ReLU）
        ffn_out = self.ffn(tgt)
        tgt = tgt + self.dropout(ffn_out)
        tgt = self.norm3(tgt)
        return tgt, a1, a2
