# src/utils.py
# 位置编码与掩码生成
import math
import torch
import torch.nn as nn

# 为Transformer的输入添加位置信息
class PositionalEncoding(nn.Module):
    # d_model：每个词的向量维度，max_len：最长序列长度
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        # position是[max_len, 1]的列向量，代表每个位置编号(0, 1, 2, ...)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # 正弦-余弦位置编码公式
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]

# 为decoder自注意力层生成“后续屏蔽掩码”，防止模型看到未来词，用-inf表示屏蔽词，返回(sz, sz)
def generate_square_subsequent_mask(sz: int, device=None) -> torch.Tensor:
    mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
    float_mask = torch.zeros(sz, sz, device=device).float()
    float_mask = float_mask.masked_fill(mask, float("-inf"))
    return float_mask

# 生成padding掩码，在注意力中屏蔽掉补齐（pad）的位置
# seq: (batch, seq_len)。不是pad即为1，是即为0
def make_pad_mask(seq, pad_id=0):
    return (seq != pad_id).to(torch.int64)
