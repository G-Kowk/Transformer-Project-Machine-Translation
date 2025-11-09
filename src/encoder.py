# src/encoder.py
# 堆叠多个编码层（形成完整的编码器）。encoder将若干个encoder_layer叠加起来，形成完整的编码器结构，每一层的输出作为下一层的输入
import torch.nn as nn
from src.encoder_layer import EncoderLayer

class Encoder(nn.Module):
    # 参数含义：编码层层数、每个token的表示维度、注意力头数、前馈层隐藏维度、dropout概率
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = src
        attns = []
        for layer in self.layers:
            x, attn = layer(x, src_mask)
            attns.append(attn)
        x = self.norm(x)
        # 返回：最终编码器输出、每层的注意力矩阵
        return x, attns
