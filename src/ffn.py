# src/ffn.py
# 前馈全连接网络（Feed-Forward Network, FFN），Position-wise FFN（逐位置前馈网络）
# 在每个token的表示上，单独执行两层线性变换+非线性激活，用来增强模型的特征提取能力
import torch.nn as nn

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # 全连接升维，ReLU激活，dropout，降维还原
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
