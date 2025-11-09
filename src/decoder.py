# src/decoder.py

import torch.nn as nn
from src.decoder_layer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = tgt
        self_attns = []
        encdec_attns = []
        for layer in self.layers:
            x, a1, a2 = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            self_attns.append(a1)
            encdec_attns.append(a2)
        x = self.norm(x)
        return x, self_attns, encdec_attns
