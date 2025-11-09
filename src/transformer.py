# src/transformer.py
# 定义完整的transformer神经网络架构，包括嵌入层（词嵌入+位置编码）、编码器（Encoder）、解码器（Decoder）、输出层（线性层预测词分布）
import torch
import torch.nn as nn
import math
from src.utils import PositionalEncoding, generate_square_subsequent_mask, make_pad_mask
from src.encoder import Encoder
from src.decoder import Decoder

class TransformerFromScratch(nn.Module):
    # 初始化：src_vocab_size源语言词汇表大小（输入），tgt_vocab_size目标语言词汇表大小（输出），d_model模型隐层维度（embedding维度），d_ff前馈层隐藏维度，max_len最大序列长度，pad_id是padding token的ID，用于掩码生成
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 num_encoder_layers: int = 2, num_decoder_layers: int = 2, d_ff: int = 512,
                 max_len: int = 256, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        # 词嵌入，将token ID转换为稠密向量，padding_idx表示填充位置不会被更新（始终是零向量）
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)

        # 位置编码，transformer不使用循环结构，需显式加入位置信息
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.pos_dec = PositionalEncoding(d_model, max_len=max_len)

        self.encoder = Encoder(num_encoder_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, n_heads, d_ff, dropout)

        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        src_tokens: (batch, src_len)
        tgt_tokens: (batch, tgt_len)
        src_mask: (batch, src_len) int mask (1 valid, 0 pad) or None
        tgt_mask: (tgt_len, tgt_len) subsequent mask or None
        memory_mask: (tgt_len, src_len) float mask with -inf at padded keys (optional)
        """
        device = src_tokens.device

        # 嵌入 + 位置编码，输出的src tgt是编码/解码器输入向量（batch, src/tgt_len, d_model）
        src = self.src_emb(src_tokens) * math.sqrt(self.d_model)
        src = self.pos_enc(src)  # (b, src_len, d_model)

        tgt = self.tgt_emb(tgt_tokens) * math.sqrt(self.d_model)
        tgt = self.pos_dec(tgt)  # (b, tgt_len, d_model)

        # 构造掩码mask
        # 1) 目标序列未来掩码tgt_mask (tgt_len, tgt_len)的方阵，是下三角矩阵，未来信息用“-inf”，softmax≈0，无法关注未来词
        if tgt_mask is None:
            tgt_len = tgt_tokens.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_len, device=device)

        # 2) 源序列填充掩码memory_mask_float，用于Decoder的Encoder-Decoder Attention
        # src_mask原始形状是(batch, src_len)，1表示该token有效，为正常词，0表示是padding，需要屏蔽。
        # 要把它转换为注意力模块能理解的形状(batch, 1, 1, src_len)，正常token对应0，PAD token对应-inf。
        memory_mask_float = None
        if src_mask is not None:
            pad_positions = (src_mask == 0)  # 需要屏蔽
            # 变换维度，mask dims (batch, 1, 1, src_len)
            memory_mask_float = torch.zeros((src_tokens.size(0), 1, 1, src_tokens.size(1)), device=device)
            memory_mask_float = memory_mask_float.masked_fill(pad_positions.unsqueeze(1).unsqueeze(1), float("-inf"))

        # 3) encoder
        memory, enc_attn = self.encoder(src, src_mask=None)  # src（输入序列）在之前已经归一化处理

        # 4) decoder
        out, self_attns, encdec_attns = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask_float)

        # 5) 输出层，将每个时间步的向量映射到目标词表的维度
        logits = self.output_linear(out)  # (batch, tgt_len, tgt_vocab)
        return logits, {"enc_attns":enc_attn, "self_attns":self_attns, "encdec_attns":encdec_attns}
