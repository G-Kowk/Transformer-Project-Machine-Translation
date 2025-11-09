# train_encoder_only.py
# 实验目的：验证仅使用 Encoder 的 Transformer 在机器翻译任务上的性能下降

import os
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data.iwslt_dataset import IWSLTSubDataset
from src.utils import make_pad_mask, generate_square_subsequent_mask
from transformers import AutoTokenizer
import argparse
import time
import matplotlib.pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR


# ===============================
# 固定随机种子，保证可复现性
# ===============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] Random seed set to: {seed}")


# ===============================
# 简化数据batch函数
# ===============================
def collate_stack(batch):
    srcs = torch.stack([b[0] for b in batch], dim=0)
    tgts = torch.stack([b[1] for b in batch], dim=0)
    return srcs, tgts


# ===============================
# Encoder-only Transformer
# ===============================
class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, num_layers=2, d_ff=512, max_len=48, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # 输出层：直接预测下一个词（无decoder）
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        B, T = src.size()
        pos = torch.arange(0, T, device=src.device).unsqueeze(0).expand(B, T)
        x = self.embed(src) + self.pos_embed(pos)

        src_key_padding_mask = (src == self.pad_id)
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        # 预测下一个token（shift）
        logits = self.output(x)
        return logits


# ===============================
# 主训练逻辑
# ===============================
def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    vocab_size = tokenizer.vocab_size
    print("Tokenizer loaded. Vocab size:", vocab_size, "pad_id:", pad_id)

    # 数据加载
    train_ds = IWSLTSubDataset(split='train', num_samples=args.train_samples,
                               max_len=args.max_len, tokenizer_name=args.tokenizer_name)
    val_ds = IWSLTSubDataset(split='validation', num_samples=args.val_samples,
                             max_len=args.max_len, tokenizer_name=args.tokenizer_name)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_stack)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_stack)

    # 模型
    model = EncoderOnlyTransformer(vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                                   num_layers=args.num_layers, d_ff=args.d_ff, max_len=args.max_len, pad_id=pad_id).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = 1e9
    train_losses, val_losses = [], []

    print("Start encoder-only training ...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0
        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            # 目标右移一位
            decoder_input = tgt[:, :-1]
            decoder_target = tgt[:, 1:]

            optimizer.zero_grad()
            logits = model(decoder_input)  # 无decoder，直接预测
            loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                decoder_input = tgt[:, :-1]
                decoder_target = tgt[:, 1:]
                logits = model(decoder_input)
                loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step()

        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  time={time.time()-t0:.1f}s")

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_encoder_only.pth"))

    print("Training finished. Best val loss:", best_val)

    # 绘制loss曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Encoder-Only Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, "encoder_only_loss_curve.png"))
    plt.show()


# ===============================
# 运行入口
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_samples", type=int, default=20000)
    parser.add_argument("--val_samples", type=int, default=4000)
    parser.add_argument("--max_len", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tokenizer_name", type=str, default="t5-small")
    parser.add_argument("--save_dir", type=str, default="./results_encoder_only")
    args = parser.parse_args()
    train(args)
