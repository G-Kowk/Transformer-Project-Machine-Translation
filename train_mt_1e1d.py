# train_mt.py
# 训练流程，包括数据加载、模型构建、训练验证、贪心解码测试、早停和loss曲线绘制
import os
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data.iwslt_dataset import IWSLTSubDataset
from src.transformer import TransformerFromScratch
from src.utils import make_pad_mask, generate_square_subsequent_mask
from transformers import AutoTokenizer
import argparse
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

# 固定随机种子，保证可复现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] Random seed set to: {seed}")

# 将Dataset中返回的(src_ids, tgt_ids)样本堆叠成batch。返回值形状(batch, seq_len)
def collate_stack(batch):
    srcs = torch.stack([b[0] for b in batch], dim=0)
    tgts = torch.stack([b[1] for b in batch], dim=0)
    return srcs, tgts

# 贪心解码（即每次选取概率最高的词），实现自回归翻译
# 输入单个源句子(1, src_len)，输出生成的token id列表（如[BOS, 13, 254, 72, EOS]）
def greedy_decode(model, src_tensor, tokenizer, max_len, device):
    model.eval()
    src = src_tensor.to(device)
    src_mask = (src != tokenizer.pad_token_id).to(device)  # (1, src_len) int mask

    # 对编码器输入仅进行一次编码
    # 准备初始解码输入：若tokenizer包含BOS/CLS标记，则使用该标记作为起始
    if tokenizer.cls_token_id is not None:
        start_token = tokenizer.cls_token_id
    elif tokenizer.bos_token_id is not None:
        start_token = tokenizer.bos_token_id
    else:
        # 若无BOS或CLS，则使用pad作为占位符
        start_token = tokenizer.pad_token_id

    generated = [start_token]
    for step in range(max_len):
        dec_input = torch.tensor([generated], dtype=torch.long, device=device)  # (1, seq)
        # 构造目标序列的自回归mask（上三角为-inf）
        tgt_mask = generate_square_subsequent_mask(dec_input.size(1), device=device)
        # 前向传播
        logits, _ = model(src, dec_input, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=None)
        next_token_logits = logits[:, -1, :]  # (1, vocab)
        next_token = torch.argmax(next_token_logits, dim=-1).item()
        generated.append(next_token)
        # 若生成到EOS则提前停止
        if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
            break
    return generated

def train(args):
    set_seed(args.seed)

    # 环境与数据初始化
    device = torch.device(args.device)
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    src_vocab = tokenizer.vocab_size
    tgt_vocab = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    print("Tokenizer loaded. Vocab size:", tokenizer.vocab_size, "pad_id:", pad_id)

    # 加载数据
    train_ds = IWSLTSubDataset(split='train', num_samples=args.train_samples, max_len=args.max_len, tokenizer_name=args.tokenizer_name)
    val_ds = IWSLTSubDataset(split='validation', num_samples=args.val_samples, max_len=args.max_len, tokenizer_name=args.tokenizer_name)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_stack)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_stack)

    # 构建模型
    model = TransformerFromScratch(src_vocab_size=src_vocab, tgt_vocab_size=tgt_vocab,
                                   d_model=args.d_model, n_heads=args.n_heads,
                                   num_encoder_layers=args.num_layers, num_decoder_layers=args.num_layers,
                                   d_ff=args.d_ff, max_len=args.max_len, pad_id=pad_id).to(device)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

    # 学习率调度器（StepLR，每5轮衰减0.8倍）
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    # 余弦退火学习率调度器，T_max表示一个完整余弦周期的epoch数，eta_min为最小学习率（退火最低点）
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 如果存在之前保存的模型，自动加载继续训练
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "best_transformer_mt.pth")
    if os.path.exists(best_model_path):
        print(f"Loading existing model from {best_model_path} ...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Model weights loaded successfully.")

    patience_counter = 0  # 初始化
    train_losses, val_losses = [], []  # 用于记录每轮loss
    print("Start training ...")
    best_val = 1e9
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            # 解码器输入: [B, T]
            # 简化处理，直接使用tgt作为输入，并将目标右移一位
            decoder_input = tgt[:, :-1]
            decoder_target = tgt[:, 1:]

            # 构造掩码
            src_pad_mask = (src != pad_id).to(device)  # (B, src_len)
            memory_mask = torch.zeros((src.size(0), 1, 1, src.size(1)), device=device)
            memory_mask = memory_mask.masked_fill((src_pad_mask==0).unsqueeze(1).unsqueeze(1), float("-inf"))
            # 生成目标序列的后续mask（用于decoder的自注意力）
            tgt_mask_subseq = generate_square_subsequent_mask(decoder_input.size(1), device=device)

            logits, _ = model(src, decoder_input, src_mask=src_pad_mask, tgt_mask=tgt_mask_subseq, memory_mask=memory_mask)
            # logits: (B, T_dec, V)
            # loss = criterion(logits.view(-1, logits.size(-1)), decoder_target.view(-1))
            loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                decoder_input = tgt[:, :-1]
                decoder_target = tgt[:, 1:]
                src_pad_mask = (src != pad_id).to(device)
                memory_mask = torch.zeros((src.size(0),1,1,src.size(1)), device=device)
                memory_mask = memory_mask.masked_fill((src_pad_mask==0).unsqueeze(1).unsqueeze(1), float("-inf"))
                tgt_mask_subseq = generate_square_subsequent_mask(decoder_input.size(1), device=device)

                logits, _ = model(src, decoder_input, src_mask=src_pad_mask, tgt_mask=tgt_mask_subseq, memory_mask=memory_mask)
                # loss = criterion(logits.view(-1, logits.size(-1)), decoder_target.view(-1))
                loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 更新学习率调度器
        scheduler.step()

        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  time={time.time()-t0:.1f}s")

        # sample generation on one example
        model.eval()
        with torch.no_grad():
            src0, tgt0 = val_ds[0]
            print("SRC text (token ids):", src0.tolist())
            print("REFERENCE (token ids):", tgt0.tolist())
            gen_ids = greedy_decode(model, src0.unsqueeze(0), tokenizer, max_len=args.max_len, device=device)
            print("GENERATED ids:", gen_ids)
            try:
                print("GENERATED text:", tokenizer.decode(gen_ids, skip_special_tokens=True))
                print("REFERENCE text:", tokenizer.decode(tgt0.tolist(), skip_special_tokens=True))
            except Exception:
                pass

        '''
        # save best
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_transformer_mt.pth"))
            print("Saved best model.")
        '''
        # early stopping
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            patience_counter = 0
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_transformer_mt.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}. Best val_loss={best_val:.4f}")
                break

    print("Training finished. Best val loss:", best_val)

    # 确保 results 文件夹存在
    os.makedirs(args.save_dir, exist_ok=True)

    # 绘制loss曲线
    plt.figure(figsize=(8, 5))
    # plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss", marker='o')
    # plt.plot(range(1, args.epochs + 1), val_losses, label="Validation Loss", marker='x')
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, len(train_losses)+1), val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, "loss_curve.png"))
    plt.show()

    '''
    # 保存图像到 results 文件夹
    loss_plot_path = os.path.join(args.save_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss curve saved to {loss_plot_path}")
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--train_samples", type=int, default=20000)
    parser.add_argument("--val_samples", type=int, default=4000)
    parser.add_argument("--max_len", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)  # 早停容忍轮数
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tokenizer_name", type=str, default="t5-small")
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()
    train(args)
