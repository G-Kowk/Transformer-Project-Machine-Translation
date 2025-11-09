# 从零实现 Transformer 的英德机器翻译模型（EN→DE）

本项目基于 PyTorch **从零实现 Transformer 模型**，包括：

* 多头自注意力（Multi-Head Self-Attention）
* 缩放点积注意力
* 前馈网络（Position-wise FFN）
* 残差连接 + LayerNorm
* 位置编码（Positional Encoding）
* 编码器-解码器结构（Encoder–Decoder）
* 训练流程、掩码机制、余弦退火学习率调度
* Greedy Decoding 翻译
* 多组消融实验（移除解码器、更换头数、移除位置编码等）

项目目标在于 **深入理解 Transformer 的内部机制**，避免直接使用 `torch.nn.Transformer` 或高层 API。

---

## 项目特性

* 自实现 Transformer（Encoder + Decoder）
* 使用 Hugging Face `AutoTokenizer` 进行分词
* 支持 IWSLT2017 EN–DE 全量数据
* 实现 padding mask 与 future mask
* 使用 AdamW + 余弦退火学习率调度（CosineAnnealingLR）
* 训练早停（Early Stopping）
* Loss 曲线自动保存
* Greedy auto-regressive 翻译
* 多种模型消融实验：

  * 2 层 → 1 层 Transformer
  * 4 头 → 2 头注意力
  * Encoder-only 模型
  * 无位置编码模型

---

## 项目目录结构

```
Transformer Project——Machine Translation/
├── src/                     # 源代码目录
│   ├── attention.py         # 注意力机制实现
│   ├── decoder.py           # 解码器主结构
│   ├── decoder_layer.py     # 单层解码器结构
│   ├── encoder.py           # 编码器主结构
│   ├── encoder_layer.py     # 单层编码器结构
│   ├── ffn.py               # 前馈网络实现
│   ├── transformer.py       # Transformer整体模型封装
│   └── utils.py             # 工具函数（mask生成等）
├── data/                    # 数据相关文件
│   └── iwslt_dataset.py     # IWSLT2017 EN-DE数据集加载与处理
├── results/                 # 训练结果、模型权重、loss曲线
├── scripts/                 # 运行脚本
│   └── run.sh               # 一键训练脚本示例
├── train_mt.py              # 主训练脚本
├── train_mt_1e1d.py         # 单编码单解码实验脚本
├── train_mt_2heads.py       # 双编码双解码实验脚本
├── train_mt_alldata.py      # 使用全量数据训练脚本
├── train_mt_encoder_only.py # 仅编码器实验脚本
├── README.md                # 项目说明文件
└── requirements.txt         # Python依赖环境列表
```

---

## 依赖环境与安装

### 1. 创建环境

```bash
conda create -n transformer python=3.10
conda activate transformer
```

### 2. 安装依赖

```bash
pip install torch>=2.1.0
pip install transformers>=4.44.0
pip install datasets>=3.0.0
pip install numpy tqdm sentencepiece sacrebleu scikit-learn matplotlib
```

### 依赖列表汇总

* Python 3.10+
* PyTorch ≥ 2.1
* Transformers ≥ 4.44（用于 tokenizer）
* Datasets ≥ 3.0（用于加载 IWSLT）
* numpy、tqdm、sentencepiece
* sacrebleu、scikit-learn（BLEU 计算可选）
* matplotlib（绘制曲线）

---

## 训练方式示例

最简单的训练命令：

```bash
python train_mt.py --seed 42 --epochs 50 --batch_size 16 --max_len 48 --device cuda
# 若无 GPU，可将 --device cuda 改为 --device cpu
```

---

## 使用完整 IWSLT2017 数据集

```bash
python train_mt_alldata.py --batch_size 16 --epochs 30 --max_len 64
```

内部自动按 **80% 训练 / 20% 验证** 划分。

---

## 翻译示例（Greedy Decoding）

```python
gen_ids = greedy_decode(model, src_ids, tokenizer, max_len=48, device=device)
print(tokenizer.decode(gen_ids, skip_special_tokens=True))
```

---

## 结果输出位置

训练结果存放于：

```
./results/
```

包括：

* `loss_curve.png`
* `best_transformer_mt.pth`

---

## 可重复性（Reproducibility）

训练脚本中设置：

```python
torch.manual_seed(42)
numpy.random.seed(42)
random.seed(42)
```

以及：

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

确保实验结果可复现。

---

## 许可信息（License）

本项目用于课程学习与研究目的。

---
