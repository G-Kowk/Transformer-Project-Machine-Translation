# data/iwslt_dataset.py
# 加载IWSLT2017英德翻译数据集（en→de）
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

class IWSLTSubDataset(Dataset):
    # 加载IWSLT2017 en->de数据集，并用指定的tokenizer（默认t5-small）将句子转换为input_ids，每个句子会被截断或填充到max_len长度
    # 初始化，参数：数据划分（train/validation/test）、取样本条数、每个句子的最大token数、使用的HuggingFace分词器（默认：t5-small）
    def __init__(self, split='train', num_samples=5000, max_len=48, tokenizer_name="t5-small"):
        ds = load_dataset("iwslt2017", "iwslt2017-en-de", split=split, trust_remote_code=True)
        # 截取子集，确保不要超过实际数据集大小
        # self.num_samples = min(num_samples, len(ds))

        # 兼容原有写法：num_samples 为 None 时加载全部数据
        if num_samples is None:
            # 使用全部样本
            self.num_samples = len(ds)
        else:
            # 使用原有逻辑
            self.num_samples = min(num_samples, len(ds))

        # 提取英德句子对
        self.src_texts = [ds[i]['translation']['en'] for i in range(self.num_samples)]
        self.tgt_texts = [ds[i]['translation']['de'] for i in range(self.num_samples)]
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        # 文本 -> token ID 序列。将原始句子转换为token ID向量，每个句子被截断到max_len、填充到固定长度、返回pytorch tensor格式
        self.src_enc = self.tokenizer(self.src_texts, padding='max_length', truncation=True,
                                      max_length=max_len, return_tensors='pt')
        self.tgt_enc = self.tokenizer(self.tgt_texts, padding='max_length', truncation=True,
                                      max_length=max_len, return_tensors='pt')

    def __len__(self):
        return self.num_samples

    # 取单条样本
    def __getitem__(self, idx):
        src_ids = self.src_enc["input_ids"][idx]    # tensor (max_len,)
        tgt_ids = self.tgt_enc["input_ids"][idx]
        return src_ids, tgt_ids
