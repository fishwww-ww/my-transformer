import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Transformer import Transformer
from torch.utils.data import Dataset, DataLoader
import math

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, attention_type='autoformer'):
        super(EncoderLayer, self).__init__()
        if attention_type == 'autoformer':
            self.self_attn = AutoformerAttention(d_model, num_heads)
        else:
            raise ValueError("Unsupported attention type")

        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自适应注意力机制
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
# 前馈网络
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

# 自适应注意力机制
# 未实现Autoformer的核心特性
# 缺少时间序列专用组件
# 自适应性的体现不足
class AutoformerAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(AutoformerAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 自适应注意力机制
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.W_o(attn_output)
        return output

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)  # 初始化位置编码矩阵
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数
        self.register_buffer('pe', pe.unsqueeze(0))  # 注册为缓冲区
    def forward(self, x):
        # 将位置编码添加到输入中
        return x + self.pe[:, :x.size(1)]

class PVOTransformer(Transformer):
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=2, d_ff=256, max_seq_length=100, dropout=0.1):
        super(PVOTransformer, self).__init__(src_vocab_size=1, tgt_vocab_size=1, d_model=d_model, num_heads=num_heads,
                                             num_layers=num_layers, d_ff=d_ff, max_seq_length=max_seq_length, dropout=dropout)
        self.input_dim = input_dim
        self.token_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)

        # 使用自适应注意力机制
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout, attention_type='autoformer') for _ in range(num_layers)]
        )

    def forward(self, src):
        token_embedded = self.token_embedding(src)
        position_encoded = self.positional_encoding(token_embedded)
        src_embedded = self.dropout(position_encoded)

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, None)

        output = self.fc(enc_output)
        return output[:, -1, :]