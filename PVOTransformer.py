import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Transformer import Transformer
from torch.utils.data import Dataset, DataLoader


class PVOTransformer(Transformer):
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=2, d_ff=256, max_seq_length=100, dropout=0.1):
        # 假设src_vocab_size和tgt_vocab_size都为1，因为我们处理的是连续数据而不是离散词汇
        super(PVOTransformer, self).__init__(src_vocab_size=1, tgt_vocab_size=1, d_model=d_model, num_heads=num_heads,
                                             num_layers=num_layers, d_ff=d_ff, max_seq_length=max_seq_length, dropout=dropout)
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, d_model)  # 输入特征到模型维度的线性变换

    def forward(self, src):
        # src shape: [batch_size, seq_length, input_dim]
        src_embedded = self.embedding(src)  # 将输入特征映射到模型维度
        src_embedded = self.positional_encoding(src_embedded)  # 添加位置编码
        src_embedded = self.dropout(src_embedded)  # 应用Dropout

        # 编码器部分
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, None)  # 不使用掩码

        # 只使用编码器的输出进行预测
        output = self.fc(enc_output)
        return output[:, -1, :]  # 返回最后一个时间步的输出