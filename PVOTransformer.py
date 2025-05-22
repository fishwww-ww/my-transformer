import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Transformer import Transformer

class EnhancedAutoformerAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(EnhancedAutoformerAttention, self).__init__()
        self.num_heads = num_heads  # 注意力头数
        self.d_model = d_model      # 模型维度
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model)  # 查询矩阵
        self.W_k = nn.Linear(d_model, d_model)  # 键矩阵
        self.W_v = nn.Linear(d_model, d_model)  # 值矩阵
        self.W_o = nn.Linear(d_model, d_model)  # 输出矩阵
        
        # 增强注意力的额外组件
        self.temperature = nn.Parameter(torch.sqrt(torch.FloatTensor([self.d_k])))  # 温度参数
        self.dropout = nn.Dropout(dropout)  # dropout层
        
        # 时间感知组件
        self.time_weights = nn.Parameter(torch.randn(1, num_heads, 1, self.d_k))  # 时间权重
        
        # 修正后的门控机制 - 作用于头维度
        self.gate = nn.Linear(self.d_k, 1)  # 注意力门控
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)  # 输出层归一化

    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len, _ = Q.size()  # 获取批量大小和序列长度
        
        # 线性投影
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # 查询向量
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # 键向量
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # 值向量
        
        # 时间感知注意力增强
        Q = Q + self.time_weights  # 添加时间敏感组件
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.temperature  # 缩放点积注意力
        
        # 如果提供掩码则应用
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)  # 掩码处理
        
        # 注意力门控 - 作用于每个头的特征维度
        gate_score = torch.sigmoid(self.gate(Q))  # 形状: [batch_size, num_heads, seq_len, 1]
        attn_scores = attn_scores * gate_score  # 应用门控
        
        # Softmax和dropout
        attn_probs = self.dropout(F.softmax(attn_scores, dim=-1))  # 注意力概率
        
        # 上下文向量
        context = torch.matmul(attn_probs, V)  # 加权求和
        
        # 拼接多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        # 最终线性投影
        output = self.W_o(context)  # 输出变换
        
        return self.layer_norm(output)  # 层归一化后返回

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, attention_type='autoformer'):
        super(EncoderLayer, self).__init__()
        if attention_type == 'autoformer':
            self.self_attn = EnhancedAutoformerAttention(d_model, num_heads, dropout)  # 自注意力层
        else:
            raise ValueError("不支持的注意力类型")

        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)  # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)  # 第一层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二层归一化
        self.dropout = nn.Dropout(dropout)  # dropout层

    def forward(self, x, mask=None):
        # 增强注意力机制
        attn_output = self.self_attn(x, x, x, mask)  # 自注意力计算
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接+归一化

        # 前馈网络
        ff_output = self.feed_forward(x)  # 前向传播
        x = self.norm2(x + self.dropout(ff_output))  # 残差连接+归一化
        return x

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 第一线性层
        self.fc2 = nn.Linear(d_ff, d_model)  # 第二线性层
        self.gelu = nn.GELU()  # GELU激活函数
        self.dropout = nn.Dropout(0.1)  # dropout层

    def forward(self, x):
        return self.fc2(self.dropout(self.gelu(self.fc1(x))))  # 前向传播

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)  # 初始化位置编码矩阵
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)  # 位置序列
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # 除数项
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数
        self.register_buffer('pe', pe.unsqueeze(0))  # 注册为缓冲区

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]  # 添加位置编码到输入

class PVOTransformer(Transformer):
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=2, d_ff=256, max_seq_length=100, dropout=0.1):
        super(PVOTransformer, self).__init__(src_vocab_size=1, tgt_vocab_size=1, d_model=d_model, num_heads=num_heads,
                                             num_layers=num_layers, d_ff=d_ff, max_seq_length=max_seq_length, dropout=dropout)
        self.input_dim = input_dim
        self.output_dim = 1  # 添加输出维度属性，用于预测单个值
        self.token_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)

        # 使用增强型注意力机制
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout, attention_type='autoformer') for _ in range(num_layers)]
        )
        
        # 输出层
        self.fc = nn.Linear(d_model, self.output_dim)

    def forward(self, src):
        # 输入形状: [batch_size, seq_len, input_dim]
        token_embedded = self.token_embedding(src)  # [batch_size, seq_len, d_model]
        position_encoded = self.positional_encoding(token_embedded)
        src_embedded = self.dropout(position_encoded)

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, None)  # [batch_size, seq_len, d_model]

        # 只取最后一个时间步的特征
        final_step_feature = enc_output[:, -1, :]  # [batch_size, d_model]
        output = self.fc(final_step_feature)  # [batch_size, output_dim]
        
        return output