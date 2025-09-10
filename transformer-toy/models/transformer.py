import torch
import torch.nn as nn
import math

# ----------------------------------------------------------------
#  1. 핵심 부품 (Core Components)
# ----------------------------------------------------------------
# 1-1. Rotary Positional Embedding (RoPE)
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()

        # RoPE의 주파수(theta) 계산
        # θ_i = 10000^(-2(i-1)/d) for i = 1, 2, ..., d/2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # 위치(t)와 주파수(freq)를 미리 계산하여 sin, cos 값을 만듦
        t = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # (max_seq_len, dim/2)

        # 복소수 형태로 (cos + i*sin) 저장. polar는 복소수를 극좌표 형식으로 표현
        # freqs_cis.real은 cos, freqs_cis.imag는 sin
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x):
        # x: (batch, nhead, seq_len, d_k)
        seq_len = x.shape[-2]
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]

        # x의 절반은 그대로 두고, 나머지 절반은 부호를 바꿔서 회전 준비
        # x = (x_1, x_2, x_3, x_4, ...) -> x_rotated = (-x_2, x_1, -x_4, x_3, ...)
        x_rotated = torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1)

        # 회전 행렬 곱셈 적용
        # y_i = x_i*cos(mθ) + rot(x_i) * sin(mθ)
        return x * cos + x_rotated * sin
    
# 1-2. Position-wise Feed-Forward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x Shape: [Batch, Seq_Len, d_model]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
# ----------------------------------------------------------------
# 2. Multi-Head Attention (RoPE 적용 버전)
# ----------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead # 각 헤드의 차원

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.rotary_emb = RotaryPositionalEmbedding(dim=self.d_k)
    
    def forward(self, query, key, value, mask=None):
        # query, key, value Shape: [Batch, Seq_Len, d_model]
        batch_size = query.size(0)

        # 1. Linear projection
        Q = self.w_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # Q, K에 RoPE 적용
        Q = self.rotary_emb(Q)
        K = self.rotary_emb(K)

        # 3. Scaled Dot-Product Attention 계산
        # [B, nhead, Seq_Len, d_k] @ [B, nhead, d_k, Seq_Len] -> [B, nhead, Seq_Len, Seq_Len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # 마스크가 있는 위치에 아주 작은 값을 채워 softmax 후 0이 되도록 함
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)

        # 4. Attention weight와 V를 곱함
        # [B, nhead, Seq_Len, Seq_Len] @ [B, nhead, Seq_Len, d_k] -> [B, nhead, Seq_Len, d_k]
        context = torch.matmul(attn_weights, V)

        # 5. 헤드들을 다시 결합
        # [B, nhead, Seq_Len, d_k] -> [B, Seq_Len, nhead * d_k] -> [B, Seq_Len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 6.Final linear layer
        output = self.w_o(context)
        return output

# ----------------------------------------------------------------
# 3. 트랜스포머 레이어 (Transform Layers)
# ----------------------------------------------------------------
# 3-1. Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, src, src_mask=None):
        # 1. Norm -> Self-Attention -> Add
        # 원본 src를 norm1에 통과시킨 결과를 어텐션에 입력
        norm_src = self.norm1(src)
        attn_output = self.self_attn(norm_src, norm_src, norm_src, src_mask)
        # 어텐션 결과는 원본 src에 더해줌
        src = src + self.dropout(attn_output)

        # 2. Norm -> Feed Forward -> Add
        # 위에서 나온 결과를 norm2에 통과시켜 FFNN에 입력
        norm_src = self.norm2(src)
        ff_output = self.feed_forward(norm_src)
        # FFNN 결과는 다시 원본 src에 더해줌
        src = src + self.dropout(ff_output)

        return src

# 3-2. Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 1. Norm -> Self-Attention -> Add
        norm_tgt = self.norm1(tgt)
        attn_output = self.self_attn(norm_tgt, norm_tgt, norm_tgt, tgt_mask)
        tgt = tgt + self.dropout(attn_output)

        # 2. Norm -> Cross-Attention -> Add
        norm_tgt = self.norm2(tgt)
        attn_output = self.cross_attn(norm_tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout(attn_output)

        # 3. Norm -> Feed Forward -> Add
        norm_tgt = self.norm3(tgt)
        ff_output = self.feed_forward(norm_tgt)
        tgt = tgt + self.dropout(ff_output)

        return tgt

# ----------------------------------------------------------------
# 4. 최종 모델 조립 (Final Model Assembly)
# ----------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, d_ff, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_decoder_layers)
        ])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=src.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        src_emb = self.dropout(self.src_embedding(src))
        tgt_emb = self.dropout(self.tgt_embedding(tgt))

        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)
        
        output = tgt_emb
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask, src_mask)
        
        return self.fc_out(output)