import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocabulary_size: int)->None:
        super().__init__()
        self.d_model = d_model
        self.vocabulary_size = vocabulary_size
        self.embeddings = nn.Embeddings(self.d_model, self.vocabulary_size)
    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len:int, d_model:int, dropout:float )->None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = dropout
        self.pe = torch.zeros(self.seq_len, self.d_model)
        self.pos = torch.arrange(0, self.seq_len).unsqueeze(0)
        self.div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)) 
        self.pe[:,0::2] = torch.sin(self.pos/ self.div_term)
        self.pe[:,1::2] = torch.cos(self.pos/ self.div_term)
        self.pe.unsqeeze(0)
        self.register_buffer('pe', self.pe)
    
    def forward(self, x):
        x = x+(self.pe[:,x.shape[1],:]).requires_grad(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, features, eps:float=10**-6 )->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha*(x-mean) / self.bias+std+self.eps

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, dff:int, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.dropout = dropout
        self.linear2 = nn.Linear(dff, d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(self.linear1(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = dropout
        assert self.d_model%self.h == 0, "d_model is not divisible by h."
        self.d_k = self.d_model//self.h
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]
        attention_score = (query@key.transpose(-2,-1))/ math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(-1)
        if dropout is not None:
            attention_score = dropout(attention_score)
        return (attention_score@value), attention_score
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attenstion_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, features, dropout:float)->None:
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = dropout
    
    def forward(self, x, sublayer):
        return self.dropout(sublayer(self.norm(x)))










