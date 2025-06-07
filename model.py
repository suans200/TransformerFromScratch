import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocabulary_size: int)->None:
        super.__init__()
        self.d_model = d_model
        self.vocabulary_size = vocabulary_size
        self.embeddings = nn.Embeddings(self.d_model, self.vocabulary_size)
    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len:int, d_model:int, dropout:float )->None:
        super.__init__()
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




