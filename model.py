import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int)->None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.Embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self,x):
        return self.Embedding(x) @ math.sqrt(self.d_model)


class PositionalEncodings(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float ) ->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout= nn.Dropout(dropout)

        ## Important to understand
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

        ## Up to this


    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) 
        return self.dropout(x)
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout: float)->None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, self.d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    


class LayerNormalization(nn.Module):
    def __init__(self, features:int, eps:float=10**-6)->None:
        super().__init__()
        self.features = features
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(self.features))
        self.bias = nn.Parameter(torch.zeros(self.features))

    def forward(self, x):
        mean = x.min(dim = -1, keepdim = True) #  batch, sequencelength,1
        std = x.std(dim = -1, keepdim = True) # batch, sequencelength, 1
        return self.alpha @ (x-min)/ (std+self.eps) + self.bias

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float)->None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert self.d_model % self.h == 0, "d_model is not divisible by h"
        self.d_k = self.d_model // self.h

        self.w_q = nn.Linear(self.d_model, self.d_model, bias = False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias = False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias = False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias = False)
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention.masked_fill_(mask==0, -1e9)
        
        if dropout is not None:
            attention = dropout(attention)
        
        return attention @ value , attention
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        self.mask = mask

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contigous().view(x.shape[0], x.shape[1], self.h @ self.d_k)
        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout:float):
        super().__init__()
        self.features = features
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(features)

    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.layer_norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, feed_forward: FeedForwardBlock, attention: MultiHeadAttentionBlock, droput:float, feature:int):
        super().__init__()
        self.feed_forward = feed_forward
        self.attention = attention
        self.dropout = nn.Dropout(droput)
        self.feature = feature
        self.residual_connection = nn.ModuleList([ResidualConnection(self.feature, self.dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.attention(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward)

        return x
    

class Encoder(nn.Module):
    def __init__(self, features:int, layers:int)->None:
        super().__init__()
        self.norm = LayerNormalization(features=features)
        self.layers = layers
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)



class DecoderBlock(nn.Module):
    def __init__(self, self_attention:MultiHeadAttentionBlock, cross_attention:MultiHeadAttentionBlock, features:int, feed_foward_block: FeedForwardBlock, dropout:nn.Dropout)->None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.features = features
        self.feed_forward_block = feed_foward_block
        self.dropout= nn.DropOut(dropout)
        self.residula_connection = nn.ModuleList([ResidualConnection(self.features, self.dropout) for _ in range(3)])
    
    def forward(self, x, target_mask, src_mask, encoder_output):
        x = self.residula_connection[0](x, lambda x : self.self_attention(x, x, x, target_mask ))
        x = self.residula_connection[1](x, lambda x: self.cross_attention(x,encoder_output,encoder_output,src_mask))
        x = self.residula_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int)->None:
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.projection = nn.Linear(self.d_model, self.vocab_size)
    
    def forward(self, x):
        return self.projection(x)
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder:Decoder, src_embedding:InputEmbeddings, target_embedding:InputEmbeddings, src_position_emnbedding:PositionalEncodings, target_position_encoding:PositionalEncodings,  projection_layer:ProjectionLayer):
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.target_embedding = target_embedding
        self.src_positional_embedding = src_position_emnbedding
        self.target_positional_embedding = target_position_encoding
        self.projectionlayer = projection_layer
    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_positional_embedding(src)
        return self.encoder(src, src_mask)
    def decode(self, traget: torch.Tensor, encoder_output:torch.Tensor, src_mark,  target_mask ):
        target = self.target_embedding(target)
        target = self.target_positional_embedding(traget)
        return self.decoder(target,
                            encoder_output,
                            src_mark,
                            target_mask)
    def projection(self, x):
        return self.projectionlayer(x)
    


def build_transformer(src_vocab_size: int, target_vocab_size: int, source_sequence_len : int, target_sequence_len : int, d_model:int=512, N:int=6, h:int=6, dropout:float=0.1, dff:int = 2048):
    src_embed = InputEmbeddings(d_model=d_model, vocab_size= src_vocab_size)
    target_embed = InputEmbeddings(d_model=d_model, vocab_size=target_vocab_size)
    src_pos_embed = PositionalEncodings(d_model=d_model, seq_len=source_sequence_len)
    traget_pos_embed = PositionalEncodings(d_model=d_model, seq_len=target_sequence_len)
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, dff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, dff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, target_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, target_embed, src_pos_embed, traget_pos_embed, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer