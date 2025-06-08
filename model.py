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
    def __init__(self, d_model:int, dff:int, dropout:float)->None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.dropout = dropout
        self.linear2 = nn.Linear(dff, d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(self.linear1(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h:int, dropout:float)->None:
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
        return x+self.dropout(sublayer(self.norm(x)))

class Encoder_Block(nn.Module):
    def __init__(self, features, dropout, self_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock)->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features=features, dropout=dropout) for _ in range(2)])
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, features)->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features=features)
    
    def __format__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self,features, dropout, self_attention_block: MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock)->None:
        super().__init__()
        self.features = features
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_block = nn.ModuleList([ResidualConnection(features=features, dropout=dropout) for _ in range(3)])
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection_block[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection_block[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection_block[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList, features)->None:
        super().__init__()
        self.layers = layers
        self.features = features
        self.norm = LayerNormalization(features=features)
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer():
    def __init__(self, d_model:int, vocab_size:int)->None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        return self.projection(x)

class Transformer(nn.Module):
    def __init__(self, src_embeddings:InputEmbeddings, tgt_embedidngs:InputEmbeddings, src_pos:PositionalEncoding, tgt_pos:PositionalEncoding, projection:ProjectionLayer)->None:
        super().__init__()
        self.src_embedding = src_embeddings
        self.tgt_embedding = tgt_embedidngs
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection
    def encode(self, x, src_mask):
        x = self.src_embedding(x)
        x = self.src_pos(x)
        return self.encode(x, src_mask=src_mask)
    def decoder(self, src_mask, tgt_mask, encoder_output, tgt):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(encoder_output=encoder_output, src_mask=src_mask, tgt_mask=tgt_mask, x= tgt)
    def projection(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab: int, tgt_vocab: int, src_sequence_len: int, tgt_sequence_len: int, d_model: int = 512, h: int = 8, N:int=6, dropout:float=0.01, dff:int=2048):
    src_embed = InputEmbeddings(d_model=d_model, vocabulary_size=src_vocab)
    tgt_embed = InputEmbeddings(d_model=d_model, vocabulary_size=tgt_vocab)
    src_pos = PositionalEncoding(seq_len=src_sequence_len, d_model=d_model, dropout=dropout)
    tgt_pos = PositionalEncoding(seq_len=tgt_sequence_len, d_model=d_model, dropout=dropout)

    encoder_block = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model=d_model, dff=dff, dropout=dropout)
        encoder_blocks = Encoder_Block(features=d_model, self_attention_block=encoder_self_attention, feed_forward_block=encoder_feed_forward_block, dropout=dropout)
        encoder_block.append(encoder_blocks)
    
    decoder_block = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttentionBlock(d_model=d_model, dropout=dropout, h=h)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model=d_model, dff=dff, dropout=dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, dff, dropout)
        decoder_blocks= DecoderBlock(features=d_model, self_attention_block=decoder_self_attention, cross_attention_block=decoder_cross_attention, feed_forward_block=decoder_feed_forward_block, dropout=dropout)
        decoder_block.append(decoder_blocks)
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_block))
    decoder = Decoder(d_model, nn.ModuleList(decoder_block))
    projection_layer = ProjectionLayer(d_model, tgt_vocab)
    transformer = Transformer(encoder= encoder, decoder=decoder, projection=projection_layer,src_embeddings=src_embed, tgt_embedidngs=tgt_embed, src_pos=src_pos, tgt_pos=tgt_pos)

    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
        
    return transformer
