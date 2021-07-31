"""Transformer Modules.

Defines class Transformer(nn.Module) and several other classes to
implement a transformer model having overall same interface as Pytorch's
implement (Transformer, TransformerEncoder/Decoder and 
TransformerEncoder/DecoderLayer). So docstring of the modules above is omitted
"""
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """Transformer Module"""
    def __init__(self, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                memory_mask=None, src_key_padding_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory = self.encoder(
            src, 
            mask=src_mask, 
            src_key_padding_mask=src_key_padding_mask
            )
        output = self.decoder.forward(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
            )
        return output
                

class TransformerEncoder(nn.Module):
    """The encoder in transformer"""
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, mask, src_key_padding_mask)
        
        if self.norm is not None:
            src = self.norm(src)
        return src


class TransformerEncoderLayer(nn.Module):
    """One encoder layer in the transformer"""
    def __init__(self, 
                 d_model, 
                 nhead, 
                 dim_feedforward=2048, 
                 dropout=0.1):
        super().__init__()
        self.self_attn = SublayerWrapper(
            MultiheadAttention(d_model, nhead, nn.Dropout(dropout)),
            d_model=d_model,
            dropout_r=dropout)

        self.ffn = SublayerWrapper(
            FeedForwardNet(d_model, dim_feedforward, dropout),
            d_model=d_model,
            dropout_r=dropout)
    
    # src_mask will be directly added onto attention scores.
    # src_key_padding_mask is a ByteTensor places where True located will be masked.
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn = self.self_attn(src, src, src, 
                              mask=src_mask, 
                              key_padding_mask=src_key_padding_mask)
        return self.ffn(attn)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, 
                memory_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)
            
            if self.norm is not None:
                tgt = self.norm(tgt)
            return tgt


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, 
                 dim_feedforward=2048, 
                 dropout=0.1):
        super().__init__()
        self.src_attn = SublayerWrapper(
            MultiheadAttention(d_model, nhead, nn.Dropout(dropout)),
            d_model=d_model,
            dropout_r=dropout)

        self.self_attn = SublayerWrapper(
            MultiheadAttention(d_model, nhead, nn.Dropout(dropout)),
            d_model=d_model,
            dropout_r=dropout)

        self.ffn = SublayerWrapper(
            FeedForwardNet(d_model, dim_feedforward, dropout),
            d_model=d_model,
            dropout_r=dropout)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, 
                memory_key_padding_mask=None):
        tgt = self.self_attn(tgt, tgt, tgt, 
                             mask=tgt_mask, 
                             key_padding_mask=tgt_key_padding_mask)

        tgt = self.src_attn(tgt, memory, memory, 
                            mask=memory_mask, 
                            key_padding_mask=memory_key_padding_mask)

        return self.ffn(tgt)


class PositionalEncoding(nn.Module):
    """Add positional encoding to token embedding vector"""
    def __init__(self, embedding_dim, dropout_r, max_len=5000):
        """Add positional encoding to token embedding vector

        :param embedding_dim: The embedding dim of positional encoding
        :param dropout_r: Ratio of combined embedding vector dropout
        :param max_len: Max length positinal encoding can be generated, defaults to 5000
        """
        super().__init__()
        den = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000)) / embedding_dim)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, embedding_dim))
        pos_embedding[:, 0::2] = torch.sin(den * pos)
        pos_embedding[:, 1::2] = torch.cos(den * pos)
        pos_embedding = pos_embedding.unsqueeze(-2) # Reshape into [max_len, 1, emb_size]

        self.dropout = nn.Dropout(dropout_r)

        # Register some variables as buffers when
        # 1) The values don't join compute graph and don't require gradients
        # 2) Wish model.save() could save the variables
        # 3) Wish model.to(device) could apply on the variables
        self.register_buffer('pos_embedding', pos_embedding)
    
    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class SublayerWrapper(nn.Module):
    def __init__(self, sub_layer, d_model, dropout_r):
        super().__init__()
        self.dropout = nn.Dropout(dropout_r)
        self.layernorm = nn.LayerNorm(d_model)
        self.sub_layer = sub_layer
    
    def forward(self, *args, **kwargs):
        output = self.sub_layer(*args, **kwargs)
        return self.layernorm(args[0] + self.dropout(output))


def attention(query, key, value, mask=None, key_padding_mask=None, dropout=None):
    """Compute scale dot product attention for given Q, K, V

    Below S is the length of query, N is the batch size and E is the feature numbers,
    T is the length of key and value. Particularly in self attention, S=T; in source attention, S!=T.
    Assume in any condition, length of key and value is the same.

    :param query: Q tensor :math:`(S, N, E)` or `(S, N_HEAD, N, E)`
    :param key: K tensor :math:`(T, N, E)` or `(T, N_HEAD, N, E)`
    :param value: V tensor :math:`(T, N, E)` or `(T, N_HEAD, N, E)`
    :param mask: Mask of QKV tensor :math:`(S, T)` or `(N, N_HEAD, S, T)`, this mask 
    will be added onto scores directly.
    :param key_padding_mask: ByteTensor mask indicating padding tokens' place.
    place where is one shows there is a padding tokens and will be maksed.
    shape: :math:`(N, N_Head, 1, T)`
    :param dropout: dropout module will be applied defaults to None
    :return: Attention values with shape:math:`(S, N, E)` or `(*, N, E)`
    and global align weights with shape :math:`(S, N, N)` or `(*, N, N)`
    """
    d_k = query.size(-1)

    # 3d tensor matrix multiplication in torch
    # The last two dimension will function as normal
    # matrix multiplication while other dimensions will
    # act as element-wise action.
    # In order to correctly make the mask action batch-wised, 
    # First is to turn all qkv into batch_first. After we turned qkv into 
    # (N, S/T, E), we multiply (N, S, E) and (N, E, T) will get (N, S, T),
    # which means for all N batches, the T key scores(last dim) for all S queries(second dim)

    # 1) permute batch dim to the first place (S/T, N_HEAD, N, E) to (N, N_HEAD, S/T, E).
    query = query.transpose(0, -2)
    key = key.transpose(0, -2)
    value = value.transpose(0, -2)

    # 2) Use batched matrix multiplication to get a score tensor(N, N_HEAD, S, T)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

    # 3) Use mask to set some scores to -inf and softmax
    # For key_padding_mask, set score to -infinity will make its weight being zero
    # As key_padding_mask shape:(N, 1, 1, T), directly using masked_fill wil broadcast
    # it to (N, N_HEAD, S, T) and all scores will be masked correctly
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask, -1e9)

    # For mask, the only thing need to do is to added it onto scores directly so it
    # will be broadcast to (N, N_HEAD, S, T), applying same sequential mask to all
    # batches equally.
    if mask is not None:
        scores += mask
    weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        weights = dropout(weights)
    
    # 4)Compute all weighted values and transpose
    # Final attention shape:(N, N_HEAD, T, E), transpose() is needed.
    return (weights @ value).transpose(0, -2), weights.transpose(0, -2)


class MultiheadAttention(nn.Module):
    """MultiHead Attention module"""
    def __init__(self, d_model, nhead, dropout):
        """MultiHead Attention module"""
        super().__init__()

        # All nhead features will be combined back to one feature with d_model dim
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.d_model = d_model
        self.nhead = nhead
        self.att_weights = None

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, query, key, value, mask=None, key_padding_mask=None):
        """Forward propagation with multihead attention
        
        :param query: Q tensor :math:`(L, N, E)`
        :param key: K tensor :math:`(L, N, E)`
        :param value: V tensor :math:`(L, N, E)`
        :param mask: Mask of QKV tensor :math:`(L, N, E)`, places where
                     mask value is zero will not be computed, defaults to None
        """
        if mask is not None:
            # Expand mask shape from (L, L) to (1, 1, L, L) for the convenience to broadcast
            # to (N, N_HEAD, L, L) to apply on all heads equally
            mask = mask.unsqueeze(0).unsqueeze(0)
        
        if key_padding_mask is not None:
            # Expand padding mask shape from (N, L) to (N, 1, 1, L) in order to euqally mask 
            # different padding tokens for the correspnding batch.
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
        n_qlen, n_batch, d_model = query.shape
        n_klen = key.size(0)

        # Compute nhead features from q, k and v.
        # Transpose them into shape (L, N_HEAD, N, K) for convenience of parallelization
        query = self.q_linear(query).view(n_qlen, n_batch, self.nhead, self.d_k).transpose(1, 2)
        key = self.k_linear(key).view(n_klen, n_batch, self.nhead, self.d_k).transpose(1, 2)
        value = self.v_linear(value).view(n_klen, n_batch, self.nhead, self.d_k).transpose(1, 2)

        # Do attention calculation and concatenate
        att_value, self.att_weights = attention(query, key, value, mask, key_padding_mask, self.dropout)
        att_value = att_value.transpose(1, 2).contiguous().view(n_qlen, n_batch, d_model)

        return self.out(att_value)


class FeedForwardNet(nn.Module):
    """A two-layer relu feedforward network following a multihead attention"""
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input):
        return self.fc2(self.dropout(F.relu(self.fc1(input))))


class TokenEmbedding(nn.Module):
    """Word embedding but scaled"""
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
    
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_dim)


class Generator(nn.Module):
    """A softmax regression at the top of the decoder"""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.out = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(-1)
    
    def forward(self, input):
        return self.softmax(self.out(input))


class Seq2seqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 num_encoder_layers=6, num_decoder_layers=6,
                 d_model=512, num_heads=8, dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        self.transformer = Transformer(
            d_model=d_model, nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout)
        
        self.generator = Generator(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask, tgt_mask,
                src_padding_mask, tgt_padding_mask):
        src_embedded = self.positional_encoding(self.src_embedding(src))
        tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt))

        output = self.transformer.forward(
            src=src_embedded, tgt=tgt_embedded,
            src_mask=src_mask, tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask)
        
        return self.generator(output)