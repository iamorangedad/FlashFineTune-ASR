import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        cross_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class ASRModel(nn.Module):
    def __init__(
        self,
        n_mels=80,
        d_model=256,
        n_heads=4,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_ff=1024,
        vocab_size=1000,
        max_len=5000,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.audio_embedding = nn.Linear(n_mels, d_model)

        self.pos_encoding = self._create_positional_encoding(max_len, d_model)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_encoder_layers)
            ]
        )

        self.text_embedding = nn.Embedding(vocab_size, d_model)

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_decoder_layers)
            ]
        )

        self.output_linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def encode(self, audio_features, src_mask=None):
        x = self.audio_embedding(audio_features)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        x = self.text_embedding(tgt) * (self.d_model**0.5)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return self.output_linear(x)

    def forward(self, audio_features, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encode(audio_features, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return dec_output

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask
