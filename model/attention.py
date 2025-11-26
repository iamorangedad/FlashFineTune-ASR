import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = (
            self.q_linear(q)
            .reshape(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(k)
            .reshape(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(v)
            .reshape(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k**0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        output = (
            output.transpose(1, 2).contiguous().reshape(batch_size, -1, self.d_model)
        )
        return self.out_linear(output)
