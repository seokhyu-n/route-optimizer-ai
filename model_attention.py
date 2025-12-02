import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value):
        out, _ = self.attention(query, key, value)
        return out


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.linear = nn.Linear(2, embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, coords):
        x = self.linear(coords)
        x = self.norm(self.mha(x, x, x) + x)
        return x


class PointerDecoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.W1 = nn.Linear(embed_dim, embed_dim)
        self.W2 = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, 1)

    def forward(self, encoder_outputs):
        batch = encoder_outputs.size(0)
        query = self.query.repeat(batch, 1, 1)

        Wq = self.W1(query)
        We = self.W2(encoder_outputs)

        scores = self.v(torch.tanh(Wq + We)).squeeze(-1)  # (B, N)
        return scores


class TransformerPointer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.encoder = Encoder(embed_dim, num_heads)
        self.decoder = PointerDecoder(embed_dim)

    def forward(self, coords):
        x = self.encoder(coords)
        scores = self.decoder(x)
        return scores
