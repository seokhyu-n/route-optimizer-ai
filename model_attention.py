import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------
# Multi-head attention block
# ---------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Multi-head attention
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed Forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


# ---------------------------------------------------
# Pointer Decoder (attention scoring)
# ---------------------------------------------------
class PointerDecoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.W1 = nn.Linear(embed_dim, embed_dim)
        self.W2 = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, 1)

    def forward(self, context, enc_outputs):
        """
        context: (B, E)
        enc_outputs: (B, N, E)
        """

        # Expand context to match encoder outputs
        context = self.W1(context).unsqueeze(1)     # (B,1,E)
        enc_proj = self.W2(enc_outputs)             # (B,N,E)

        scores = self.v(torch.tanh(context + enc_proj)).squeeze(-1)
        return scores  # (B, N)


# ---------------------------------------------------
# Full Transformer Pointer Network
# ---------------------------------------------------
class TransformerPointer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, ff_hidden=256, num_layers=3):
        super().__init__()

        # Input projection (2D → embed_dim)
        self.input_proj = nn.Linear(2, embed_dim)

        # Transformer encoder stack
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_hidden)
            for _ in range(num_layers)
        ])

        # context vector (mean pooling)
        self.context_proj = nn.Linear(embed_dim, embed_dim)

        # pointer network decoder
        self.pointer = PointerDecoder(embed_dim)

    def forward(self, coords):
        """
        coords: (B, N, 2)
        """

        x = self.input_proj(coords)   # (B,N,E)

        # Transformer encoder
        for layer in self.layers:
            x = layer(x)

        # Global context vector
        context = x.mean(dim=1)            # (B,E)
        context = self.context_proj(context)

        # Pointer scoring
        scores = self.pointer(context, x)  # (B,N)

        return scores
