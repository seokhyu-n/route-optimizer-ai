import torch
import numpy as np

class AttentionPolicy(torch.nn.Module):
    def __init__(self, n_nodes=10, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.embed = torch.nn.Linear(2, d_model)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc_out = torch.nn.Linear(d_model, 1)

    def forward(self, coords):
        x = self.embed(coords)
        h = self.encoder(x)
        logits = self.fc_out(h).squeeze(-1)
        return logits


def load_model(model_path="attn_pointer_rl.pt", device="cpu"):
    model = AttentionPolicy()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
