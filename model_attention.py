import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        output, hidden = self.lstm(x)
        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, enc_outputs, dec_hidden, mask):
        """ enc_outputs: (B,N,H), dec_hidden: (B,H), mask: (B,N) """

        B, N, H = enc_outputs.shape

        hidden_expanded = dec_hidden.unsqueeze(1).repeat(1, N, 1)

        energy = torch.tanh(self.W1(enc_outputs) + self.W2(hidden_expanded))

        scores = self.v(energy).squeeze(-1)

        scores = scores - mask * 1e9

        probs = F.softmax(scores, dim=-1)

        return scores, probs


class Decoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)

    def forward(self, enc_outputs, hidden, mask):
        B, N, H = enc_outputs.shape

        dec_input = torch.zeros(B, 1, H, device=enc_outputs.device)
        dec_output, hidden = self.lstm(dec_input, hidden)

        dec_hidden = dec_output.squeeze(1)

        logits, probs, hidden = self.attention(enc_outputs, dec_hidden, mask)
        return logits, probs, hidden


class PointerNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim)

    def forward(self, coords, teacher=None):
        B, N, _ = coords.shape

        enc_outputs, hidden = self.encoder(coords)

        mask = torch.zeros(B, N, device=coords.device)
        actions = []
        log_probs = []

        for step in range(N):
            logits, probs, hidden = self.decoder(enc_outputs, hidden, mask)

            if teacher is None:
                action = torch.multinomial(probs, 1).squeeze(-1)
            else:
                action = teacher[:, step]

            log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-10)
            log_probs.append(log_prob)

            for b in range(B):
                mask[b, action[b]] = 1

            actions.append(action)

        actions = torch.stack(actions, dim=1)
        log_probs = torch.stack(log_probs, dim=1).squeeze(-1)

        return actions, log_probs
