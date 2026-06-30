"""
07_bilstm_attention.py — Day 7: BiLSTM + temporal attention model.

Comparison axis 1 from THESIS_PLAN.md:
  BiLSTM (last-timestep pooling)  vs  BiLSTM + temporal attention

Identical backbone to 03_bilstm_model.py EXCEPT the final pooling step:
  - baseline (03):  takes out[:, -1, :]   (last timestep only)
  - this (07):      additive attention over all T timesteps -> weighted sum

Original 03_bilstm_model.py is NOT modified.
"""

import torch
import torch.nn as nn


class BiLSTMAttentionIntentPredictor(nn.Module):
    """
    Same backbone as baseline (input proj -> 2-layer BiLSTM), but instead
    of using only the last timestep, computes additive (Bahdanau-style)
    attention over all 16 timesteps and uses the weighted context vector.

    Why additive attention (not dot-product):
      - simpler to explain in the thesis (one small MLP scores each frame)
      - no scaling factor needed, stable for a small model
      - attention weights are directly interpretable -> good thesis figures

    return_attn=True returns per-frame weights (B, T) so we can visualize
    WHICH frames the model focused on for a given pedestrian.
    """

    def __init__(self, input_dim: int = 5, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3,
                 attn_dim: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 64)

        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        # Additive attention: score_t = v^T * tanh(W * h_t)
        # W projects each 256-d BiLSTM output to attn_dim
        # v scores it down to a scalar
        self.attn_W = nn.Linear(hidden_size * 2, attn_dim)   # 256 -> 64
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)     # 64  -> 1

        self.head = nn.Linear(hidden_size * 2, 1)            # 256 -> 1
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        x           : (batch, seq_len, input_dim)
        return_attn : if True, also return attention weights (batch, T)
        returns     : raw logits (batch, 1)
        """
        x = self.drop(torch.relu(self.input_proj(x)))    # (B, T, 64)
        H, _ = self.bilstm(x)                            # (B, T, 256)

        # ── temporal attention over T timesteps ──────────────────────────────
        scores = self.attn_v(torch.tanh(self.attn_W(H)))  # (B, T, 1)
        # (B, T, 1) sums to 1
        weights = torch.softmax(scores, dim=1)
        # (B, 256) weighted sum
        context = (weights * H).sum(dim=1)
        # ─────────────────────────────────────────────────────────────────────

        logit = self.head(self.drop(context))              # (B, 1)

        if return_attn:
            return logit, weights.squeeze(-1)              # weights: (B, T)
        return logit
