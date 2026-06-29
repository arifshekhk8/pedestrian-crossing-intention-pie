"""
03b_bilstm_model_flex.py — Flexible BiLSTM (input_dim is a parameter).

Identical architecture to 03_bilstm_model.py but accepts any input_dim.
Used for ablation: pass input_dim=4 for bbox-only, input_dim=5 for bbox+speed.

Architecture (locked from THESIS_PLAN.md):
  input_dim -> Linear(input_dim, 64) -> 2-layer BiLSTM(hidden=128) ->
  last timestep -> Linear(256, 1) -> raw logit
"""

import torch
import torch.nn as nn


class BiLSTMIntentPredictorFlex(nn.Module):
    """
    Same as BiLSTMIntentPredictor but input_dim is configurable.
    Default input_dim=5 keeps it compatible with the original baseline.
    Pass input_dim=4 for the bbox-only ablation (no ego-speed column).
    """

    def __init__(self, input_dim: int = 5, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 64)

        # bidirectional=True doubles the output size: hidden_size * 2
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        # 128 * 2 = 256 because bidirectional
        self.head = nn.Linear(hidden_size * 2, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, input_dim)
        returns: (batch, 1) raw logits — sigmoid applied outside in loss / eval
        """
        x = self.drop(torch.relu(self.input_proj(x)))   # (B, T, 64)
        out, _ = self.bilstm(x)                          # (B, T, 256)
        last = out[:, -1, :]                             # (B, 256) last timestep
        return self.head(self.drop(last))                # (B, 1)