"""Primary AAM LSTM model: 2-layer LSTM with MLP head."""

import torch
import torch.nn as nn


class AAMLSTM(nn.Module):
    """LSTM-based Adaptive Arm Module for factor estimation.

    Pipeline:
        [B, T, 20] -> 2-layer LSTM -> last hidden state -> MLP -> [B, 7]
    """

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 7,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.LayerNorm(32),
            nn.Linear(32, output_size),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                start = n // 4
                end = n // 2
                param.data[start:end].fill_(1.0)

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, input_size) input features

        Returns:
            y: (B, output_size) predicted factors
        """
        lstm_out, _ = self.lstm(x)           # (B, T, hidden_size)
        last_hidden = lstm_out[:, -1, :]      # (B, hidden_size)
        last_hidden = self.layer_norm(last_hidden)
        y = self.mlp(last_hidden)             # (B, output_size)
        return y


class AAMLSTMConfig:
    """Configuration for AAM LSTM model."""

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 7,
        history_length: int = 10,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        self.history_length = history_length
