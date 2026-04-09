"""
Deep Learning Laboratory — Architecture Implementations
========================================================
Three alternative architectures to PatchTST, all accepting the same
feature inputs: (batch, lookback=60, n_features=10).

Architecture Theory
-------------------
MLPBaseline:
    Feedforward NN — the simplest possible learner.
    Flattens the entire (60 × 10) window into a 600-dim vector, then
    applies stacked nn.Linear layers with an activation and dropout.
    No sense of temporal order — treats day 1 and day 60 as equal peers.
    Useful as a lower bound to confirm that architectures which exploit
    temporal structure add real value.

CNN1DModel:
    Convolutional 1-D — extracts local temporal patterns.
    nn.Conv1d slides a filter of width `kernel_size` (e.g., 5 days) along
    the time axis, learning short-range dependencies (e.g., "3-day reversal",
    "momentum burst").  Multiple stacked conv layers increase the receptive
    field: layer 1 sees 5 days, layer 2 sees ~25 days, etc.
    Global average pooling then collapses the temporal dimension so the head
    is input-length-agnostic.  Faster and more parameter-efficient than LSTM
    for fixed-window problems.

LSTMModel:
    Recurrent NN — processes time steps sequentially.
    Hidden state h_t = f(x_t, h_{t-1}) carries a "memory" of all past steps.
    Great at capturing long-range dependencies but:
      - Sequential — cannot be parallelised like Conv or Attention.
      - Vanishing gradient for very long sequences (partially solved by gates).
    PatchTST outperforms LSTM on long horizons because Attention can directly
    attend to any step without routing through every intermediate hidden state.

All models share the same output: a single scalar (5-day forward return).
"""

import torch
import torch.nn as nn


# ── MLP Baseline ─────────────────────────────────────────────────────────────

class MLPBaseline(nn.Module):
    """
    Multi-Layer Perceptron — fully connected feedforward network.

    Input  : (batch, lookback, n_features)   — flattened internally
    Output : (batch,)                         — scalar return forecast

    Key hyperparameters
    -------------------
    hidden_sizes : list of ints — neuron counts per hidden layer
                   e.g. [256, 128, 64] → 3 hidden layers
    activation   : nn.Module class — e.g. nn.ReLU, nn.Tanh
    dropout      : float [0, 1] — fraction of neurons zeroed per forward pass
                   0.0 → no regularisation (likely to overfit on small datasets)
                   0.3 → moderate; forces redundant representations
    """

    def __init__(
        self,
        lookback: int = 60,
        n_features: int = 10,
        hidden_sizes: list = None,
        activation=None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]
        if activation is None:
            activation = nn.ReLU

        in_dim = lookback * n_features       # flatten entire window

        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))   # stabilises training depth
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))    # output head

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)  →  flatten  →  (B, T*C)
        B = x.shape[0]
        return self.net(x.reshape(B, -1)).squeeze(-1)


# ── 1-D CNN ──────────────────────────────────────────────────────────────────

class CNN1DModel(nn.Module):
    """
    1-D Convolutional network for temporal feature extraction.

    Input  : (batch, lookback, n_features)
    Output : (batch,)

    Conv1d convention: PyTorch Conv1d expects (batch, channels, length).
    Here features = channels, time = length.

    Key hyperparameters
    -------------------
    channels     : list of ints — out-channels per conv layer
    kernel_size  : int — temporal receptive field per layer (default 5 days)
    dropout      : float — applied after each conv block
    activation   : nn.Module class

    Receptive field math
    --------------------
    After L stacked Conv1d(kernel_size=k) layers with no dilation:
        effective_rf = 1 + L*(k-1)
    With 3 layers and k=5: rf = 1 + 3*(5-1) = 13 trading days.
    Increase dilation=[1,2,4] to reach 25+ days with fewer params.
    """

    def __init__(
        self,
        n_features: int = 10,
        channels: list = None,
        kernel_size: int = 5,
        dropout: float = 0.2,
        activation=None,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]
        if activation is None:
            activation = nn.ReLU

        conv_blocks = []
        in_ch = n_features
        for out_ch in channels:
            conv_blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),   # BatchNorm after conv is standard practice
                activation(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch

        self.conv_stack = nn.Sequential(*conv_blocks)
        # Global Average Pooling: average over time dimension → (B, C)
        # Makes the model agnostic to input length (useful for variable windows).
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(channels[-1], 64),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)  →  (B, C, T)  for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv_stack(x)       # (B, out_ch, T)
        x = self.gap(x).squeeze(-1)  # (B, out_ch)
        return self.head(x).squeeze(-1)


# ── LSTM ─────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Long Short-Term Memory recurrent network.

    Input  : (batch, lookback, n_features)
    Output : (batch,)

    LSTM gate equations (for intuition):
        f_t = σ(W_f [h_{t-1}, x_t] + b_f)   ← forget gate
        i_t = σ(W_i [h_{t-1}, x_t] + b_i)   ← input gate
        c_t = f_t * c_{t-1} + i_t * tanh(W_c [h_{t-1}, x_t] + b_c)   ← cell
        o_t = σ(W_o [h_{t-1}, x_t] + b_o)   ← output gate
        h_t = o_t * tanh(c_t)                ← hidden state

    Key hyperparameters
    -------------------
    hidden_size  : int — width of the hidden state vector
    num_layers   : int — stacked LSTM layers (deep LSTM)
                   More layers → more abstraction but harder to train.
    dropout      : float — applied between LSTM layers (Zaremba dropout)
                   Note: dropout is NOT applied on the last layer's output
                   (PyTorch behaviour) — we add an explicit Dropout before head.
    bidirectional: bool — processes sequence forward AND backward.
                   Doubles hidden state size; useful when future context
                   helps (e.g., mid-window pattern recognition).
                   For causal/real-time prediction, keep False.
    """

    def __init__(
        self,
        n_features: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        activation=None,   # kept for API consistency — LSTM uses its own gates
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        lstm_out_dim = hidden_size * direction_factor

        self.post_lstm_drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU() if activation is None else activation(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_features)
        out, _ = self.lstm(x)        # out: (B, T, hidden * directions)
        last    = out[:, -1, :]      # take final hidden state only
        last    = self.post_lstm_drop(last)
        return self.head(last).squeeze(-1)
