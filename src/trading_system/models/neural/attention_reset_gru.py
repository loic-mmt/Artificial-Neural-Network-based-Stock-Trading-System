"""Explicit recurrent controls and a documented channel-attention reset gate.

The paper's single-key temporal attention is degenerate. Our attention variant
normalizes elementwise query/key scores over hidden channels, not over time.
It is an adaptation of the reset-gate idea, not the complete MCI-GRU model.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .gru_base import GRUVariantClassifier


class ExplicitGRU(nn.Module):
    """One-layer unidirectional GRU with native-compatible gate conventions."""

    def __init__(self, native: nn.GRU, *, attention_reset=False):
        super().__init__()
        if native.num_layers != 1 or native.bidirectional or not native.batch_first or native.dropout:
            raise ValueError("Explicit-cell benchmark requires one layer, batch_first, no bidirection/dropout.")
        self.input_size = native.input_size
        self.hidden_size = native.hidden_size
        self.attention_reset = attention_reset
        if attention_reset:
            # Native ordering is reset/update/candidate. Keep update/candidate
            # initial weights exactly, but remove the unused reset parameters.
            for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
                self.register_parameter(name, nn.Parameter(
                    getattr(native, name)[self.hidden_size:].detach().clone()))
            self.query = nn.Linear(self.hidden_size, self.hidden_size)
            self.key = nn.Linear(self.input_size, self.hidden_size)
            self.value = nn.Linear(self.input_size, self.hidden_size)
            for layer in (self.query, self.key, self.value):
                nn.init.zeros_(layer.bias)
        else:
            for name, parameter in native.named_parameters():
                self.register_parameter(name, parameter)

    def reset_gate(self, values, hidden):
        """Shapes [B,F], [B,H] -> [B,H]; softmax is only along H."""
        scores = self.query(hidden) * self.key(values) / math.sqrt(self.hidden_size)
        alpha = torch.softmax(scores, dim=-1)
        # H compensates the average 1/H attention scale. The additional sigmoid
        # keeps the replacement a bounded reset gate, unlike raw alpha * value.
        return torch.sigmoid(self.hidden_size * alpha * self.value(values))

    def forward(self, sequences, hx=None):
        if sequences.ndim != 3 or sequences.shape[-1] != self.input_size or sequences.shape[1] < 1:
            raise ValueError("Expected non-empty [batch,time,input_size] sequences.")
        if hx is not None and hx.shape != (1, len(sequences), self.hidden_size):
            raise ValueError("Initial hidden shape must be [1,batch,hidden].")
        hidden = (sequences.new_zeros((len(sequences), self.hidden_size)) if hx is None
                  else hx.unbind(dim=0)[0])
        projected = F.linear(sequences, self.weight_ih_l0, self.bias_ih_l0)
        outputs = []
        for values, input_gates in zip(sequences.unbind(1), projected.unbind(1), strict=True):
            hidden_gates = F.linear(hidden, self.weight_hh_l0, self.bias_hh_l0)
            if self.attention_reset:
                x_z, x_n = input_gates.chunk(2, dim=-1)
                h_z, h_n = hidden_gates.chunk(2, dim=-1)
                reset = self.reset_gate(values, hidden)
            else:
                x_r, x_z, x_n = input_gates.chunk(3, dim=-1)
                h_r, h_z, h_n = hidden_gates.chunk(3, dim=-1)
                reset = torch.sigmoid(x_r + h_r)
            update = torch.sigmoid(x_z + h_z)
            candidate = torch.tanh(x_n + reset * h_n)
            hidden = (1 - update) * candidate + update * hidden
            outputs.append(hidden)
        return torch.stack(outputs, dim=1), hidden.unsqueeze(0)


class ManualGRUClassifier(GRUVariantClassifier):
    model_name = "gru_manual_cell"
    attention_reset = False

    def _build_module(self, torch_module, nn_module):
        module = super()._build_module(torch_module, nn_module)
        # Build the complete native model first: common pooling/head and native
        # recurrent weights have identical initialization for each paired seed.
        module.recurrent = ExplicitGRU(module.recurrent, attention_reset=self.attention_reset)
        return module


class AttentionResetGRUClassifier(ManualGRUClassifier):
    model_name = "gru_channel_attention_reset"
    attention_reset = True
