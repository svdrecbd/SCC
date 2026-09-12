"""Ordinary GRU reference for the persistent-task interface; no SCC mechanism.

Permanent input/recurrent/readout parameters and a live hidden vector are
deliberately separate. START and READ have no reset or hand-coded behavior.
"""
import torch
from torch import nn
from torch.nn import functional as F

from scc.persistent_tasks import TOKEN_COUNT, TOKENS_PER_REQUEST


class PersistentGRU(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        if width < 1:
            raise ValueError('Width must be positive')
        self.width = width
        self.recurrent = nn.GRU(TOKEN_COUNT, width, batch_first=True)
        self.readout = nn.Linear(width, 4)

    def forward(self, token_ids, state=None):
        if token_ids.ndim != 3 or token_ids.shape[-1] != TOKENS_PER_REQUEST:
            raise ValueError('Expected batch x requests x 19 tokens')
        batch, requests, ticks = token_ids.shape
        encoded = F.one_hot(token_ids.reshape(batch, requests*ticks), TOKEN_COUNT)
        encoded = encoded.to(dtype=self.readout.weight.dtype)
        sequence, state = self.recurrent(encoded, state)
        # Emit at READ positions without modifying the hidden state.
        return self.readout(sequence[:, ticks-1::ticks]), state

    def manual_tick(self, token_ids, state):
        """Independent explicit GRU equations, used only to audit execution.

        PyTorch's reset gate multiplies the projected recurrent candidate,
        including its bias. No use of GRU/GRUCell or fused RNN operators here.
        """
        encoded = F.one_hot(token_ids, TOKEN_COUNT).to(self.readout.weight.dtype)
        rnn = self.recurrent
        ir, iz, inn = F.linear(encoded, rnn.weight_ih_l0, rnn.bias_ih_l0).chunk(3, -1)
        hr, hz, hn = F.linear(state, rnn.weight_hh_l0, rnn.bias_hh_l0).chunk(3, -1)
        reset, update = (ir+hr).sigmoid(), (iz+hz).sigmoid()
        candidate = (inn + reset*hn).tanh()
        state = (1-update)*candidate + update*state
        return self.readout(state), state
