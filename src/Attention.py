import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import pdb
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        self.v.data.normal_(mean=0, std=1. / np.sqrt(self.v.size(0)))

    def forward(self, hidden, encoder_outputs):
        #  encoder_outputs:(seq_len, batch_size, hidden_size)
        #  hidden:(num_layers * num_directions, batch_size, hidden_size)
        max_len = encoder_outputs.size(0)

        hh = hidden[-1].repeat(max_len, 1, 1)
        # (seq_len, batch_size, hidden_size)

        attn_energies = self.score(hh, encoder_outputs)  # compute attention score

        dot_attn_energies = self.score(hh , encoder_outputs)

        # pdb.set_trace()

        return F.softmax(dot_attn_energies, dim=1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        # (seq_len, batch_size, 2*hidden_size)-> (seq_len, batch_size, hidden_size)
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.permute(1, 2, 0)  # (batch_size, hidden_size, seq_len)
        v = self.v.repeat(encoder_outputs.size(1), 1).unsqueeze(1)  # (batch_size, 1, hidden_size)
        energy = torch.bmm(v, energy)  # (batch_size, 1, seq_len)
        return energy.squeeze(1)  # (batch_size, seq_len)

    def dot_score(self , hidden , encoder_outputs):
        c = torch.sum(hidden * encoder_outputs , dim = 2)
        return c
