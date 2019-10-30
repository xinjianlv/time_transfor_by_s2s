import torch
from torch import nn
import pdb
from Attention import *
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        # input_dim = vocab_size + 1
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.GRU(embedding_dim, hidden_dim,
                          num_layers=num_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # src = [sent len, batch size]
        embedded = self.dropout(self.embedding(input_seqs))

        # embedded = [sent len, batch size, emb dim]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # outputs, hidden = self.rnn(embedded, hidden)
        # outputs = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers, batch size, hid dim]
        # outputs are always from the last layer
        return outputs, hidden




import torch.nn.functional as F
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hid_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.attention = Attention(hidden_dim)
        self.rnn = nn.GRU(embedding_dim + hidden_dim, hidden_dim,
                          num_layers=num_layers, dropout=dropout)
        self.out = nn.Linear(embedding_dim + hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [bsz]
        # hidden = [n layers * n directions, batch size, hid dim]
        # encoder_outputs = [sent len, batch size, hid dim * n directions]
        input = input.unsqueeze(0)
        # input = [1, bsz]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, bsz, emb dim]
        attn_weight = self.attention(hidden, encoder_outputs)
        # (batch_size, seq_len)
        context = attn_weight.unsqueeze(1).bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)
        # (batch_size, 1, hidden_dim * n_directions)
        # (1, batch_size, hidden_dim * n_directions)
        emb_con = torch.cat((embedded, context), dim=2)
        # emb_con = [1, bsz, emb dim + hid dim]
        _, hidden = self.rnn(emb_con, hidden)
        # outputs = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        output = torch.cat((embedded.squeeze(0), hidden[-1], context.squeeze(0)), dim=1)
        output = F.log_softmax(self.out(output), 1)
        # outputs = [sent len, batch size, vocab_size]
        return output, hidden, attn_weight