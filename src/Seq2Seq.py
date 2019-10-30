from numpy import *
import torch
from torch import nn
import pdb
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.5):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src_seqs, src_lengths, trg_seqs):
        # src_seqs = [sent len, batch size]
        # trg_seqs = [sent len, batch size]
        batch_size = src_seqs.shape[1]
        max_len = trg_seqs.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        # hidden used as the initial hidden state of the decoder
        # encoder_outputs used to compute context
        encoder_outputs, hidden = self.encoder(src_seqs, src_lengths)
        # first input to the decoder is the <sos> tokens
        output = trg_seqs[0, :]
        for t in range(1, max_len):  # skip sos
            output, hidden, _ = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < self.teacher_forcing_ratio
            output = (trg_seqs[t] if teacher_force else output.max(1)[1])
        return outputs

    def predict(self, src_seqs, src_lengths, max_trg_len=20, start_ix=1):
        max_src_len = src_seqs.shape[0]
        batch_size = src_seqs.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src_seqs, src_lengths)
        output = torch.LongTensor([start_ix] * batch_size).to(self.device)
        attn_weights = torch.zeros((max_trg_len, batch_size, max_src_len))
        for t in range(1, max_trg_len):
            output, hidden, attn_weight = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            output = output.max(1)[1]
            # attn_weights[t] = attn_weight
        return outputs, attn_weights

