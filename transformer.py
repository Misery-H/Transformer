import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        return: [batch_size, seq_len, d_model]
        """
        return self.embedding(x) * math.sqrt(self.d_model)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super(PositionalEncoding, self).__init__()


        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd cos

        pe = pe.unsqueeze(0)
        #register as a const buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        return: same shape, with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

