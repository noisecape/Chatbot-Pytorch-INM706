import torch
import torchvision
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, embedding_size, hidden_size, voc_len, num_layers=1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(voc_len, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers)

    def forward(self, input_seq):
        # convert words of the sequence to embeddings
        # seq_embedded should have dimension (input_seq, embedding_size) when given in input.
        # the shape of seq_embedded when embedded is (*, H) where H is the embedding_size.
        seq_embedded = self.embedding(input_seq)  # of shape (seq_len, batch_size, embedding_size)
        outputs, h_n, c_n = self.lstm(seq_embedded)
        return h_n, c_n

class Decoder(nn.Module):

    def __init__(self, embedding_size, hidden_size, voc_len, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(voc_len, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers)

    def forward(self, x, h, c):
        """
        The forward method for the decoder.
        :param x: the input word embedded to be processed by the decoder. Initially is '<S>'
        :param h: the previous hidden state. Initially is the hidden state from the encoder's output
        :param c: the previous cell state. Initially is the cell state from the encoder's output
        :return output, (h_n, c_n):
        """
        pass
