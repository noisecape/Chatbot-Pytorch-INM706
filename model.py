import torch
import torchvision
import torch.nn as nn
import numpy as np


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
        outputs, (h_n, c_n) = self.lstm(seq_embedded)
        return h_n, c_n


class Decoder(nn.Module):

    def __init__(self, embedding_size, hidden_size, voc_len, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(voc_len, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers)
        self.fc_1 = nn.Linear(hidden_size, voc_len)

    def forward(self, x, h, c):
        """
        The forward method for the decoder.
        :param x: the input word embedded to be processed by the decoder. Initially is '<S>'
        :param h: the previous hidden state. Initially is the hidden state from the encoder's output
        :param c: the previous cell state. Initially is the cell state from the encoder's output
        :return output, (h_n, c_n):
        """
        # at this stage x has the dimension of (64). Hovewer it must be (1,64)
        x = x.unsqueeze(0)
        embedded_word = self.embedding(x)
        output, (h, c) = self.lstm(embedded_word, (h, c))
        predictions = self.fc_1(output)
        # remove the 'extra' dimension
        return predictions.squeeze(0), h, c


class ChatbotModel(nn.Module):

    def __init__(self, encoder, decoder, vocab_size):
        super(ChatbotModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size

    def forward(self, X, y, tf_ratio=0.5):
        seq_len = y.shape[0]
        batch_size = X.shape[1]
        # this will store all the outputs for the batches
        outputs = torch.zeros(seq_len, batch_size, self.vocab_size)
        # compute the hidden and cell state from the encoder
        h_n, c_n = self.encoder(X)
        # initially consider the <S> token for all the batches
        word_t = y[0]
        # compute the predictions through the decoder
        for t in range(seq_len):
            # compute output, hidden state and cell state
            output, h_n, c_n = self.decoder(word_t, h_n, c_n)
            # update the data structure to hold outputs
            outputs[t] = output
            # take the best prediction from the vocabulary.
            # Since the dimension in (batch_size, len_voc), take the argmax of the second dimension
            # to get the best prediction for each sentence in the batch.
            prediction = output.argmax(1)
            # use teaching forcing to randomly chose the next input for the decoder
            probabilities = [tf_ratio, 1-tf_ratio]
            idx_choice = np.argmax(np.random.multinomial(1, probabilities))
            word_t = y[t] if idx_choice == 0 else prediction

        return outputs








