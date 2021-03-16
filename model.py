import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, embedding_size, hidden_size, voc_len, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(voc_len, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers)

    def forward(self, input_seq):
        # convert words of the sequence to embeddings
        # seq_embedded should have dimension (input_seq, embedding_size) when given in input.
        # the shape of seq_embedded when embedded is (*, H) where H is the embedding_size.
        seq_embedded = self.embedding(input_seq)  # of shape (seq_len, batch_size, embedding_size)
        output, (h_n, c_n) = self.lstm(seq_embedded)
        return h_n, c_n


class EncoderAttention(nn.Module):

    def __init__(self, embedding_size, hidden_size, voc_len, num_layers=1):
        super(EncoderAttention, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(voc_len, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True)

    def forward(self, input_seq):
        seq_embedded = self.embedding(input_seq)
        encoder_outputs, (h_n, c_n) = self.lstm(seq_embedded)

        # because of the bidirection both h_n and c_n have 2 tensors (forward, backward), but
        # the decoder is not bidirection, thus only accepts one tensor.
        # to solve the problem, add them together and obtain one unified tensor.
        h_n = h_n[0:1, :, :] + h_n[1:2, :, :]
        c_n = c_n[0:1, :, :] + c_n[1:2, :, :]

        return encoder_outputs, h_n, c_n


class Decoder(nn.Module):

    def __init__(self, embedding_size, hidden_size, voc_len, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(voc_len, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=False)
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
        return predictions.squeeze(0), h


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        # hidden_size*3 is given by the fact that we concatenate the prev_hidden (of size hidden_size)
        # with the encoder states (of size hidden_size*2 because it's bidirectional)
        # the output is one because we want to output one value for each word in the batch (attention weight)
        self.attn = nn.Linear(self.hidden_size*3, 1)

    def forward(self, prev_hidden, encoder_outputs):
        """
        This function computes the attention vector which will be used to measure
        the relevance of different words in the sentence.
        :param prev_hidden: the previous hiddes state of the decoder (s_t-1)
        :param encoder_outputs: the outputs from the encoder (H)
        :return attention: the vector of shape [seq_len, batch] that contains the probabilities which reflects the
        level of 'relevance' for each word.
        """
        # encoder_outputs -> [seq_len, batch, hidden_size*2]
        # prev_hidden -> [seq_len, batch, hidden_size]

        # concatenate the previous hidden state and the encoder's output
        input_concat = torch.cat((prev_hidden, encoder_outputs), dim=2)
        # compute the energy values through the 'small' neural network attention.
        energy = self.attn(input_concat)
        # compute the so called 'attentional hidden state'
        energy = torch.tanh(energy)
        # apply the score function (dot product) to compute the score
        score = torch.sum(prev_hidden * energy, dim=2)
        # apply softmax layer to obtain the probability distribution.
        attention = F.softmax(score, dim=0)
        # finally we'd like to append one dimension at the end for later operations.
        return attention.unsqueeze(2)


class LuongAttentionDecoder(nn.Module):

    def __init__(self, embedding_size, hidden_size, voc_len, attention, n_layers=1):
        super(LuongAttentionDecoder, self).__init__()

        self.embedding = nn.Embedding(voc_len, embedding_size)
        self.hidden_size = hidden_size
        self.attention = attention
        self.lstm = nn.LSTM((self.hidden_size*2) + embedding_size, hidden_size, num_layers=n_layers)
        self.fc = nn.Linear(hidden_size, voc_len)

    def forward(self, word, prev_hidden, prev_cell, encoder_outputs):
        """
        This function realized the attention mechanism applying the Luong et al calculation method.
        :param word: the word per batch taken in input by the Decoder.
        :param prev_hidden: the previous hidden state (s_t-1) of the Decoder
        :param prev_cell: the previous cell state of the Decoder
        :param encoder_outputs: the Encoder's output.
        :return predictions, h, c: the predictions for the next word; the new hidden and cell state.
        """
        # word -> [1, batch_size]

        # remove the first dimension to perform the embedding
        word = word.unsqueeze(0)
        # embed each word of the batch
        embedded = self.embedding(word)
        # repeat the prev_hidden because it contains one tensor but we want to concatenate it
        # with encoder_outputs which has T tensors
        prev_hidden_repeated = prev_hidden.repeat(encoder_outputs.shape[0], 1, 1)
        # compute the attention values that will specify what part of the input sentence is more relevant
        attention = self.attention(prev_hidden_repeated, encoder_outputs)

        # in order to combine the attention weights with the encoder outputs we want to multiply
        # them element wise. To do that we have to adjust the dimensions of those data structures.
        # the multiplication is achieved by the 'torch.bmm' operator. This will output the new context vector.
        attention = attention.permute(1, 2, 0)  # -> [batch, 1, seq_len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # -> [batch, seq_len, hidden*2]
        context_vector = torch.bmm(attention, encoder_outputs)  # -> [batch, 1, hidden*2]
        context_vector = context_vector.permute(1, 0, 2) # -> [1, batch, hidden*2]
        # Finally, concatenate the context vector and the embedded word to build the
        # new input to feed forward pass to the lstm network.
        new_input = torch.cat((context_vector, embedded), dim=2)
        outputs, (h, c) = self.lstm(new_input, (prev_hidden, prev_cell))
        # to get the predictions feed forward pass the outputs from the decoder to a final fully connected layer.
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        return predictions, h, c

class ChatbotModel(nn.Module):

    def __init__(self, encoder, decoder, vocab_size, attention):
        super(ChatbotModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.attention = attention

    def forward(self, X, y, tf_ratio=0.5):
        """
        This function implement the forward method for both the regular Decoder and the
        one with the Attention mechanism implemented. This aspect is handled by a simple else-if control.
        In order to achieve a better convergence, the teaching force technique is used with probability of 0.5
        :param X: the batch of input sentences.
        :param y: the batch of target senteces.
        :param tf_ratio: teaching force ratio used to choose the next word to pass to the decoder.
        :return outputs: the outputs of the Decoder for each batch.
        """
        seq_len = y.shape[0]
        batch_size = X.shape[1]
        # this will store all the outputs for the batches
        outputs = torch.zeros(seq_len, batch_size, self.vocab_size)
        # compute the hidden and cell state from the encoder
        if self.attention:
            encoder_outputs, h_n, c_n = self.encoder(X)
            word_t = y[0]
            for t in range(seq_len):
                output, h_n, c_n = self.decoder(word_t, h_n, c_n, encoder_outputs)
                outputs[t] = output
                prediction = output.argmax(1)
                probabilities = [tf_ratio, 1 - tf_ratio]
                idx_choice = np.argmax(np.random.multinomial(1, probabilities))
                word_t = y[t] if idx_choice == 0 else prediction
        else:
            h_n, c_n = self.encoder(X)
            # initially consider the <S> token for all the batches
            word_t = y[0]
            # compute the predictions through the decoder
            for t in range(seq_len):
                # compute output, hidden state and cell state
                output, h_n = self.decoder(word_t, h_n, c_n)
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








