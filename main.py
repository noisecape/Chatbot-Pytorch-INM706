import re
import os
from collections import OrderedDict
from dataset import CornellCorpus
import torch
from torch.utils.data import DataLoader
from model import Encoder, Decoder, ChatbotModel, EncoderAttention, LuongAttentionDecoder, Attention, GreedySearch
from torch import optim
import torch.nn as nn
import numpy as np
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# check cuda availability
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

print(device)

# define the dir where to save the trained model
save_model_dir = 'saved_models'
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)


def get_movie_lines(path):
    """
    This function extracts the movie lines id and the text associated
    and store them in a dictionary.
    :param path: the path where to find the file 'movie_lines.txt'
    :return line_to_phrase: the dictionary that maps each line id to the corresponding text
    """
    file = open(path, 'r', encoding='iso-8859-1')
    dialog_data = []
    line_to_phrase = {}
    for line in file.readlines():
        dialog_data.append(line.strip().split(sep=' +++$+++ '))
    for information in dialog_data:
        line_to_phrase[information[0]] = information[-1].replace('\n', '')
    return line_to_phrase


def extract_dialogs():
    """
    This function extracts dialogs from each movie. A dialog is represented by
    a list on lineid which identifies a unique conversation in the dataset.
    :return conversation:
    """
    PATH_CONVERSATION = os.path.join(os.curdir, 'cornell-movie-dialogs-corpus/movie_conversations.txt')
    file = open(PATH_CONVERSATION, 'r', encoding='iso-8859-1')
    dialog_list = []

    # extract conversations info from 'movie_conversation.txt'
    for line in file.readlines():
        line = line.split(' +++$+++ ')
        regex = re.compile('[^a-zA-Z0-9.,!?]')
        line = regex.sub('', line[-1])
        line = line.split(',')
        dialog_list.append(line)

    return dialog_list


def create_pair_dialogs(dialogs):
    # dictionary that stores the following [question] -> [answer] for each line in a dialog
    dialogs_pairs = []
    for dialog in dialogs:
        question_to_answer = {}
        for id in range(len(dialog) - 1):
            question_line = dialog[id]
            answer_line = dialog[id+1]
            # check if either the answer or the question is empty and if that's the case don't append it.
            if question_line and answer_line:
                question_to_answer[question_line] = answer_line
                dialogs_pairs.append(question_to_answer)
    return dialogs_pairs


def save_models(model):
    PATH = save_model_dir + '/chatbot_model'
    torch.save(model.state_dict(), PATH)


def load_model(model):
    PATH = save_model_dir + '/chatbot_model'
    model.load_state_dict(torch.load(PATH))


def format_time(start, end):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_secs, elapsed_mins


class Vocabulary:

    def __init__(self, idx_to_text, dialogs_ids):
        self.dialogs_ids = dialogs_ids
        self.idx_to_text = self.normalize_sentence(idx_to_text)
        self.word_to_idx = self.map_word_to_idx()
        self.vocab = self.map_idx_to_word()
        # self.delete_unused_word()

    def __len__(self):
        return len(self.vocab)

    def normalize_sentence(self, idx_to_text):
        normalized_idx_to_sentence = {}
        for line_id, sentence in zip(idx_to_text.keys(), idx_to_text.values()):
            # convert each word in the sentence to a lower case
            sentence = sentence.lower()
            # eliminate extra spaces for punctuation
            sentence = re.sub(r"([.,!?])", r" \1", sentence)
            # remove non-letter characters but keep regular punctuation
            sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
            normalized_idx_to_sentence[line_id] = sentence
        return normalized_idx_to_sentence

    def map_idx_to_word(self):
        words = self.word_to_idx.keys()
        index = self.word_to_idx.values()
        idx_to_word = OrderedDict()
        for w, i in zip(words, index):
            idx_to_word[i] = w
        return idx_to_word

    def map_word_to_idx(self):
        word_to_idx = OrderedDict()
        count_words = 0
        pad_token = '<PAD>'
        word_to_idx[pad_token] = count_words
        count_words += 1
        for dialogs in self.dialogs_ids:
            for line in dialogs:
                sentence = self.idx_to_text[line]
                for word in sentence.strip().split():
                    if word not in word_to_idx:
                        word_to_idx[word] = count_words
                        count_words += 1
        start_token = '<S>'
        word_to_idx[start_token] = count_words
        count_words += 1
        end_token = '</S>'
        word_to_idx[end_token] = count_words
        count_words += 1
        unknown_token = '<UNK>'
        word_to_idx[unknown_token] = count_words
        return word_to_idx


def select_n_pairs(data, limit):
    """
    This function limits the number of pairs to use in the dataset
    :param data: the list of pairs, of shape [question, answer]
    :param limit: the integer that represents the number of pairs to select from the data
    :return data: the trimmed data with 'limit' number of pairs.
    """
    return data[:limit]


def train_loop():
    batch_history = []
    model.train()
    avg_loss = 0
    for idx, X in enumerate(train_dataloader):
        # transpose both input sentence and target sentence to access using the first dimension
        # the the i-th word for each batch at each given time step t.
        question = torch.transpose(X[0].to(device), 0, 1)
        answer = torch.transpose(X[1].to(device), 0, 1)
        # compute the output. Recall the output size should be (seq_len, batch_size, voc_size)
        output = model(question, answer)
        # don't consider the first element in all batches because it's the '<S>' token
        output = output[1:].to(device)
        answer = answer[1:].to(device)
        # reshape both question and answer to the correct size for the loss function
        output = output.reshape(-1, output.shape[2])
        answer = answer.reshape(-1)
        # default previous weights values
        optim.zero_grad()
        # compute loss to backpropagate
        loss = criterion(output, answer)
        loss = loss / X[0].shape[1]
        # backpropagate to compute gradients
        loss.backward()
        # clip gradients to avoid exploding values
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optim.step()
        batch_history.append(loss.item())
        # print current loss every 500 processed batches
        if idx % 10 == 0:
            path_checkpoint = os.path.join(os.curdir, 'saved_models/checkpoint.pth')
            print('BATCH [{}/{}], LOSS: {}'.format(idx, train_dataloader.__len__(), loss))
    avg_loss = np.sum(batch_history) / train_dataloader.__len__()
    return avg_loss

def val_loop():
    model.eval()
    batch_history = []
    avg_loss = 0
    for idx, X in enumerate(val_dataloader):
        # transpose both input sentence and target sentence to access using the first dimension
        # the the i-th word for each batch at each given time step t.
        question = torch.transpose(X[0].to(device), 0, 1)
        answer = torch.transpose(X[1].to(device), 0, 1)
        # compute the output. Recall the output size should be (seq_len, batch_size, voc_size)
        output = model(question, answer)
        # don't consider the first element in all batches because it's the '<S>' token
        output = output[1:].to(device)
        answer = answer[1:].to(device)
        # reshape both question and answer to the correct size for the loss function
        output = output.reshape(-1, output.shape[2])
        answer = answer.reshape(-1)
        # compute the loss
        loss = criterion(output, answer)

        # keep track of the loss
        batch_history.append(loss.item())
        # print current loss every 500 processed batches
        if idx % 10 == 0:
            print('BATCH [{}/{}], LOSS: {}'.format(idx, val_dataloader.__len__(), loss))
    avg_loss = np.sum(batch_history) / val_dataloader.__len__()
    return avg_loss


dir = os.path.join(os.curdir, 'cornell-movie-dialogs-corpus/movie_lines.txt')
idx_to_text = get_movie_lines(dir)
# create a list of dialogs for each movie.
dialogs = extract_dialogs()
# for each movie, create pairs dialogs (Q/A). This is the actual data used for training.
pair_dialogs_idx = create_pair_dialogs(dialogs)
# limit pairs for batch building
pair_dialogs_idx = select_n_pairs(pair_dialogs_idx, 100000)
# instantiate the vocabulary
vocabulary = Vocabulary(idx_to_text, dialogs)
print('Total words counted in the vocabulary: {}'.format(vocabulary.__len__()))

# create the dataset class to store the data
train_data = CornellCorpus(pair_dialogs_idx, vocabulary, train_data=True)
val_data = CornellCorpus(pair_dialogs_idx, vocabulary, train_data=False)

# hyperparameters
batch_size = 512
hidden_size = 128
embedding_size = 128
epochs = 25
optim_parameters = {'lr': 1e-5, 'weight_decay': 1e-3}

# init dataloader
load_args = {'batch_size': batch_size, 'shuffle': True}
train_dataloader = DataLoader(train_data, **load_args)
val_dataloader = DataLoader(val_data, **load_args)
print(train_dataloader.__len__())
print(val_dataloader.__len__())
# init seq2seq model, the parameters needed are
# embedding_size -> the size of the embedding for each word
# hidden_size -> the number of hidden neurons per unit
# voc_size -> the size of the vocabulary to embed each word
# encoder = Encoder(embedding_size, hidden_size, vocabulary.__len__()).to(device)
# decoder = Decoder(embedding_size, hidden_size, vocabulary.__len__()).to(device)

encoder = EncoderAttention(embedding_size, hidden_size, vocabulary.__len__())
attention = Attention(hidden_size)
decoder = LuongAttentionDecoder(embedding_size, hidden_size, vocabulary.__len__(), attention=attention)

model = ChatbotModel(encoder, decoder, vocabulary.__len__(), attention=True).to(device)

print(model.state_dict())
#init the optimizer
optim = optim.Adam(model.parameters(), **optim_parameters)
# init loss function
# when computing the loss you want to ignore the '<PAD>' token.
pad_index = vocabulary.word_to_idx['<PAD>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
epoch_history = []

### TRAIN LOOP ###

checkpoint_path = os.path.join(os.curdir, 'saved_models/checkpoint.pth')
path_saved_model = os.path.join(os.curdir, 'saved_models/trained_model.pth')
# check if the model is already trained
if os.path.exists(path_saved_model):
    # load state_dict
    model.load_state_dict(torch.load(path_saved_model, map_location=torch.device(device)))
else:
    # check if a training phase was already started
    if os.path.exists(checkpoint_path):
        # load trained values
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        # restore previous values
        epoch = loaded_checkpoint['epoch']
        model.load_state_dict(loaded_checkpoint['model_sd'])
        optim.load_state_dict(loaded_checkpoint['optim_sd'])

    # UNCOMMENT THE BELOW TO CONTINUE TRAINING

    # print('Start training...')
    # for epoch in range(epochs):
    #     # start counting epoch time
    #     start_time = time.time()
    #     # compute train loss
    #     train_loss = train_loop()
    #     # compute val loss
    #     val_loss = val_loop()
    #     # store train and val loss for later analysis
    #     epoch_history.append((train_loss, val_loss))
    #     # end of epoch
    #     end_time = time.time()
    #     # format elapsed time
    #     elapsed_secs, elapsed_mins = format_time(start_time, end_time)
    #     checkpoint = {'epoch': epoch,
    #                   'optim_sd': optim.state_dict(),
    #                   'model_sd':model.state_dict(),
    #                   'train_loss': train_loss,
    #                   'val_loss': val_loss
    #                   }
    #     torch.save(checkpoint, checkpoint_path)
    #     print("EPOCH [{}/{}] | Train Loss: {} | Val. Loss: {} | time: {}m {}s".format(epoch+1,
    #                                                                                        epochs,
    #                                                                                        train_loss,
    #                                                                                        val_loss,
    #                                                                                        elapsed_mins,
    #                                                                                        elapsed_secs))
    # # save training model.
    # print('Training completed.')
    # torch.save(model.state_dict(), path_saved_model)

def pad_sequence(sequence, max_length):
    pad_token_idx = vocabulary.word_to_idx['<PAD>']
    while len(sequence) != max_length:
        sequence.append(pad_token_idx)
    return sequence


def format_user_input(sequence, max_length=10):
    # convert each word into index from the vocabulary
    regex = re.compile('[^a-zA-Z0-9.!?]')
    sequence = regex.sub(' ', sequence)
    # remove extra space
    sequence = re.sub(r"([.!?])", r" \1", sequence)
    # remove non-letter characters but keep regular punctuation
    sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sequence)
    sequence = sequence.lower()
    sequence = sequence.strip().split()
    user_seq_indices = []
    for word in sequence:
        if word in vocabulary.word_to_idx.keys():
            user_seq_indices.append(vocabulary.word_to_idx[word])
        else:
            user_seq_indices.append(vocabulary.word_to_idx['<UNK>'])

    # pad or trim the sentence
    if len(user_seq_indices) > max_length:  # trim
        user_seq_indices = user_seq_indices[:max_length]
    elif len(user_seq_indices) < max_length:  # pad
        user_seq_indices = pad_sequence(user_seq_indices, max_length)
    user_seq_indices.insert(0, vocabulary.word_to_idx['<S>'])
    user_seq_indices.append(vocabulary.word_to_idx['</S>'])
    user_seq_indices = torch.tensor(user_seq_indices).unsqueeze(1)
    return user_seq_indices


def convert_to_string(reply):
    parsed_reply = []
    for word_idx in reply:
        # if the word is PAD or END of Sentence token, ignore it.
        word_idx = word_idx.cpu().numpy()[0][0]
        reply = vocabulary.vocab[word_idx]
        if word_idx == vocabulary.word_to_idx['<PAD>'] or word_idx == vocabulary.word_to_idx['</S>']:
            break
        else:
            parsed_reply.append(reply)

    return ' '.join(parsed_reply)


def map_to_idx(sequence):
    seq_idx = []
    for word in sequence:
        seq_idx.append(vocabulary.word_to_idx[word])
    return seq_idx


# evaluate the model to talk with it.
def evaluate(seq, searcher):
    # tensor should have shape [seq, 1]
    seq = seq.to(device)
    # feedforward to the searcher to get the list of most likely indices of words
    bot_reply = searcher(seq)
    # discard first element which is the start token
    bot_reply = bot_reply[1:].to(device)
    bot_reply = torch.topk(bot_reply, 1)
    bot_reply = bot_reply.indices
    # convert indices to words.
    bot_reply = convert_to_string(bot_reply)
    return bot_reply


def run_bot(searcher, max_length=10):
    user_input = ""
    while(True):
        try:
            # ask the user for the input
            user_input = input('> ')
            # format the user input.
            if user_input == 'quit':
                break
            user_input_idx = format_user_input(user_input, max_length)
            # run the evaluate function to get bot's reply
            reply = evaluate(user_input_idx, searcher)
            print(reply)
        except KeyError:
            print('Error: While parsing the input sentence...')


# run bot and begin dialog
searcher = GreedySearch(encoder, decoder, vocabulary, attention=True)
run_bot(searcher)

