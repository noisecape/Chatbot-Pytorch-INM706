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
        for idx in range(len(dialog) - 1):
            question_to_answer = []
            question_to_answer.append(dialog[idx])
            question_to_answer.append(dialog[idx+1])
            # check if either the answer or the question is empty and if that's the case don't append it.
            if dialog[idx] and dialog[idx+1]:
                dialogs_pairs.append(question_to_answer)
    return dialogs_pairs


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

    def __len__(self):
        return len(self.vocab)

    def normalize_sentence(self, idx_to_text):
        normalized_idx_to_sentence = {}
        for line_id, sentence in zip(idx_to_text.keys(), idx_to_text.values()):
            # convert each word in the sentence to a lower case
            sentence = sentence.lower()
            # eliminate extra spaces for punctuation
            sentence = re.sub(r"([.,!?'])", r" \1", sentence)
            # remove non-letter characters but keep regular punctuation
            sentence = re.sub(r"[^a-zA-Z.,!?]+", r" ", sentence)
            normalized_idx_to_sentence[line_id] = sentence

        # filters empty q-a pairs
        filtered_sentences = {}
        for line_id, text in zip(normalized_idx_to_sentence.keys(), normalized_idx_to_sentence.values()):
            if text != ' ':
                filtered_sentences[line_id] = text
            else:
                print('empty')
        print('Filtered sentences: {}'.format(len(normalized_idx_to_sentence) - len(filtered_sentences)))
        return normalized_idx_to_sentence

    def map_idx_to_word(self):
        words = self.word_to_idx.keys()
        index = self.word_to_idx.values()
        idx_to_word = OrderedDict()
        for w, i in zip(words, index):
            idx_to_word[i] = w
        return idx_to_word

    def map_word_to_idx(self):
        word_to_idx = {}
        count_words = 0
        for dialogs in self.dialogs_ids:
            for line in dialogs:
                sentence = self.idx_to_text[line]
                for word in sentence.strip().split():
                    if word not in word_to_idx:
                        word_to_idx[word] = count_words
                        count_words += 1
        start_token = '<S>'
        end_token = '</S>'
        unknown_token = '<UNK>'
        pad_token = '<PAD>'
        word_to_idx = sorted(word_to_idx)
        word_to_idx.append(start_token)
        word_to_idx.append(end_token)
        word_to_idx.append(unknown_token)
        word_to_idx.append(pad_token)
        word_to_idx = self.build_dictionary(word_to_idx)
        return word_to_idx

    def build_dictionary(self, word_collection):
        word_to_idx = OrderedDict()
        for idx, word in enumerate(word_collection):
            word_to_idx[word] = idx
        return word_to_idx

# USEFULL METHODS

def pad_sequence(sequence, max_length):
    pad_token_idx = vocabulary.word_to_idx['<PAD>']
    while len(sequence) <= max_length:
        sequence.append(pad_token_idx)
    return sequence


def format_user_input(sequence, max_length=10):
    # convert each word into index from the vocabulary
    regex = re.compile('[^a-zA-Z0-9.!?]')
    sequence = regex.sub(' ', sequence)
    # remove extra space
    sequence = re.sub(r"([.!?'])", r" \1", sequence)
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
        user_seq_indices.append(vocabulary.word_to_idx['</S>'])
    elif len(user_seq_indices) <= max_length:  # pad
        user_seq_indices.append(vocabulary.word_to_idx['</S>'])
        user_seq_indices = pad_sequence(user_seq_indices, max_length)
    user_seq_indices = torch.tensor(user_seq_indices).unsqueeze(1)
    user_seq_indices = user_seq_indices.to(device)
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


def run_bot(searcher, testing=True, max_length=15):
    if testing:
        user_input = 'Hey'
        user_input_idx = format_user_input(user_input, max_length)
        reply = evaluate(user_input_idx, searcher)
        print(reply)
    else:
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


def init_model(with_attention=False):
    if with_attention:
        # init with attention
        encoder = EncoderAttention(embedding_size, hidden_size, vocabulary.__len__()).to(device)
        attention = Attention(hidden_size).to(device)
        decoder = LuongAttentionDecoder(embedding_size, hidden_size, vocabulary.__len__(), attention=attention).to(
            device)
        model = ChatbotModel(encoder, decoder, vocabulary.__len__(), with_attention=True).to(device)
        return encoder, decoder, model
    else:
        # init with no attention
        encoder = Encoder(embedding_size, hidden_size, vocabulary.__len__()).to(device)
        decoder = Decoder(embedding_size, hidden_size, vocabulary.__len__()).to(device)
        model = ChatbotModel(encoder, decoder, vocabulary.__len__(), with_attention=False).to(device)
        return encoder, decoder, model

def train_loop():
    batch_history = []
    model.train()
    avg_loss = 0
    # start timer
    start_time = time.time()
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
        loss = 0
        for t, batch_out in enumerate(output):
            loss += criterion(batch_out, answer[t])

        loss /= output.shape[0]
        # default previous weights values
        model.optim.zero_grad()
        # backpropagate to compute gradients
        loss.backward()
        # clip gradients to avoid exploding values
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50)
        model.optim.step()
        batch_history.append(loss.item())
        # print current loss every 500 processed batches
        if idx % 500 == 0:
            # end of epoch
            end_time = time.time()
            # format elapsed time
            elapsed_secs, elapsed_mins = format_time(start_time, end_time)
            print('BATCH [{}/{}], LOSS TRAINING: {}, eta: {}m {}s'.format(idx,
                                                                          train_dataloader.__len__(),
                                                                          loss,
                                                                          elapsed_mins,
                                                                          elapsed_secs))
            # start timer
            start_time = time.time()
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
    avg_loss = np.sum(batch_history) / val_dataloader.__len__()
    # test learning with dummy phrase
    print("Test learning, input phrase: 'Hey what's your name?'")
    if model.attention:
        searcher = GreedySearch(encoder, decoder, vocabulary, attention=True).to(device)
    else:
        searcher = GreedySearch(encoder, decoder, vocabulary, attention=False).to(device)
    run_bot(searcher)
    return avg_loss

# Define all the parameters for the model

dirs = os.path.join(os.curdir, 'cornell-movie-dialogs-corpus/movie_lines.txt')
idx_to_text = get_movie_lines(dirs)
# create a list of dialogs for each movie.
dialogs = extract_dialogs()
# for each movie, create pairs dialogs (Q/A). This is the actual data used for training.
pair_dialogs_idx = create_pair_dialogs(dialogs)
# instantiate the vocabulary
vocabulary = Vocabulary(idx_to_text, dialogs)
print('Total words counted in the vocabulary: {}'.format(vocabulary.__len__()))

# shuffle data
np.random.shuffle(pair_dialogs_idx)
train_data = CornellCorpus(pair_dialogs_idx, vocabulary, max_length=10, stage='train')
val_data = CornellCorpus(pair_dialogs_idx, vocabulary, max_length=10, stage='val')
test_data = CornellCorpus(pair_dialogs_idx, vocabulary, max_length=10, stage='test')

print('Total training batches: {}, Total val batches: {}, Total test batches: {}'.format(train_data.__len__(),
                                                                                         val_data.__len__(),
                                                                                         test_data.__len__()))

# hyperparameters
batch_size = 64
hidden_size = 256
embedding_size = 300
epochs = 10

# init dataloader
load_args = {'batch_size': batch_size, 'shuffle': True}
train_dataloader = DataLoader(train_data, **load_args)
val_dataloader = DataLoader(val_data, **load_args)
test_dataloader = DataLoader(test_data, **load_args)

# init model
encoder, decoder, model = init_model(with_attention=True)

# init loss function
criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.word_to_idx['<PAD>'])
epoch_history = []

### TRAIN LOOP ###

epoch_idx = 0

if model.attention:
    checkpoint_path = os.path.join(os.curdir, 'saved_models/checkpoint.pth')
    path_saved_model = os.path.join(os.curdir, 'saved_models/trained_model.pth')
else:
    checkpoint_path = os.path.join(os.curdir, 'saved_models/checkpoint_no_att.pth')
    path_saved_model = os.path.join(os.curdir, 'saved_models/trained_model_no_att.pth')
    # check if the model is already trained
if os.path.exists(path_saved_model):
    # load state_dict
    print('Trained model found. Loading...')
    model.load_state_dict(torch.load(path_saved_model, map_location=torch.device(device)))
    print('Model loaded.')
else:
    # check if a training phase was already started
    if os.path.exists(checkpoint_path):
        # load trained values
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        print('Checkpoint found. Restore from [{}/{}] epoch.'.format(loaded_checkpoint['epoch'], epochs))
        # restore previous values
        epoch_idx = loaded_checkpoint['epoch']
        model.load_state_dict(loaded_checkpoint['model_sd'])
        model.optim.load_state_dict(loaded_checkpoint['optim_sd'])

    # UNCOMMENT THE BELOW TO CONTINUE TRAINING

    print('Start training...')
    for epoch in range(epoch_idx, epochs):
        # start counting epoch time
        start_time = time.time()
        # compute train loss
        train_loss = train_loop()
        # compute val loss
        val_loss = val_loop()
        # store train and val loss for later analysis
        epoch_history.append((train_loss, val_loss))
        # end of epoch
        end_time = time.time()
        # format elapsed time
        elapsed_secs, elapsed_mins = format_time(start_time, end_time)
        checkpoint = {'epoch': epoch,
                      'optim_sd': model.optim.state_dict(),
                      'model_sd': model.state_dict(),
                      'train_loss': train_loss,
                      'val_loss': val_loss
                      }
        torch.save(checkpoint, checkpoint_path)
        print("EPOCH [{}/{}] | Train Loss: {} | Val. Loss: {} | time: {}m {}s".format(epoch+1,
                                                                                           epochs,
                                                                                           train_loss,
                                                                                           val_loss,
                                                                                           elapsed_mins,
                                                                                           elapsed_secs))
    # save training model.
    print('Training completed.')
    torch.save(model.state_dict(), path_saved_model)

searcher = GreedySearch(encoder, decoder, vocabulary, attention=True).to(device)
run_bot(searcher, testing=False)


