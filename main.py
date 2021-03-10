import re
import os
from collections import OrderedDict
from dataset import CornellCorpus
from torch.utils.data import DataLoader
from model import Encoder, Decoder


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
        dialog_data.append(line.split(sep=' +++$+++ '))
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
        regex = re.compile('[^a-zA-Z0-9,]')
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



class Vocabulary:

    def __init__(self, idx_to_text, dialogs_ids):
        self.dialogs_ids = dialogs_ids
        self.idx_to_text = self.normalize_sentence(idx_to_text)
        self.word_to_idx = self.map_word_to_idx()
        self.vocab = self.map_idx_to_word()

    def __len__(self):
        return len(self.word_to_idx)

    def normalize_sentence(self, idx_to_text):
        normalized_idx_to_sentence = {}
        for line_id, sentence in zip(idx_to_text.keys(), idx_to_text.values()):
            # convert each word in the sentence to a lower case
            sentence = sentence.lower()
            # eliminate extra spaces for punctuation
            sentence = re.sub(r"([.!?])", r" \1", sentence)
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
                for word in sentence.split():
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


dir = os.path.join(os.curdir, 'cornell-movie-dialogs-corpus/movie_lines.txt')
idx_to_text = get_movie_lines(dir)
# create a list of dialogs for each movie.
dialogs = extract_dialogs()
# for each movie, create pairs dialogs (Q/A). This is the actual data used for training.
pair_dialogs_idx = create_pair_dialogs(dialogs)
# instantiate the vocabulary
vocabulary = Vocabulary(idx_to_text, dialogs)
print('Total words counted in the vocabulary: {}'.format(vocabulary.__len__()))
# create the dataset class to store the data
dataset = CornellCorpus(pair_dialogs_idx, vocabulary, max_length=10)

# check if the batch is well construncted
batch = dataset.__getitem__(5)

# hyperparameters
batch_size = 64
hidden_size = 128
embedding_size = 256
epochs = 1000

# init dataloader
load_args = {'batch_size': batch_size, 'shuffle': True}
dataloader = DataLoader(dataset.data, **load_args)

# init seq2seq model, the parameters needed are
# embedding_size -> the size of the embedding for each word
# hidden_size -> the number of hidden neurons per unit
# voc_size -> the size of the vocabulary to embed each word
encoder = Encoder(embedding_size, hidden_size, vocabulary.__len__())
decoder = Decoder(embedding_size, hidden_size, vocabulary.__len__())

### TRAIN LOOP ###
for epoch in range(epochs):
    for id, X in enumerate(dataloader):
        print()


# create dataloader to load batches for the training


# vocabulary = Vocabulary(dialogs_pair) # extract and map each word from a dialog to an index.
# data = CornellDialogs(dialogs_pair...) # wrap the dialog inside the Dataset class which subclasses Dataset from pytorch
# dataloader = Dataloader() # create data loader to load the data
