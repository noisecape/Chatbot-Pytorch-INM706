import torch
import torchvision
from torch.utils.data import Dataset


class CornellCorpus(Dataset):

    def __init__(self, dialogs, vocabulary, max_length=10):
        super(CornellCorpus, self).__init__()
        self.dialogs_pair_idx = dialogs
        self.vocabulary = vocabulary
        self.data = self.build_dataset()
        self.max_length = max_length
        print('Dataset built')
        print('Dataset dimension: {}'.format(self.__len__()))

    def build_dataset(self):
        dataset = []
        for dialog in self.dialogs_pair_idx:
            for question, answer in zip(dialog.keys(), dialog.values()):
                q_a_pair = [self.vocabulary.idx_to_text[question], self.vocabulary.idx_to_text[answer]]
                dataset.append(q_a_pair)
        return dataset

    def pad_sequence(self, sequence):
        pad_token_idx = self.vocabulary.word_to_idx['<PAD>']
        end_token_idx = self.vocabulary.word_to_idx['</S>']
        start_token_idx = self.vocabulary.word_to_idx['<S>']
        while len(sequence) != self.max_length:
            sequence.append(pad_token_idx)
        sequence.append(end_token_idx)
        sequence.insert(0, start_token_idx)
        return sequence

    def check_length_requirement(self, sequence):
        if len(sequence) < self.max_length:
            # pad sequence and append EOS and S
            return self.pad_sequence(sequence)
        elif len(sequence) > self.max_length:
            # trunk sequence and append EOS and S
            sequence = sequence[:self.max_length]
            end_token_idx = self.vocabulary.word_to_idx['</S>']
            start_token_idx = self.vocabulary.word_to_idx['<S>']
            sequence.append(end_token_idx)
            sequence.insert(0, start_token_idx)
            return sequence
        else:
            end_token_idx = self.vocabulary.word_to_idx['</S>']
            start_token_idx = self.vocabulary.word_to_idx['<S>']
            sequence.append(end_token_idx)
            sequence.insert(0, start_token_idx)
            return sequence

    def process_batch(self, batch):
        """
        This function takes a batch and for both the query and the answer
        converts each word to the relative index in the vocabulary, pads
        the sentences that are shorter than 'max_length' and append the
        <EOS> token. The <S> is inserted at the beginning of the sentence.
        Finally the sentences are converted to tensors.
        :param batch: the batch composed by a pair of question/answer
        :return: the batch containing the pair of tensor relatives to the question/answer.
        """
        question = batch[0]
        answer = batch[1]
        question_idx = []
        answer_idx = []
        for q_word in question.split(" "):
            # fill the list with the corresponding index2word mapping for the question
            question_idx.append(self.vocabulary.word_to_idx[q_word])
        for a_word in answer.split(" "):
            # fill the list with the corresponding index2word mapping for the answer
            answer_idx.append(self.vocabulary.word_to_idx[a_word])

        # check if either or both the pair has a length greater or smaller than max_length
        question_idx = self.check_length_requirement(question_idx)
        answer_idx = self.check_length_requirement(answer_idx)
        # convert to tensors
        question_idx = torch.tensor(question_idx)
        answer_idx = torch.tensor(answer_idx)
        return [question_idx, answer_idx]

    def __getitem__(self, idx):
        return self.process_batch(self.data[idx])

    def __len__(self):
        return len(self.data)
