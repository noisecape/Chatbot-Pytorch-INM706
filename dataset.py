import torch
import torchvision
from torch.utils.data import Dataset


class CornellCorpus(Dataset):

    def __init__(self, dialogs, idx_to_text):
        super(CornellCorpus, self).__init__()
        self.dialogs_pair_idx = dialogs
        self.idx_to_text = idx_to_text
        self.data = self.build_dataset()
        print('Dataset built')
        print('Dataset dimension: {}'.format(self.__len__()))

    def build_dataset(self):
        dataset = []
        for dialog in self.dialogs_pair_idx:
            for question, answer in zip(dialog.keys(), dialog.values()):
                q_a_pair = [self.idx_to_text[question], self.idx_to_text[answer]]
                dataset.append(q_a_pair)
        return dataset

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
