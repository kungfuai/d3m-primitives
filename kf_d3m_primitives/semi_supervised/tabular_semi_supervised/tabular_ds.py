import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDs(Dataset):
    """
    For precise proportion between labeled and unlabeled
    """

    def __init__(self, data, relative_len, labels=None):

        # init data
        self.data = data
        self.labels = labels

        # make length of dataset equal for iterating
        self.indexing = [np.arange(self.data.shape[0]) for _ in range(relative_len)]
        self.indexing = np.concatenate(self.indexing, axis=0)

    def __len__(self):
        return len(self.indexing)

    def __getitem__(self, idx):

        true_idx = self.indexing[idx]
        data = torch.from_numpy(self.data[true_idx]).float()
        if self.labels is not None:
            label = int(self.labels[true_idx])
            return data, label
        else:
            return data


class SimpleDs(Dataset):
    """
    Just normal
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx]).float()
        return data, idx