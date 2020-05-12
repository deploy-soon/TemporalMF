import torch
import numpy as np
from os.path import join as pjoin
from torch.utils.data import Dataset


def col_normalize(array, nor_dim=0):
    mean = array.mean(axis=0)
    std = np.std(array, axis=0)
    normalized = (array - mean) / std
    return normalized, mean, std

def col_min_max(array, _min=0, _max=1):
    nom = (array - array.min(axis=0)) * (_max - _min)
    denom = X.max(axis=0) - X.min(axis=0)
    return _min + nom / denom

class COOMatrix(Dataset):

    def __init__(self, file_name, data_path="../data", norm=True):
        self.data_path = data_path
        self.file_name = file_name
        self.load_txt()
        self.norm = norm

    def load_txt(self):
        file_path = pjoin(self.data_path, self.file_name)
        with open(file_path, "r") as fin:
            array = np.loadtxt(fin, delimiter=",")
            array, mean, std = col_normalize(array)
            self.mean = mean
            self.std = std
        shape = array.shape
        self.users = shape[0]
        self.items = shape[1]
        self.data_len = self.users * self.items
        self.data = array
        print("load data: {}".format(shape))

    def denormalize(self, item, col):
        col = col.item()
        return self.mean[col] + item * self.std[col]

    def __getitem__(self, index):
        assert index < self.data_len
        row = int(index / self.items)
        col = index % self.items
        val = self.data[row][col]
        return (row, col), val

    def __len__(self):
        return self.data_len


class COOMatrixForecasting(Dataset):

    def __init__(self, file_name, data_path="../data", norm=True):
        self.data_path = data_path
        self.file_name = file_name
        self.load_txt()
        self.norm = norm

    def load_txt(self):
        file_path = pjoin(self.data_path, self.file_name)
        with open(file_path, "r") as fin:
            array = np.loadtxt(fin, delimiter=",")
            array, mean, std = col_normalize(array)
            self.mean = torch.from_numpy(mean).cuda().type(torch.cuda.FloatTensor)
            self.std = torch.from_numpy(std).cuda().type(torch.cuda.FloatTensor)
        shape = array.shape
        self.users = shape[0]
        self.items = shape[1]
        self.data_len = self.users * self.items
        self.data = torch.from_numpy(array).cuda().type(torch.cuda.FloatTensor)
        print("load data: {}".format(shape))

    def denormalize(self, item):
        return self.mean + item * self.std

    def __getitem__(self, index):
        assert index < self.users
        val = self.data[index]
        return index, val

    def __len__(self):
        return self.users

if __name__ == "__main__":
    data = COOMatrix("exchange_rate")

