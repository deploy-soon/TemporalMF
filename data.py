import numpy as np
from os.path import join as pjoin
from torch.utils.data import Dataset


class COOMatrix(Dataset):

    def __init__(self, file_name):
        self.file_name = file_name
        self.load_txt()

    def load_txt(self):
        file_path = pjoin("data", self.file_name)
        with open(file_path, "r") as fin:
            array = np.loadtxt(fin, delimiter=",")
        shape = array.shape
        self.users = shape[0]
        self.items = shape[1]
        self.data_len = self.users * self.items
        self.data = array
        print("load data: {}".format(shape))

    def __getitem__(self, index):
        assert index < self.data_len
        row = int(index / self.items)
        col = index % self.items
        val = self.data[row][col]
        return (row, col), val

    def __len__(self):
        return self.data_len

