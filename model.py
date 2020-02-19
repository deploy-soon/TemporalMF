from os.path import join as pjoin
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader,random_split


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

    def __getitem__(self, index):
        assert index < self.data_len
        row = int(index / self.items)
        col = index % self.items
        val = data[row][col]
        return (row, col), val

    def __len__(self):
        return self.data_len

class BaseMF(nn.Module):

    def __init__(self, users, items, factors):
        self.user_factor = nn.Embedding(users, factors, sparse=True)
        self.item_factor = nn.Embedding(items, factors, sparse=True)

    def forward(self, user, item):
        return (self.user_factor(user) * self.item_factor(item)).sum(1)

class Train:

    def __init__(self, factors, train_ratio=0.8):

        self.factors = factors

        self.train_ratio = train_ratio
        data = COOMatrix("exchange_rate.txt")

        data_len = len(data)
        train_num = int(data_len * self.train_ratio)
        vali_num = data_len - train_num
        train_set, vali_set = random_split(data, [train_num, vali_num])
        self.train_loader = DataLoader(train_set, batch_size=batch_size,
                                       shuffle=True)
        self.vali_loader = DataLoader(vali_set, batch_size=batch_size,
                                      shuffle=True)
        self.users = array.shape[0]

    def run(self):
        model = BaseMF(self.data.users, self.data.items, self.factors)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)


