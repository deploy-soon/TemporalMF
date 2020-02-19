from os.path import join as pjoin
import numpy as np
from tqdm import tqdm
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
        print("load data: {}".format(shape))

    def __getitem__(self, index):
        assert index < self.data_len
        row = int(index / self.items)
        col = index % self.items
        val = self.data[row][col]
        return (row, col), val

    def __len__(self):
        return self.data_len

class BaseMF(nn.Module):

    def __init__(self, users, items, factors):
        super().__init__()
        self.user_factor = nn.Embedding(users, factors)
        self.item_factor = nn.Embedding(items, factors)

    def forward(self, user, item):
        preds = (self.user_factor(user) * self.item_factor(item)).sum(1, keepdim=True)
        return preds.squeeze()

class Train:

    def __init__(self, factors, batch_size=16, train_ratio=0.8, epochs=10):

        self.factors = factors

        self.train_ratio = train_ratio
        self.data = COOMatrix("exchange_rate.txt")

        data_len = len(self.data)
        self.train_num = int(data_len * self.train_ratio)
        self.vali_num = data_len - self.train_num
        train_set, vali_set = random_split(self.data, [self.train_num, self.vali_num])
        self.train_loader = DataLoader(train_set, batch_size=batch_size,
                                       shuffle=True)
        self.vali_loader = DataLoader(vali_set, batch_size=batch_size,
                                      shuffle=True)
        self.model = BaseMF(self.data.users, self.data.items, self.factors)

        self.epochs = epochs

    def train(self, epoch, loss_func, optimizer):
        self.model.train()

        total_loss = torch.Tensor([0])
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc="({0:^3})".format(epoch))
        for batch, ((row, col), val) in pbar:
            optimizer.zero_grad()
            preds = self.model(row, col)
            loss = loss_func(preds, val)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            batch_loss = loss.item() / len(row)
            pbar.set_postfix(train_loss=batch_loss)
        total_loss /= self.train_num
        return total_loss[0]

    def validate(self, epoch, loss_func):
        self.model.eval()
        total_loss = torch.Tensor([0])
        for batch, ((row, col), val) in enumerate(self.vali_loader):
            preds = self.model(row, col)
            loss = loss_func(preds, val)
            total_loss += loss.item()
        total_loss /= self.vali_num
        return total_loss[0]

    def run(self):
        loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-6)
        for epoch in range(self.epochs):
            train_loss = self.train(epoch, loss_func, optimizer)
            vali_loss = self.validate(epoch, loss_func)
            print("train loss: {} vali loss: {}".format(train_loss, vali_loss))

if __name__ == "__main__":
    train = Train(factors=3)
    train.run()

