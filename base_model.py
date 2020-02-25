import fire
from os.path import join as pjoin
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader,random_split

from data import COOMatrix
from misc import get_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseMF(nn.Module):

    def __init__(self, users, items, factors):
        super().__init__()
        self.user_factor = nn.Embedding(users, factors)
        self.item_factor = nn.Embedding(items, factors)

    def forward(self, user, item):
        preds = (self.user_factor(user) * self.item_factor(item)).sum(1, keepdim=True)
        return preds.squeeze()


class BaseBiasMF(nn.Module):

    def __init__(self, users, items, factors):
        super().__init__()
        self.user_factor = nn.Embedding(users, factors)
        self.item_factor = nn.Embedding(items, factors)
        self.user_biases = nn.Embedding(users, 1)
        self.item_biases = nn.Embedding(items, 1)

    def forward(self, user, item):
        user_bias = self.user_biases(user)
        item_bias = self.item_biases(item)
        user_factor = self.user_factor(user)
        item_factor = self.item_factor(item)
        preds = (user_factor * item_factor).sum(1, keepdim=True) +\
            user_bias + item_bias
        return preds.squeeze()

class BaseTrain:

    def __init__(self,
                 factors=20,
                 file_name="exchange_rate.txt",
                 batch_size=128,
                 train_ratio=0.8,
                 learning_rate=0.0005,
                 epochs=10,
                 test_inference=20,
                 **kwargs):
        self.logger = get_logger()
        self.factors = factors
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.learning_rate = learning_rate
        self.data = COOMatrix(file_name)

        data_len = len(self.data)
        self.train_num = int(data_len * self.train_ratio)
        self.vali_num = data_len - self.train_num
        train_set, vali_set = random_split(self.data, [self.train_num, self.vali_num])
        self.train_loader = DataLoader(train_set, batch_size=batch_size,
                                       shuffle=True)
        self.vali_loader = DataLoader(vali_set, batch_size=batch_size,
                                      shuffle=False)

        self.test_inference = test_inference
        self.epochs = epochs

        self.model = None

    def get_loss(self, loss_func, row, col, pred, y):
        raise NotImplemented

    def train(self, epoch, loss_func, optimizer):
        self.model.train()

        total_loss = torch.Tensor([0])
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc="({0:^3})".format(epoch))
        for batch, ((row, col), val) in pbar:
            row = row.to(device)
            col = col.to(device)
            val = val.to(device)
            optimizer.zero_grad()
            pred = self.model(row, col)
            mse, loss = self.get_loss(loss_func, row, col, pred, val)
            loss.backward()
            optimizer.step()
            total_loss += mse.item()
            batch_loss = loss.item()
            pbar.set_postfix(train_loss=batch_loss)
        total_loss /= (self.train_num)
        return total_loss[0]

    def validate(self, epoch, loss_func):
        self.model.eval()
        total_loss = torch.Tensor([0])
        for batch, ((row, col), val) in enumerate(self.vali_loader):
            row = row.to(device)
            col = col.to(device)
            val = val.to(device)
            pred = self.model(row, col)
            mse, loss = self.get_loss(loss_func, row, col, pred, val)
            total_loss += mse.item()
        total_loss /= (self.vali_num)
        return total_loss[0]

    def test(self):
        self.model.eval()
        for (row, col), val in self.vali_loader:
            row = row.to(device)
            col = col.to(device)
            val = val.to(device)
            preds = self.model(row, col)
            count = 0
            for v, p in zip(val, preds):
                print("actual: {:.5}, predict: {:.5}".format(v, p))
                count += 1
                if count > self.test_inference:
                    break
            break

    def log_hyperparameter(self):
        hyperparam = list()
        for key, value in self.__dict__.items():
            if isinstance(value, (str, int, float)):
                hyperparam.append((key, value))
        hyperparam = sorted(hyperparam, key=lambda x: x[0])
        hyperparam = ["{}: {}".format(h[0], h[1]) for h in hyperparam]
        self.logger.info(",".join(hyperparam))

    def run(self):
        loss_func = nn.MSELoss(reduction="sum")
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)
        train_loss, vali_loss = 0.0, 0.0
        for epoch in range(self.epochs):
            train_loss = self.train(epoch, loss_func, optimizer)
            vali_loss = self.validate(epoch, loss_func)
            print("train loss: {:.4} vali loss: {:.4}".format(train_loss, vali_loss))
        self.log_hyperparameter()
        self.logger.info("train_loss: {:.5}, vali_loss: {:.5}".format(train_loss, vali_loss))
        self.test()


class Train(BaseTrain):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = BaseBiasMF(self.data.users, self.data.items, self.factors).to(device)

    def get_loss(self, loss_func, row, col, pred, y):
        loss = loss_func(pred, y)
        return loss, loss


if __name__ == "__main__":
    fire.Fire(Train)

