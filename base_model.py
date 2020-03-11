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
        """
        TRMF embedding vectors
        :param times: length of time series
        :param items: dimensions of time series
        :param factors: dimensions of latent factors
        """
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
                 train_ratio=0.7,
                 learning_rate=0.005,
                 epochs=10,
                 test_inference=10,
                 **kwargs):
        self.logger = get_logger(name=self.__class__.__name__)
        self.factors = factors
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.learning_rate = learning_rate
        self.file_name=file_name
        self.data = COOMatrix(file_name)

        data_len = len(self.data)
        self.train_num = int(data_len * self.train_ratio)
        self.vali_num = int((data_len - self.train_num) / 2)
        self.test_num = data_len - self.train_num - self.vali_num
        self.logger.debug("TRAIN NUM: {} VALIDATION NUM: {} TEST NUM: {}"
                          .format(self.train_num, self.vali_num, self.test_num))

        train_set, vali_set, test_set = random_split(self.data,
                                                     [self.train_num, self.vali_num, self.test_num])
        self.train_loader = DataLoader(train_set, batch_size=batch_size,
                                       shuffle=True)
        self.vali_loader = DataLoader(vali_set, batch_size=batch_size,
                                      shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size=batch_size,
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

    def validate(self, epoch, iterator, loss_func):
        self.model.eval()
        total_loss = torch.Tensor([0])
        for batch, ((row, col), val) in enumerate(iterator):
            row = row.to(device)
            col = col.to(device)
            val = val.to(device)
            pred = self.model(row, col)
            mse, loss = self.get_loss(loss_func, row, col, pred, val)
            total_loss += mse.item()
        total_loss /= len(iterator)
        return total_loss[0]

    def test(self):
        self.model.eval()
        for (row, col), val in self.test_loader:
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

    def _cache_l1_norm(self):
        # to get NRMSE at each epoch
        train_abs = 0.0
        vali_abs = 0.0
        test_abs = 0.0
        for (row, col), val in self.train_loader:
            train_abs += torch.sum(val.abs()) / self.train_num
        for (row, col), val in self.vali_loader:
            vali_abs += torch.sum(val.abs()) / self.vali_num
        for (row, col), val in self.test_loader:
            test_abs += torch.sum(val.abs()) / self.test_num

        self.train_abs = train_abs
        self.vali_abs = vali_abs
        self.test_abs = test_abs
        self.logger.debug("absolute value: {}, {}, {}".format(train_abs, vali_abs, test_abs))

    def run(self):
        loss_func = nn.MSELoss(reduction="sum")
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=0)
        print(self.model)
        self._cache_l1_norm()
        train_nrmse, vali_nrmse, test_nrmse = 0.0, 99999.0, 0.0
        for epoch in range(self.epochs):
            train_mse = self.train(epoch, loss_func, optimizer)
            vali_mse = self.validate(epoch, self.vali_loader, loss_func)
            epoch_train_loss = torch.sqrt(train_mse) / self.train_abs
            epoch_vali_loss = torch.sqrt(vali_mse) / self.vali_abs
            if epoch_vali_loss < vali_nrmse:
                vali_nrmse = epoch_vali_loss
                train_nrmse = epoch_train_loss
                test_mse = self.validate(epoch, self.test_loader, loss_func)
                test_nrmse = torch.sqrt(test_mse) / self.test_abs
            print("train loss: {:.4} vali loss: {:.4}".format(epoch_train_loss, epoch_vali_loss))
        self.log_hyperparameter()
        self.logger.info("train_loss: {:.5}, vali_loss: {:.5}, test_loss: {:.5}"
                         .format(train_nrmse, vali_nrmse, test_nrmse))
        self.test()


class Train(BaseTrain):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = BaseMF(self.data.users, self.data.items, self.factors).to(device)

    def get_loss(self, loss_func, row, col, pred, y):
        loss = loss_func(pred, y)
        return loss, loss


if __name__ == "__main__":
    fire.Fire(Train)

