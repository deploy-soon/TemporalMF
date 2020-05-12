import os
import sys
import json
import fire
import datetime
from os.path import join as pjoin
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.append("../")
from data import COOMatrix
from misc import get_logger


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
                 file_name="exchange_rate",
                 batch_size=128,
                 train_ratio=0.7,
                 learning_rate=0.005,
                 epochs=10,
                 test_inference=10,
                 gpu=0,
                 verbose=False,
                 **kwargs):
        self.name = self.__class__.__name__
        self.logger = get_logger(name=self.name)
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

        if gpu is not None and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu))
        else:
            self.device = torch.device("cpu")
        self.verbose = verbose
        self.model = None

    def get_loss(self, loss_func, row, col, pred, y):
        raise NotImplemented

    def denormalized(self, val, col):
        for i, c in enumerate(col):
            val[i] = self.data.denormalize(val[i], c)
        return val

    def train(self, epoch, loss_func, optimizer):
        self.model.train()

        total_loss = torch.Tensor([0])
        if self.verbose:
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                        desc="({0:^3})".format(epoch))
        else:
            pbar = enumerate(self.train_loader)

        abs_diff, abs_true, mse_diff = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
        for batch, ((row, col), val) in pbar:
            row = row.to(self.device)
            col = col.to(self.device)
            val = val.to(self.device)
            optimizer.zero_grad()
            pred = self.model(row, col)
            mse, loss = self.get_loss(loss_func, row, col, pred, val)
            loss.backward()
            optimizer.step()
            #total_loss += mse.item()
            batch_loss = loss.item()
            pred = self.denormalized(pred, col)
            val = self.denormalized(val, col)
            abs_diff += torch.sum(torch.abs(pred-val))
            abs_true += torch.sum(torch.abs(val))
            mse_diff += torch.sum((pred-val)**2)
            if self.verbose:
                pbar.set_postfix(train_loss=batch_loss)

        nd = abs_diff / abs_true
        nrmse = torch.sqrt(mse_diff * self.train_num) / abs_true
        #print(nd, nrmse)
        return nd, nrmse
        #total_loss /= (self.train_num)
        #return total_loss[0]

    def validate(self, epoch, iterator, loss_func):
        self.model.eval()
        total_loss = torch.Tensor([0])
        abs_diff, abs_true, mse_diff = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
        for batch, ((row, col), val) in enumerate(iterator):
            row = row.to(self.device)
            col = col.to(self.device)
            val = val.to(self.device)
            pred = self.model(row, col)
            pred = self.denormalized(pred, col)
            val = self.denormalized(val, col)
            abs_diff += torch.sum(torch.abs(pred-val))
            abs_true += torch.sum(torch.abs(val))
            mse_diff += torch.sum((pred-val)**2)
            #mse, _ = self.get_loss(loss_func, row, col, pred, val)
            #total_loss += mse.item()
        #total_loss /= (len(iterator) * self.batch_size)
        nd = abs_diff / abs_true
        nrmse = torch.sqrt(mse_diff * len(iterator) * self.batch_size) / abs_true
        #print(nd, nrmse)
        return nd, nrmse

    def test(self):
        self.model.eval()
        for (row, col), val in self.test_loader:
            row = row.to(self.device)
            col = col.to(self.device)
            val = val.to(self.device)
            preds = self.model(row, col)
            count = 0
            for v, p in zip(val, preds):
                print("actual: {:.5}, predict: {:.5}".format(v, p))
                count += 1
                if count > self.test_inference:
                    break
            break

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_hyperparameter(self):
        hyperparam = dict()
        for key, value in self.__dict__.items():
            if isinstance(value, (str, int, float, bool)):
                hyperparam[key] = value
        hyperparam_list = ["{}: {}".format(k, v) for k, v in hyperparam.items()]
        self.logger.info(",".join(hyperparam_list))
        return hyperparam

    def save_snapshot(self, info):
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        res_path = pjoin("res", self.file_name, "{}.json".format(now))
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        with open(res_path, "w") as fout:
            json.dump(info, fout, indent=4, sort_keys=True)

    def run(self):
        loss_func = nn.MSELoss(reduction="sum")
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=0)
        print(self.model)
        self.num_params = self.count_parameters()

        train_nrmse, vali_nrmse, test_nrmse = 0.0, 99999.0, 0.0
        for epoch in range(self.epochs):
            #train_mse = self.train(epoch, loss_func, optimizer)
            #vali_mse = self.validate(epoch, self.vali_loader, loss_func)
            _train_nd, _train_nrmse = self.train(epoch, loss_func, optimizer)
            _vali_nd, _vali_nrmse = self.validate(epoch, self.vali_loader, loss_func)
            if _vali_nrmse < vali_nrmse:
                vali_nrmse = _vali_nrmse
                train_nrmse = _train_nrmse
                test_nd, test_nrmse = self.validate(epoch, self.test_loader, loss_func)
            if self.verbose:
                print("trn loss: {:.4} vali loss:{:.4}".format(float(train_nrmse), float(vali_nrmse)))
        hyperparams = self.get_hyperparameter()
        hyperparams.update({
            "train_loss": float(train_nrmse),
            "vali_loss": float(vali_nrmse),
            "test_loss": float(test_nrmse),
        })
        self.logger.info("train_loss: {:.5}, vali_loss: {:.5}, test_loss: {:.5}"
                         .format(float(train_nrmse), float(vali_nrmse), float(test_nrmse)))
        self.save_snapshot(hyperparams)

        if self.verbose:
            self.test()


class Train(BaseTrain):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = BaseMF(self.data.users, self.data.items, self.factors).to(self.device)

    def get_loss(self, loss_func, row, col, pred, y):
        loss = loss_func(pred, y)
        return loss, loss


if __name__ == "__main__":
    fire.Fire(Train)

