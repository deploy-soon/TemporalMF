import os
import abc
import sys
import json
import fire
import datetime
from os.path import join as pjoin
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

sys.path.append("../")
from data import COOMatrixForecasting
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


class MatrixEmbedding(nn.Module):

    def __init__(self, lag_set, factors):
        super().__init__()
        lags = len(lag_set)
        self.lag_set = sorted(lag_set, reverse=True)
        self.lag_factor = nn.Parameter(torch.rand(lags, factors))

    def regularizer(self):
        return torch.sum(self.lag_factor ** 2)

    def forward(self, lags_vectors):
        embedding_lags_dot = lags_vectors * self.lag_factor
        #embedding_lags_dot = (batch, lags, factors)
        target_vectors = torch.sum(embedding_lags_dot, dim=1)
        #target_vectors = (batch, factors)
        return target_vectors


class TemporalMF(nn.Module):

    def __init__(self, users, items, factors, temporal_model):
        """
        TRMF embedding vectors
        :param times: length of time series
        :param items: dimensions of time series
        :param factors: dimensions of latent factors
        """
        super().__init__()
        self.user_factor = nn.Embedding(users, factors)
        self.item_factor = nn.Embedding(items, factors)
        self.temporal_model = temporal_model

    def predict(self, user, item, device):
        lag_set = self.temporal_model.lag_set
        m = max(lag_set) + 1
        #filtered_row = row[row >= m]
        filtered_row = user
        filtered_batch_num = filtered_row.size()[0]
        repeated_lag_set = torch.LongTensor([lag_set for _ in range(filtered_batch_num)]).to(device)
        # repeated_lag_set = (batch, lags)
        filtered_row_lags = filtered_row.expand(m - 1, filtered_batch_num).transpose(1, 0)
        filtered_row_lags = filtered_row_lags - repeated_lag_set
        #filtered_row_lags = (batch, lags)
        embedding_lags = self.user_factor(filtered_row_lags)
        #embedding_target = (batch, factors)
        lag_pred = self.temporal_model(embedding_lags)
        preds = (lag_pred * self.item_factor(item)).sum(1, keepdim=True)
        return preds.squeeze()

    def forward(self, user):
        preds = torch.mm(self.user_factor(user), self.item_factor.weight.T)
        return preds

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Embedding):
            m.weight.data.normal_(0, 0.01)
        if isinstance(m, nn.Parameter):
            m.data.normal_(0, 0.01)


class BaseTrain(metaclass=abc.ABCMeta):

    def __init__(self,
                 factors=20,
                 file_name="exchange_rate",
                 batch_size=128,
                 window_size=24,
                 nr_windows_vali=7,
                 nr_windows_test=3,
                 learning_rate=0.005,
                 epochs=10,
                 gpu=0,
                 verbose=False,
                 lambda_x=0.5, lambda_f=0.005, lambda_theta=0.005, mu_x=0.005,
                 lag_set=list(range(24)) + list(range(24 * 7, 24 * 8)),
                 is_pred_sub=True, **kwargs):
        super().__init__(**kwargs)
        self.name = self.__class__.__name__
        self.logger = get_logger(name=self.name)
        self.factors = factors
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.file_name=file_name
        self.data = COOMatrixForecasting(file_name)

        data_len = len(self.data)
        self.window_size = window_size
        self.nr_windows_vali = nr_windows_vali
        self.nr_windows_test = nr_windows_test
        self.train_num = data_len - window_size * (nr_windows_vali + nr_windows_test)
        self.vali_num = window_size * nr_windows_vali
        self.test_num = window_size * nr_windows_test
        self.logger.debug("TRAIN NUM: {} VALIDATION NUM: {} TEST_NUM: {}"
                          .format(self.train_num, self.vali_num, self.test_num))

        #train_set = Subset(self.data, list(range(self.train_num)))
        #vali_set = Subset(self.data, list(range(self.train_num, self.train_num+self.vali_num)))
        #test_set = Subset(self.data, list(range(self.train_num, self.train_num+self.vali_num)))
        #self.train_loader = DataLoader(train_set, batch_size=batch_size,
        #                               shuffle=True)
        #self.vali_loader = DataLoader(vali_set, batch_size=self.vali_num,
        #                              shuffle=False)

        self.epochs = epochs

        if gpu is not None and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu))
        else:
            self.device = torch.device("cpu")
        self.verbose = verbose

        self.lambda_x = lambda_x
        self.lambda_f = lambda_f
        self.lambda_theta = lambda_theta
        self.mu_x=mu_x

        self.lag_set = lag_set
        self.lags = len(lag_set)
        self.is_pred_sub = is_pred_sub
        assert max(self.lag_set) < self.data.users, "Lag set is too big"

        self.temporal_model = MatrixEmbedding(lag_set=self.lag_set,
                                              factors=self.factors).to(self.device)
        self.model = TemporalMF(users=self.data.users,
                                items=self.data.items,
                                factors=self.factors,
                                temporal_model=self.temporal_model).to(self.device)

    def get_loss(self, loss_func, row, pred, y, is_pred_sub=True):
        """
        loss function has four parts which contain inference and regularization.
        1. Products of time factor and item factor
            should be close to time series data. This derived with MSE loss
        2. Regularization term with item factors,
            usually from squared Frobenius norm.
        3. In this model, we specialize the time series model to AR model.
            Time series latent factors are fit to lag set AR model.
            Temporal regularizer should be contained graph regularization and
            squared Frobenuis norm
        4. Regularization term with time weight factors
        """
        loss = loss_func(pred, y)
        #item_loss = self.lambda_f * torch.sum(self.model.item_factor**2)
        lag_loss = self.lambda_theta * self.model.temporal_model.regularizer()

        L = max(self.lag_set)
        m = 1 + L

        filtered_row = row[row >= m]
        filtered_batch_num = filtered_row.size(0)
        repeated_lag_set = torch.LongTensor([self.lag_set for _ in range(filtered_batch_num)]).to(self.device)
        # repeated_lag_set = (batch, lags)
        filtered_row_lags = filtered_row.expand(self.lags, filtered_batch_num).transpose(1, 0)
        filtered_row_lags = filtered_row_lags - repeated_lag_set
        #filtered_row_lags = (batch, lags)
        if is_pred_sub:
            filtered_row_one_prev = filtered_row - torch.LongTensor([1]).to(self.device)
            embedding_target = self.model.user_factor(filtered_row) - self.model.user_factor(filtered_row_one_prev)
        else:
            embedding_target = self.model.user_factor(filtered_row)
        #embedding_target = (batch, factors)
        embedding_lags = self.model.user_factor(filtered_row_lags)
        #embedding_lags = (batch, lags, factors)
        lag_pred = self.model.temporal_model(embedding_lags)
        AR_residual = embedding_target - lag_pred
        #AR_residual = (batch, factors)
        time_loss = torch.sum(AR_residual ** 2)

        time_loss = self.lambda_x * time_loss + self.mu_x * torch.sum(self.model.user_factor(row) ** 2)

        return loss, time_loss, lag_loss

    def train(self, epoch, loss_func, optimizer):
        self.model.train()

        total_loss = torch.Tensor([0])
        total_time_loss = torch.Tensor([0])
        #total_item_loss = torch.Tensor([0])
        total_lag_loss = torch.Tensor([0])
        if self.verbose:
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                        desc="({0:^3})".format(epoch))
        else:
            pbar = enumerate(self.train_loader)
        for batch, (row, val) in pbar:
            row = row.to(self.device)
            val = val.to(self.device)
            optimizer.zero_grad()
            pred = self.model(row)
            loss, time_loss, lag_loss = self.get_loss(loss_func,
                    row, pred, val, self.is_pred_sub)
            #cost_func = loss + item_loss + time_loss + lag_loss
            cost_func = loss + time_loss + lag_loss
            cost_func.backward()
            optimizer.step()

            total_loss += loss.item()
            #total_item_loss += item_loss.item()
            total_lag_loss += lag_loss.item()
            total_time_loss += time_loss.item()
            batch_loss = loss.item()
            if self.verbose:
                pbar.set_postfix(train_loss=batch_loss)
        if self.verbose:
            print(total_loss[0] / self.train_num / self.data.items,
                  #total_item_loss[0] / self.train_num,
                  total_time_loss[0] / self.train_num / self.data.items,
                  total_lag_loss[0] / self.train_num / self.data.items)
        total_loss /= (self.train_num * self.data.items)
        return total_loss[0]

    def validate(self, epoch, iterator, loss_func):
        self.model.eval()
        total_loss = torch.Tensor([0])

        user_factors = self.model.user_factor.weight.clone()
        preds, trues = None,  None
        for batch, (row, val) in enumerate(iterator):
            row = row.to(self.device)
            val = val.to(self.device)
            repeated_lag_set = torch.LongTensor(self.lag_set).to(self.device)
            for r in row:
                row_lags = r.expand(self.lags)
                row_lags = row_lags - repeated_lag_set
                embedding_lags = user_factors[row_lags]
                lag_pred = self.model.temporal_model(embedding_lags.unsqueeze(0)).squeeze()
                row_one_prev = r - torch.LongTensor([1]).to(self.device)
                lag_pred = user_factors[row_one_prev] + lag_pred
                user_factors[r] = lag_pred
            pred = torch.mm(user_factors[row], self.model.item_factor.weight.T)

            pred = self.denormalized(pred)
            val = self.denormalized(val)
            preds = pred if preds is None else torch.cat((preds, pred), 0)
            trues = val if trues is None else torch.cat((trues, val), 0)
            loss = loss_func(pred, val)
            total_loss += loss.item()
        total_loss /= (self.vali_num * self.data.items)
        return total_loss[0], trues, preds

    def denormalized(self, val):
        val = self.data.denormalize(val)
        return val

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

    def _cache_l1_norm(self):
        # to get NRMSE at each epoch
        train_abs = 0.0
        vali_abs = 0.0
        vali_ori_abs = 0.0
        for _, val in self.train_loader:
            train_abs += torch.sum(val.abs()) / self.train_num / self.data.items
        for _, val in self.vali_loader:
            vali_abs += torch.sum(val.abs()) / self.vali_num / self.data.items
            val = self.denormalized(val)
            vali_ori_abs += torch.sum(val.abs()) / self.vali_num / self.data.items

        self.train_abs = train_abs
        self.vali_abs = vali_abs
        self.vali_ori_abs = vali_ori_abs
        self.logger.debug("absolute value: {:.5}, {:.5} {:.5}".format(train_abs, vali_abs, vali_ori_abs))

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

        trues_vali, preds_vali = None, None
        for w in range(self.nr_windows_vali):
            init_weights(self.model.modules())
            init_weights(self.temporal_model.modules())
            nr_windows = self.nr_windows_vali + self.nr_windows_test
            train_num = len(self.data) - self.window_size * (nr_windows - w)
            print("train num: ", train_num)
            train_set = Subset(self.data, list(range(train_num)))
            vali_set = Subset(self.data, list(range(train_num, train_num+self.window_size)))
            self.train_loader = DataLoader(train_set, batch_size=self.batch_size,
                                       shuffle=True)
            self.vali_loader = DataLoader(vali_set, batch_size=self.window_size,
                                      shuffle=False)
            self._cache_l1_norm()

            train_nrmse, vali_nrmse, vali_ori_nrmse = 0.0, 99999.0, 0.0
            for epoch in range(self.epochs):
                train_mse = self.train(epoch, loss_func, optimizer)
                vali_mse, true, pred = self.validate(epoch, self.vali_loader, loss_func)
                epoch_train_loss = torch.sqrt(train_mse) / self.train_abs
                epoch_vali_loss = torch.sqrt(vali_mse) / self.vali_ori_abs

                if self.verbose:
                    print("train loss: {:.4} vali loss: {:.4}".format(epoch_train_loss, epoch_vali_loss))
            vali_mse, true, pred = self.validate(epoch, self.vali_loader, loss_func)
            preds_vali = pred if preds_vali is None else torch.cat((preds_vali, pred), 0)
            trues_vali = true if trues_vali is None else torch.cat((trues_vali, true), 0)
        #print(preds_vali)
        #print(trues_vali)
        #print(preds_vali.size(), trues_vali.size())
        nrmse = torch.sqrt(torch.mean((preds_vali-trues_vali)**2))/torch.mean(torch.abs(trues_vali))
        print(nrmse)

        hyperparams = self.get_hyperparameter()
        hyperparams.update({
            "train_loss": float(train_nrmse),
            "vali_loss": float(vali_nrmse),
            "vali_ori_loss": float(vali_ori_nrmse),
        })
        #self.logger.info("train_loss: {:.5}, vali_loss: {:.5}, origin_loss: {:.5}"
        #                 .format(train_nrmse, vali_nrmse, vali_ori_nrmse))
        self.save_snapshot(hyperparams)


class Train(BaseTrain):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = BaseMF(self.data.users, self.data.items, self.factors).to(self.device)

    def get_loss(self, loss_func, row, col, pred, y):
        loss = loss_func(pred, y)
        return loss, loss


if __name__ == "__main__":
    #fire.Fire(Train)
    train = BaseTrain(file_name="electricity", epochs=150, factors=40, verbose=False)
    train.run()

