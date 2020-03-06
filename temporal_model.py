import fire
from os.path import join as pjoin
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable

from base_model import BaseTrain, BaseMF, BaseBiasMF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MatrixEmbedding(nn.Module):

    def __init__(self, lag_set, factors):
        super().__init__()
        lags = len(lag_set)
        self.lag_set = lag_set
        self.lag_factor = nn.Parameter(torch.rand(lags, factors))

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

    def forward(self, user, item):
        preds = (self.user_factor(user) * self.item_factor(item)).sum(1, keepdim=True)
        return preds.squeeze()


class TemporalTrain(BaseTrain):

    def __init__(self, lambda_x=0.025, lambda_f=0.1, lambda_theta=0.1, mu_x=4.0,
                 lag_set = [1, 2, 3, 4, 5, 6], **kwargs):
        super().__init__(**kwargs)
        self.lambda_x = lambda_x
        self.lambda_f = lambda_f
        self.lambda_theta = lambda_theta
        self.mu_x=mu_x
        lag_set.sort()
        self.lag_set = lag_set
        self.lags = len(lag_set)
        assert self.lags < self.data.users, "Lag set is too big"

        self.temporal_model = MatrixEmbedding(lag_set = self.lag_set,
                                              factors = self.factors).to(device)
        self.model = TemporalMF(users=self.data.users,
                            items=self.data.items,
                            factors=self.factors,
                            temporal_model=self.temporal_model).to(device)

    def get_loss(self, loss_func, row, col, pred, y):
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
        item_loss = self.lambda_f * torch.sum(self.model.item_factor(col)**2)
        lag_loss = self.lambda_theta * torch.sum(self.model.temporal_model.lag_factor**2)

        L = max(self.lag_set)
        m = 1 + L

        filtered_row = row[row >= m]
        filtered_batch_num = filtered_row.size()[0]
        repeated_lag_set = torch.LongTensor([self.lag_set for _ in range(filtered_batch_num)]).to(device)
        # repeated_lag_set = (batch, lags)
        filtered_row_lags = filtered_row.expand(self.lags, filtered_batch_num).transpose(1, 0)
        filtered_row_lags = filtered_row_lags - repeated_lag_set
        #filtered_row_lags = (batch, lags)
        embedding_target = self.model.user_factor(filtered_row)
        #embedding_target = (batch, factors)
        embedding_lags = self.model.user_factor(filtered_row_lags)
        #embedding_lags = (batch, lags, factors)
        lag_pred = self.model.temporal_model(embedding_lags)
        AR_residual = embedding_target - lag_pred
        #AR_residual = (batch, factors)
        time_loss = 0.5 * torch.sum(AR_residual ** 2)

        time_loss = time_loss + self.mu_x * torch.sum(self.model.user_factor(row) ** 2)
        time_loss = self.lambda_x * time_loss

        return loss, item_loss, time_loss, lag_loss

    def train(self, epoch, loss_func, optimizer):
        self.model.train()

        total_loss = torch.Tensor([0])
        total_time_loss = torch.Tensor([0])
        total_item_loss = torch.Tensor([0])
        total_lag_loss = torch.Tensor([0])
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc="({0:^3})".format(epoch))
        for batch, ((row, col), val) in pbar:
            row = row.to(device)
            col = col.to(device)
            val = val.to(device)
            optimizer.zero_grad()
            pred = self.model(row, col)
            loss, item_loss, time_loss, lag_loss = self.get_loss(loss_func, row, col, pred, val)
            cost_func = loss + item_loss + time_loss + lag_loss
            cost_func.backward()
            optimizer.step()

            total_loss += loss.item()
            total_item_loss += item_loss.item()
            total_lag_loss += lag_loss.item()
            total_time_loss += time_loss.item()
            batch_loss = loss.item()
            pbar.set_postfix(train_loss=batch_loss)
        print(total_loss[0] / self.train_num,
              total_item_loss[0] / self.train_num,
              total_time_loss[0] / self.train_num,
              total_lag_loss[0] / self.train_num)
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
            loss, _, _, _ = self.get_loss(loss_func, row, col, pred, val)
            total_loss += loss.item()
        total_loss /= (self.vali_num)
        return total_loss[0]

if __name__ == "__main__":
    fire.Fire(TemporalTrain)

