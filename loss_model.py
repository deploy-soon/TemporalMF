from os.path import join as pjoin
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader,random_split

from data import COOMatrix
from base_model import BaseTrain, BaseMF, BaseBiasMF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(BaseTrain):

    def __init__(self, lmbd=4.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lmbd = lmbd
        self.model = BaseBiasMF(self.data.users, self.data.items, self.factors).to(device)

    def get_loss(self, loss_func, row, col, pred, y):
        mse = loss_func(pred, y)
        reg_loss = 0.0
        for name, param in self.model.named_parameters():
            if "user_factor" in name:
                partial_loss = torch.sum(param[row]**2)
            elif "item_factor" in name:
                partial_loss = torch.sum(param[col]**2)
            reg_loss = reg_loss + partial_loss
        total_loss = mse + self.lmbd * reg_loss
        return mse, total_loss


if __name__ == "__main__":
    train = Train(factors=20, file_name="exchange_rate.txt", epochs=100, train_ratio=0.8)
    train.run()

