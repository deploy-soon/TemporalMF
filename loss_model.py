import fire
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

    def __init__(self, lmbd=0.04, **kwargs):
        super().__init__(**kwargs)
        self.lmbd = lmbd
        self.model = BaseMF(self.data.users, self.data.items, self.factors).to(device)

    def get_loss(self, loss_func, row, col, pred, y):
        mse = loss_func(pred, y)
        reg_loss = torch.sum(self.model.user_factor(row) ** 2) +\
            torch.sum(self.model.item_factor(col) ** 2)
        total_loss = mse + self.lmbd * reg_loss
        return mse, total_loss


if __name__ == "__main__":
    fire.Fire(Train)

