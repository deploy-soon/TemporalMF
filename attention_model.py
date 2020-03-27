import fire
import torch
from torch import nn

from temporal_model import TemporalTrain, TemporalMF



"""
Shun-Yao Shih, Fan-Keng Sun, Hung-yi Lee
Temporal Pattern Attention for Multivariate Time Series Forecasting
Journal track of ECML/PKDD 2019
"""
class AttnLSTM(nn.Module):

    def __init__(self, factors, lag_set, hidden_dim, n_layers, kernels, dropout, device):
        super().__init__()
        lags = len(lag_set)
        self.lag_set = lag_set
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.rnn = nn.LSTM(factors, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.conv = nn.Conv2d(in_channels=1, out_channels=kernels,
                              kernel_size=(1, lags), bias=False)

        self.attn_weight = nn.Linear(kernels, hidden_dim, bias=False)
        self.fc_v = nn.Linear(kernels, hidden_dim)
        self.fc_h = nn.Linear(hidden_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim * 2, factors, bias=False)

    def regularizer(self):
        return torch.FloatTensor([0.0]).to(self.device)

    def forward(self, lags_vectors):
        # lags_vectors = (batch, lags, factors)
        embedded = self.dropout(lags_vectors)
        out, hidden = self.rnn(embedded)
        # out = (batch, lags, hidden_dim)

        ipt_embedded = out.permute(0, 2, 1)
        ipt_embedded = ipt_embedded.unsqueeze(1)
        # ipt_embedded = (batch, 1, hidden_dim, lags)
        opt_embedded = out[:, -1, :]
        # opt_embedded = (batch, hidden_dim)
        conv = self.conv(ipt_embedded).squeeze()
        # conv = (batch, kernels, hidden_dim)
        conv = conv.permute(0, 2, 1)
        # conv = (batch, hidden_dim, kernels)
        attn_weight = self.attn_weight(conv)
        # attn_weight = (batch, hidden_dim, hidden_dim)
        attn_weight = torch.bmm(attn_weight, opt_embedded.unsqueeze(2))
        # attn_weight = (batch, hidden_dim, 1)
        attn_weight = torch.sigmoid(attn_weight)
        # attn_weight = (batch, hidden_dim, 1)
        v = torch.sum(conv * attn_weight, dim=1)

        v = self.fc_v(v)
        h = self.fc_h(opt_embedded)
        out = self.fc(torch.cat((v, h), dim=1))
        return out


class AttnEmbedding(nn.Module):
    """
    Since the input of temporal model is embedding layer,
    AttnEmbedding model use the embedding layer to get attention weight.
    """

    def __init__(self, factors, lag_set, kernels, device):
        super().__init__()
        lags = len(lag_set)
        self.lag_set = lag_set
        self.device = device

        self.conv = nn.Conv2d(in_channels=1, out_channels=kernels,
                              kernel_size=(1, lags), bias=False)

        self.attn_weight = nn.Linear(kernels, factors, bias=False)
        self.fc_v = nn.Linear(kernels, factors)
        self.fc_h = nn.Linear(factors, factors)

        self.fc = nn.Linear(factors * 2, factors, bias=False)

    def regularizer(self):
        return torch.FloatTensor([0.0]).to(self.device)

    def forward(self, lags_vectors):
        # lags_vectors = (batch, lags, factors)

        ipt_embedded = lags_vectors.permute(0, 2, 1)
        ipt_embedded = ipt_embedded.unsqueeze(1)
        # ipt_embedded = (batch, 1, factors, lags - 1)
        opt_embedded = lags_vectors[:, -1, :]
        conv = self.conv(ipt_embedded)
        # conv = (batch, kernels, factors, 1)
        conv = conv.squeeze().permute(0, 2, 1)
        # conv = (batch, factors, kernels)
        attn_weight = self.attn_weight(conv)
        attn_weight = torch.bmm(attn_weight, opt_embedded.unsqueeze(2))
        attn_weight = torch.sigmoid(attn_weight)
        v = torch.sum(conv * attn_weight, dim=1)

        v = self.fc_v(v)
        h = self.fc_h(opt_embedded)
        out = self.fc(torch.cat((v, h), dim=1))
        return out


class AttnLSTMMF(TemporalTrain):

    def __init__(self, hidden_dim=128, n_layers=1, kernels=64, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.hidden_dim=hidden_dim
        self.kernels = kernels
        self.dropout = dropout
        self.temporal_model = AttnLSTM(self.factors,
                                       self.lag_set,
                                       hidden_dim,
                                       n_layers=self.n_layers,
                                       kernels=self.kernels,
                                       dropout=self.dropout,
                                       device=self.device).to(self.device)
        self.model = TemporalMF(users=self.data.users,
                            items=self.data.items,
                            factors=self.factors,
                            temporal_model=self.temporal_model).to(self.device)


class AttnMF(TemporalTrain):

    def __init__(self, kernels=64, **kwargs):
        super().__init__(**kwargs)
        self.kernels = kernels
        self.temporal_model = AttnEmbedding(self.factors,
                                            self.lag_set,
                                            kernels=self.kernels,
                                            device=self.device).to(self.device)
        self.model = TemporalMF(users=self.data.users,
                            items=self.data.items,
                            factors=self.factors,
                            temporal_model=self.temporal_model).to(self.device)


if __name__ == "__main__":
    fire.Fire({
        "attnlstm": AttnLSTMMF,
        "attn": AttnMF,
    })

