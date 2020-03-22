import fire
import torch
from torch import nn

from temporal_model import TemporalTrain, TemporalMF


class LagRNN(nn.Module):

    def __init__(self, factors, lag_set, hidden_dim, n_layers, dropout, device):
        super().__init__()
        lags = len(lag_set)
        self.lag_set = lag_set
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.rnn = nn.RNN(factors, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, factors)

    def regularizer(self):
        return torch.FloatTensor([0.0]).to(self.device)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

    def forward(self, lags_vectors):
        batch_size = lags_vectors.size(0)
        hidden = self.init_hidden(batch_size)
        embedded = self.dropout(lags_vectors)
        out, hidden = self.rnn(embedded, hidden)
        out = out[:,-1,:]
        out = self.fc(out)
        return out


class LagLSTM(nn.Module):

    def __init__(self, factors, lag_set, hidden_dim, n_layers, dropout, device):
        super().__init__()
        lags = len(lag_set)
        self.lag_set = lag_set
        self.hidden_dim = hidden_dim
        self.device = device

        self.rnn = nn.LSTM(factors, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, factors)

    def regularizer(self):
        return torch.FloatTensor([0.0]).to(self.device)

    def forward(self, lags_vectors):
        embedded = self.dropout(lags_vectors)
        out, (hidden, cell) = self.rnn(embedded)
        out = out[:,-1,:]
        out = self.fc(out)
        return out


class LagGRU(nn.Module):

    def __init__(self, factors, lag_set, hidden_dim, n_layers, dropout, device):
        super().__init__()
        lags = len(lag_set)
        self.lag_set = lag_set
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.rnn = nn.GRU(factors, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, factors)

    def regularizer(self):
        return torch.FloatTensor([0.0]).to(self.device)

    def forward(self, lags_vectors):
        embedded = self.dropout(lags_vectors)
        out, hidden = self.rnn(embedded)
        out = out[:,-1,:]
        out = self.fc(out)
        return out


class RNNMF(TemporalTrain):

    def __init__(self, hidden_dim=64, n_layers=1, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.temporal_model = LagRNN(self.factors,
                                     self.lag_set,
                                     hidden_dim,
                                     n_layers=self.n_layers,
                                     dropout=self.dropout,
                                     device=self.device).to(self.device)
        self.model = TemporalMF(users=self.data.users,
                            items=self.data.items,
                            factors=self.factors,
                            temporal_model=self.temporal_model).to(self.device)


class LSTMMF(TemporalTrain):

    def __init__(self, hidden_dim=64, n_layers=1, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.temporal_model = LagLSTM(self.factors,
                                      self.lag_set,
                                      hidden_dim,
                                      n_layers=self.n_layers,
                                      dropout=self.dropout,
                                      device=self.device).to(self.device)
        self.model = TemporalMF(users=self.data.users,
                            items=self.data.items,
                            factors=self.factors,
                            temporal_model=self.temporal_model).to(self.device)


class GRUMF(TemporalTrain):

    def __init__(self, hidden_dim=64, n_layers=1, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.temporal_model = LagGRU(self.factors,
                                     self.lag_set,
                                     hidden_dim,
                                     n_layers=self.n_layers,
                                     dropout=self.dropout,
                                     device=self.device).to(self.device)
        self.model = TemporalMF(users=self.data.users,
                            items=self.data.items,
                            factors=self.factors,
                            temporal_model=self.temporal_model).to(self.device)


if __name__ == "__main__":
    fire.Fire({
        "rnn": RNNMF,
        "lstm": LSTMMF,
        "gru": GRUMF,
    })

