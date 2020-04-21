import fire
import torch
from torch import nn
from torch.nn import functional as F

from temporal_model import TemporalTrain, TemporalMF



class BaseAttnLSTM(nn.Module):

    def __init__(self, factors, hid_dim, n_layers, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.device = device

        self.rnn = nn.LSTM(factors, hid_dim, n_layers,
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attn_weight = nn.Linear(hid_dim, factors, bias=False)
        self.fc = nn.Linear(hid_dim, factors)

    def regularizer(self):
        return torch.FloatTensor([0.0]).to(self.device)

    def attention(self, opts, hidden):
        hidden = hidden.squeeze(0)
        # hidden = (batch_size, hid_dim)
        attn_weights = torch.bmm(opts, hidden.unsqueeze(2)).squeeze(2)
        # attn_weights = (batch_size, lags, 1) -> (batch_size, lags)
        norm_attn = F.softmax(attn_weights, dim=1)
        attn_hidden = torch.bmm(opts.permute(0, 2, 1),
                                norm_attn.unsqueeze(2)).squeeze(2)
        return attn_hidden

    def forward(self, lags_vectors):
        # lags_vectors = (batch, lags, factors)
        embedded = self.dropout(lags_vectors)
        opts, (hidden, cell) = self.rnn(embedded)
        # opts = (batch, lags, hidden_dim)
        # hidden = (n_layers, batch_size, hid_dim)
        opts = self.attention(opts, hidden)
        return self.fc(opts)


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


class LSTNet(nn.Module):
    """
    LSTNet
    """
    def __init__(self, factors, lag_set, device):
        super().__init__()
        self.device = device
        self.lags = len(lag_set)
        self.factors = factors
        self.hidR = 100
        self.hidC = 100
        self.hidS = 4
        self.Ck = 7 # CNN kernel
        self.skip = 10
        self.pt = int((self.lags - self.Ck)/self.skip) # 6
        self.hw = 21 # window size of highway component
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.factors))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p = 0.2)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.factors)
        else:
            self.linear1 = nn.Linear(self.hidR, self.factors)

        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        #if (args.output_fun == 'sigmoid'):
        #    self.output = F.sigmoid
        #if (args.output_fun == 'tanh'):
        #    self.output = F.tanh

    def regularizer(self):
        return torch.FloatTensor([0.0]).to(self.device)

    def forward(self, lags_vectors):
        # lags_vectors = (batch, lags, factors)
        batch_size = lags_vectors.size(0)
        #CNN
        c = lags_vectors.view(-1, 1, self.lags, self.factors)
        # c = (batch_size, channels, lags - Ck)
        c = F.relu(self.conv1(c)).squeeze()
        c = self.dropout(c)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r);
        # r = (n_layer, batch size, hidden dim)
        r = r.squeeze()
        r = self.dropout(r)

        #skip-rnn
        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)
        res = self.linear1(r)

        #highway
        if (self.hw > 0):
            z = lags_vectors[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            #print("z ", z.size())
            z = self.highway(z)
            z = z.view(-1, self.factors)
            res = res + z
        if (self.output):
            res = self.output(res)
        return res


class LSTNetMF(TemporalTrain):

    def __init__(self, hid_dim=128, n_layers=1, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.temporal_model = LSTNet(factors=self.factors,
                                     lag_set=self.lag_set,
                                     device=self.device).to(self.device)
        self.model = TemporalMF(users=self.data.users,
                            items=self.data.items,
                            factors=self.factors,
                            temporal_model=self.temporal_model).to(self.device)

class BaseAttnLSTMMF(TemporalTrain):

    def __init__(self, hid_dim=128, n_layers=1, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.temporal_model = BaseAttnLSTM(factors=self.factors,
                                           hid_dim=self.hid_dim,
                                           n_layers=self.n_layers,
                                           dropout=self.dropout,
                                           device=self.device).to(self.device)
        self.model = TemporalMF(users=self.data.users,
                            items=self.data.items,
                            factors=self.factors,
                            temporal_model=self.temporal_model).to(self.device)


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
        "baseattn": BaseAttnLSTMMF,
        "attnlstm": AttnLSTMMF,
        "attn": AttnMF,
        "lstnet": LSTNetMF,
    })

