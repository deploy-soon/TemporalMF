import fire
import torch
from torch import nn

from temporal_model import TemporalTrain, TemporalMF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorEmbedding(nn.Module):

    def __init__(self, lag_set):
        super().__init__()
        lags = len(lag_set)
        self.lag_set = lag_set
        self.lag_factor = nn.Parameter(torch.rand(lags))

    def regularizer(self):
        return torch.sum(self.lag_factor ** 2)

    def forward(self, lags_vectors):
        temp = lags_vectors.permute(0, 2, 1)
        embedding_lags_dot = temp * self.lag_factor
        embedding_lags_dot = embedding_lags_dot.permute(0, 2, 1)
        #embedding_lags_dot = (batch, lags, factors)
        target_vectors = torch.sum(embedding_lags_dot, dim=1)
        #target_vectors = (batch, factors)
        return target_vectors


class MatrixEmbedding(nn.Module):

    def __init__(self, lag_set, factors):
        super().__init__()
        lags = len(lag_set)
        self.lag_set = lag_set
        self.lag_factor = nn.Parameter(torch.rand(lags, factors))

    def regularizer(self):
        return torch.sum(self.lag_factor ** 2)

    def forward(self, lags_vectors):
        embedding_lags_dot = lags_vectors * self.lag_factor
        #embedding_lags_dot = (batch, lags, factors)
        target_vectors = torch.sum(embedding_lags_dot, dim=1)
        #target_vectors = (batch, factors)
        return target_vectors


class TensorEmbedding(nn.Module):

    def __init__(self, lag_set, factors):
        super().__init__()
        lags = len(lag_set)
        self.lags = lags
        self.lag_set = lag_set
        self.factors = factors
        self.lag_factor = nn.Parameter(torch.rand(factors, lags, factors))

    def regularizer(self):
        return torch.sum(self.lag_factor ** 2)

    def forward(self, lags_vectors):
        factors_lags_vectors = lags_vectors.repeat(1, self.factors, 1)
        factors_lags_vectors = factors_lags_vectors.view(-1, self.factors, self.lags, self.factors)
        embedding_lags_dot = factors_lags_vectors * self.lag_factor
        #embedding_lags_dot = (batch, facors, lags, factors)
        target_vectors = torch.sum(embedding_lags_dot, dim=(2, 3))
        #target_vectors = (batch, factors)
        return target_vectors


class VectorMF(TemporalTrain):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temporal_model = VectorEmbedding(lag_set=self.lag_set).to(device)
        self.model = TemporalMF(users=self.data.users,
                            items=self.data.items,
                            factors=self.factors,
                            temporal_model=self.temporal_model).to(device)


class MatrixMF(TemporalTrain):

    def __init__(self, **kwargs):
        super().__init__(** kwargs)
        self.temporal_model = MatrixEmbedding(lag_set=self.lag_set,
                                              factors=self.factors).to(device)
        self.model = TemporalMF(users=self.data.users,
                            items=self.data.items,
                            factors=self.factors,
                            temporal_model=self.temporal_model).to(device)


class TensorMF(TemporalTrain):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temporal_model = TensorEmbedding(lag_set=self.lag_set,
                                              factors=self.factors).to(device)
        self.model = TemporalMF(users=self.data.users,
                            items=self.data.items,
                            factors=self.factors,
                            temporal_model=self.temporal_model).to(device)


if __name__ == "__main__":
    fire.Fire({
        "vector": VectorMF,
        "matrix": MatrixMF,
        "tensor": TensorMF,
    })

