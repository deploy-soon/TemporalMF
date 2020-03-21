import fire
import torch
from torch import nn

from temporal_model import TemporalTrain, TemporalMF


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


class MLPVectorEmbedding(nn.Module):

    def __init__(self, lags, factors, hid_dim, device):
        super().__init__()
        self.lags = lags
        self.factors = factors
        self.device = device

        self.fc1 = nn.Linear(lags, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 1)

    def regularizer(self):
        return torch.FloatTensor([0.0]).to(self.device)

    def forward(self, lags_vectors):
        lags_vectors = lags_vectors.permute(0, 2, 1)
        #lags_vectors = (batch, factors, lags)
        target_vectors = self.fc1(lags_vectors)
        # target_vectros = (batch, factors, hid_dim)
        target_vectors = self.fc2(target_vectors)
        # target_vectors = (batch, factors, 1)
        return target_vectors.squeeze()


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


class MLPMatrixEmbedding(nn.Module):

    def __init__(self, lags, factors, hid_dim, device):
        super().__init__()
        self.lags = lags
        self.factors = factors
        self.device = device

        self.fc = nn.ModuleList()

        for factor in range(factors):
            self.fc.append(
                nn.Sequential(
                    nn.Linear(lags, hid_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hid_dim, 1)
                )
            )

    def regularizer(self):
        return torch.FloatTensor([0.0]).to(self.device)

    def forward(self, lags_vectors):
        lags_vectors = lags_vectors.permute(0, 2, 1)
        #lags_vectors = (batch, factors, lags)
        embeddings = []
        #TODO: ineffective matrix concat
        for factor in range(self.factors):
            linear = self.fc[factor](lags_vectors[:, factor, :])
            embeddings.append(linear)
        target_vectors = torch.cat(embeddings, dim=-1)
        # target_vectors = (batch, factors)
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


class MLPTensorEmbedding(nn.Module):

    def __init__(self, lags, factors, hid_dim, device):
        super().__init__()
        self.lags = lags
        self.factors = factors
        self.device = device

        self.fc1 = nn.Linear(lags * factors, hid_dim)
        self.fc2 = nn.Linear(hid_dim, factors)

    def regularizer(self):
        return torch.FloatTensor([0.0]).to(self.device)

    def forward(self, lags_vectors):
        # flatten batch vectors
        lags_vector = lags_vectors.view(-1, self.factors * self.lags)
        # lags_vector = (batch, factors * lags)
        target_vectors = self.fc1(lags_vector)
        # target_vectros = (batch, hid_dim)
        target_vectors = self.fc2(target_vectors)
        # target_vectors = (batch, factors)
        return target_vectors


class VectorMF(TemporalTrain):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temporal_model = VectorEmbedding(lag_set=self.lag_set).to(self.device)
        self.model = TemporalMF(users=self.data.users,
                            items=self.data.items,
                            factors=self.factors,
                            temporal_model=self.temporal_model).to(self.device)


class MatrixMF(TemporalTrain):

    def __init__(self, **kwargs):
        super().__init__(** kwargs)
        self.temporal_model = MatrixEmbedding(lag_set=self.lag_set,
                                              factors=self.factors).to(self.device)
        self.model = TemporalMF(users=self.data.users,
                            items=self.data.items,
                            factors=self.factors,
                            temporal_model=self.temporal_model).to(self.device)


class TensorMF(TemporalTrain):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temporal_model = TensorEmbedding(lag_set=self.lag_set,
                                              factors=self.factors).to(self.device)
        self.model = TemporalMF(users=self.data.users,
                            items=self.data.items,
                            factors=self.factors,
                            temporal_model=self.temporal_model).to(self.device)


class MLPVectorMF(TemporalTrain):

    def __init__(self, hid_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.hid_dim = hid_dim
        self.temporal_model = MLPVectorEmbedding(lags=self.lags,
                                                 factors=self.factors,
                                                 hid_dim=hid_dim,
                                                 device=self.device).to(self.device)
        self.model = TemporalMF(users=self.data.users,
                                items=self.data.items,
                                factors=self.factors,
                                temporal_model=self.temporal_model).to(self.device)


class MLPMatrixMF(TemporalTrain):

    def __init__(self, hid_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.hid_dim = hid_dim
        self.temporal_model = MLPMatrixEmbedding(lags=self.lags,
                                                 factors=self.factors,
                                                 hid_dim=hid_dim,
                                                 device=self.device).to(self.device)
        self.model = TemporalMF(users=self.data.users,
                                items=self.data.items,
                                factors=self.factors,
                                temporal_model=self.temporal_model).to(self.device)


class MLPTensorMF(TemporalTrain):

    def __init__(self, hid_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.hid_dim = hid_dim
        self.temporal_model = MLPTensorEmbedding(lags=self.lags,
                                                 factors=self.factors,
                                                 hid_dim=hid_dim,
                                                 device=self.device).to(self.device)
        self.model = TemporalMF(users=self.data.users,
                                items=self.data.items,
                                factors=self.factors,
                                temporal_model=self.temporal_model).to(self.device)


if __name__ == "__main__":
    fire.Fire({
        "vector": VectorMF,
        "matrix": MatrixMF,
        "tensor": TensorMF,
        "mlpvector": MLPVectorMF,
        "mlpmatrix": MLPMatrixMF,
        "mlptensor": MLPTensorMF,
    })

