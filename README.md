# Temporal Matrix Factorization

Implementations of the following paper

```
H.-F. Yu, N. Rao, and I. S. Dhillon. Temporal Regularized
  Matrix Factoriztion for High-dimensional Time Series Prediction. Advances
  in Neural Information Processing Systems (NIPS) 29, 2016.
```

with torch framework


## How to run

```
# load temporal data
$ git submodule update --init
$ bash load_data.sh

# run mf
$ python temporal_model.py run
$ python autoregressive_model.py vector run
$ python autoregressive_model.py matrix run
$ python autoregressive_model.py tensor run

# train RNN
$ python recurrent_model.py rnn run --lags=100
$ python recurrent_model.py lstm run --hid_dim=128
$ python recurrent_model.py gru run --n_layers=2

```
