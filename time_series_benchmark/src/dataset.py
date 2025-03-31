import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as sliding_window

import torch
from torch.utils.data import Dataset, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def train_val_test_split(X, fractions, n_history, n_horizon, normalize=True, d_embedding=4, embedding=None):

    n_window = n_history + n_horizon
    
    n_steps, n_channels = X.shape
    if d_embedding > n_channels:
        raise ValueError("Embedding dimension should be less or equals the number of channels.")

    if len(fractions)==2:
        f_train, f_test = fractions
    elif len(fractions)==3:
        f_train, f_val_, f_test = fractions
    
    n_train = int(n_steps * f_train)
    n_test = int(n_steps * f_test)
    n_val = n_steps - n_train - n_test

    start_train, end_train = 0, n_train
    start_val, end_val = n_train, n_train + n_val
    start_test, end_test = n_train + n_val, n_train + n_val + n_test

    if normalize:
        scaler = StandardScaler().fit(X[start_train:end_train])
        X = scaler.transform(X)

    # C = torch.tensor(np.corrcoef(X[start_train:end_train].T), dtype=torch.float)
    # _, eigenvectors = torch.linalg.eigh(C)
    # embedding = eigenvectors[:,-d_embedding:]

    # TODO: pass embedding
    embedding = PCA(n_components=d_embedding).fit_transform(np.corrcoef(X[start_train:end_train].T))
    # # embedding = PCA(n_components=d_embedding, whiten=True).fit_transform(X[start_train:end_train].T)
    embedding = torch.tensor(embedding, dtype=torch.float)

    XY = sliding_window(X, n_window, axis=0) # n_windows, n_channels, n_window
    XY_train = XY[:end_train-n_window+1]
    XY_val = XY[start_val-n_history:end_val-n_window+1]
    XY_test = XY[start_test-n_history:end_test-n_window+1]

    n_points_train = XY_train.shape[0]
    n_points_val = XY_val.shape[0]
    n_points_test = XY_test.shape[0]

    X_train = np.zeros((n_points_train, n_channels, n_history))
    Y_train = np.zeros((n_points_train, n_channels, n_horizon))
    for i in range(n_points_train):
        X_train[i] = XY_train[i,:,:n_history]
        Y_train[i] = XY_train[i,:,n_history:]

    X_val = np.zeros((n_points_val, n_channels, n_history))
    Y_val = np.zeros((n_points_val, n_channels, n_horizon))
    for i in range(n_points_val):
        X_val[i] = XY_val[i,:,:n_history]
        Y_val[i] = XY_val[i,:,n_history:]

    X_test = np.zeros((n_points_test, n_channels, n_history))
    Y_test = np.zeros((n_points_test, n_channels, n_horizon))
    for i in range(n_points_test):
        X_test[i] = XY_test[i,:,:n_history]
        Y_test[i] = XY_test[i,:,n_history:]

    dataset_train = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(Y_train, dtype=torch.float))
    dataset_val = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(Y_val, dtype=torch.float))
    dataset_test = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(Y_test, dtype=torch.float))

    return dataset_train, dataset_val, dataset_test, embedding # TODO


class TimeSeriesKNNDataset(Dataset):
    def __init__(self, X, n_history, n_horizon, k_neighbors):
        self.X = torch.tensor(X, dtype=torch.float)
        self.n_history = n_history
        self.n_horizon = n_horizon
        self.k_neighbors = k_neighbors

        X_flat = self.X.reshape(len(self.X), -1)
        dist_mat = torch.cdist(X_flat, X_flat)
        self.nn_indices = torch.topk(dist_mat, k=k_neighbors + 1, 
                                   largest=False, sorted=True).indices[:, 1:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_center = self.X[idx]
        neighbor_indices = self.nn_indices[idx]
        x_neighbors = self.X[neighbor_indices]

        x_c = x_center[:, :self.n_history].flatten()
        x_n = x_neighbors[:, :, :self.n_history].reshape(self.k_neighbors, -1)

        return {
            'x_center': x_c,
            'x_neighbors': x_n,
            'neighbor_indices': neighbor_indices
        }


def train_val_test_split_with_knn(X, fractions, n_history, n_horizon, k_neighbors=5, normalize=True, d_embedding=4):
    n_window = n_history + n_horizon
    
    n_steps, n_channels = X.shape
    if d_embedding > n_channels:
        raise ValueError("Embedding dimension should be less or equals the number of channels.")

    f_train, f_val, f_test = fractions
    
    n_train = int(n_steps * f_train)
    n_val = int(n_steps * f_val)
    n_test = int(n_steps * f_test)

    start_train, end_train = 0, n_train
    start_val, end_val = n_train, n_train + n_val
    start_test, end_test = n_train + n_val, n_steps

    if normalize:
        scaler = StandardScaler().fit(X[start_train:end_train])
        X = scaler.transform(X)

    embedding = PCA(n_components=d_embedding).fit_transform(np.corrcoef(X[start_train:end_train].T))
    embedding = torch.tensor(embedding, dtype=torch.float)

    XY = sliding_window(X, n_window, axis=0)
    XY_train = XY[:end_train-n_window+1]
    XY_val = XY[start_val-n_history:end_val-n_window+1]
    XY_test = XY[start_test-n_history:end_test-n_window+1]

    dataset_train = TimeSeriesKNNDataset(XY_train, n_history, n_horizon, k_neighbors)
    dataset_val = TimeSeriesKNNDataset(XY_val, n_history, n_horizon, k_neighbors)
    dataset_test = TimeSeriesKNNDataset(XY_test, n_history, n_horizon, k_neighbors)

    return dataset_train, dataset_val, dataset_test, embedding