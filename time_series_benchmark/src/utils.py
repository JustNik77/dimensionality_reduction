import numpy as np
from scipy.signal import convolve2d

def moving_average(x, n_window=3):
    if len(x) <= n_window:
        return x
    h_window = n_window // 2
    kernel = (np.hanning(n_window+2) / np.hanning(n_window+2).sum())[1:-1]
    x_padded = np.pad(x, h_window, mode="edge")
    x_convolved = np.convolve(x_padded, kernel, mode="same")
    return x_convolved[h_window:-h_window]

def moving_average2d(x, n_window=3):
    if x.shape[-1] <= n_window:
        return x
    h_window = n_window // 2
    kernel = np.array([(np.hanning(n_window+2) / np.hanning(n_window+2).sum())[1:-1]])
    x_padded = np.pad(x, ((0, 0), (h_window, h_window)), mode="edge")
    x_convolved = convolve2d(x_padded, kernel, mode="same")
    return x_convolved[:,h_window:-h_window]


