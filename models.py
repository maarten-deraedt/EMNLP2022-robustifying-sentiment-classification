from sklearn.linear_model import LinearRegression
from random import gauss

import numpy as np


"""Main models"""


class MeanOffset(object):
    def __init__(self, X, y, src_label: int):
        X_train_src = X[y == src_label]
        X_train_tgt = X[y != src_label]
        self.offset = np.mean(X_train_tgt - X_train_src, axis=0)

    def transform(self, X, y):
        return (X + self.offset), np.abs(1 - y)


class MeanOffsetRegressor(object):
    def __init__(self, X, y, src_label: int):
        X_train_src = X[y == src_label]
        X_train_tgt = X[y != src_label]
        self.mean_offset = np.mean(X_train_tgt - X_train_src, axis=0)
        self.residuals = (X_train_tgt - X_train_src) - self.mean_offset
        self.regressor = LinearRegression(n_jobs=-1).fit(X_train_src, self.residuals)

    def transform(self, X, y):
        residuals = self.regressor.predict(X)
        return (X + self.mean_offset + residuals), np.abs(1 - y)


"""
Ablation Models
"""


class OriginalMeanOffset(object):
    """
    Ablation model.
    The mean offset calculated between id (original) samples with opposite labels (no counterfactuals are used).
    The resulting offset is then added to original samples to produce dummy 'counterfactuals'.
    """
    def __init__(self, X_orig, y_orig, src_label: int):
        X_train_src = X_orig[y_orig == src_label]
        X_train_tgt = X_orig[y_orig != src_label]
        if X_train_src.shape[0] >= X_train_tgt.shape[0]:
            n_min = X_train_tgt.shape[0]
        else:
            n_min = X_train_src.shape[0]
        self.offset = np.mean(X_train_tgt[:n_min, :] - X_train_src[:n_min, :], axis=0)

    def transform(self, X, y):
        return (X + self.offset), np.abs(1 - y)


class RandomOffset(object):
    """
    Ablation model.
    Samples a random offset with the same L2 norm as the mean offset calculated from the k available (original, counterfactual)-pairs.
    The random offset is then added to the original samples to produce dummy 'counterfactuals'.
    """
    def __init__(self, X, y, src_label: int):
        X_train_src = X[y == src_label]
        X_train_tgt = X[y != src_label]
        self.target_norm = np.linalg.norm(np.mean(X_train_tgt - X_train_src, axis=0))
        self.d = X_train_src.shape[1]
        self.offset = self.sample_random_offset()

    def sample_random_offset(self):
        vec = [gauss(0, 1) for _ in range(self.d)]
        mag = sum(x ** 2 for x in vec) ** .5
        unit = np.array([x / mag for x in vec])
        scaled = unit * self.target_norm
        assert np.abs(np.linalg.norm(scaled) - self.target_norm) < 0.0001
        return scaled

    def transform(self, X, y):
        return (X + self.offset), np.abs(1 - y)


class LinearRegressor(object):
    """
    Ablation model.
    Directly learns a linear mapping in the embedding space to transform original samples to counterfactuals.
    Trained based on the k available (original, counterfactual)-pairs
    """
    def __init__(self, X, y, src_label: int):
        X_train_src = X[y == src_label]
        X_train_tgt = X[y != src_label]
        self.regressor = LinearRegression(n_jobs=-1).fit(X_train_src, X_train_tgt)

    def transform(self, X, y):
        return self.regressor.predict(X), np.abs(1 - y)