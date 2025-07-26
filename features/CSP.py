import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4, log=True):
        self.n_components = n_components
        self.filters = None
        self.log = log
    def fit(self, X, y):
        def calculate_covariance(X, y, class_label):
            trials = [X[i] for i in range(len(X)) if y[i] == class_label]
            cov_matrices = []
            for trial in trials:
                trial = trial - trial.mean(axis=1, keepdims=True)  # DC offset removal
                cov = trial @ trial.T
                cov /= np.trace(cov)
                cov_matrices.append(cov)
            return np.mean(cov_matrices, axis=0)
        N, C, T = X.shape
        labels = np.unique(y)
        if len(labels) != 2:
            raise ValueError("CSP requires exactly 2 classes.")
        cov1 = calculate_covariance(X, y, labels[0])
        cov2 = calculate_covariance(X, y, labels[1])
        composite_cov = cov1 + cov2
        eigvals, eigvecs = np.linalg.eigh(composite_cov)
        eigvals[eigvals < 1e-12] = 1e-12
        whitening_mat = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        S1 = whitening_mat @ cov1 @ whitening_mat.T
        eigvals_s1, eigvecs_s1 = np.linalg.eigh(S1)
        idx = np.argsort(eigvals_s1)[::-1]
        eigvecs_s1 = eigvecs_s1[:, idx]

        if self.n_components % 2 != 0 or self.n_components > C:
            raise ValueError("n_components must be even and <= number of channels")

        self.filters = (eigvecs_s1.T @ whitening_mat)[:self.n_components]
        return self
    def transform(self, X):
        if self.filters is None:
            raise ValueError("Model not fitted.")
        N, _, T = X.shape
        # Apply CSP filters to get 3D output (samples, components, time)
        X_csp = np.zeros((N, self.n_components, T))
        for i in range(N):
            X_csp[i] = self.filters @ X[i]
        return X_csp

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
