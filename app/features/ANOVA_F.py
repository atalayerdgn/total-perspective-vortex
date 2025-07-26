from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ANOVA_F(BaseEstimator, TransformerMixin):
    def __init__(self, k=10):
        self.k = k
        self.selector = None
        self.selected_channels_ = None

    def fit(self, X, y):
        """
        Fit ANOVA_F selector on EEG data
        X: shape (n_samples, n_channels, n_timepoints)
        y: shape (n_samples,)
        """
        # Compute channel-wise features (variance across time)
        n_samples, n_channels, n_timepoints = X.shape
        X_features = np.var(X, axis=2)  # Shape: (n_samples, n_channels)
        
        # Apply feature selection
        self.selector = SelectKBest(score_func=f_classif, k=self.k)
        self.selector.fit(X_features, y)
        self.selected_channels_ = self.selector.get_support(indices=True)
        return self

    def transform(self, X):
        """
        Transform EEG data by selecting best channels
        X: shape (n_samples, n_channels, n_timepoints)
        Returns: shape (n_samples, k_selected, n_timepoints)
        """
        if self.selector is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Select the best channels directly from the 3D data
        return X[:, self.selected_channels_, :]
