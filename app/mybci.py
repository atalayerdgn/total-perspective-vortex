from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import joblib
from mne.io import concatenate_raws, read_raw_edf
from sklearn.base import BaseEstimator, TransformerMixin
import sys
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import time

class MyCSP(BaseEstimator, TransformerMixin):
    """Custom CSP transformer that accepts X shaped (n_epochs, n_channels, n_times)
    and returns (n_epochs, n_components, n_times)."""
    def __init__(self, n_components=4, log=True, norm_trace=False):
        self.n_components = n_components
        self.filters = None
        self.log = log
        self.norm_trace = norm_trace
    def fit(self, X, y):
        def calculate_covariance(X, y, class_label):
            trials = [X[i] for i in range(len(X)) if y[i] == class_label]
            cov_matrices = []
            for trial in trials:
                trial = trial - trial.mean(axis=1, keepdims=True)  # DC offset removal
                cov = trial @ trial.T
                # norm_trace etkisi: kovaryans matrisini normalize et
                if self.norm_trace:
                    cov /= np.trace(cov)
                cov_matrices.append(cov)
            return np.mean(cov_matrices, axis=0)
            
        N, C, T = X.shape
        labels = np.unique(y)
        if len(labels) != 2:
            raise ValueError("CSP requires exactly 2 classes.")
            
        # Her iki sınıf için kovaryans matrislerini hesapla
        cov1 = calculate_covariance(X, y, labels[0])
        cov2 = calculate_covariance(X, y, labels[1])
        
        # Kompozit kovaryans matrisi
        composite_cov = cov1 + cov2
        
        # Whitening matrisi hesapla
        eigvals, eigvecs = np.linalg.eigh(composite_cov)
        eigvals[eigvals < 1e-12] = 1e-12
        whitening_mat = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # S1 matrisini hesapla ve özdeğer ayrıştırması yap
        S1 = whitening_mat @ cov1 @ whitening_mat.T
        eigvals_s1, eigvecs_s1 = np.linalg.eigh(S1)
        idx = np.argsort(eigvals_s1)[::-1]
        eigvecs_s1 = eigvecs_s1[:, idx]

        if self.n_components % 2 != 0 or self.n_components > C:
            raise ValueError("n_components must be even and <= number of channels")

        # CSP filtrelerini hesapla
        self.filters = (eigvecs_s1.T @ whitening_mat)[:self.n_components]
        
        # norm_trace=False durumunda ek normalizasyon yapma
        if not self.norm_trace:
            # Filtrelerin doğal varyansını koru
            pass
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
class LoadEEGBCI:
    def __init__(self, runs: list[int], subjects: list[int]):
        self.runs = runs
        self.subjects = subjects
    def load_and_process_data(self):
        tmin, tmax = -1.0, 4.0
        raw_fnames = eegbci.load_data(self.subjects, self.runs, path="./data")
        raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
        eegbci.standardize(raw)  # set channel names
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)
        raw.annotations.rename(dict(T1="hands", T2="feet"))
        raw.set_eeg_reference(projection=True)
        raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        epochs = Epochs(
            raw,
            event_id=["hands", "feet"],
            tmin=tmin,
            tmax=tmax,
            proj=True,
            picks=picks,
            baseline=None,
            preload=True,
        )
        return epochs
    def transform_data(self, epochs):
        # For pipeline compatibility return raw epoch array and mapped labels.
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        raw_labels = epochs.events[:, -1]
        unique = np.unique(raw_labels)
        label_map = {v: i for i, v in enumerate(unique)}
        labels = np.array([label_map[v] for v in raw_labels], dtype=int)
        print(f"Event id mapping: {label_map}")
        return data, labels
    
class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Convert CSP output (n_epochs, n_components, n_times) to 2D features.
    Default feature: log-variance across time for each component."""
    def __init__(self, feature='logvar'):
        self.feature = feature
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Expect X shape: (n_epochs, n_components, n_times)
        if X.ndim != 3:
            raise ValueError('FeatureExtractor expects 3D input (n_epochs, n_components, n_times)')
        n_epochs, n_comp, T = X.shape
        feats = np.zeros((n_epochs, n_comp))
        for i in range(n_epochs):
            for j in range(n_comp):
                x = X[i, j, :]
                if self.feature == 'logvar':
                    feats[i, j] = np.log(np.var(x) + 1e-10)
                elif self.feature == 'rms':
                    feats[i, j] = np.sqrt(np.mean(x**2))
                else:
                    feats[i, j] = np.mean(x)
        return feats
def main():

    subjects = [i for i in range(int(sys.argv[1]), int(sys.argv[2]))]
    runs = [6, 10, 14]
    data_loader = LoadEEGBCI(runs, subjects)
    epochs = data_loader.load_and_process_data()

    # Use raw epochs as pipeline input
    X_raw, y = data_loader.transform_data(epochs)

    # Build processing pipeline: CSP -> FeatureExtractor -> Scaler -> Classifier
    pipeline = Pipeline([
        ('csp', MyCSP(n_components=8)),
        ('fe', FeatureExtractor(feature='logvar')),
        ('scaler', StandardScaler()),
        ('clf', xgb.XGBClassifier(eval_metric='logloss',
                                  n_estimators=1000, max_depth=8, random_state=42, learning_rate=0.1))
    ])

    if len(sys.argv) < 4:
        print('Usage: python mybci.py <subj_start> <subj_end> <train|predict>')
        return

    mode = sys.argv[3].lower()

    if mode == 'train':
        # create an explicit holdout so saved model is not trained on everything
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X_raw, y, test_size=0.2, random_state=42, stratify=y)
        cv = StratifiedKFold(n_splits=int(sys.argv[2]) - int(sys.argv[1]))
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        print(scores)
        print(f"cross_val_score: {scores.mean()}")
        # fit on train partition and save model + holdout
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, 'pipeline_xgb.joblib')
        joblib.dump((X_holdout, y_holdout), 'holdout.joblib')
        print('Saved trained pipeline to pipeline_xgb.joblib and holdout.joblib')
    elif mode == 'predict':
        # load if exists, otherwise train on a train split
        try:
            pipeline = joblib.load('pipeline_xgb.joblib')
            print('Loaded pipeline_xgb.joblib')
        except Exception:
            print('No saved model found, training a fresh pipeline on a train split')
            X_train, X_holdout, y_train, y_holdout = train_test_split(
                X_raw, y, test_size=0.2, random_state=42, stratify=y)
            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, 'pipeline_xgb.joblib')
            joblib.dump((X_holdout, y_holdout), 'holdout.joblib')

        # Prefer evaluating on saved holdout if present
        try:
            X_test, y_test = joblib.load('holdout.joblib')
            print('Loaded holdout set from holdout.joblib')
        except Exception:
            # fallback: create a temporary split (results may be optimistic)
            print('No holdout found, creating a fresh train/test split (results may be optimistic)')
            X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)

        predictions = pipeline.predict(X_test)
        for i, (pred, true) in enumerate(zip(predictions, y_test)):
            equal = 'True' if pred == true else 'False'
            print(f'epoch {i:02d}: [{pred}] [{true}] {equal}')

        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {accuracy:.4f}')
    else:
         print('Unknown mode: use train or predict')


if __name__ == "__main__":
     
     main()
