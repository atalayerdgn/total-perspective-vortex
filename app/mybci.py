import sys
import numpy as np
import mne
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from features.CSP import CSP
from features.ANOVA_F import ANOVA_F
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
import joblib
import os
from pathlib import Path
from sklearn.metrics import accuracy_score

def load_epochs(file_path):
    return mne.read_epochs(file_path, preload=True, verbose=False)

def create_pipeline():
    return Pipeline([
        ('ANOVA_F', ANOVA_F(k=10)),
        ('CSP', CSP(n_components=6)),
        ('Covariances', Covariances(estimator='lwf')),
        ('TangentSpace', TangentSpace(metric='riemann')),
        ('SVC', SVC(kernel='linear', C=0.1, probability=True)),
    ])

def get_data_from_runs(runs):
    epochs_data = []
    labels = []
    
    epochs_dir = Path("../data/epochs/train")
    if not epochs_dir.exists():
        return None, None
    
    for run in runs:
        file_pattern = f"*S{run:03d}-epo.fif"
        epoch_files = list(epochs_dir.glob(file_pattern))
        
        if not epoch_files:
            continue
        
        for file_path in epoch_files:
                epochs = load_epochs(file_path)
                X = epochs.get_data(copy=True)
                y = epochs.events[:, 2]
                epochs_data.append(X)
                labels.append(y)
    
    if epochs_data:
        X_combined = np.concatenate(epochs_data, axis=0)
        y_combined = np.concatenate(labels, axis=0)
        return X_combined, y_combined
    return None, None

def train_mode(runs):
    train_dir = Path("../data/train")
    if not train_dir.exists():
        return
    
    model_path = Path("../models/model.pkl")
    if model_path.exists():
        clf = joblib.load(model_path)
    else:
        clf = create_pipeline()
        os.makedirs("../models", exist_ok=True)
    
    for run in runs:
        file_pattern = f"S{run:03d}-epo.fif"
        epoch_files = list(train_dir.glob(file_pattern))
        
        if not epoch_files:
            continue
        
        for file_path in epoch_files:
                epochs = load_epochs(file_path)
                X = epochs.get_data(copy=True)
                y = epochs.events[:, 2]
                
                clf.fit(X, y)
                print(f"S{run:03d} accuracy: {clf.score(X, y):.4f}")
                joblib.dump(clf, model_path)                


def predict_mode(runs):
    model_path = Path("../models/model.pkl")
    if not model_path.exists():
        return
    
    test_dir = Path("../data/test")
    if not test_dir.exists():
        return
    clf = joblib.load(model_path)
    
    for run in runs:
        file_pattern = f"S{run:03d}-epo.fif"
        epoch_files = list(test_dir.glob(file_pattern))
        
        if not epoch_files:
            continue
        
        for file_path in epoch_files:
                epochs = load_epochs(file_path)
                X = epochs.get_data(copy=True)
                y = epochs.events[:, 2]
                
                predictions = clf.predict(X)
                
                print("epoch nb: [prediction] [truth] equal?")
                file_correct = 0
                for i, (pred, true) in enumerate(zip(predictions, y)):
                    is_correct = pred == true
                    if is_correct:
                        file_correct += 1
                    print(f"epoch {i:02d}: [{pred}] [{true}] {is_correct}")
                
                file_accuracy = file_correct / len(y)
                print(f"Bu dosya için accuracy: {file_accuracy:.4f}")
                
        overall_accuracy = file_correct / len(y)
        print(f"Genel accuracy: {overall_accuracy:.4f}")
    
def main():
    if len(sys.argv) == 4:
        run1, run2, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
        runs = range(run1, run2)
        
        if mode == "train":
            train_mode(runs)
        elif mode == "predict":
            predict_mode(runs)

if __name__ == "__main__":
    main()
