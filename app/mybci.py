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
        ('CSP', CSP(n_components=8)),
        ('Covariances', Covariances(estimator='lwf')),
        ('TangentSpace', TangentSpace(metric='riemann')),
        ('SVC', SVC(kernel='linear', C=0.15, probability=True)),
    ])

def get_data_from_runs(runs):
    epochs_data = []
    labels = []
    
    for run in runs:
        file_pattern = f"*R{run:02d}*-epo.fif"
        epoch_files = list(Path("data/epochs").glob(file_pattern))
        
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
    X, y = get_data_from_runs(runs)
    if X is None:
        return
    
    clf = create_pipeline()
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    print(f"[{' '.join([f'{score:.4f}' for score in scores])}]")
    print(f"cross_val_score: {scores.mean():.4f}")
    
    clf.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/model.pkl")

def predict_mode(runs):
    if not os.path.exists("models/model.pkl"):
        return
    
    clf = joblib.load("models/model.pkl")
    X, y = get_data_from_runs(runs)
    if X is None:
        return
    
    predictions = clf.predict(X)
    
    print("epoch nb: [prediction] [truth] equal?")
    correct = 0
    for i, (pred, true) in enumerate(zip(predictions, y)):
        is_correct = pred == true
        if is_correct:
            correct += 1
        print(f"epoch {i:02d}: [{pred}] [{true}] {is_correct}")
    
    accuracy = correct / len(y)
    print(f"Accuracy: {accuracy:.4f}")

def run_all_experiments():
    experiment_configs = [
        [3, 7, 11],  # left fist vs right fist (real)
        [4, 8, 12],  # left fist vs right fist (imagery)
        [5, 9, 13],  # both fists vs both feet (real)
        [6, 10, 14], # both fists vs both feet (imagery)
        [3, 4],      # left vs right (mixed)
        [5, 6]       # fists vs feet (mixed)
    ]
    
    all_accuracies = []
    
    for exp_idx, runs in enumerate(experiment_configs):
        exp_accuracies = []
        
        subject_files = sorted(Path("data/epochs").glob("S*-epo.fif"))
        
        for subject_file in subject_files:
            subject_num = subject_file.name.split('S')[1][:3]
            
            try:
                epochs = load_epochs(subject_file)
                
                run_mask = np.isin(epochs.metadata['run'] if epochs.metadata is not None else [4], runs)
                if not run_mask.any():
                    continue
                
                X = epochs.get_data(copy=True)
                y = epochs.events[:, 2]
                
                if len(np.unique(y)) < 2:
                    continue
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                clf = create_pipeline()
                clf.fit(X_train, y_train)
                
                predictions = clf.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                
                exp_accuracies.append(accuracy)
                print(f"experiment {exp_idx}: subject {subject_num}: accuracy = {accuracy:.1f}")
                
            except:
                continue
        
        if exp_accuracies:
            mean_acc = np.mean(exp_accuracies)
            all_accuracies.append(mean_acc)
            print(f"experiment {exp_idx}: accuracy = {mean_acc:.4f}")
        else:
            all_accuracies.append(0.0)
            print(f"experiment {exp_idx}: accuracy = 0.0000")
    
    if all_accuracies:
        overall_mean = np.mean(all_accuracies)
        print(f"Mean accuracy of 6 experiments: {overall_mean:.4f}")

def main():
    if len(sys.argv) == 4:
        run1, run2, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
        runs = [run1, run2]
        
        if mode == "train":
            train_mode(runs)
        elif mode == "predict":
            predict_mode(runs)
    else:
        run_all_experiments()

if __name__ == "__main__":
    main()
