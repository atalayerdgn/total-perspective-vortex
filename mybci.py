from sklearn.model_selection import train_test_split
import mne
import numpy as np
from sklearn.pipeline import Pipeline
from features.CSP import CSP
from features.ANOVA_F import ANOVA_F
import joblib 
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
import os
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

def load_epochs(file_path):
    """Load epochs from .fif file"""
    return mne.read_epochs(file_path, preload=True)

def split_data(epochs, test_size=0.2, random_state=42):
    X = epochs.get_data(copy=True) 
    y = epochs.events[:, 2]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def create_pipeline() -> Pipeline:
    """
    Pipeline sırası:
    1. ANOVA_F - En uygun kanalları seç (tüm kanallar arasından)
    2. CSP - Seçilen kanallar üzerinde Common Spatial Pattern uygula
    3. Riemann - Covariance ve TangentSpace
    4. SVC - Support Vector Classifier
    """
    return Pipeline([
        ('ANOVA_F', ANOVA_F(k=10)),  # En uygun 10 kanalı seç
        ('CSP', CSP(n_components=8)),  # Seçilen kanallar üzerinde CSP (8 component)
        ('Covariances', Covariances(estimator='lwf')),
        ('TangentSpace', TangentSpace(metric='riemann')),
        ('SVC', SVC(kernel='linear', C=0.15, probability=True)),
    ])

def train_with_channel_info(X_train, y_train, channel_names):
    """
    Train pipeline with detailed channel selection information
    """
    print("\n=== Channel Selection Process ===")
    
    # Step 1: Apply ANOVA_F to select best channels
    print("1. Applying ANOVA_F for channel selection...")
    anova_f = ANOVA_F(k=10)
    X_selected = anova_f.fit_transform(X_train, y_train)
    selected_channels = anova_f.selected_channels_
    
    print(f"   Original channels: {X_train.shape[1]}")
    print(f"   Selected channels: {len(selected_channels)}")
    print(f"   Selected channel indices: {selected_channels}")
    
    # Show selected channel names
    if channel_names:
        selected_channel_names = [channel_names[i] for i in selected_channels]
        print(f"   Selected channel names: {selected_channel_names}")
    
    # Step 2: Apply CSP on selected channels
    print("\n2. Applying CSP on selected channels...")
    csp = CSP(n_components=8)
    X_csp = csp.fit_transform(X_selected, y_train)
    
    print(f"   CSP output shape: {X_csp.shape}")
    
    # Step 3: Apply Riemann
    print("\n3. Applying Riemann covariance and tangent space...")
    cov = Covariances(estimator='lwf')
    X_cov = cov.fit_transform(X_csp)
    
    tangent = TangentSpace(metric='riemann')
    X_tangent = tangent.fit_transform(X_cov)
    
    print(f"   Tangent space shape: {X_tangent.shape}")
    
    # Step 4: Train SVC
    print("\n4. Training Support Vector Classifier...")
    svc = SVC(kernel='linear', C=0.15, probability=True)
    svc.fit(X_tangent, y_train)
    
    # Create the full pipeline for saving
    full_pipeline = Pipeline([
        ('ANOVA_F', anova_f),
        ('CSP', csp),
        ('Covariances', cov),
        ('TangentSpace', tangent),
        ('SVC', svc),
    ])
    
    return full_pipeline, selected_channels

def load_all_files_from_directory(directory_path):
    """Load all .fif files from a directory and return combined data"""
    train_data = []
    train_labels = []
    channel_names = None
    
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Directory not found: {directory_path}")
        return None, None, None
    
    fif_files = list(directory.glob("*.fif"))
    if not fif_files:
        print(f"No .fif files found in {directory_path}")
        return None, None, None
    
    print(f"Loading {len(fif_files)} files from {directory_path}:")
    
    for fif_path in fif_files:
        try:
            print(f"  Loading: {fif_path.name}")
            epochs = mne.read_epochs(fif_path, preload=True)
            X = epochs.get_data(copy=True)
            y = epochs.events[:, 2]
            
            # Get channel names from first file
            if channel_names is None:
                channel_names = epochs.info['ch_names']
            
            train_data.append(X)
            train_labels.append(y)
            
            print(f"    Shape: {X.shape}, Labels: {y.shape}")
            
        except Exception as e:
            print(f"    Error loading {fif_path.name}: {e}")
            continue
    if not train_data:
        print("No data loaded!")
        return None, None, None
    X_combined = np.concatenate(train_data, axis=0)
    y_combined = np.concatenate(train_labels, axis=0)
    
    print(f"Combined data shape: {X_combined.shape}")
    print(f"Combined labels shape: {y_combined.shape}")
    print(f"Channel names: {channel_names}")
    
    return X_combined, y_combined, channel_names

def main():
    train_dir = "data/epochs/train"
    test_dir = "data/epochs/test"
    
    print("=== BCI Training and Testing System ===")
    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    print()
    
    choice = input("Train or Test? (t/T for train, any other key for test): ")
    
    if choice.lower() == 't':
        print("\n=== TRAINING MODE ===")
        print("Loading training data...")
        X_train, y_train, channel_names = load_all_files_from_directory(train_dir)
        
        if X_train is None:
            print("No training data available!")
            return
        
        print(f"\nTraining data loaded successfully!")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Features per sample: {X_train.shape[1]} x {X_train.shape[2]}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        print(f"Class distribution: {np.bincount(y_train)}")
        print("\nCreating pipeline:")
        print("1. ANOVA_F - En uygun kanalları seç")
        print("2. CSP - Seçilen kanallar üzerinde Common Spatial Pattern")
        print("3. Riemann - Covariance estimation and tangent space")
        print("4. SVC - Support Vector Classifier")
        
        # Train with detailed channel information
        clf_pipeline, selected_channels = train_with_channel_info(X_train, y_train, channel_names)
        
        os.makedirs("models", exist_ok=True)
        
        # Save model
        model_path = "models/csp_anova_f_riemann_svc_model.pkl"
        joblib.dump(clf_pipeline, model_path)
        print(f"Model saved as: {model_path}")
        
        # Training accuracy
        train_score = clf_pipeline.score(X_train, y_train)
        print(f"Training Accuracy: {train_score:.4f}")
        
        # Detailed training results
        y_train_pred = clf_pipeline.predict(X_train)
        print("\nTraining Classification Report:")
        print(classification_report(y_train, y_train_pred))
        
    else:
        print("\n=== TESTING MODE ===")
        
        # Load test data
        print("Loading test data...")
        X_test, y_test, _ = load_all_files_from_directory(test_dir)
        
        if X_test is None:
            print("No test data available!")
            return
        
        print(f"\nTest data loaded successfully!")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Features per sample: {X_test.shape[1]} x {X_test.shape[2]}")
        print(f"Number of classes: {len(np.unique(y_test))}")
        print(f"Class distribution: {np.bincount(y_test)}")
        
        # Load and test model
        model_path = "models/csp_anova_f_riemann_svc_model.pkl"
        try:
            clf_pipeline = joblib.load(model_path)
            print(f"Model loaded from: {model_path}")
            
            # Test accuracy
            test_score = clf_pipeline.score(X_test, y_test)
            print(f"Test Accuracy: {test_score:.4f}")
            
            # Detailed test results
            y_test_pred = clf_pipeline.predict(X_test)
            print("\nTest Classification Report:")
            print(classification_report(y_test, y_test_pred))
            
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_test_pred))
            
        except FileNotFoundError:
            print(f"Model file not found: {model_path}")
            print("Please train the model first using 't' option.")

if __name__ == "__main__":
    main()
