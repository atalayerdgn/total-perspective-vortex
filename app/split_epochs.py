import os
import shutil
import random
from pathlib import Path

def split_epochs_files(epochs_dir="data/epochs", train_ratio=0.8, random_seed=42):
    """
    Split epoch files into train and test sets.
    
    Args:
        epochs_dir (str): Path to the epochs directory
        train_ratio (float): Ratio of files to use for training (0.0 to 1.0)
        random_seed (int): Random seed for reproducibility
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Get all epoch files
    epochs_path = Path(epochs_dir)
    epoch_files = [f for f in epochs_path.glob("*.fif")]
    
    if not epoch_files:
        print(f"No .fif files found in {epochs_dir}")
        return
    
    print(f"Found {len(epoch_files)} epoch files:")
    for f in epoch_files:
        print(f"  - {f.name}")
    
    # Shuffle files for random split
    random.shuffle(epoch_files)
    
    # Calculate split point
    split_idx = int(len(epoch_files) * train_ratio)
    
    # Split files
    train_files = epoch_files[:split_idx]
    test_files = epoch_files[split_idx:]
    
    print(f"\nSplit results:")
    print(f"Training files ({len(train_files)}):")
    for f in train_files:
        print(f"  - {f.name}")
    
    print(f"\nTest files ({len(test_files)}):")
    for f in test_files:
        print(f"  - {f.name}")
    
    # Create train and test directories
    train_dir = epochs_path / "train"
    test_dir = epochs_path / "test"
    
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Move files to appropriate directories
    print(f"\nMoving files...")
    
    for f in train_files:
        dest = train_dir / f.name
        shutil.copy2(f, dest)
        print(f"Copied {f.name} to train/")
    
    for f in test_files:
        dest = test_dir / f.name
        shutil.copy2(f, dest)
        print(f"Copied {f.name} to test/")
    
    print(f"\nSplit completed successfully!")
    print(f"Training files: {len(train_files)} in {train_dir}")
    print(f"Test files: {len(test_files)} in {test_dir}")

if __name__ == "__main__":
    split_epochs_files() 
