
from pathlib import Path

# Paths
base_path = Path(__file__).resolve().parent 

dataset_path = base_path / "dataset"

train_path = dataset_path / "train_augmented.csv"

validation_path = dataset_path / "valid_augmented.csv"

