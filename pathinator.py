
from pathlib import Path

# -----------

base_path = Path(__file__).resolve().parent

# -----------

model_path = base_path / "model.keras"

# -----------

dataset_path = base_path / "dataset"

test_path = dataset_path / "test_unlabeled_augmented.csv"
train_path = dataset_path / "train_augmented_plus.csv"
validation_path = dataset_path / "valid_augmented_plus.csv"

test_original_path = dataset_path / "test_unlabeled.csv"
train_original_path = dataset_path / "train.csv"
validation_original_path = dataset_path / "valid.csv"

# -----------

encoders_path = base_path / "encoders"

label_encoder_path = encoders_path / "label_encoder.joblib"
categorical_encoder_path = encoders_path / "categorical_encoder.joblib"
features_list_path = encoders_path / "features_list.joblib"


