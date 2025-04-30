import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import pandas as pd
import numpy as np
from tensorflow import keras
import joblib

from pathinator import *
from data_manipulator import *
from data_augmentator import *

numerical_features = ["num_languages", 'num_geoproperties', "num_cultural_properties"]
categorical_features = ['type', 'category', 'subcategory']

model = keras.models.load_model(model_path)

categorical_encoder = joblib.load(categorical_encoder_path)
features = joblib.load(features_list_path)
label_encoder = joblib.load(label_encoder_path)


X_test_original = test_data[features].copy()
X_test = test_data[features].copy()

for col in numerical_features:
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype(int)

X_test = categorical_encoder.transform(X_test)

predictions_probabilities = model.predict(X_test[:])
predicted_classes_encoded = np.argmax(predictions_probabilities, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes_encoded)
test_names = test_data['name']

for i in range(5):
    print(f"  Istanza {i+1}:")
    print(f"   Nome: {test_names[i]}")
    print(f"    Probabilità predette: {predictions_probabilities[i]}")
    print(f"    Label predetto: '{predicted_labels[i]}'")
    print("-" * 20)

'''test_original_data = pd.read_csv(test_original_path)
test_original_data["label"] = predicted_labels

test_labeled_path = dataset_path / "test_labeled.csv"
test_original_data.to_csv(test_labeled_path, index=False, encoding='utf-8')'''
