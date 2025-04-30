import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathinator import *
from data_manipulator import *
from data_augmentator import *

model = keras.models.load_model(model_path)

label_encoder = joblib.load(label_encoder_path)
categorical_encoder = joblib.load(categorical_encoder_path)
features = joblib.load(features_list_path)

validation_data['exclusive_categories_count'] = long_descriptions.apply(lambda text: keywords_countator(text, exclusive_keywords))
validation_data['representative_categories_count'] = long_descriptions.apply(lambda text: keywords_countator(text, representative_keywords))
validation_data['agnostic_categories_count'] = long_descriptions.apply(lambda text: keywords_countator(text, agnostic_keywords))

X_test_original = validation_data[features].copy()
X_test = validation_data[features].copy()
y_test_original = validation_data["label"].copy()
y_test = validation_data["label"].copy()

for col in numerical_features:
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype(int)

X_test = categorical_encoder.transform(X_test)

y_test = label_encoder.transform(y_test)

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"\nLoss sul set di test: {loss:.4f}")
print(f"Accuracy sul set di test: {accuracy:.4f} ({accuracy*100:.2f}%)")

predictions_probabilities = model.predict(X_test)
y_pred_encoded = np.argmax(predictions_probabilities, axis=1)

# Precision, Recall, F1-score
print("\n--- Report di Classificazione ---")
class_names = label_encoder.classes_ 
report = classification_report(y_test, y_pred_encoded, target_names=class_names)
print(report)

# Matrice di Confusione
print("\n--- Matrice di Confusione ---")
conf_matrix = confusion_matrix(y_test, y_pred_encoded)

plt.figure(figsize=(12, 9))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 8})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Matrice di Confusione')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ------
print("\nEsempio di predizione sulle prime 5 istanze del set di test:")
predictions_probabilities = model.predict(X_test[:5])
predicted_classes_encoded = np.argmax(predictions_probabilities, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes_encoded)
actual_labels = label_encoder.inverse_transform(y_test[:5])

for i in range(5):
    print(f"  Istanza {i+1}:")
    print(f"    Probabilità predette: {predictions_probabilities[i]}")
    print(f"    Label predetto: '{predicted_labels[i]}'")
    print(f"    Label reale: '{actual_labels[i]}'")
    print("-" * 20)
