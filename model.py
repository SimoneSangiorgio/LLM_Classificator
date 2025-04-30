from pathinator import *
from data_manipulator import *
from data_augmentator import keywords_countator

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt

import joblib

features = categorical_features + numerical_features #+ keyword_features  

label_encoder = LabelEncoder()

X = train_data[features].copy()
X_encoded = X.copy() # input
y = labels.copy() 
y_encoded = label_encoder.fit_transform(y) # output

X_validation = validation_data[features].copy()
X_validation_encoded = X_validation.copy()
y_validation = validation_labels.copy()
y_validation_encoded = label_encoder.transform(y_validation)

for col in numerical_features:
    X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0).astype(int)
    X_validation_encoded[col] = pd.to_numeric(X_validation_encoded[col], errors='coerce').fillna(0).astype(int)

categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
preprocessor = ColumnTransformer([
    ('cat', categorical_encoder, categorical_features), 
    ('num', 'passthrough', numerical_features)])
pipeline = Pipeline([('encode', preprocessor)])

X_encoded = pipeline.fit_transform(X_encoded)
X_validation_encoded = pipeline.transform(X_validation_encoded)

X_train, X_test, y_train, y_test = X_encoded, X_validation_encoded, y_encoded, y_validation_encoded
num_classes = len(label_encoder.classes_)
num_features = X_train.shape[1]

model = keras.Sequential(
    [   layers.Input(shape=(num_features,)), 
     
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.05),

        layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-5)),

        layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        layers.Dropout(0.05),

        layers.Dense(num_classes, activation="softmax", name="Output_Layer"),])

optimizer_adam = Adam(learning_rate=0.00008, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(optimizer=optimizer_adam,loss="sparse_categorical_crossentropy",metrics=["accuracy"],)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,         
    patience=20,        
    min_lr=1e-6,        
    verbose=1           
)

batch_size = 256
epochs = 1000

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=250, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1)

# --- VALUTATION ---
print("\nValutazione del modello sul set di test...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nRisultati sul set di test:")
print(f"  Loss: {loss:.4f}")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# --- PLOTTING LEARNING CURVES ---

def plot_history(history):
    
    hist = history.history
    epochs_range = range(1, len(hist['loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, hist['loss'], label='Training Loss')
    plt.plot(epochs_range, hist['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, hist['accuracy'], label='Training Accuracy')
    plt.plot(epochs_range, hist['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

print("\n--- Plotting Curve di Apprendimento ---")
plot_history(history)

# Saving
joblib.dump(label_encoder, label_encoder_path)
joblib.dump(pipeline, categorical_encoder_path)
joblib.dump(features, features_list_path)
model.save(model_path)


