# --- START OF FILE NN3_integrated.py ---

import pandas as pd
import numpy as np
from pathinator import train_path # Assuming pathinator provides train_path
from tqdm import tqdm
import math

# --- ML/NN Imports ---
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder # Keep LabelEncoder
# Removed OrdinalEncoder as we don't know the categorical features
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping # Keep EarlyStopping

# --- Keyword Lists ---
EXCLUSIVE_KEYWORDS = [
    'leipzig', 'canton', 'municipalities', 'municipality', 'ministers', 'province',
    'polish', 'council', 'parliament', 'mi', 'swiss', 'county', 'located'
]
REPRESENTATIVE_KEYWORDS = [
    'hbo', 'emmy', 'disney', 'nominations', 'manga', 'globe', 'album',
    'greatest', 'albums', 'golden', 'islamic', 'career'
]
AGNOSTIC_KEYWORDS = [
    'aquatic', 'organisms', 'organism', 'cement', 'climbing', 'genus', 'plants',
    'surface', 'species', 'fish', 'systems', 'trees', 'plant', 'materials'
]

# --- Feature Extraction ---
def count_keywords(text, keywords):
    count = 0
    if not isinstance(text, str): return 0
    text_lower = text.lower()
    for keyword in keywords:
        if keyword.lower() in text_lower: count += 1
    return count

def extract_features_from_csv(row):
    summary = str(row.get('long_description', ""))
    lang_count = int(row.get('num_languages', 0)) if not pd.isna(row.get('num_languages', 0)) else 0
    desc_len = int(row.get('description_length', 0)) if not pd.isna(row.get('description_length', 0)) else 0
    num_geo = int(row.get('num_geoproperties', 0)) if not pd.isna(row.get('num_geoproperties', 0)) else 0
    category = str(row.get('category', ""))
    subcategory = str(row.get('subcategory', ""))
    summary = "" if pd.isna(summary) else summary
    summary_lower = summary.lower()
    kw_ex = count_keywords(summary_lower, EXCLUSIVE_KEYWORDS)
    kw_rp = count_keywords(summary_lower, REPRESENTATIVE_KEYWORDS)
    kw_ag = count_keywords(summary_lower, AGNOSTIC_KEYWORDS)
    return {
        'description_length': desc_len, 'num_languages': lang_count,
        'num_geoproperties': num_geo, 'exclusive_keywords': kw_ex,
        'representative_keywords': kw_rp, 'agnostic_keywords': kw_ag
    }

# --- Main Script ---

# 1. Load Data
try:
    df = pd.read_csv(train_path)
    print(f"Loaded {len(df)} rows from {train_path}")
except FileNotFoundError:
    print(f"Error: Training data file not found at {train_path}"); exit()
except Exception as e:
    print(f"Error reading training data CSV: {e}"); exit()

# essential columns
essential_cols = ['long_description', 'num_languages', 'description_length', 'num_geoproperties', 'category', 'subcategory', 'label']

missing_cols = [col for col in essential_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Missing essential columns: {', '.join(missing_cols)}"); exit()

df.dropna(subset=['label'], inplace=True)
print(f"Proceeding with {len(df)} rows after removing potential missing labels.")

# 2. Feature Engineering (Numerical + Keywords only for now)
print("Extracting features...")
features_list = [extract_features_from_csv(row) for index, row in tqdm(df.iterrows(), total=df.shape[0])]
features_df = pd.DataFrame(features_list, index=df.index)

# Define numerical and keyword features (Adapt if needed)
feature_names = ['description_length', 'num_languages', 'num_geoproperties',
                 'exclusive_keywords', 'representative_keywords', 'agnostic_keywords']
X = features_df[feature_names].values
y_raw = df['label'].values # Raw text labels
X = np.nan_to_num(X.astype(float), nan=0.0) # Impute NaNs in features

# 3. Data Preprocessing
print("Preprocessing data...")
# Scale numerical/keyword features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels to integers (0, 1, 2...) -> Required for sparse_categorical_crossentropy
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw) # <--- Use these integer labels for training/validation
num_classes = len(label_encoder.classes_)
print(f"Found classes: {label_encoder.classes_}")
print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Split data - Use integer encoded labels for y
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled,
    y_encoded, # <--- Use integer labels
    test_size=0.2,
    random_state=42,
    stratify=y_encoded # Stratify based on integer labels
)

print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")

# NOTE: No class weights used here, matching the model.py structure observed

# 4. Build the Neural Network Model 
print("Building model (architecture based on model.py)...")
num_features = X_train.shape[1]
model = keras.Sequential(
    [
        keras.layers.Input(shape=(num_features,), name="Input_Layer"),
        keras.layers.Dense(32, activation="relu", name="Hidden_Layer_1"),
        keras.layers.Dropout(0.2), 
        keras.layers.Dense(64, activation="relu", name="Hidden_Layer_2"),
        keras.layers.Dropout(0.1), 
        keras.layers.Dense(96, activation="relu", name="Hidden_Layer_3"),
        keras.layers.Dropout(0.4), 
        keras.layers.Dense(num_classes, activation="softmax", name="Output_Layer"), # Softmax still needed for multi-class probabilities
    ],
    name="Classificatore_Wikidata_Integrated"
)
model.summary()

# 5. Compile the Model
print("Compiling model...")
optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(
    optimizer=optimizer_adam,
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

# 6. Define Callbacks 
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=50, 
    min_lr=1e-6,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=200, 
    restore_best_weights=True
)

# 7. Train the Model
print("Training model...")
batch_size = 64
epochs = 1000 

history = model.fit(
    X_train,
    y_train, # Pass integer labels
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val), # Pass integer labels
    callbacks=[early_stopping, reduce_lr], 
    verbose=1
)
print("Training finished.")

# 8. Evaluate the Model
print("\nEvaluating model on validation set...")
loss, accuracy = model.evaluate(X_val, y_val, verbose=0) # Use integer labels for evaluation
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# --- Make Predictions and Display Report/Matrix ---
y_pred_probs = model.predict(X_val)
y_pred_encoded = np.argmax(y_pred_probs, axis=1) # Get integer class predictions

# Convert predictions and true validation labels back to original text labels for report
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
y_val_labels = label_encoder.inverse_transform(y_val) # y_val is already integer encoded

print("\nClassification Report:")
print(classification_report(y_val_labels, y_pred_labels, labels=label_encoder.classes_, zero_division=0))
print("Confusion Matrix:")
cm = confusion_matrix(y_val_labels, y_pred_labels, labels=label_encoder.classes_)
cm_df = pd.DataFrame(cm, index=[f'True {l.split()[-1].title()[:4]}' for l in label_encoder.classes_],
                       columns=[f'Pred {l.split()[-1].title()[:4]}' for l in label_encoder.classes_])
print(cm_df)

# --- Optional: Plotting ---
def plot_history(history):
    hist = history.history
    epochs_range = range(1, len(hist['loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, hist['loss'], label='Training Loss')
    plt.plot(epochs_range, hist['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training and Validation Loss')
    plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, hist['accuracy'], label='Training Accuracy')
    plt.plot(epochs_range, hist['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.title('Training and Validation Accuracy')
    plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

print("\n--- Plotting Learning Curves ---")
plot_history(history)

# --- Optional: Save Model (using path from model.py if available, or define) ---
model_path = "integrated_model.h5" # Define a path
model.save(model_path)
print(f"\nModel saved to: {model_path}")
