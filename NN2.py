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
# Import OneHotEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

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

    # Ensure category and subcategory are strings, handle potential NaNs
    category = str(row.get('category', "Unknown")) if not pd.isna(row.get('category')) else "Unknown"
    subcategory = str(row.get('subcategory', "Unknown")) if not pd.isna(row.get('subcategory')) else "Unknown"
    summary = "" if pd.isna(summary) else summary
    summary_lower = summary.lower()
    kw_ex = count_keywords(summary_lower, EXCLUSIVE_KEYWORDS)
    kw_rp = count_keywords(summary_lower, REPRESENTATIVE_KEYWORDS)
    kw_ag = count_keywords(summary_lower, AGNOSTIC_KEYWORDS)

    return {
        'description_length': desc_len, 'num_languages': lang_count,
        'num_geoproperties': num_geo, 'exclusive_keywords': kw_ex,
        'representative_keywords': kw_rp, 'agnostic_keywords': kw_ag,
        'category': category, # Add category
        'subcategory': subcategory # Add subcategory
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

# Handle potential not a number in label column *before* feature extraction if needed
df.dropna(subset=['label'], inplace=True)
print(f"Proceeding with {len(df)} rows after removing potential missing labels.")

# Fill not a number in category/subcategory *before* extraction (alternative: handle in extract_features_from_csv)
df['category'] = df['category'].fillna('Unknown')
df['subcategory'] = df['subcategory'].fillna('Unknown')

# 2. Feature Engineering (Numerical + Keywords + Categorical)
print("Extracting features...")
# Extract all features including category/subcategory strings
features_list = [extract_features_from_csv(row) for index, row in tqdm(df.iterrows(), total=df.shape[0])]
features_df = pd.DataFrame(features_list, index=df.index)

# Define numerical/keyword and categorical feature names
numerical_keyword_features = ['description_length', 'num_languages', 'num_geoproperties',
                              'exclusive_keywords', 'representative_keywords', 'agnostic_keywords']
categorical_features = ['category', 'subcategory']

X_num_kw = features_df[numerical_keyword_features].values.astype(float)
X_cat = features_df[categorical_features].values

# Impute NaNs in numerical/keyword features (should ideally be handled earlier or more carefully)
X_num_kw = np.nan_to_num(X_num_kw, nan=0.0)

y_raw = df['label'].values # Raw text labels

# 3. Data Preprocessing
print("Preprocessing data...")

# Scale numerical/keyword features
scaler = StandardScaler()
X_num_kw_scaled = scaler.fit_transform(X_num_kw)
print(f"Scaled numerical/keyword features shape: {X_num_kw_scaled.shape}")

# Encode categorical features using OneHotEncoder
# handle_unknown='ignore' will encode unknown categories (seen in validation/test but not train) as all zeros
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat_encoded = ohe.fit_transform(X_cat)
print(f"One-Hot Encoded categorical features shape: {X_cat_encoded.shape}")
print(f"Number of categories found by OHE: {len(ohe.categories_[0])} (category), {len(ohe.categories_[1])} (subcategory)")
# Optional: print categories found
# print("Categories found:", ohe.categories_)

# Combine scaled numerical/keyword and one-hot encoded categorical features
X_combined = np.hstack((X_num_kw_scaled, X_cat_encoded))
print(f"Combined features shape: {X_combined.shape}")

# Encode labels to integers (0, 1, 2...) -> Required for sparse_categorical_crossentropy
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw) # Use these integer labels for training/validation
num_classes = len(label_encoder.classes_)
print(f"Found classes: {label_encoder.classes_}")
print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Split data - Use the combined feature matrix and integer encoded labels for y
X_train, X_val, y_train, y_val = train_test_split(
    X_combined, # Use the combined features
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded # Stratify based on integer labels
)

print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")

# 4. Build the Neural Network Model
print("Building model...")
# --- UPDATE num_features ---
num_features = X_train.shape[1] # Get the total number of features after OHE
print(f"Model input layer expecting {num_features} features.")

model = keras.Sequential(
    [
        # --- UPDATE Input shape ---
        keras.layers.Input(shape=(num_features,), name="Input_Layer"),
        keras.layers.Dense(16, activation="relu", name="Hidden_Layer_1"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu", name="Hidden_Layer_2"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu", name="Hidden_Layer_3"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(num_classes, activation="softmax", name="Output_Layer"),
    ],
    name="Classificatore_Wikidata_Integrated_Categorical" # Updated name
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
    patience=200,
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
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
print("Training finished.")

# 8. Evaluate the Model
print("\nEvaluating model on validation set...")
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# --- Make Predictions and Display Report/Matrix ---
y_pred_probs = model.predict(X_val)
y_pred_encoded = np.argmax(y_pred_probs, axis=1) # Get integer class predictions

# Convert predictions and true validation labels back to original text labels for report
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
y_val_labels = label_encoder.inverse_transform(y_val)

print("\nClassification Report:")
# Ensure labels parameter includes all potential classes found during label encoding
print(classification_report(y_val_labels, y_pred_labels, labels=label_encoder.classes_, zero_division=0))
print("Confusion Matrix:")
cm = confusion_matrix(y_val_labels, y_pred_labels, labels=label_encoder.classes_)
# Make column/index names robust if classes are long
class_abbr = [l.split()[-1][:4].title() if ' ' in l else l[:4].title() for l in label_encoder.classes_]
cm_df = pd.DataFrame(cm, index=[f'True {abbr}' for abbr in class_abbr],
                       columns=[f'Pred {abbr}' for abbr in class_abbr])
print(cm_df)

# --- Optional: Plotting ---
def plot_history(history):
    hist = history.history
    # Handle cases where training stops very early
    if not hist or 'loss' not in hist or not hist['loss']:
        print("No history data to plot.")
        return
    epochs_range = range(1, len(hist['loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, hist['loss'], label='Training Loss')
    if 'val_loss' in hist:
        plt.plot(epochs_range, hist['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training and Validation Loss')
    plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, hist['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in hist:
        plt.plot(epochs_range, hist['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.title('Training and Validation Accuracy')
    plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

print("\n--- Plotting Learning Curves ---")
plot_history(history)

# --- Optional: Save Model ---
model_path = "model.h5" # Updated path
model.save(model_path)
print(f"\nModel saved to: {model_path}")