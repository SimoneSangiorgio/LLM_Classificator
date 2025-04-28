import pandas as pd
# Removed: import requests
# Removed: import wikipediaapi
# Removed: import random # Not used anymore
from pathinator import train_path
from tqdm import tqdm
import math # For checking NaN

# --- Removed API fetching functions: fetch_entity, get_summary ---

def extract_features_from_csv(row):
    
    # Check for essential columns
    required_cols = ['long_description', 'num_languages', 'description_length', 'num_geoproperties']
    for col in required_cols:
        if col not in row.index:
            print(f"Warning: Required column '{col}' not found in CSV row. Returning None for features.")
            return None
        
    #-------------------------------------------------------

    " --- Get features from the row ---"

    #-------------------------------------------------------
    # Use 'long_description' as the summary text
    summary = row['long_description']
    # Handle potential missing descriptions (NaN)
    if pd.isna(summary):
        summary = ""
    summary = str(summary).lower() # Ensure it's string and lowercase

    #-------------------------------------------------------

    lang_count = row['num_languages']
    if pd.isna(lang_count):
        lang_count = 0
    else:
        lang_count = int(lang_count)

    #-------------------------------------------------------

    description_length = row['description_length'] 
    if pd.isna(description_length):
        description_length = 0
    else:
        description_length = int(description_length) 

    #-------------------------------------------------------

    num_geoproperties = row['num_geoproperties'] 
    if pd.isna(num_geoproperties):
        num_geoproperties = 0
    else:
        num_geoproperties = int(num_geoproperties) 

    #-------------------------------------------------------


    # --- Calculate features based on CSV data ---
    return {
        'summary': summary,  # Still useful for manual inspection
        'lang_count': lang_count,
        'num_geoproperties': num_geoproperties,
        'description_length': description_length,

        # Cultural Exclusive Keywords
        'mentions_traditional': 'traditional' in summary,
        'mentions_indigenous': 'indigenous' in summary,
        'mentions_local': 'local' in summary,
        'mentions_ritual': 'ritual' in summary,

        # Cultural Representative Keywords
        'mentions_famous': 'famous' in summary,
        'mentions_popular': 'popular' in summary,
        'mentions_recognized': 'recognized' in summary,

        # Cultural Agnostic Keywords
        'mentions_worldwide': 'worldwide' in summary or 'historical' in summary,
        'mentions_international': 'international' in summary,
    }

def classify_cultural_category(features):
    """
    Classifies based on extracted features from CSV data.
    """
    if not features:
        return 'unknown'

    # --- Cultural Agnostic ---
    if ((features['description_length'] > 1000 and features['lang_count'] > 50) or ()) :
        return 'cultural agnostic'

    # --- Cultural Representative ---
    if ((features['description_length'] < 1000 and features['description_length'] > 500) and features['lang_count'] > 25):
        return 'cultural representative'

    # --- Cultural Exclusive ---
    if (features['description_length'] < 500 or features['lang_count'] < 25):
        return 'cultural exclusive'

    # --- Fallback ---
    return 'cultural representative'

# --- Main Script Execution ---

# Load your training data CSV using the path from pathinator
try:
    df = pd.read_csv(train_path)
    print(f"Loaded {len(df)} rows from {train_path}")
except FileNotFoundError:
    print(f"Error: Training data file not found at {train_path}")
    exit()
except Exception as e:
    print(f"Error reading training data CSV: {e}")
    exit()

# Verify essential columns exist in the DataFrame
essential_csv_cols = ['item', 'long_description', 'num_languages'] # 'label' is optional for eval
missing_cols = [col for col in essential_csv_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Missing essential columns in {train_path}: {', '.join(missing_cols)}")
    print(f"Found columns: {df.columns.tolist()}")
    exit()

# Optional: Check for 'num_geoproperties' if used in classification
if 'num_geoproperties' not in df.columns:
     print(f"Warning: Column 'num_geoproperties' not found. Classification logic using it may be affected.")


print("Starting classification using only CSV data...")

predicted = []
# Iterate through DataFrame rows instead of just links
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Classifying items", unit="item", leave=True):
    features = extract_features_from_csv(row) # Pass the whole row
    label = classify_cultural_category(features)
    predicted.append(label)

print("Classification finished.")

df['predicted_category'] = predicted

# Save or print the output
output_filename = 'classified_output_csv_only.csv' # Changed output name slightly
try:
    df.to_csv(output_filename, index=False)
    print(f"Output saved to {output_filename}")
except Exception as e:
    print(f"Error saving output CSV: {e}")


# --- Evaluation ---
print("\nSample of input data and predictions:")
# Check if 'label' exists for comparison
if 'label' in df.columns:
    print(df[['item', 'label', 'predicted_category']].head())
    
    # Calculate and print basic accuracy
    correct_predictions = (df['label'] == df['predicted_category']).sum()
    total_items = len(df)
    if total_items > 0:
        accuracy = (correct_predictions / total_items) * 100
        print(f"\nBasic Accuracy (CSV Only): {accuracy:.2f}% ({correct_predictions}/{total_items})")
    else:
        print("\nNo items to calculate accuracy.")
else:
     print(f"Warning: Column 'label' not found in input CSV for comparison.")
     print(df[['item', 'predicted_category']].head()) # Show predictions even without labels