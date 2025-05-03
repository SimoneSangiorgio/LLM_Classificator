import pandas as pd
from pathinator import train_path
from tqdm import tqdm
#import math # For checking NaN
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# --- Keyword Lists  ---
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
# --- Removed Original Keyword Lists ---
# ORIGINAL_EXCLUSIVE_KEYWORDS = ['traditional', 'indigenous', 'local', 'ritual']
# ORIGINAL_REPRESENTATIVE_KEYWORDS = ['famous', 'popular', 'recognized']
# ORIGINAL_AGNOSTIC_KEYWORDS = ['worldwide', 'historical', 'international']

# --- count_keywords function  ---
def count_keywords(text, keywords):
    """Counts occurrences of keywords (case-insensitive) in text."""
    count = 0
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    for keyword in keywords:
        if keyword.lower() in text_lower:
            count += 1
    return count

# --- extract_features_from_csv  ---
def extract_features_from_csv(row):
    """Extracts numerical features and keyword counts from a CSV row."""

    required_cols = ['long_description', 'num_languages', 'description_length', 'num_geoproperties']
    missing_instance_cols = [col for col in required_cols if col not in row.index]
    if missing_instance_cols:
         print(f"Warning: Missing columns {missing_instance_cols} in row {row.name if hasattr(row, 'name') else 'N/A'}. Using default features.")
         return {
            'summary_lower': "", 'lang_count': 0, 'num_geoproperties': 0,
            'description_length': 0, 'exclusive_keywords': 0,
            'representative_keywords': 0, 'agnostic_keywords': 0,
         }

    summary = row['long_description']
    summary = "" if pd.isna(summary) else str(summary)
    summary_lower = summary.lower()

    lang_count = row['num_languages']
    lang_count = 0 if pd.isna(lang_count) else int(lang_count)

    description_length = row['description_length']
    description_length = 0 if pd.isna(description_length) else int(description_length)

    num_geoproperties = row['num_geoproperties']
    num_geoproperties = 0 if pd.isna(num_geoproperties) else int(num_geoproperties)

    # --- Calculate keyword features ---
    exclusive_keyword_count = count_keywords(summary_lower, EXCLUSIVE_KEYWORDS)
    representative_keyword_count = count_keywords(summary_lower, REPRESENTATIVE_KEYWORDS)
    agnostic_keyword_count = count_keywords(summary_lower, AGNOSTIC_KEYWORDS)

    return {
        'summary_lower': summary_lower,
        'lang_count': lang_count,
        'num_geoproperties': num_geoproperties,
        'description_length': description_length,
        'exclusive_keywords': exclusive_keyword_count,
        'representative_keywords': representative_keyword_count,
        'agnostic_keywords': agnostic_keyword_count,
    }


# --- classify_cultural_category  ---

def classify_cultural_category(features):
    """
    Classifies based on extracted features, with refined logic for Agnostic and keyword thresholds.
    """
    if not features:
        return 'unknown'

    # --- Tuned Thresholds  ---
    LANG_LOW = 18; LANG_MID = 35; LANG_HIGH = 55
    DESC_LEN_LOW = 350; DESC_LEN_MID = 800; DESC_LEN_HIGH = 1200
    KEYWORD_THRESHOLD = 2 # Requires >= 2 keywords from the new lists

    # --- Feature Access ---
    dl = features['description_length']
    lc = features['lang_count']
    kw_ex = features['exclusive_keywords']
    kw_rp = features['representative_keywords']
    kw_ag = features['agnostic_keywords']

    # --- Rule Priority: Strongest indicators first ---

    # 1. Strong Agnostic Indicators:
    is_strongly_numeric_agnostic = (dl > DESC_LEN_HIGH and lc > LANG_HIGH)
    has_strong_agnostic_keywords = (kw_ag >= KEYWORD_THRESHOLD)
    agnostic_keywords_dominate = has_strong_agnostic_keywords and (kw_ag > kw_ex + kw_rp)

    if is_strongly_numeric_agnostic or agnostic_keywords_dominate:
         return 'cultural agnostic'
    if (dl > DESC_LEN_MID and lc > LANG_MID and kw_ex == 0) or \
       (has_strong_agnostic_keywords and kw_ex == 0 and kw_rp <= 1): # Allow 0 or 1 Rep keyword if Agnostic is strong
         return 'cultural agnostic'

    # 2. Strong Exclusive Indicators:
    is_strongly_numeric_exclusive = (dl < DESC_LEN_LOW and lc < LANG_LOW)
    has_strong_exclusive_keywords = (kw_ex >= KEYWORD_THRESHOLD)
    exclusive_keywords_dominate = has_strong_exclusive_keywords and (kw_ex > kw_rp + kw_ag)

    if is_strongly_numeric_exclusive and exclusive_keywords_dominate:
         return 'cultural exclusive'
    if exclusive_keywords_dominate and not (dl > DESC_LEN_MID and lc > LANG_MID): # If keywords dominate and not numerically agnostic
         return 'cultural exclusive'
    # Simplified: If numerically exclusive AND has *any* exclusive keywords (>=1) without strong contradiction
    if is_strongly_numeric_exclusive and kw_ex >= 1 and kw_ag < KEYWORD_THRESHOLD and kw_rp < KEYWORD_THRESHOLD:
       return 'cultural exclusive'


    # 3. Strong Representative Indicators:
    is_mid_range_numeric = (dl >= DESC_LEN_LOW and dl <= DESC_LEN_HIGH) or \
                           (lc >= LANG_LOW and lc <= LANG_HIGH)
    has_strong_representative_keywords = (kw_rp >= KEYWORD_THRESHOLD)
    representative_keywords_dominate = has_strong_representative_keywords and (kw_rp > kw_ex + kw_ag)

    if representative_keywords_dominate and is_mid_range_numeric:
        return 'cultural representative'
    if representative_keywords_dominate and not is_strongly_numeric_exclusive and not is_strongly_numeric_agnostic:
        return 'cultural representative'

    # --- Fallback Logic based on Numerical Features ---
    
    if (dl > DESC_LEN_MID or lc > LANG_MID) and not is_strongly_numeric_exclusive:
        
        if lc > LANG_MID + 10 or dl > DESC_LEN_HIGH - 100:
             return 'cultural agnostic'

    if dl < DESC_LEN_LOW + 25 and lc < LANG_LOW + 3 :
        return 'cultural exclusive'

    return 'cultural representative'


# --- Main Script Execution ---

# Load data
try:
    df = pd.read_csv(train_path)
    print(f"Loaded {len(df)} rows from {train_path}")
except FileNotFoundError:
    print(f"Error: Training data file not found at {train_path}")
    exit()
except Exception as e:
    print(f"Error reading training data CSV: {e}")
    exit()

# Verify essential columns for feature extraction
essential_feature_cols = ['long_description', 'num_languages', 'description_length', 'num_geoproperties']
missing_cols = [col for col in essential_feature_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Missing essential columns needed for feature extraction in {train_path}: {', '.join(missing_cols)}. Cannot proceed.")
    print(f"Found columns: {df.columns.tolist()}")
    exit()

# Check for label column
if 'label' not in df.columns:
    print("Warning: 'label' column not found in input CSV. Accuracy evaluation will be skipped.")
    EVALUATE = False
else:
    EVALUATE = True

print("Starting feature extraction (new keywords only) and classification...")

predicted_labels = []
feature_list_dicts = []
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing items", unit="item", leave=True):
    features = extract_features_from_csv(row)
    feature_list_dicts.append(features)
    if features:
         label = classify_cultural_category(features)
         predicted_labels.append(label)
    else:
         predicted_labels.append('unknown')

print("Processing finished.")

features_df = pd.DataFrame(feature_list_dicts)
df['predicted_category'] = predicted_labels
keyword_cols_to_add = ['exclusive_keywords', 'representative_keywords', 'agnostic_keywords']
for col in keyword_cols_to_add:
     if col in features_df.columns:
        df[col] = features_df[col]
     else:
        print(f"Warning: Feature column '{col}' not found in extracted features. Skipping addition to DataFrame.")

# Save output
output_filename = 'classified_output_csv.csv' # Updated filename
try:
    cols_to_save = df.columns.tolist()
    df.to_csv(output_filename, index=False, columns=cols_to_save)
    print(f"Output saved to {output_filename}")
except Exception as e:
    print(f"Error saving output CSV: {e}")


# --- Evaluation ---
if EVALUATE:
    print("\nSample of input data and predictions:")
    display_cols = ['item', 'label', 'predicted_category', 'description_length', 'num_languages']
    for kw_col in keyword_cols_to_add:
        if kw_col in df.columns:
            display_cols.append(kw_col)
    print(df.loc[:, display_cols].head())

    correct_predictions = (df['label'] == df['predicted_category']).sum()
    total_items = len(df)
    if total_items > 0:
        accuracy = (correct_predictions / total_items) * 100
        print(f"\nBasic Accuracy (New Keywords Only): {accuracy:.2f}% ({correct_predictions}/{total_items})")

        if SKLEARN_AVAILABLE and pd.Series(predicted_labels).nunique() > 1:
            try:
                 present_labels = sorted(list(set(df['label']) | set(df['predicted_category'])))
                 target_labels = ['cultural exclusive', 'cultural representative', 'cultural agnostic']
                 report_labels = [l for l in target_labels if l in present_labels]
                 if not report_labels:
                      print("\nNo target labels found for detailed report.")
                 else:
                      print("\nClassification Report:")
                      print(classification_report(df['label'], df['predicted_category'], labels=report_labels, zero_division=0))
                      print("\nConfusion Matrix:")
                      cm = confusion_matrix(df['label'], df['predicted_category'], labels=report_labels)
                      cm_df = pd.DataFrame(cm, index=[f'True {l.split()[-1].title()[:4]}' for l in report_labels],
                                              columns=[f'Pred {l.split()[-1].title()[:4]}' for l in report_labels])
                      print(cm_df)
            except Exception as e_eval:
                print(f"\nError during detailed evaluation: {e_eval}")
        elif not SKLEARN_AVAILABLE:
             print("\nInstall scikit-learn (pip install scikit-learn) for detailed classification report and confusion matrix.")
    else:
        print("\nNo items to calculate accuracy.")
else:
     print("\nSample of predictions (no labels for comparison):")
     display_cols_no_label = ['item', 'predicted_category', 'description_length', 'num_languages']
     for kw_col in keyword_cols_to_add:
         if kw_col in df.columns:
             display_cols_no_label.append(kw_col)
     print(df.loc[:, display_cols_no_label].head())

# Plot Confusion Matrix if sklearn is available
if SKLEARN_AVAILABLE and 'cm' in locals():
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=report_labels,
                yticklabels=report_labels)
    plt.title('Matrice di Confusione')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()