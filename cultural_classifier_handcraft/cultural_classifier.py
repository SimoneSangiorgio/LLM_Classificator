import pandas as pd
import requests
import wikipediaapi
import random
from pathinator import train_path
from tqdm import tqdm

def fetch_entity(wiki_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wiki_id}.json"

    response = requests.get(url)
    data = response.json()

    if 'entities' in data and wiki_id in data['entities']:
        return data['entities'][wiki_id]
    else:
        print(f"Warning: Unexpected JSON structure for {wiki_id}.")
        return None

def get_summary(title):
    if not title: # Handle empty title case immediately
        return ""
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='CulturalClassifierBot/1.0 (https://example.com/bot; cool-user@example.com)', # Example better user agent
        extract_format=wikipediaapi.ExtractFormat.WIKI # Using WIKI format can sometimes be more robust
    )
    page = wiki.page(title)

    return page.summary[:]


def extract_cultural_features(wikidata_url):

    wiki_id = wikidata_url.split('/')[-1]

    entity = fetch_entity(wiki_id)
    if not entity: # Check if fetch_entity returned None
        return None # Propagate the failure

    claims = entity.get("claims", {})
    sitelinks = entity.get("sitelinks", {})
    lang_count = len(sitelinks)

    # Try to get English title, fallback to any sitelink title if needed
    title = sitelinks.get('enwiki', {}).get('title')
    if not title and sitelinks:
        # Fallback: try getting the first available sitelink title
        first_sitelink_key = next(iter(sitelinks), None)
        if first_sitelink_key:
            title = sitelinks[first_sitelink_key].get('title')
            # print(f"Using fallback title '{title}' for {wiki_id}") # Optional info

    summary = get_summary(title).lower()

    def get_country_code(prop):
        # Check if the property exists and has at least one statement
        if prop in claims and claims[prop]:
             # Check structure down to the 'id'
            value_data = claims[prop][0].get('mainsnak', {}).get('datavalue', {}).get('value', {})
            if value_data and 'id' in value_data:
                return value_data['id']
        return None # Return None if structure is not as expected
        

    origin_country = get_country_code('P495') or get_country_code('P17')

    return {
        'summary': summary,
        'lang_count': lang_count,
        'origin_country': origin_country,
        'mentions_traditional': 'traditional' in summary,
        'mentions_worldwide': 'worldwide' in summary or 'international' in summary,
        'mentions_specific': 'regional' in summary or 'specific' in summary,
    }

def classify_cultural_category(features):
    if not features:
        return 'unknown'
    # Prioritize agnostic based on wide reach first
    if features['lang_count'] > 75 or (features['lang_count'] > 50 and features['mentions_worldwide']):
         return 'cultural agnostic'
    # Then check for specific cultural markers
    elif features['origin_country'] or features['mentions_traditional'] or features['mentions_specific']:
        # Distinguish between representative and exclusive based on reach
        if features['lang_count'] > 25 or features['mentions_worldwide']: # If somewhat widespread even with traditional markers
            return 'cultural representative'
        else: # Lower language count and specific markers -> exclusive
            return 'cultural exclusive'
    # Default to representative if not strongly agnostic or exclusive
    else:
        return 'cultural representative'

# Load your training data CSV using the path from pathinator
try:
    # Use train_path from pathinator
    df = pd.read_csv(train_path)
    print(f"Loaded {len(df)} rows from {train_path}")
except FileNotFoundError:
    print(f"Error: Training data file not found at {train_path}")
    exit() # Exit if the main data file can't be loaded
except Exception as e:
    print(f"Error reading training data CSV: {e}")
    exit()

# Classify
predicted = []
# Use the correct column name 'item' from train.csv
if 'item' not in df.columns:
    print(f"Error: Column 'item' not found in {train_path}. Found columns: {df.columns.tolist()}")
    exit()

print(f"Starting classification...")

for link in tqdm(df['item'], desc="Classifying items", unit="item", leave=True):
    features = extract_cultural_features(link)
    label = classify_cultural_category(features)
    predicted.append(label)

print("Classification finished.")

df['predicted_category'] = predicted

# Save or print the output
output_filename = 'classified_output.csv'
try:
    df.to_csv(output_filename, index=False)
    print(f"Output saved to {output_filename}")
except Exception as e:
    print(f"Error saving output CSV: {e}")


# Use correct input column names 'item' and 'label' for comparison
print("\nSample of input data and predictions:")
if 'label' in df.columns:
    print(df[['item', 'label', 'predicted_category']].head())
else:
     print(f"Warning: Column 'label' not found in input CSV for comparison.")
     print(df[['item', 'predicted_category']].head())

# Optional: Calculate and print basic accuracy if 'label' exists
if 'label' in df.columns:
    correct_predictions = (df['label'] == df['predicted_category']).sum()
    total_items = len(df)
    if total_items > 0:
        accuracy = (correct_predictions / total_items) * 100
        print(f"\nBasic Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_items})")
    else:
        print("\nNo items to calculate accuracy.")