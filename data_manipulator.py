from pathinator import *
import pandas as pd

#--LOAD csv--
train_original_data = pd.read_csv(train_original_path)
validation_original_data = pd.read_csv(validation_original_path)

test_data = pd.read_csv(test_path)
test_data["category"] = test_data["category"].str.lower()
test_data["type"] = test_data["type"].replace('named entity', 'entity')

train_data = pd.read_csv(train_path)
validation_data = pd.read_csv(validation_path)

#print(sorted(test_data["category"].unique()))

#--TRAIN columns--
items, names, descriptions, long_descriptions = train_data["item"], train_data["name"], train_data["description"], train_data["long_description"]
types, categories, subcategories = train_data["type"], train_data["category"], train_data["subcategory"]
descriptions_length, num_languages, num_geoproperties = train_data["description_length"], train_data["num_languages"], train_data["num_geoproperties"]
labels = train_data["label"]

#print(sorted(subcategories.unique()))

#--VALIDATION columns--
validation_items, validation_names = validation_data["item"], validation_data["name"]
validation_descriptions, validation_long_descriptions = validation_data["description"], validation_data["long_description"]
validation_types, validation_categories, validation_subcategories = validation_data["type"], validation_data["category"], validation_data["subcategory"]
validation_descriptions_length = validation_data["description_length"]
validation_num_languages, validation_num_geoproperties = validation_data["num_languages"], validation_data["num_geoproperties"]
validation_labels = validation_data["label"]

#print(sorted(validation_subcategories.unique()))

#--FEATURES--
numerical_features = ["num_languages", 'num_geoproperties', "num_cultural_properties"]
categorical_features = ['type', 'category', 'subcategory']
keyword_features = ['exclusive_categories_count', 'representative_categories_count', 'agnostic_categories_count']

#--KEYWORDS--
exclusive_keywords = ['leipzig', 'canton', 'municipalities', 'municipality', 'ministers', 'province', 
                      'polish', 'council', 'parliament', 'mi', 'swiss', 'county', 'located']
representative_keywords = ['hbo', 'emmy', 'disney', 'nominations', 'manga', 'globe', 
                           'album', 'greatest', 'albums']
agnostic_keywords = ['aquatic', 'organisms', 'organism', 'cement', 'climbing', 'genus', 'plants', 
                     'surface', 'species', 'fish', 'systems', 'trees', 'plant', 'materials']

#--AUGMENTATION--
'''train_data['exclusive_categories_count'] = long_descriptions.apply(lambda text: keywords_countator(text, exclusive_keywords))
train_data['representative_categories_count'] = long_descriptions.apply(lambda text: keywords_countator(text, representative_keywords))
train_data['agnostic_categories_count'] = long_descriptions.apply(lambda text: keywords_countator(text, agnostic_keywords))

validation_data['exclusive_categories_count'] = long_descriptions.apply(lambda text: keywords_countator(text, exclusive_keywords))
validation_data['representative_categories_count'] = long_descriptions.apply(lambda text: keywords_countator(text, representative_keywords))
validation_data['agnostic_categories_count'] = long_descriptions.apply(lambda text: keywords_countator(text, agnostic_keywords))'''
