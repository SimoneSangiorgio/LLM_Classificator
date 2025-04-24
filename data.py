from pathinator import *
import pandas as pd

train_data = pd.read_csv(train_path)
validation_data = pd.read_csv(validation_path)

items, names, descriptions, long_descriptions = train_data["item"], train_data["name"], train_data["description"], train_data["long_description"]
#train_data["text_features"] = names + " - " + descriptions
#text_feature = 'text_features'

types, categories, subcategories = train_data["type"], train_data["category"], train_data["subcategory"]
#categorical_features = ['type', 'category', 'subcategory']

descriptions_length, num_languages, num_geoproperties = train_data["description_length"], train_data["num_languages"], train_data["num_geoproperties"]

#features = train_data[['text_features', 'type', 'category', 'subcategory']]

labels = train_data["label"]
