from pathinator import *
from data import *

import pandas as pd
import wikipediaapi
import requests
import random

wikidata_link=items[random.randint(0, 6250)]
wikidata_link=items[1234]

def wiki_descriptor(wikidata_link):
    wiki = wikipediaapi.Wikipedia(language='en', user_agent='MNLPBot')
    wiki_id=wikidata_link.split('/')[-1]
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wiki_id}.json"

    response = requests.get(url)
    data = response.json()

    entity = data['entities'][wiki_id]

    #title
    title = entity['sitelinks']['enwiki']['title']

    #description
    page = wiki.page(title)

    return page.summary[:]

def wiki_originator(wikidata_link):

    def get_label(entity_id, language='en'):
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
        data = requests.get(url).json()
        return data['entities'][entity_id]['labels'].get(language, {}).get('value', None)

    wiki_id=wikidata_link.split('/')[-1]
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wiki_id}.json"

    response = requests.get(url)   
    data = response.json()
    entity = data['entities'][wiki_id]
    claims = entity.get('claims', {})
    
    properties = {
        #'P19': 'Place of birth',
        #'P20': 'Place of death',
        'P27': 'Nationality',
        'P17': 'Country',
        'P495': 'Country of origin',
        'P276': 'Location',
        'P131': 'Administrative territorial entity',
        'P1001': 'Jurisdiction',
        'P291': 'Place of publication',
        'P1071': 'Location of origin',
        'P159': 'Headquarters location',
        'P2348': 'Time or place of invention/discovery',
    }

    results = {}

    for property, description in properties.items():
        if property in claims:
            try:
                valore_id = claims[property][0]['mainsnak']['datavalue']['value']['id']
                label = get_label(valore_id)
                results[description] = label
            except Exception:
                results[description] = results[description]

    return results

def wiki_languagetor(wikidata_link):
    wiki_id=wikidata_link.split('/')[-1]
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wiki_id}.json"

    response = requests.get(url)
    data = response.json()

    entity = data['entities'][wiki_id]
    sitelinks = entity.get("sitelinks", {})
    num_languages = len(sitelinks)

    return num_languages



print(wiki_descriptor(wikidata_link))
print(wiki_originator(wikidata_link))
print(wiki_languagetor(wikidata_link))
print('')
print(wikidata_link)


'''types_list, categories_list, subcategories_list = types.unique(), categories.unique(), subcategories.unique()
labels_list = labels.unique()

exclusive_categories = train_data[(categories == categories) & (labels == "cultural exclusive")]
representative_categories = train_data[(categories == categories) & (labels == "cultural representative")]
agnostic_categories = train_data[(categories == categories) & (labels == "cultural agnostic")]

perc_exclusive_categories = len(exclusive_categories)/len(train_data)
perc_representative_categories = len(representative_categories)/len(train_data)
perc_agnostic_categories = len(agnostic_categories)/len(train_data)'''

def listinator(x):
    return x.unique()

def labelinator(x, label):
    return train_data[(x == x) & (labels == label)]

def percentageinator(x):
    return len(x)/len(train_data)

def percensortinator(x):
    data_list = []
    for element in listinator(x):
        for label in listinator(labels):
            data = train_data[(x == element) & (labels == label)]
            data_list.append({
                "element": element,
                "label": label,
                "%": percentageinator(data)*10
            })
    return pd.DataFrame(data_list).sort_values(by="%", ascending=False)



'''print(percentageinator(labelinator(categories, "cultural exclusive")))
print(percentageinator(labelinator(categories, "cultural representative")))
print(percentageinator(labelinator(categories, "cultural agnostic")))'''
