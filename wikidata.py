from pathinator import *
from data import *

import pandas as pd
import wikipediaapi
import requests
import random

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
        'P2348': 'Time or place of invention/discovery'
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

def wiki_geografic_propertinator(wikidata_link):

    wiki_id=wikidata_link.split('/')[-1]
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wiki_id}.json"

    response = requests.get(url)   
    data = response.json()
    entity = data['entities'][wiki_id]
    claims = entity.get('claims', {})
    
    properties = {'P27', 'P17', 'P495', 'P276', 'P131','P1001','P291','P1071','P159','P2348'}

    results = 0

    for property in properties:
        if property in claims:
            results += 1
        else:
            results = results

    return results

def wiki_languagetor(wikidata_link):
    wiki_id=wikidata_link.split('/')[-1]
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wiki_id}.json"

    response = requests.get(url)
    data = response.json()

    entity = data['entities'][wiki_id]
    sitelinks = entity.get("sitelinks", {})
    num_languages = len(sitelinks)

    return int(num_languages)

'''for i in range(3000,6000):
    wikidata_link=train_mod_data["item"][i]
    print(f"---{i}---")
    print(len(wiki_descriptor(wikidata_link)))
    print(wiki_languagetor(wikidata_link))'''
#print(wikidata_link)

'''print(wiki_geografic_propertinator(wikidata_link))

print(wiki_originator(wikidata_link))'''

#print(wikidata_link)

