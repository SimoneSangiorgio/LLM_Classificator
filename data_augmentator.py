from pathinator import *
from data_manipulator import *

import pandas as pd
import wikipediaapi
import requests
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import re

numero = 6100

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

#print(wiki_descriptor(items[numero]))

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

#print(wiki_originator(items[numero]))

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

#print(wiki_geografic_propertinator(items[numero]))

def wiki_culturator(wikidata_link):

    wiki_id=wikidata_link.split('/')[-1]
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wiki_id}.json"

    response = requests.get(url)   
    data = response.json()
    entity = data['entities'][wiki_id]
    claims = entity.get('claims', {})
    
    properties = {
    'P19', 'P20', 'P840',
    'P170', 'P50', 'P112', 'P127', 'P1535', 'P172', 'P140',
    'P364', 'P407',
    'P571', 'P2348',
    'P361', 'P921', 'P144'}

    results = 0

    for property in properties:
        if property in claims:
            results += 1
        else:
            results = results

    return results

#print(wiki_culturator(items[numero]))

def wiki_languagetor(wikidata_link):
    wiki_id = wikidata_link.rstrip('/').split('/')[-1]
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wiki_id}.json"

    response = requests.get(url)
    data = response.json()

    entity = data['entities'][wiki_id]
    sitelinks = entity.get("sitelinks", {})

    # Lingue Wikipedia (es. enwiki, itwiki, frwiki)
    wikipedia_langs = [
        site for site in sitelinks
        if re.fullmatch(r'[a-z]{2,3}wiki', site)]

    num_languages = len(wikipedia_langs)

    return int(num_languages)

#print(wiki_languagetor(items[numero]))

def keywords_countator(text, keywords):
    if not isinstance(text, str):
        return 0    
    text_lower = text.lower()
    count = 0
    for keyword in keywords:
         if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
              count += 1     
              
    # ALTERNATIVA: conta TUTTE le occorrenze (anche multiple della stessa keyword):
    #count = sum(len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower)) for keyword in keywords)
    
    return count

#print(keywords_countator(long_descriptions[3], representative_keywords))

def augmented_csv(data, csv_name):
    def combined_wiki_info(wikidata_link):
        long_desc = wiki_descriptor(wikidata_link)
        languages = wiki_languagetor(wikidata_link)
        cultural_properties = wiki_culturator(wikidata_link)
        num_geoproperties = wiki_geografic_propertinator(wikidata_link)

        return pd.Series({'long_description': long_desc, 'num_geoproperties': num_geoproperties,
                          'num_languages': languages, 'num_cultural_properties': cultural_properties})

    tqdm.pandas(desc="Fetching Wikidata Properties")
    data_to_process = data["item"][:]
    results_df = data_to_process.progress_apply(combined_wiki_info)
    data = data.copy()  # evitare side effects
    data.loc[data_to_process.index, results_df.columns] = results_df


    output_file_path = dataset_path / csv_name
    data.to_csv(output_file_path, index=False, encoding='utf-8')

#augmented_csv(test_data, "test_unlabeled_augmented.csv")

def keywords_finder(long_descriptions, labels, string_label):
    nltk.download('stopwords')

    # Unisci il testo
    all_text = " ".join(long_descriptions.dropna())
    words = re.findall(r'\b\w+\b', all_text.lower())

    # Stopwords
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'also', 'known', 'one', 'used', 'may', 'often', 'two', 'first', 'new', 'including', 'world', 'century'}
    stop_words.update(custom_stopwords)

    # Filtro parole
    filtered_words = [word for word in words if word not in stop_words]

    # Conta occorrenze globali
    word_counts = Counter(filtered_words)
    most_common_words = word_counts.most_common(3000)

    # Prepara struttura per contare per label
    word_label_counts = defaultdict(lambda: Counter())

    # Per ogni descrizione e label associata
    for description, label in zip(long_descriptions, labels):
        if pd.isna(description):
            continue
        desc_words = re.findall(r'\b\w+\b', description.lower())
        for word in desc_words:
            if word not in stop_words:
                word_label_counts[word][label] += 1

    # Salviamo parole filtrate e ordinate per percentuale
    filtered_entries = []

    for word, count in most_common_words:
        label_counter = word_label_counts[word]
        if label_counter:
            most_common_label, label_count = label_counter.most_common(1)[0]
            total = sum(label_counter.values())
            percentage = (label_count / total) * 100
            if most_common_label == string_label:
                filtered_entries.append((word, count, most_common_label, percentage))

    # Ora ordiniamo per percentuale decrescente
    filtered_entries.sort(key=lambda x: x[3], reverse=True)

    # Stampa finale
    for word, count, most_common_label, percentage in filtered_entries[:20]:
        print(f"Parola: '{word}' - Occorrenze: {count} - Label prevalente: {most_common_label} ({percentage:.1f}%)")

#print(keywords_finder(long_descriptions, labels, "cultural exclusive"))

def keyword_label_percentages(long_descriptions, labels, keywords):
    nltk.download('stopwords')

    # Prepara struttura per contare label associate a ogni parola
    word_label_counts = defaultdict(lambda: Counter())

    # Stopwords
    stop_words = set(stopwords.words('english'))
    #custom_stopwords = {'also', 'known', 'one', 'used', 'may', 'often', 'two', 'first', 'new', 'including', 'world', 'century'}
    custom_stopwords = {}
    stop_words.update(custom_stopwords)

    # Per ogni descrizione e label associata
    for description, label in zip(long_descriptions, labels):
        if pd.isna(description):
            continue
        desc_words = re.findall(r'\b\w+\b', description.lower())
        for word in desc_words:
            if word not in stop_words:
                word_label_counts[word][label] += 1

    # Ora analizziamo le parole di interesse
    for word in keywords:
        label_counter = word_label_counts[word]
        if label_counter:
            total = sum(label_counter.values())
            print(f"\nParola: '{word}' (totale occorrenze: {total})")
            for label, count in label_counter.items():
                percentage = (count / total) * 100
                print(f"  - {label}: {percentage:.1f}% ({count} occorrenze)")
        else:
            print(f"\nParola: '{word}' non trovata nel dataset.")

#print(keyword_label_percentages(long_descriptions, labels, agnostic_keywords))

def labels_percentage(labels):
    label_counts = labels.value_counts()
    label_percentages = (label_counts / len(labels)) * 100
    print(label_percentages)

#print(labels_percentage(labels))

#print(labels[numero])

'''for i in range(0,300):
    wikidata_link=validation_data["item"][i]
    print(f"---{i}---")
    print(len(wiki_descriptor(wikidata_link)))
    print(wiki_languagetor(wikidata_link))'''