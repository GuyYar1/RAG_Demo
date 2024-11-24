import nltk
import pandas as pd
import re
import numpy as np
from nltk.corpus import movie_reviews
from gensim.models import Word2Vec
import requests
from bs4 import BeautifulSoup
import os
import re
from collections import OrderedDict


# Function to scrape and save text from URLs
def scrape_and_save_text(url, output_dir, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = ' '.join([para.get_text() for para in paragraphs])
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(text_content)
        print(f"Saved content from {url} to {filepath}")
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")


# Function to scrape and save corpus
def scrape_and_save_Corpus():
    urls = [
        "https://www.verywellhealth.com/search?q=Diabetic+retinopathy",
        "https://www.verywellhealth.com/search?q=Diabetic+retinopathy",
        "https://app.pulsenotes.com/medicine/diabetes/notes/type-2-diabetes#note-9",
        "https://stanfordhealthcare.org/medical-conditions/eyes-and-vision/diabetic-retinopathy/treatments.html",
        "https://www2.hse.ie/conditions/diabetic-retinopathy/treatment/",
        "https://www.hopkinsmedicine.org/health/conditions-and-diseases/diabetes/diabetic-retinopathy",
        "https://emedicine.medscape.com/article/1225122-treatment#d14",
        "https://www.nhs.uk/conditions/diabetic-retinopathy/"
    ]

    output_directory = './scraped_data'
    for idx, url in enumerate(urls[:1]):  # Process only the first URL
        filename = f"article_{idx}.txt"
        scrape_and_save_text(url, output_directory, filename)


# Function to concatenate corpus into a DataFrame
def concat_corpus(directory):
    concatenated_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                file_content = file.read()
                concatenated_list.append(file_content)
    return pd.DataFrame(concatenated_list, columns=['Document'])

# Function to create corpus document
def create_corpus_doc():
    directory = './scraped_data'
    documents = concat_corpus(directory)
    print("Corpus created successfully.")
    return documents

# Function to download NLTK resources if not already downloaded
def ensure_nltk_resources():
    try:
        # Check if the resource is already downloaded
        movie_reviews.fileids()
    except LookupError:
        # If not downloaded, download the resources
        nltk.download('movie_reviews')



def download_imdb_data(url):
    """
    Download IMDb data from a URL.
    Replace with actual implementation if needed.
    """
    print("Downloading IMDb data...")
    # Read the CSV file into a DataFrame
    return pd.read_csv(url, delimiter=',', low_memory=False)


def filter_non_empty_overview(df):
    """
    Filter rows where the 'Overview' column is not empty.

    Parameters:
        df (pd.DataFrame): The DataFrame to filter.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    #df[...]: Uses the boolean Series to filter the DataFrame. Only rows where the boolean Series is True are included in the result.
    # This means only rows where the 'Overview' column is not empty are retained.
    return df[df['overview'].str.strip() != '']



def clean_and_deduplicate(text):
    # Remove repeated lines
    lines = text.split('\n')
    seen = OrderedDict.fromkeys(lines)
    cleaned_text = '\n'.join(seen.keys())

    # Additional cleaning if necessary
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize whitespace
    return cleaned_text


# Preprocess documents to generate embeddings
def preprocess_documents(SelctIMDB=False, Diabitis=True):
    print("preprocess_documents")

    scrape_and_save_Corpus()
    documents_df = create_corpus_doc()
    # Assuming documents_df is your DataFrame containing the documents
    documents_df['Cleaned_Document'] = documents_df['Document'].apply(clean_and_deduplicate)
    # Check the cleaned documents
    print("Cleaned Documents:")
    print(documents_df['Cleaned_Document'].head())


    tokenized_docs = [re.sub(r'[^\w\s]', '', doc.lower()).split() for doc in documents_df['Cleaned_Document'] ]
    print("tokenized_docs", tokenized_docs)
    print("len tokenized_docs", len(tokenized_docs))

    w2v_model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)
    document_embeddings = []

    vector_size = w2v_model.vector_size
    for doc in tokenized_docs:
        valid_embeddings = [w2v_model.wv[word] for word in doc if word in w2v_model.wv]
        if valid_embeddings:
            doc_embedding = np.mean(valid_embeddings, axis=0)
        else:
            doc_embedding = np.zeros(vector_size)
        document_embeddings.append(doc_embedding)

    document_embeddings = np.array(document_embeddings)
    vector_db = {i: doc for i, doc in enumerate(documents_df['Cleaned_Document'])}

    return tokenized_docs, document_embeddings, vector_db, w2v_model

    # After calculating embeddings and performing a similarity search, you might get indices of the most similar documents.
    # You can use vector_db to map these indices back to the original document texts.
    # Raw Documents Storage: vector_db stores the raw, unprocessed documents. It does not include the embeddings of these documents;
    # it only contains the original text data.


