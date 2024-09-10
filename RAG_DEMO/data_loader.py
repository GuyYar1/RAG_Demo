import nltk
import pandas as pd
import re
import numpy as np
from nltk.corpus import movie_reviews
from gensim.models import Word2Vec

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

def preprocess_documents(SelctIMDB=False):
    """
    Preprocess documents by tokenizing and generating document embeddings.

    Args:
    SelctIMDB (bool): Flag to choose between IMDb data or NLTK movie reviews.

    Returns:
    tokenized_docs (list of list of str): Tokenized documents.
    document_embeddings (numpy array): Embeddings for each document.
    vector_db (dict): Dictionary mapping document indices to raw documents.
    """
    print("preprocess_documents")

    if SelctIMDB:
        url = "TMDB_tv_dataset_v3.csv"  # Replace with actual path
        documents = download_imdb_data(url)
        # Iterate over each row in the DataFrame
        concatenated_list = []
        documents1 = pd
        for index, row in documents.iterrows():
            # Concatenate all columns in the row, delimited by newlines
            concatenated_string = '\n'.join(row.astype(str))
            # Append the result to the list
            concatenated_list.append(concatenated_string)
            # Adjust based on the actual column name

        documents = concatenated_list
    else:
        # Ensure NLTK resources are downloaded
        ensure_nltk_resources()
        # Load the movie review dataset from NLTK
        documents = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]

    # Preprocess documents by removing punctuation and tokenizing
    tokenized_docs = [re.sub(r'[^\w\s]', '', doc.lower()).split() for doc in documents]

    # Train Word2Vec model to generate document embeddings
    w2v_model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

    # Generate embeddings for each document
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

    # Dictionary for storing document database
    vector_db = {i: doc for i, doc in enumerate(documents)}

    return tokenized_docs, document_embeddings, vector_db, w2v_model

    # After calculating embeddings and performing a similarity search, you might get indices of the most similar documents.
    # You can use vector_db to map these indices back to the original document texts.
    # Raw Documents Storage: vector_db stores the raw, unprocessed documents. It does not include the embeddings of these documents;
    # it only contains the original text data.


