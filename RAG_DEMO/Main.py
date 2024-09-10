from nltk.corpus import movie_reviews
from gensim.models import Word2Vec
from flask import Flask, request, jsonify, session
from flask_session import Session
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np
import re

import nltk
import pandas as pd
import requests
import gzip
import io
from nltk.corpus import movie_reviews


# Function to download and load TSV file from IMDb dataset
def download_imdb_data(url):
    response = requests.get(url)
    compressed_file = io.BytesIO(response.content)
    with gzip.GzipFile(fileobj=compressed_file) as uncompressed_file:
        return pd.read_csv(uncompressed_file, delimiter='\t', low_memory=False)


# Function to retrieve IMDb TV show data (2020-2023) and compare format with NLTK movie reviews
def retrieve_imdb_data_and_compare():
    # IMDb dataset URLs
    basics_url = 'https://datasets.imdbws.com/title.basics.tsv.gz'
    ratings_url = 'https://datasets.imdbws.com/title.ratings.tsv.gz'

    # Download datasets
    df_basics = download_imdb_data(basics_url)
    df_ratings = download_imdb_data(ratings_url)

    # Filter for TV shows from 2020-2023
    df_tv_shows = df_basics[(df_basics['titleType'] == 'tvSeries') &
                            (df_basics['startYear'].apply(lambda x: x.isdigit() and 2022 <= int(x) <= 2023))]

    # Merge with ratings data
    df_tv_shows_with_ratings = pd.merge(df_tv_shows, df_ratings, on='tconst', how='left')
    df_tv_shows_filtered = df_tv_shows_with_ratings[
        ['tconst', 'primaryTitle', 'startYear', 'genres', 'averageRating', 'numVotes']]

    # Convert the filtered DataFrame to a format similar to NLTK movie_reviews
    imdb_documents = df_tv_shows_filtered['primaryTitle'].tolist()

    # Load the movie review dataset from NLTK
    nltk_documents = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]

    # Compare shapes/formats
    print(f"IMDb TV Shows (2022-2023): {len(imdb_documents)} documents")
    print(f"NLTK Movie Reviews: {len(nltk_documents)} documents")

    if len(imdb_documents) == len(nltk_documents):
        print("Both datasets have the same number of documents.")
    else:
        print("The datasets do not have the same number of documents.")

    # Optionally return the IMDb documents for further use
    return imdb_documents


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def check_relevance(imdb_df, rag_result, threshold=0.5):
    """
    Check how relevant the RAG result is to the IMDb dataset.

    Parameters:
    - imdb_df (pd.DataFrame): DataFrame containing IMDb dataset with a 'description' column.
    - rag_result (str): The result from the RAG system.
    - threshold (float): The similarity threshold for considering a result valid.

    Returns:
    - bool: True if the similarity is above the threshold, otherwise False.
    - float: The highest similarity score found.
    """

    # Extract descriptions from the IMDb dataset
    descriptions = imdb_df['description'].tolist()

    # Append the RAG result to the descriptions for vectorization
    texts = descriptions + [rag_result]

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Find the maximum similarity score
    max_similarity = cosine_sim.max()

    # Determine if the maximum similarity score meets the threshold
    is_valid = max_similarity >= threshold

    return is_valid, max_similarity



# Start ...

SelctIMDB=False

if SelctIMDB==False:

    # Call the function
    documents = retrieve_imdb_data_and_compare()
else:
    nltk.download('movie_reviews')
    #Load the movie review dataset from NLTK
    documents = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]



# Preprocess documents: Remove punctuation
tokenized_docs = [re.sub(r'[^\w\s]', '', doc.lower()).split() for doc in documents]

# Train Word2Vec model
w2v_model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)


# # Generate embeddings for each document
# document_embeddings = []
# for doc in tokenized_docs:
#     doc_embedding = np.mean([w2v_model.wv[word] for word in doc if word in w2v_model.wv], axis=0)
#     document_embeddings.append(doc_embedding)
# document_embeddings = np.array(document_embeddings)

# Generate embeddings for each document
document_embeddings = []
vector_size = w2v_model.vector_size

for doc in tokenized_docs:
    # Compute the average embedding for each document
    valid_embeddings = [w2v_model.wv[word] for word in doc if word in w2v_model.wv]

    if valid_embeddings:
        doc_embedding = np.mean(valid_embeddings, axis=0)
    else:
        doc_embedding = np.zeros(vector_size)

    document_embeddings.append(doc_embedding)

# Convert the list of embeddings into a numpy array
document_embeddings = np.array(document_embeddings)



# Create a dictionary for document database
vector_db = {i: doc for i, doc in enumerate(documents)}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('flask_app.log'), logging.StreamHandler()])

# Initialize the flask app
app = Flask(__name__)
app.secret_key = '@@GYyour_secret_key123%@@'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define the maximum sequence length
MAX_LENGTH = 1024


# Function to get the embedding of a query
def get_embedding(query_text):
    """
    get_embedding(user_message):

get_embedding is a function or method that takes the user_message and converts it into an embedding. This typically involves:
Tokenization: Breaking down the input text into tokens.
Transformation: Using a model (like BERT, GPT, or other embedding models) to convert these tokens into a high-dimensional vector.
query_embedding:

The resulting variable from the get_embedding function is query_embedding, which is a numerical representation (vector) of the user_message. This vector captures the meaning of the text in a format suitable for further processing by machine learning models.

    :param query_text:
    :return:
    """
    words = query_text.lower().split()
    valid_words = [word for word in words if word in w2v_model.wv]

    if not valid_words:
        raise ValueError("None of the words in the query are in the vocabulary.")

    query_embedding = np.mean([w2v_model.wv[word] for word in valid_words], axis=0)
    return query_embedding


# Function to generate text in chunks
def generate_text_in_chunks(input_text, model, tokenizer, max_length=1024):
    chunks = [input_text[i:i + max_length] for i in range(0, len(input_text), max_length)]
    generated_text = ""

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_length)
        outputs = model.generate(
            **inputs,
            max_length=max_length + 50,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text += tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    logging.info(f"User received message: {user_message}")

    if 'conversation' not in session:
        session['conversation'] = []

    # Update conversation history
    session['conversation'].append({'role': 'user', 'content': user_message})

    # Convert query to embedding (if applicable)
    query_embedding = get_embedding(user_message)

    # Compute cosine similarity and retrieve documents (if applicable)
    similarities = cosine_similarity(query_embedding.reshape(1, 100), document_embeddings)
    top_k_indices = np.argsort(similarities[0])[-5:][::-1]
    retrieved_docs = [vector_db[idx] for idx in top_k_indices]

    # Concatenate retrieved documents and previous conversation
    input_text = " ".join(retrieved_docs) + "%^%^% content: ".join([message['content'] for message in session['conversation']])

    # Use the function to generate text in chunks
    generated_text = generate_text_in_chunks(input_text, model, tokenizer, max_length=MAX_LENGTH)

    logging.info(f"Generated response: {generated_text}")

    # Store bot response in the session
    session['conversation'].append({'role': 'system', 'content': generated_text})

    # Example usage
    # Assuming you have loaded IMDb dataset into imdb_df and have the RAG result as rag_result
    imdb_df = pd.read_csv('TMDB_tv_dataset_v3.csv')  # Example file
    rag_result = generated_text
    is_valid, similarity_score = check_relevance(imdb_df, rag_result, threshold=0.5)
    print(f"Is the result valid? {is_valid}")
    print(f"Similarity score: {similarity_score}")

    return jsonify({'response': generated_text})


if __name__ == '__main__':
    app.run()
