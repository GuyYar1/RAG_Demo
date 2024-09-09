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


# Call the function
documents = retrieve_imdb_data_and_compare()


#nltk.download('movie_reviews')
# Load the movie review dataset from NLTK
# documents = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]



# Preprocess documents: Remove punctuation
tokenized_docs = [re.sub(r'[^\w\s]', '', doc.lower()).split() for doc in documents]

# Train Word2Vec model
w2v_model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

# Generate embeddings for each document
document_embeddings = []
for doc in tokenized_docs:
    doc_embedding = np.mean([w2v_model.wv[word] for word in doc if word in w2v_model.wv], axis=0)
    document_embeddings.append(doc_embedding)
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

    return jsonify({'response': generated_text})


if __name__ == '__main__':
    app.run()

# from nltk.corpus import movie_reviews
# from gensim.models import Word2Vec
# from flask import Flask, request, jsonify,session
# from flask_session import Session
# from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from flask_ngrok import run_with_ngrok
# import threading
# import time
# import requests
# import os
# import logging
# import numpy as np
# import json
# import numpy
# import nltk
# import re
#
# nltk.download('movie_reviews')
#
# #This line of code reads in the movie review dataset from NLTK.
# # 1.movie_reviews.fileids(): This returns a list of all file IDs in the movie review dataset.
# # 2.movie_reviews.raw(fileid): This reads the raw text of the movie review corresponding to a particular file ID.
# # [ ... for fileid in movie_reviews.fileids()]: This is a list comprehension that iterates over all file IDs and applies the movie_reviews.raw() function to each of them
#
# documents = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
#
# ## "\n'," - sign for new item in the list.
# documents[:2]
#
# #Preprocess documents  - Remove Punctuation: Punctuation could interfere with tokenization, so you might want to remove it.
# tokenized_docs = [re.sub(r'[^\w\s]', '', doc.lower()).split() for doc in documents]
#
# # This code trains a Word2Vec model using the Gensim library.
# #
# # 1.sentences=tokenized_docs: This specifies the input data for the model. In this case, it's tokenized_docs, which is a list of lists, where each inner list represents a sentence and contains the individual words (tokens) of that sentence.
# #
# # vector_size=100: This determines the dimensionality of the word vectors. Each word will be represented by a vector of 100 dimensions.
# #
# # explnation :The vector_size parameter in Word2Vec determines the number of dimensions used to represent each word as a vector. This dimension represents the features or characteristics of words. The choice of vector size can impact the model's performance and the quality of the word embeddings.
# # 2.1 Lower dimensional vectors capture less information about the word and its context. 2.2. May lead to faster training and reduced memory usage. 2.3.Could result in lower accuracy, especially for tasks that require fine-grained semantic distinctions.
# #
# # window=5: This defines the size of the context window. The context window determines the number of words before and after a target word that are considered when training the model. In this case, a window of 5 means that 5 words before and 5 words after the target word are included.
# #
# # min_count=1: This sets the minimum frequency for words to be included in the model's vocabulary. Words that appear less than this number of times will be ignored.
# #
# # workers=4: This specifies the number of worker threads to use for training the model. This can speed up the training process, especially for large datasets.
# #
# #
#
# #Train Word2Vec model
# w2v_model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)
#
# # This code calculates embeddings for each document by averaging the word embeddings of the words present in the document.
# # 1.document_embeddings = []: Initializes an empty list to store the document embeddings.
# # 2.for doc in tokenized_docs:: Iterates through each document in the tokenized_docs list.
# # 3.doc_embedding = np.mean([w2v_model.wv[word] for word in doc if word in w2v_model.wv], axis=0): Calculates the embedding for the current document (doc) by taking the average of the word embeddings of words present in the document's vocabulary.
# # 4.document_embeddings.append(doc_embedding): Appends the calculated document embedding to the document_embeddings list.
# # 5. document_embeddings = np.array(document_embeddings): Converts the document_embeddings list to a NumPy array for efficient computation.
#
# #Generate embeddings for each document
# document_embeddings = []
# for doc in tokenized_docs:
#     doc_embedding = np.mean([w2v_model.wv[word] for word in doc if word in w2v_model.wv], axis=0)
#     document_embeddings.append(doc_embedding)
# document_embeddings = np.array(document_embeddings)
#
# len(documents)
# len(document_embeddings)
#
# #Example of Tuples:
#
# # my_list_of_tuples = [(1, "apple"), (2, "banana"), (3, "cherry")]
# # print(my_list_of_tuples)
# #
# # # Get the size of the list of tuples (number of tuples)
# # list_size = len(my_list_of_tuples)
# # print(f"Size of the list of tuples: {list_size}")
# #
# # # Get the size of a specific tuple (number of elements in the tuple)
# # specific_tuple_size = len(my_list_of_tuples[0])  # Accessing the first tuple
# # print(f"Size of the first tuple: {specific_tuple_size}")
#
#
# a = list(enumerate(documents))
# print (f"list of Tuples which are list of  Docs", len(a)) # list of tuple
#
# first_tuple = a[0] # take the first element of the list . take the first tuple
# print(f"Tuple size element",len(first_tuple))
# first_element_in_first_tuple = first_tuple[0]  ## eqvivalent to a[0][0]
# print(f"first_element_in_first_tuple ", first_element_in_first_tuple)
#
# second_element_in_first_tuple = first_tuple[1] # # eqvivalent to a[0][1]
# print(f"second_element_in_first_tuple ", second_element_in_first_tuple)
#
# print(f"Tuple element, in the second location ,its size is: ", len(second_element_in_first_tuple))
# print ("There is no len to the:first_element_in_first_tuple due to int type")
#
# ## Document database
# vector_db = {i: doc for i, doc in enumerate(documents)}  ## creates a dictionary.
# len(vector_db)
#
#
# # *document_embeddings Vs. documents * represent the same data but in different forms.
# # documents: Contains the raw text of movie reviews. document_embeddings: Contains numerical representations of those movie reviews.
# # Common: Both variables represent the movie review dataset.
# # Difference: documents stores the data in its original text form, while document_embeddings stores the data as numerical vectors.
# # This code snippet configures logging for a Flask application. Before I explain that, let's address your question about Flask.
# # Flask is a lightweight and flexible web framework for Python. It's designed to make it quick and easy to build web applications, APIs, and microservices. Here's a breakdown of its key features and benefits:
# # Lightweight: Flask has a small core and relies on extensions for additional functionality, keeping it minimal and efficient. Flexible: It doesn't impose strict rules on project structure, giving developers freedom to organize their applications as they see fit. Easy to learn: Flask has a simple and intuitive API, making it beginner-friendly. Large community: It has a vibrant community of users and developers, offering ample resources, tutorials, and support. Extensible: A wide range of extensions are available to add features like database integration, authentication, and more.
#
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
#                     handlers=[logging.FileHandler('flask_app.log'), logging.StreamHandler()
#                               ])
#
# # logging.basicConfig(...): This line configures the root logger, which is the base logger for all loggers in your application. level=logging.INFO: This sets the minimum logging level to INFO. Only log messages with severity INFO or higher (WARNING, ERROR, CRITICAL) will be recorded. format='%(asctime)s - %(levelname)s - %(message)s %(message)s': This defines the format for log messages. It includes: %(asctime)s: Timestamp %(levelname)s: Log level (e.g., INFO, WARNING) %(message)s: The actual log message (repeated twice in this case) handlers=[...]: This specifies a list of handlers that will process log messages. The code includes two handlers: logging.FileHandler('flask_app.log'): This handler writes log messages to a file named "flask_app.log". logging.StreamHandler(): This handler prints log messages to the console (standard output). This configuration ensures that log messages are both displayed on the console for immediate feedback and saved to a file for later analysis.
#
# # Initialize the flask app
# app = Flask(__name__)
# # import os # app.secret_key = os.environ.get('SECRET_KEY')
# app.secret_key = '@@GYyour_secret_key123%@@'
# #app.config['SESSION_TYPE'] = 'filesystem': This line configures the session type to use the filesystem for storing session data.
#
# #Alternatives: Other session types include 'redis', 'memcached', and 'mongodb'.
# # Filesystem sessions: For development or small applications, using the filesystem is convenient. However, it might not be the best choice for production as it can lead to performance issues with many concurrent users.
# app.config['SESSION_TYPE'] = 'filesystem'
# Session(app)
#
#
# # After adding those lines to your Flask app, you'll have enabled session management. This means your application can now:
# # Store data for each user: You can store user-specific information like login status, shopping cart contents, or preferences within a session. This data is accessible across different requests from the same user. Maintain user state: Sessions allow you to track whether a user is logged in or has performed certain actions, creating a more interactive experience. How to interact with sessions in Flask:
#
# # expalin the text_generator:
# # Here's a simplified analogy: Imagine you have a magic box (the text_generator) that can write stories. You give it the beginning of a sentence (the prompt), and the box uses its knowledge of stories (the GPT-2 model) to write the rest of the story.
# # This line is a concise way to access the powerful text generation capabilities of GPT-2 within your code.
# # his line of code creates a text generation pipeline using the GPT-2 model from the Hugging Face Transformers library.
# # In essence, this line sets up a ready-to-use tool for generating text based on the GPT-2 model. You can then provide a prompt or starting text to the text_generator and it will generate continuation text based on the patterns it learned during training.
#
#
# # The error indicates that the input sequence length for your text generation model exceeds the maximum sequence length
# # supported by the model, which is causing indexing errors. For GPT-2, the maximum sequence length is 1024 tokens.
# # Here’s how you can handle this issue:
#
# # Initialize the tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')
#
# # Define the maximum sequence length
# MAX_LENGTH = 1024
#
# text_generator = pipeline("text-generation", model="gpt2")
#
# # Function to get the embedding of a query
#
# def get_embedding(query_text):
#     words = query_text.lower().split()
#
#     # Check for words not in the vocabulary
#     valid_words = [word for word in words if word in w2v_model.wv]
#
#     if not valid_words:
#         raise ValueError("None of the words in the query are in the vocabulary.")
#
#     # Compute the embedding for the query
#     query_embedding = np.mean([w2v_model.wv[word] for word in valid_words], axis=0)
#
#     return query_embedding
#
#
# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.json.get('message')
#     logging.info(f"User received message: {user_message}")
#
#     if 'conversation' not in session:
#         session['conversation'] = []
#
#     # Update conversation history
#     session['conversation'].append({'role': 'user', 'content': user_message})
#
#     # Convert query to embedding (if applicable)
#     query_embedding = get_embedding(user_message)
#
#     # Compute cosine similarity and retrieve documents (if applicable)
#     similarities = cosine_similarity(query_embedding.reshape(1, 100), document_embeddings)
#     top_k_indices = np.argsort(similarities[0])[-5:][::-1]
#     retrieved_docs = [vector_db[idx] for idx in top_k_indices]
#
#     # Concatenate retrieved documents and previous conversation
#     input_text = " ".join(retrieved_docs) + " ".join([message['content'] for message in session['conversation']])
#
#     # Tokenize input text and ensure it fits within the model's maximum length
#     inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)
#     print(f"inputs", inputs)
#     # Calculate the maximum number of new tokens to generate
#     max_new_tokens = MAX_LENGTH - inputs['input_ids'].size(1)
#     print(f"max_new_tokens",max_new_tokens)
#     if max_new_tokens <= 0:
#         # If input length already exceeds or is close to the limit, adjust or handle accordingly
#         return jsonify({'response': "Input text is too long, please shorten it."})
#
#     # Generate response with controlled max_new_tokens
#     outputs = model.generate(
#         **inputs,
#         max_length=MAX_LENGTH + 50,  # Adjust based on needs
#         max_new_tokens=50,
#         pad_token_id=tokenizer.eos_token_id
#     )
#
#     # Decode the generated tokens to text
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     logging.info(f"Generated response: {generated_text}")
#
#     # Store bot response in the session
#     session['conversation'].append({'role': 'system', 'content': generated_text})
#
#     return jsonify({'response': generated_text})
#
#
# # imagine query_embedding is like a tool that can cut a piece of cake (embedding). You give it instructions on where to cut by providing the starting and ending points (1 and -1 in this case). The function then returns the slice of cake you requested (sub-embedding).
#
# if __name__ == '__main__':
#     app.run()