from gensim.models import Word2Vec
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from data_loader import preprocess_documents



def data_oN_model(model):
    global vocab_size
    # Assuming w2v_model is loaded
    vocab_size = len(model.wv.index_to_key)
    print(f"Vocabulary Size: {vocab_size}")
    # Assuming w2v_model is loaded
    embedding_dim = model.wv.vector_size
    print(f"Embedding Dimension: {embedding_dim}")

# Function to train or load a Word2Vec model
# Function to train or load a Word2Vec model
def load_w2v_model():
    print("load_w2v_model")
    # Load the preprocessed data
    tokenized_docs, document_embeddings, vector_db, w2v_model = preprocess_documents()

    # w2v_model is Trained with tokenized_docs and  load Word2Vec model

    # print data on model
    data_oN_model(w2v_model)

    return w2v_model


# Function to load GPT-2 model
def load_gpt2_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    return tokenizer, model


# Function to get the embedding of a query
def get_embedding(query_text, w2v_model):
    words = query_text.lower().split()
    valid_words = [word for word in words if word in w2v_model.wv]

    if not valid_words:
        raise ValueError("None of the words in the query are in the vocabulary.")

    query_embedding = np.mean([w2v_model.wv[word] for word in valid_words], axis=0)
    return query_embedding