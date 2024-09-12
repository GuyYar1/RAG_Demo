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
    tokenized_docs, document_embeddings, vector_db, w2v_model = preprocess_documents(True)

    # w2v_model is Trained with tokenized_docs and  load Word2Vec model

    # print data on model
    data_oN_model(w2v_model)

    return w2v_model


# Function to load GPT-2 model
def load_gpt2_model():
    """
    Tokenizer: Converts text into tokens that the model can process.
    Model: The actual GPT-2 model that performs tasks like text generation based on the input tokens.

    A tokenizer is in charge of preparing the inputs for a model. The library contains tokenizers for all the models. Most of the tokenizers are available in two flavors: a full python implementation and a
    “Fast” implementation based on the Rust library 🤗 Tokenizers. The “Fast” implementations allows:
    :return:
    """

    # Load the pre-trained GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Load the pre-trained GPT-2 model
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