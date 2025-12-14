import ollama
import numpy as np

# Global variables for caching models
# For Ollama, the model is managed by the service, so we just define the name
EMBEDDING_MODEL = "nomic-embed-text"

def get_query_embedding(query_text):
    """
    Generate embedding for a query using Ollama (nomic-embed-text).
    """
    try:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query_text)
        embedding = response["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"Error generating embedding via Ollama: {e}")
        # Return empty array or raise, depending on desired fail state
        return np.zeros(768) 

# Deprecated functions for compatibility
def load_w2v_model():
    pass

def load_models():
    pass