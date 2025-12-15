import ollama
import numpy as np
import logging

# Initialize logger for model_service module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Or DEBUG if you want more details
ch = logging.StreamHandler()  # Logs to the console
ch.setLevel(logging.INFO)  # Set the logging level for the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # Log format
ch.setFormatter(formatter)
logger.addHandler(ch)

# Global variables for caching models
# For Ollama, the model is managed by the service, so we just define the name
EMBEDDING_MODEL = "nomic-embed-text"

def get_query_embedding(query_text):
    """
    Generate embedding for a query using Ollama (nomic-embed-text).
    """
    try:
        logger.info(f"Generating embedding for query: {query_text[:50]}...")  # Log the first 50 chars of the query
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query_text)
        embedding = response["embedding"]
        return np.array(embedding)
    except Exception as e:
        logger.error(f"Error generating embedding via Ollama: {e}", exc_info=True)
        # Return empty array or raise, depending on desired fail state
        return np.zeros(768)

# Deprecated functions for compatibility
def load_w2v_model():
    pass

def load_models():
    pass
