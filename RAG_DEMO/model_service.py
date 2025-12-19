import numpy as np
import logging
import os
from sentence_transformers import SentenceTransformer

# Initialize logger for model_service module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Use sentence-transformers for embeddings (works on HF Spaces)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embedding_model = None

def get_embedding_model():
    """Load embedding model (cached)"""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model

def get_query_embedding(query_text):
    """
    Generate embedding for a query using sentence-transformers.
    """
    try:
        logger.info(f"Generating embedding for query: {query_text[:50]}...")
        model = get_embedding_model()
        embedding = model.encode(query_text, convert_to_numpy=True)
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}", exc_info=True)
        return np.zeros(384)  # all-MiniLM-L6-v2 outputs 384-dim vectors

# Deprecated functions for compatibility
def load_w2v_model():
    pass

def load_models():
    pass
