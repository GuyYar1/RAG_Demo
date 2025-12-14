from ollama_service import generate_response

def generate_text_in_chunks(input_text, query, model="mistral", max_length=512):
    """
    Generate text using Ollama (BioMistral/Mistral).
    Arguments 'max_length' etc kept for signature compatibility but unused.
    """
    # Simply call the Ollama service
    return generate_response(context=input_text, query=query, model=model)

def rerank_response(response, query):
    """
    Pass-through. Reranking is now handled by the LLM's instruction following.
    """
    return response

def extract_sentences_with_keyword(text, keyword):
    """
    Utility helper.
    """
    sentences = text.split('.')
    filtered = [s.strip() for s in sentences if keyword.lower() in s.lower()]
    return '. '.join(filtered)


