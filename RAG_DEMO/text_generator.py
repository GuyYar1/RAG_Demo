from ollama_service import generate_response

def generate_text_in_chunks(input_text, query, model="mistral", max_length=512):
    """
    Generate text using Ollama (BioMistral/Mistral).
    Arguments 'max_length' etc kept for signature compatibility but unused.
    """
    # Simply call the Ollama service
    return generate_response(context=input_text, query=query, model=model)

def validate_severe_response(response: str, query: str) -> dict:
    """
    Validate if response for severe queries contains required urgent information.
    Returns: {'valid': bool, 'missing': list of missing elements}
    """
    query_lower = query.lower()
    response_lower = response.lower()
    
    # Only validate for severe queries
    severe_terms = ['severe', 'proliferative', 'PDR', 'severe NPDR']
    if not any(term.lower() in query_lower for term in severe_terms):
        return {'valid': True, 'missing': []}
    
    required_elements = {
        'referral': ['refer', 'referral', 'ophthalmologist', 'specialist', 'retina specialist'],
        'treatment': ['PRP', 'laser', 'anti-VEGF', 'photocoagulation', 'ranibizumab', 'aflibercept', 'bevacizumab'],
        'urgency': ['immediate', 'urgent', 'weeks', 'within', 'promptly', 'as soon as']
    }
    
    missing = []
    for category, keywords in required_elements.items():
        if not any(keyword.lower() in response_lower for keyword in keywords):
            missing.append(category)
    
    is_valid = len(missing) == 0
    
    return {'valid': is_valid, 'missing': missing}


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


