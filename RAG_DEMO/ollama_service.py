import ollama

def generate_response(context: str, query: str, model: str = "mistral") -> str:
    """
    Generate a response using the local Ollama LLM.
    Attempts the requested model first, falls back to 'tinyllama' if it fails.
    """
    prompt = f"""You are a helpful and accurate medical assistant. 
Use the following context to answer the user's question. 
Provide a detailed and comprehensive answer.
If the answer is not in the context, say you don't know, do not hallucinate.

Context:
{context}

Question: 
{query}

Answer:"""
    
    # Helper to call API
    def call_ollama(model_name):
        return ollama.chat(model=model_name, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])

    try:
        response = call_ollama(model)
        return response['message']['content']
    except Exception as e:
        print(f"Error calling Ollama with '{model}': {e}")
        if model != "tinyllama":
            print("Attempting fallback to 'tinyllama'...")
            try:
                response = call_ollama("tinyllama")
                return response['message']['content']
            except Exception as e2:
                 print(f"Error calling Ollama with fallback: {e2}")
                 
        return "I apologize, but I encountered an error generating the response."
