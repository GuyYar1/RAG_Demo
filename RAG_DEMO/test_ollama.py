import ollama
import time

print("Testing Ollama embedding for 'nomic-embed-text'...")

try:
    # 1. Simple test
    start = time.time()
    resp = ollama.embeddings(model="nomic-embed-text", prompt="hello world")
    print(f"SUCCESS: Embedding generated in {time.time()-start:.2f}s")
    print(f"Vector length: {len(resp['embedding'])}")
    
    # 2. Longer text test
    long_text = "medical " * 100
    start = time.time()
    resp = ollama.embeddings(model="nomic-embed-text", prompt=long_text)
    print(f"SUCCESS: Long text embedding generated in {time.time()-start:.2f}s")

except Exception as e:
    print(f"FAILURE: {e}")
