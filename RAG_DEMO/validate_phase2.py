from data_loader import preprocess_documents
from model_service import get_query_embedding
import numpy as np
import ollama

print("--- Starting Phase 2 Validation (Ollama Pivot) ---")

# 0. Check Ollama Model
print("\n[0/3] Checking if nomic-embed-text is available...")
try:
    ollama.show('nomic-embed-text')
    print("SUCCESS: nomic-embed-text is available.")
except:
    print("WARNING: nomic-embed-text might not be pulled yet. Attempting pull...")
    ollama.pull('nomic-embed-text')

# 1. Test Data Loading & Encoding
print("\n[1/3] Running preprocess_documents()...")
print("(This will scrape data, use Ollama for embeddings, and index in SimpleVectorStore)")
collection = preprocess_documents()
count = len(collection.documents)
print(f"SUCCESS: SimpleVectorStore Collection contains {count} documents.")

# 2. Test Query Encoding
print("\n[2/3] Testing Query Encoding (nomic-embed-text)...")
query = "diabetic retinopathy treatment"
query_vec = get_query_embedding(query)
print(f"SUCCESS: Query embedding shape: {query_vec.shape}")

# 3. Test Retrieval
print("\n[3/3] Testing Retrieval...")
results = collection.query(
    query_embeddings=[query_vec.tolist()],
    n_results=1
)
doc_snippet = results['documents'][0][0][:200]
print(f"SUCCESS: Retrieved document snippet:\n{doc_snippet}...")

print("\n--- Phase 2 Verification COMPLETE ---")
