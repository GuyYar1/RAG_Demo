import chromadb
import numpy as np

print("Testing ChromaDB...")

try:
    client = chromadb.PersistentClient(path="./chroma_db_test")
    print("Client initialized.")
    
    try:
        client.delete_collection("test_col")
    except:
        pass
        
    collection = client.create_collection("test_col")
    print("Collection created.")
    
    docs = ["hello world"]
    # 768 dim vector
    emb = [list(np.zeros(768))] 
    ids = ["id1"]
    
    print("Adding document...")
    collection.add(
        documents=docs,
        embeddings=emb,
        ids=ids
    )
    print("Document added.")
    
    print("Querying...")
    res = collection.query(
        query_embeddings=emb,
        n_results=1
    )
    print("Query success.")
    print(res)

except Exception as e:
    print(f"FAILURE: {e}")
