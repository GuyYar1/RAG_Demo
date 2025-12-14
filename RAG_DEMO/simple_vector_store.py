import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimpleVectorStore:
    def __init__(self, storage_path="vector_store.pkl"):
        self.storage_path = storage_path
        self.documents = []
        self.embeddings = []
        self.ids = []
        self.metadatas = []
        self._load()

    def add(self, documents, embeddings, ids, metadatas=None):
        """
        Add documents and their embeddings to the store.
        """
        # Ensure embeddings is a numpy array for easier handling
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
            
        self.documents.extend(documents)
        if len(self.embeddings) == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            
        self.ids.extend(ids)
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{} for _ in range(len(documents))])
            
        self._save()
        print(f"SimpleVectorStore: Added {len(documents)} items. Total: {len(self.documents)}")

    def query(self, query_embeddings, n_results=3):
        """
        Find top k nearest neighbors for query embeddings.
        """
        if len(self.embeddings) == 0:
            return {"documents": [], "ids": [], "metadatas": [], "distances": []}

        # Calculate cosine similarity
        # query_embeddings shape: (n_queries, dim)
        # self.embeddings shape: (n_docs, dim)
        similarities = cosine_similarity(query_embeddings, self.embeddings) 
        
        results = {
            "documents": [],
            "ids": [],
            "metadatas": [],
            "distances": [] # We return distances for compatibility (1 - similarity)
        }

        for i in range(len(query_embeddings)):
            # Get top k indices
            # argsort returns typically ascending, so we take last n and reverse
            top_k_indices = np.argsort(similarities[i])[-n_results:][::-1]
            
            results["documents"].append([self.documents[idx] for idx in top_k_indices])
            results["ids"].append([self.ids[idx] for idx in top_k_indices])
            results["metadatas"].append([self.metadatas[idx] for idx in top_k_indices])
            results["distances"].append([1 - similarities[i][idx] for idx in top_k_indices])
            
        return results

    def _save(self):
        with open(self.storage_path, 'wb') as f:
            data = {
                "documents": self.documents,
                "embeddings": self.embeddings,
                "ids": self.ids,
                "metadatas": self.metadatas
            }
            pickle.dump(data, f)

    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.embeddings = data["embeddings"]
                self.ids = data["ids"]
                self.metadatas = data["metadatas"]
            print(f"SimpleVectorStore: Loaded {len(self.documents)} items from {self.storage_path}")

    def reset(self):
        self.documents = []
        self.embeddings = []
        self.ids = []
        self.metadatas = []
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
