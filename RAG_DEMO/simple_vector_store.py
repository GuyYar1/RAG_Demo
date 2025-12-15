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

    def add(self, documents, embeddings, ids, metadatas, weights=None):
        """
        Adds documents to the vector store with optional weights.
        `weights` is a list of weight values corresponding to each document.
        """
        for i, doc in enumerate(documents):
            weight = weights[i] if weights else 1  # Default weight is 1
            self.documents.append(doc)
            self.embeddings.append(embeddings[i] * weight)  # Apply weight to embeddings
            self.ids.append(ids[i])
            self.metadatas.append(metadatas[i])

        self._save()
        print(f"SimpleVectorStore: Added {len(documents)} items. Total: {len(self.documents)}")

    def query(self, query_embeddings, n_results=3):
        """
        Find top k nearest neighbors for query embeddings.
        """
        if len(self.embeddings) == 0:
            return {"documents": [], "ids": [], "metadatas": [], "distances": []}

        # Calculate cosine similarity
        similarities = cosine_similarity(query_embeddings, self.embeddings)
        
        results = {
            "documents": [],
            "ids": [],
            "metadatas": [],
            "distances": []  # We return distances for compatibility (1 - similarity)
        }

        for i in range(len(query_embeddings)):
            # Get top k indices
            top_k_indices = np.argsort(similarities[i])[-n_results:][::-1]
            
            weighted_documents = []
            weighted_scores = []
            weighted_metadatas = []
            
            for idx in top_k_indices:
                weight = 1  # Default weight is 1
                if self.metadatas[idx].get('source') == 'offline':  # Give offline documents higher weight
                    weight = 2  # You can adjust this value as needed
                
                weighted_documents.append(self.documents[idx])
                weighted_scores.append(similarities[i][idx] * weight)
                weighted_metadatas.append(self.metadatas[idx])
            
            results["documents"].append(weighted_documents)
            results["ids"].append([self.ids[idx] for idx in top_k_indices])
            results["metadatas"].append(weighted_metadatas)
            results["distances"].append([1 - score for score in weighted_scores])  # Convert similarity to distance

        return results  # This line must be **outside** the `for` loop

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
