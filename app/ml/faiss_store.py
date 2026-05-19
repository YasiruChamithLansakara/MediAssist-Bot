import faiss
import numpy as np
import pickle

class FAISSStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []

    # -----------------------------
    # ADD DATA
    # -----------------------------
    def add(self, embeddings, documents):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.documents.extend(documents)

    # -----------------------------
    # SEARCH
    # -----------------------------
    def search(self, query_vector, top_k=5):
        vector = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(vector, top_k)

        results = []
        for i in indices[0]:
            if 0 <= i < len(self.documents):
                results.append(self.documents[i])

        return results

    # -----------------------------
    # SAVE INDEX
    # -----------------------------
    def save(self, path):
        faiss.write_index(self.index, path + ".index")
        with open(path + ".pkl", "wb") as f:
            pickle.dump(self.documents, f)

    # -----------------------------
    # LOAD INDEX
    # -----------------------------
    def load(self, path):
        self.index = faiss.read_index(path + ".index")
        with open(path + ".pkl", "rb") as f:
            self.documents = pickle.load(f)