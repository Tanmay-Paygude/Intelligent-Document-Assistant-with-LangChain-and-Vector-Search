
import numpy as np
import faiss

class FaissVectorStore:
    """
    Cosine similarity search using FAISS IndexFlatIP.
    Vectors are expected to be L2-normalized (we normalize again defensively).
    """
    def __init__(self, vectors: np.ndarray, texts):
        if len(vectors.shape) != 2:
            raise ValueError("vectors must be 2D array [n, dim]")
        self.texts = list(texts)
        # Normalize for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        self.vectors = vectors / norms
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors.astype("float32"))

    def search(self, query_vec, k=4):
        q = np.array(query_vec, dtype="float32").reshape(1, -1)
        # normalize query
        q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        scores, idxs = self.index.search(q, k)
        idxs = idxs[0]
        return [self.texts[i] for i in idxs if i >= 0]
