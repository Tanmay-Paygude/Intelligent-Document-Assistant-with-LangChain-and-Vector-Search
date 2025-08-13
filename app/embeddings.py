
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def chunk_text(self, text: str, max_words: int = 180) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i+max_words]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def embed_texts(self, chunks: List[str]) -> np.ndarray:
        vecs = self.model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
        return np.array(vecs, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], show_progress_bar=False, normalize_embeddings=True)
        return np.array(vec, dtype="float32")[0]

# Optional: OpenAI for answer synthesis
import os
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

class OpenAIAnswerer:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if (OpenAI and self.api_key) else None

    def is_ready(self) -> bool:
        return self.client is not None

    def answer(self, prompt: str) -> str:
        if not self.is_ready():
            return "OpenAI client not initialized."
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
