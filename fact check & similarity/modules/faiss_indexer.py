# faiss_indexer.py

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class FaissIndexer:
    def __init__(self, index_path="faiss.index", corpus_path="corpus.pkl"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.index = None
        self.corpus = []
        self._load()

    def _load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.corpus_path, "rb") as f:
                self.corpus = pickle.load(f)
        else:
            dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(dim)

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.corpus_path, "wb") as f:
            pickle.dump(self.corpus, f)

    def add_documents(self, docs):
        embeddings = self.model.encode(docs, normalize_embeddings=True)
        self.index.add(embeddings)
        self.corpus.extend(docs)
        self.save()

    def search(self, query, k=5):
        query_emb = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.array(query_emb), k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            results.append({"doc": self.corpus[idx], "score": float(score)})
        return results
