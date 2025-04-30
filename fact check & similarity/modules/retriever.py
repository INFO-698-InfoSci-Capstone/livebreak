# retriever.py

import numpy as np
from faiss_indexer import FaissIndexer
from web_search import web_search  # now imported from separate module

# Initialize FAISS
indexer = FaissIndexer()

def retrieve_evidence(claim, top_k_faiss=5, top_k_web=5):
    """
    Retrieve evidence for a claim: first from FAISS, fallback to web search if needed.
    """
    faiss_results = indexer.search(claim, k=top_k_faiss)
    faiss_texts = [r["doc"] for r in faiss_results]
    avg_faiss_score = np.mean([r["score"] for r in faiss_results]) if faiss_results else 0

    if avg_faiss_score < 0.5 or len(faiss_texts) < 3:
        # FAISS evidence weak → fallback to web search
        web_results = web_search(claim, top_k=top_k_web)
        return web_results
    else:
        return faiss_texts
