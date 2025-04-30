# hybrid_similarity.py

from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Custom SBERT model 
def create_custom_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Embed long documents by chunking and averaging
def embed_long_doc(text, model, max_chunk_tokens=200):
    """
    Splits a long text into smaller chunks, embeds each chunk,
    and returns the mean-pooled vector.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len((current_chunk + " " + sentence).split()) <= max_chunk_tokens:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    embeddings = model.encode(chunks, normalize_embeddings=True)
    return np.mean(embeddings, axis=0)

def jaccard_similarity(a, b):
    """
    Computes Jaccard similarity between two texts based on word overlap.
    """
    a_tokens = set(re.findall(r'\w+', a.lower()))
    b_tokens = set(re.findall(r'\w+', b.lower()))
    
    if not a_tokens or not b_tokens:
        return 0.0
    
    intersection = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(intersection) / len(union)
