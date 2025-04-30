import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from hybrid_similarity import create_custom_sbert, embed_long_doc, jaccard_similarity

# Get Top-k corpus embeddings most similar to the article 
def get_top_k_similar_docs(article_vector, corpus_vectors, k=5):
    scores = cosine_similarity([article_vector], corpus_vectors)[0]
    top_k_idx = np.argsort(scores)[-k:][::-1]
    top_k_vectors = corpus_vectors[top_k_idx]
    return top_k_vectors, top_k_idx, scores[top_k_idx]

# Compare article to top-k aggregated corpus subset using hybrid score
def compare_article_to_top_k_corpus(article_text, corpus_texts, model, k=5):
    # Embed the article
    article_vector = embed_long_doc(article_text, model)

    # Embed the corpus documents
    corpus_vectors = model.encode(corpus_texts, normalize_embeddings=True)

    # Find top-k similar vectors
    top_k_vectors, top_k_indices, top_k_embed_scores = get_top_k_similar_docs(article_vector, corpus_vectors, k)

    # Aggregate top-k embeddings
    top_k_mean_vector = np.mean(top_k_vectors, axis=0)

    # Compute similarity scores
    emb_sim = float(cosine_similarity([article_vector], [top_k_mean_vector])[0][0])
    jaccard_scores = [jaccard_similarity(article_text, corpus_texts[i]) for i in top_k_indices]
    avg_jaccard = float(np.mean(jaccard_scores))

    # Final score - can change the coefficients
    hybrid_score = 0.7 * emb_sim + 0.3 * avg_jaccard

    # Log top-k docs and scores
    top_k_logs = []
    for idx, embed_score, jac_score in zip(top_k_indices, top_k_embed_scores, jaccard_scores):
        top_k_logs.append({
            "doc_index": int(idx),
            "doc_text": corpus_texts[idx],
            "embedding_similarity": float(embed_score),
            "jaccard_similarity": float(jac_score)
        })

    return {
        "embedding_similarity": emb_sim,
        "average_jaccard_similarity": avg_jaccard,
        "combined_hybrid_score": hybrid_score,
        "top_k_results": top_k_logs
    }


# === Example usage ===
if __name__ == "__main__":
    model = create_custom_sbert()

    article = "OpenAI announced GPT-4, capable of reasoning through complex prompts."
    corpus = [
        "OpenAI released GPT-3 in 2020.",
        "Machine learning models are evolving fast.",
        "GPT-4 improves over its predecessor in reasoning and language tasks.",
        "Weather in Paris is sunny today.",
        "Transformers are used widely in NLP."
    ]

    result = compare_article_to_top_k_corpus(article, corpus, model, k=3)

    print(f"\n🔍 Final Hybrid Score: {result['combined_hybrid_score']:.4f}")
    print("📄 Top-k Most Similar Documents:")
    for entry in result["top_k_results"]:
        print(f"\n→ Doc #{entry['doc_index']}")
        print(f"   Embedding Sim: {entry['embedding_similarity']:.4f}")
        print(f"   Jaccard Sim:   {entry['jaccard_similarity']:.4f}")
        print(f"   Content: {entry['doc_text']}")