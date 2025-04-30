# web_search.py
import random

def web_search(query, top_k=5):
    """
    Simulates a web search.

    Args:
        query (str): search query (claim).
        top_k (int): Number of results.

    Returns:
        List[str]: List of snippets/articles relevant to the query.
    """
    # Replace this with web search or LLM API calls
    fake_database = [
        
    ]

    # Randomly pick some snippets - replace with top results
    results = random.sample(fake_database, min(top_k, len(fake_database)))
    return results
