from datasets import load_dataset
from faiss_indexer import FaissIndexer

# Uncomment to build FAISS Index at once
'''
def bootstrap_faiss_from_wikipedia(n_samples=1000):
    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=False)
    
    # Grab midsized wikipedia articles
    texts = []
    for item in dataset.select(range(n_samples)):
        text = item["text"]
        if text and len(text) > 100 and len(text) < 1000:
            texts.append(text.strip())
    
    print(f"Collected {len(texts)} documents for bootstrapping FAISS.")

    indexer = FaissIndexer()
    indexer.add_documents(texts)
    print("FAISS index bootstrapped!")

if __name__ == "__main__":
    bootstrap_faiss_from_wikipedia()
'''

from datasets import load_dataset
from faiss_indexer import FaissIndexer

def bootstrap_faiss_from_wikipedia(n_samples=1000):
    # Streaming dataset (iterator)
    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

    texts = []
    for i, item in enumerate(dataset):
        text = item["text"]
        if text and 100 < len(text) < 1000:
            texts.append(text.strip())
        if len(texts) >= n_samples:
            break

    print(f"✅ Collected {len(texts)} documents for bootstrapping FAISS.")

    indexer = FaissIndexer()
    indexer.add_documents(texts)
    print("✅ FAISS index bootstrapped!")

if __name__ == "__main__":
    bootstrap_faiss_from_wikipedia()

