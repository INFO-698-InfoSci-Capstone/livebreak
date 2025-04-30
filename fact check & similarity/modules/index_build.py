from faiss_indexer import FaissIndexer

# Load existing index (or create new one)
indexer = FaissIndexer()

# Prepare new documents
new_docs = [
  
]

# Add to index
indexer.add_documents(new_docs)

print("Added new documents to FAISS index!")