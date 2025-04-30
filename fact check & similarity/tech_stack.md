
# Fact Verification and Similarity Scoring System Technology Stack

---

## Programming Language

- **Python 3.11**

---

## NLP & Linguistic Processing

| Tool / Library      | Purpose                                     |
|---------------------|---------------------------------------------|
| `spaCy`             | Sentence segmentation, POS tagging, NER     |
| `en_core_web_sm`    | Pretrained English NLP pipeline in spaCy    |

---

## Sentence Embeddings & Similarity

| Tool / Library                 | Purpose                                      |
|--------------------------------|----------------------------------------------|
| `sentence-transformers`       | Embedding sentences using SBERT              |
| `all-MiniLM-L6-v2`            | Lightweight SBERT model for encoding         |
| `scikit-learn` (`cosine_similarity`) | Cosine similarity scoring            |
| `numpy`                       | Numerical operations, vector math            |

---

## Vector Search & Indexing

| Tool / Library | Purpose                                 |
|----------------|------------------------------------------|
| `faiss`        | Fast Approximate Nearest Neighbor Search |
| `pickle`       | Serialize/deserialize FAISS corpus       |

---

## Web Search (Fallback)

| Tool / Module     | Purpose                               |
|-------------------|----------------------------------------|
| `web_search.py`   | Returns fallback snippets (mocked or API) |

---

## Natural Language Inference (NLI)

| Tool / Library            | Purpose                                         |
|---------------------------|-------------------------------------------------|
| `transformers` (HuggingFace) | Load transformer-based NLI model             |
| `facebook/bart-large-mnli`  | MultiNLI model for fact verification          |

---

## Scoring & Evaluation

| Tool / Function       | Purpose                                      |
|------------------------|----------------------------------------------|
| `fact_score_exp.py`    | Document-level score with exponential penalty|
| `math` module          | Logarithm and exponential functions          |

---

## Visualization & Analysis (Optional)

| Tool / Library  | Purpose                       |
|------------------|-------------------------------|
| `matplotlib`     | Visualize penalty curves and scoring functions |
| `tqdm` (optional)| Progress bar for streaming datasets             |

---

## Dataset Utilities

| Tool / Library        | Purpose                            |
|------------------------|------------------------------------|
| `datasets` (HuggingFace) | Load Wikipedia dataset (streaming mode) |

