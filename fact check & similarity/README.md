# Fact Verification and Similarity Scoring System: Full Pipeline Summary (Detailed)

------------------------------------------------------------------------

## 1. Preprocess and Extraction

### **Input:** Raw article text

### Stage 1: Preprocessing with Regex

``` python
clean_text(text)
├─ Remove HTML tags
├─ Normalize newline, tab, and carriage return characters
└─ Collapse extra whitespace
```

------------------------------------------------------------------------

### Stage 2: NLP with spaCy

``` python
preprocess_with_spacy(text)
├─ Sentence segmentation
├─ POS tagging
├─ Named Entity Recognition
└─ Dependency parsing
→ Output: list(doc.sents)
```

------------------------------------------------------------------------

### Stage 3: Claim Extraction

``` python
extract_claims(text)
for sent in doc.sents:
    ├─ Has subject? (nsubj / nsubjpass)
    ├─ Has verb? (VERB POS)
    └─ Has named entity? (ORG, PERSON, GPE, DATE, etc.)
```

→ **Output: List of factual claim candidates**

------------------------------------------------------------------------

## 2. Evidence Database

### ⚡ Stage 1: FAISS Indexer

``` python
class FaissIndexer
├─ Transformer specification (all-MiniLM-L6-v2)
├─ add_documents method
└─ FAISS search method
```

------------------------------------------------------------------------

### Stage 2: Build Index

``` python
├─ Load existing index (or create new one)
├─ Bootstrap sample wikipedia subset (iteratively)
├─ Use add_documents(List of documents)
```

→ **Output: Corpus of vectorized embedding of trusted documents**

------------------------------------------------------------------------

## 3. Evidence Retrieval

### **Input:** A single claim (string)

### ⚡ Stage 1: FAISS Retrieval

``` python
FaissIndexer.search(claim)
├─ Encode claim using SentenceTransformer
├─ Search FAISS index for top-k cosine matches
└─ Return: List[{"doc": ..., "score": ...}]
```

------------------------------------------------------------------------

### Stage 2: Fallback Web Search (conditional)

``` python
if FAISS score < 0.5 or results < 3:
    → web_search(claim)
```

→ **Output: List of evidence snippets**

------------------------------------------------------------------------

## 3. `fact_verification_system.py`

### **Input:** List of extracted claims

### Stage 1: Natural Language Inference (NLI)

``` python
run_nli(premise=evidence, hypothesis=claim)
→ Output: label (ENTAILMENT, CONTRADICTION, NEUTRAL) + confidence
```

------------------------------------------------------------------------

### Stage 2: Aggregate NLI Verdicts

``` python
aggregate_nli_verdicts(claim, evidences[:k])
for each evidence:
    ├─ run_nli(...)
    └─ sum confidence by label
→ Final label = max total confidence
→ Final confidence = normalized average
```

------------------------------------------------------------------------

### Output Format:

``` json
[
  {
    "claim": "...",
    "verdict": {
      "label": "ENTAILMENT",
      "confidence": 0.91
    },
    "evidences": ["text1", "text2", ...]
  }
]
```

------------------------------------------------------------------------

## 4. `fact_score_exp.py` – Document-Level Scoring

### **Input:** JSON output from `fact_verification_system.py`

### Scoring: Exponential Trust Penalty

``` python
compute_exp_fact_score(results)
for each claim:
    ├─ ENTAILMENT:     + confidence
    ├─ CONTRADICTION:  - α · exp(λ · confidence)
    └─ NEUTRAL:        - β · log(1 + confidence)
```
Nonlinear metric rewards true statements moderately, penalizes neutral statements mildly, and false statements harshly. 

**Default Parameters:** - α = 1.5 - λ = 3.0 - β = 0.5

→ **Output:** Final document-level factuality score (float)

------------------------------------------------------------------------

## 5. `similarity_module.py` + `hybrid_similarity.py`

### **Input:** Full article + corpus of documents

### Stage 1: Embed & Retrieve Top-k

``` python
embed_long_doc(article, model)
→ SBERT mean-pooled vector

get_top_k_similar_docs(article_vector, corpus_vectors)
→ Top-k by cosine similarity
```

------------------------------------------------------------------------

### Stage 2: Hybrid Similarity Calculation

``` python
for each top-k doc:
    ├─ cosine_sim = SBERT similarity (semantic comparison)
    ├─ jaccard_sim = token overlap (syntactics comparison)
    └─ hybrid_score = 0.7 · cosine + 0.3 · jaccard (adjustable coefficients)
```
Hybrid score combines both syntactic and semantic comparison.

→ **Output:** Ranked list of top-k documents with combined similarity scores
