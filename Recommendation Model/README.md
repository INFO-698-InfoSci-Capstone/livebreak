# News Recommendation System

This project is a hybrid news recommendation system that leverages user interaction data and news content to generate personalized category recommendations. It implements both collaborative filtering and content-based filtering, enhanced with deep learning and NLP techniques.

---

## Features

- **User Interaction Processing**: Parses user-news click data to identify interests.
- **Category Clustering**: Associates news articles with their respective categories.
- **Synthetic Data Generation**: Expands the dataset with synthetic user preferences.
- **Neural Collaborative Filtering (NCF)**: Learns user preferences using a deep learning model.
- **Cold Start Handling (RAG Approach)**: Recommends categories using text embeddings and cosine similarity for new users.
- **Model Evaluation**: Uses train-test split and batch training for performance monitoring.

---

## Files and Data

- `user_news_clicks.csv`: Contains user interactions with news articles.
- `news_text.csv`: Metadata about each news article (title, abstract, category).
- `user_categories_V2.xlsx`: Final user-category preferences including synthetic data.

---

## Requirements

```bash
pip install pandas numpy torch scikit-learn sentence-transformers
```

## How It Works
1. Preprocessing
- Merges user interaction and news metadata.
- Filters users who clicked on articles (click == 1).
- Maps users to categories based on clicked articles.

2. Synthetic Data
- Generates 10,000 synthetic users with random category preferences.
- Combines with real user data for robust training.

3. Collaborative Filtering with NCF
- Encodes user and category IDs.
- Generates negative samples for implicit feedback learning.
- Trains an NCF model using PyTorch to predict user-category interaction likelihood.

4. Recommendations
- recommend_categories(user_id) – Returns top-5 categories for a known user.
- recommend_for_new_user(prompt) – Uses sentence embeddings to match new users to categories based on interests.

## Model Architecture (NCF)
- Embedding layers for users and items.
- Fully connected layers with ReLU activation.
- Final sigmoid output for interaction prediction.

## Cold Start Handling
Implemented using Sentence Transformers (all-MiniLM-L6-v2) to embed combined title and abstract text for each category. For new users, input prompts are embedded and compared using cosine similarity to recommend categories.









