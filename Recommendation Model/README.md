# 📰 News Recommendation System

This project is a hybrid news recommendation system that leverages user interaction data and news content to generate personalized category recommendations. It implements both collaborative filtering and content-based filtering, enhanced with deep learning and NLP techniques.

---

## 📌 Features

- **User Interaction Processing**: Parses user-news click data to identify interests.
- **Category Clustering**: Associates news articles with their respective categories.
- **Synthetic Data Generation**: Expands the dataset with synthetic user preferences.
- **Neural Collaborative Filtering (NCF)**: Learns user preferences using a deep learning model.
- **Cold Start Handling (RAG Approach)**: Recommends categories using text embeddings and cosine similarity for new users.
- **Model Evaluation**: Uses train-test split and batch training for performance monitoring.

---

## 📁 Files and Data

- `user_news_clicks.csv`: Contains user interactions with news articles.
- `news_text.csv`: Metadata about each news article (title, abstract, category).
- `user_categories_V2.xlsx`: Final user-category preferences including synthetic data.

---

## 🛠️ Requirements

```bash
pip install pandas numpy torch scikit-learn sentence-transformers
