#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np


# In[23]:


# Load the CSV file into a pandas DataFrame
df1=pd.read_csv("user_news_clicks.csv")
df2=pd.read_csv("news_text.csv")

combined_df = pd.concat([df1,df2],axis=1)

combined_df


# In[24]:


combined_df.tail(5)


# In[25]:


df1.tail(5)


# In[26]:


df2.tail(5)


# In[27]:


df1.head(15)


# In[28]:


filtered_df = df1[df1['click'] == 1]
filtered_df


# In[29]:


user_items = filtered_df.groupby('user_id')['item'].apply(list).reset_index()
user_items.columns = ['user_id', 'items_used']


# In[30]:


user_items


# In[31]:


df2_reduced = df2.drop(columns=['title', 'abstract'])
df2_sorted = df2_reduced.sort_values(by='category').reset_index(drop=True)


# In[32]:


df2_sorted


# In[33]:


category_news = df2_reduced.groupby('category')['news_id'].apply(list).reset_index()
category_news.columns = ['category', 'news_ids']


# In[34]:


category_news


# In[35]:


category_news_exploded = category_news.explode('news_ids')
category_news_exploded.columns = ['category', 'news_id']


# In[36]:


user_items_exploded = user_items.explode('items_used')

#renaming the column
user_items_exploded.columns = ['user_id', 'news_id']


# In[37]:


merged_df = pd.merge(user_items_exploded, category_news_exploded, on='news_id', how='left')


# In[48]:


merged_df


# In[38]:


user_categories = merged_df.groupby('user_id')['category'].apply(lambda x: list(set(x.dropna()))).reset_index()
user_categories.columns = ['user_id', 'categories']


# In[39]:


user_categories


# In[40]:


import pandas as pd
import random

# Example list of all possible categories from your dataset
all_categories = [
    "sports", "music", "tv", "lifestyle", "news", "entertainment", "finance",
    "foodanddrink", "travel", "autos", "health", "weather", "video"
]

# Function to create a random set of categories
def generate_random_categories():
    n = random.randint(3, 7)  # choose between 3 and 7 categories
    return random.sample(all_categories, n)

# Generate synthetic data
synthetic_data = {
    "user_id": [f"SYNTH{i}" for i in range(100000, 110000)],
    "categories": [generate_random_categories() for _ in range(10000)]
}

# Create a DataFrame
synthetic_df = pd.DataFrame(synthetic_data)

# Preview
print(synthetic_df.head())


# In[41]:


synthetic_df


# In[42]:


combined_synt_real = pd.concat([user_categories,synthetic_df],axis=1)

combined_synt_real


# In[43]:


# Concatenate the real and synthetic data vertically
combined_synt_real = pd.concat([user_categories, synthetic_df], axis=0).reset_index(drop=True)

# Check the result
print(combined_synt_real.head())


# In[49]:


combined_synt_real


# In[46]:


combined_synt_real.to_excel("user_categories_V2.xlsx", index=False)


# In[20]:


# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd

# # Build user-category matrix
# user_category_df = merged_df.groupby(['user_id', 'category']).size().unstack(fill_value=0)

# # Compute cosine similarity between users
# user_similarity = pd.DataFrame(cosine_similarity(user_category_df), 
#                                index=user_category_df.index, 
#                                columns=user_category_df.index)

# # For a user, find top similar users
# similar_users = user_similarity['U12345'].sort_values(ascending=False)[1:6]  # top 5 similar users


# In[21]:


# similar_users


# In[51]:


import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import random

# Step 1: Explode category lists into individual user-category pairs
exploded_df = combined_synt_real.explode('categories').dropna()
exploded_df = exploded_df.drop_duplicates()
exploded_df['interaction'] = 1

# Step 2: Encode users and categories to integer IDs
user2id = {u: i for i, u in enumerate(exploded_df['user_id'].unique())}
cat2id = {c: i for i, c in enumerate(exploded_df['categories'].unique())}

exploded_df['user'] = exploded_df['user_id'].map(user2id)
exploded_df['categories'] = exploded_df['categories'].map(cat2id)

# Step 3: Generate negative samples
num_users = len(user2id)
num_cats = len(cat2id)
negatives = []

for user in exploded_df['user'].unique():
    used_cats = set(exploded_df[exploded_df['user'] == user]['categories'])
    while len(used_cats) < min(num_cats, len(used_cats) + 5):  # Add 5 negatives per user (adjustable)
        neg_cat = random.randint(0, num_cats - 1)
        if neg_cat not in used_cats:
            negatives.append([user, neg_cat, 0])
            used_cats.add(neg_cat)

negative_df = pd.DataFrame(negatives, columns=['user', 'categories', 'interaction'])

# Step 4: Combine positive and negative samples
positive_df = exploded_df[['user', 'categories', 'interaction']]
all_data = pd.concat([positive_df, negative_df])

# Step 5: Train/test split
train_df, test_df = train_test_split(all_data, test_size=0.2, random_state=42)

print(train_df.head())


# In[52]:


import torch.nn as nn

#after trying cosine similarity for the users i came to a conclusion to use the neural collaborative filtering
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NCF, self).__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user, item):
        u = self.user_embed(user)
        i = self.item_embed(item)
        x = torch.cat([u, i], dim=-1)
        return self.fc_layers(x)


# In[53]:


from torch.utils.data import Dataset, DataLoader

class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user'].values, dtype=torch.long)
        self.items = torch.tensor(df['categories'].values, dtype=torch.long)
        self.labels = torch.tensor(df['interaction'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

train_loader = DataLoader(InteractionDataset(train_df), batch_size=256, shuffle=True)
test_loader = DataLoader(InteractionDataset(test_df), batch_size=256)

#recommendation model
model = NCF(num_users, num_cats)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#epoch training
for epoch in range(5):
    model.train()
    total_loss = 0
    for user, item, label in train_loader:
        optimizer.zero_grad()
        preds = model(user, item).squeeze()
        loss = criterion(preds, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# In[57]:


#predicting top categories for the users from the user db
def recommend_categories(user_str, top_k=5):
    user_id = user2id.get(user_str)
    if user_id is None:
        print("User not found")
        return []
    
    model.eval()
    items = torch.tensor(range(num_cats), dtype=torch.long)
    users = torch.tensor([user_id] * num_cats, dtype=torch.long)
    
    with torch.no_grad():
        scores = model(users, items).squeeze()
    
    top_items = torch.topk(scores, top_k).indices.numpy()
    id2cat = {v: k for k, v in cat2id.items()}
    return [id2cat[i] for i in top_items]

#please type our example User Id over here
recommend_categories('SYNTH113655')


# RAG Model for the Cold Start Problems

# In[26]:


get_ipython().system('pip install --user -U sentence-transformers')


# In[27]:


from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Example: Use average of title + abstract for each category
category_texts = []
category_labels = []
for i, row in category_news.iterrows():
    texts = []
    for news_id in row['news_ids']:
        row_match = df2[df2['news_id'] == news_id]
        if not row_match.empty:
            title = str(row_match.iloc[0]['title'])
            abstract = str(row_match.iloc[0]['abstract'])
            texts.append(title + " " + abstract)
    if texts:
        combined = " ".join(texts[:5])  # Use top 5 articles to represent the category
        category_texts.append(combined)
        category_labels.append(row['category'])

# Compute category embeddings
category_embeddings = model.encode(category_texts)


# In[28]:


from sklearn.metrics.pairwise import cosine_similarity

def recommend_for_new_user(prompt, top_k=5):
    prompt_embedding = model.encode([prompt])
    sims = cosine_similarity(prompt_embedding, category_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [category_labels[i] for i in top_indices]

# Example
recommend_for_new_user("bicycle, sports, legos")


# In[29]:


# !pip install --user openai


# In[30]:


# import openai

# def generate_response(categories):
#     prompt = f"A new user is interested in these topics. Recommend some categories: {categories}."
#     messages = [
#         {"role": "system", "content": "You are a helpful AI news recommender."},
#         {"role": "user", "content": prompt}
#     ]
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=messages
#     )
#     return response.choices[0].message['content']


# ###I am not doing this as this requires the Open AI Key
