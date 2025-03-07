import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
df = pd.read_csv("libraries_dataset.csv")

# Combine name and description for each library
df["combined_text"] = df["name"] + " " + df["description"] + " " + df["use_case"]

# Preprocess text (lowercase, remove special characters, etc.)
def preprocess(text):
    return text.lower()

df["processed_text"] = df["combined_text"].apply(preprocess)

# Tokenize corpus for BM25
corpus = [doc.split() for doc in df["processed_text"]]

# Initialize BM25 with tuned parameters
bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)

# Function to get recommendations based on a user query
def get_recommendations(query, top_n=5):
    query = preprocess(query)
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    
    # Re-rank based on average ratings from feedback if available (for example purposes, use dummy ratings)
    # Here you might join with your feedback data to adjust scores
    dummy_ratings = np.random.rand(len(scores))  # Replace with actual ratings integration
    enhanced_scores = scores + (dummy_ratings * 0.1)
    
    top_indices = enhanced_scores.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# Test the recommendation function
query = "backend framework for banking with authentication and payment gateway"
recommendations = get_recommendations(query, top_n=5)
print(recommendations[["name", "description", "use_case", "language", "documentation_link"]])
