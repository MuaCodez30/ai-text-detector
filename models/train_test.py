import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# -----------------------------
# Load dataset 
# -----------------------------
with open("data/combined/combined_dataset_new.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Total samples loaded: {len(dataset)}")

# -----------------------------
# Balance dataset (AI = human)
# -----------------------------
ai_articles = [a for a in dataset if a["label"].lower() == "ai"]
human_articles = [a for a in dataset if a["label"].lower() == "human"]

min_count = min(len(ai_articles), len(human_articles))

ai_articles = ai_articles[:min_count]
human_articles = human_articles[:min_count]

balanced_dataset = ai_articles + human_articles
balanced_dataset = shuffle(balanced_dataset, random_state=42)

print(f"Balanced dataset → AI: {len(ai_articles)}, Human: {len(human_articles)}")
print(f"Total balanced: {len(balanced_dataset)}")

# -----------------------------
# Use title + content (IMPORTANT for accuracy)
# -----------------------------
texts = [article["content"].strip() for article in balanced_dataset]

labels = [article["label"] for article in balanced_dataset]

# -----------------------------
# Encode labels ("AI" → 1, "human" → 0)
# -----------------------------
le = LabelEncoder()
y = le.fit_transform(labels)

print(f"Label classes: {le.classes_}")

# -----------------------------
# TF-IDF Vectorizer
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=8000,       # more features = better accuracy
    ngram_range=(1, 2),      # include bigrams → big accuracy boost
    sublinear_tf=True,       # improves performance
    min_df=3                 # remove very rare noise words
)

X = vectorizer.fit_transform(texts)

print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# -----------------------------
# Split into train & test sets
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")