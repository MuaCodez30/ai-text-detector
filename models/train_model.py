import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle

# -----------------------------
# Load dataset
# -----------------------------
with open("data/combined/combined_dataset_new.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Normalize labels
for a in dataset:
    a["label"] = a["label"].lower()

# Balance dataset
ai_articles = [a for a in dataset if a["label"] == "ai"]
human_articles = [a for a in dataset if a["label"] == "human"]
n = min(len(ai_articles), len(human_articles))
ai_articles = ai_articles[:n]
human_articles = human_articles[:n]
balanced_dataset = ai_articles + human_articles
balanced_dataset = shuffle(balanced_dataset, random_state=42)

print("Balanced dataset → AI:", len(ai_articles), "Human:", len(human_articles))
print("Total balanced:", len(balanced_dataset))

# Extract texts & labels
texts = [a["content"] for a in balanced_dataset]
labels = [a["label"] for a in balanced_dataset]

# Remove empty texts only (not short)
texts, labels = zip(*[(t, l) for t, l in zip(texts, labels) if t.strip()])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)  # 'ai' -> 1, 'human' -> 0

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2), sublinear_tf=True, min_df=3)
X = vectorizer.fit_transform(texts)

print("Feature matrix shape:", X.shape)
print("Label vector shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# -----------------------------
# Train Logistic Regression
# -----------------------------
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# -----------------------------
# Train SVM (linear kernel)
# -----------------------------
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
print("\n--- SVM (Linear Kernel) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

# -----------------------------
# Save models + vectorizer
# -----------------------------
with open("models/saved_models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/saved_models/logreg_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

with open("models/saved_models/svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

with open("models/saved_models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nAll models and vectorizer saved to 'models/' folder.")