import json
import pickle
from sklearn.utils import shuffle

# -----------------------------
# Load saved SVM model and vectorizer
# -----------------------------
with open("models/saved_models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/saved_models/svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("models/saved_models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -----------------------------
# Batch prediction function
# -----------------------------
def predict_articles_svm(articles):
    """
    Predict AI/Human labels for multiple articles using SVM.

    Args:
        articles (list of dict): Each dict must have 'content' key.

    Returns:
        list of dict: [{'content': ..., 'predicted_label': ..., 'probability': ...}, ...]
    """
    results = []
    for article in articles:
        text = article["content"].strip()
        if not text:
            continue  # skip empty content
        text_vector = vectorizer.transform([text])
        pred = svm_model.predict(text_vector)[0]
        prob = svm_model.predict_proba(text_vector).max()
        label = le.inverse_transform([pred])[0].capitalize()
        results.append({
            "content": article["content"],
            "predicted_label": label,
            "probability": prob
        })
    return results

# -----------------------------
# Example usage
# -----------------------------
# Load your JSON file with articles
with open("data/raw/ai_sample_articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

predictions = predict_articles_svm(articles)

# Save predictions to a new JSON
with open("data/combined/ai_sample_articles_predictions.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

print(f"Predicted {len(predictions)} articles. Results saved to 'sample_articles_predictions.json'")