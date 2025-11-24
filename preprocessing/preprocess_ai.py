# preprocessing/preprocess_human.py
import json
import re
import os

INPUT_PATH = "data/raw/ai.json"
OUTPUT_PATH = "data/processed/ai_clean_extra.json"

def clean_text(text):
    """Clean article text: remove extra whitespace, HTML entities, and weird characters."""
    text = re.sub(r'\s+', ' ', text)          # collapse multiple spaces/newlines
    text = text.replace('\xa0', ' ')          # non-breaking spaces
    text = text.strip()
    return text

def preprocess_article(article):
    """Preprocess a single article dictionary."""
    clean_article = {
        "title": clean_text(article.get("title", "")),
        "content": clean_text(article.get("content", "")),
    }
    return clean_article

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Load raw human dataset
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw_articles = json.load(f)
    
    print(f"Loaded {len(raw_articles)} articles from {INPUT_PATH}")

    # Clean and preprocess each article
    clean_articles = [preprocess_article(a) for a in raw_articles]

    # Save cleaned dataset
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_articles, f, ensure_ascii=False, indent=2)

    print(f"Saved cleaned dataset with {len(clean_articles)} articles to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()