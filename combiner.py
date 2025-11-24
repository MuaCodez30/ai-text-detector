import json
from random import shuffle, sample

# Load your human and AI articles
with open("data/processed/human_clean_combined.json", "r", encoding="utf-8") as f:
    human_articles = json.load(f)

with open("data/processed/ai_clean_combined.json", "r", encoding="utf-8") as f:
    ai_articles = json.load(f)

# Downsample human articles to match AI count
ai_count = len(ai_articles)
human_articles_balanced = sample(human_articles, ai_count)

# Label articles
human_labeled = [{"content": a["content"], "label": "human"} for a in human_articles_balanced]
ai_labeled = [{"content": a["content"], "label": "ai"} for a in ai_articles]

# Combine and shuffle
dataset = human_labeled + ai_labeled
shuffle(dataset)

# Save combined dataset
with open("data/combined/combined_dataset_new.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Combined dataset saved! Total articles: {len(dataset)}")