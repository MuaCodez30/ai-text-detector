# ai_generation/generate_ai_articles_extra.py
import json
import os
import time
import openai

# -----------------------------
# CONFIG
# -----------------------------
INPUT_PATH = "data/raw/sample_articles.json"  # your 4000 human articles
OUTPUT_PATH = "data/raw/ai_sample_articles.json"    # where AI articles will be saved
MODEL = "gpt-3.5-turbo"                   # or "gpt-4" if available
DELAY = 3  # seconds between requests
MAX_RETRIES = 3

# -----------------------------
# SETUP OPENAI API
# -----------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")  # make sure this is set

# -----------------------------
# LOAD DATA
# -----------------------------
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    human_articles = json.load(f)

# Load previously generated AI articles to resume
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        ai_articles = json.load(f)
        processed_titles = {a["title"] for a in ai_articles}
else:
    ai_articles = []
    processed_titles = set()

print(f"Loaded {len(human_articles)} human articles")
print(f"{len(processed_titles)} articles already processed")

# -----------------------------
# FUNCTION TO GENERATE AI ARTICLE
# -----------------------------
def generate_ai_content(title, content):
    prompt = f"Rewrite this news article in an engaging AI-enhanced style in Azerbaijani:\n\nTitle: {title}\nContent: {content}"
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating AI content for '{title}': {e}")
        return None

# -----------------------------
# MAIN LOOP
# -----------------------------
for idx, article in enumerate(human_articles, start=1):
    title = article["title"]

    if title in processed_titles:
        print(f"[{idx}/{len(human_articles)}] Already processed, skipping: {title[:50]}...")
        continue

    print(f"[{idx}/{len(human_articles)}] Generating AI article for: {title[:50]}...")

    content = None
    retries = MAX_RETRIES
    while retries > 0 and content is None:
        content = generate_ai_content(title, article["content"])
        if content is None:
            retries -= 1
            print(f"  Retry remaining: {retries}")
            time.sleep(DELAY)

    if content:
        ai_articles.append({
            "title": title,
            "content": content
        })
        processed_titles.add(title)

        # Incrementally save
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(ai_articles, f, ensure_ascii=False, indent=2)
        print(f"  Saved successfully!")
    else:
        print(f"  Skipped '{title}' after {MAX_RETRIES} failed attempts")

    time.sleep(DELAY)

print(f"Finished! Total AI-generated articles saved: {len(ai_articles)}")