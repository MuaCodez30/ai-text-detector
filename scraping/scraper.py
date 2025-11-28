# scraping/scraper.py
import requests
from bs4 import BeautifulSoup
import time
import json
import re
import random
import os

# -----------------------------
# CONFIG
# -----------------------------
CATEGORIES = ["world", "politics", "incedent", "economy", "society",
              "science", "culture", "sport", "tech", "showbiz", "health", "regional"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

SAVE_INTERVAL = 50  # Save JSON every 50 articles
SLEEP_MIN = 0.7     # minimum sleep in seconds
SLEEP_MAX = 1.2     # maximum sleep in seconds

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def scrape_article(url):
    """Scrape a single article and return structured data."""
    try:
        print(f"Scraping: {url}")
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            print(f"  Status code: {r.status_code}")
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        # Title
        title_tag = soup.find("h1")
        if not title_tag:
            print("  No title found")
            return None
        title = title_tag.get_text(strip=True)

        # Content
        content_div = soup.find("div", class_="article_text")
        if not content_div:
            print("  No content div found")
            return None
        paragraphs = content_div.find_all("p")
        article_text = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        if len(article_text) < 50:
            print(f"  Content too short ({len(article_text)} chars)")
            return None

        # Date
        date_div = soup.find("div", class_="date-info")
        date = date_div.get_text(strip=True) if date_div else None

        # Category
        category_span = soup.find("span", class_="category")
        category = category_span.get_text(strip=True) if category_span else "unknown"

        # Views
        view_div = soup.find("div", class_="view-info")
        views_text = view_div.get_text(strip=True) if view_div else "Baxış: 0"
        views = int(re.search(r'\d+', views_text).group()) if re.search(r'\d+', views_text) else 0

        # Tags
        tags_div = soup.find("div", class_="tags_list")
        tags = []
        if tags_div:
            tag_links = tags_div.find_all("a", href=re.compile(r"/tag/"))
            tags = [tag.get_text(strip=True) for tag in tag_links]

        return {
            "url": url,
            "title": title,
            "content": article_text,
            "date": date,
            "category": category,
            "views": views,
            "tags": tags,
            "content_length": len(article_text)
        }

    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None


def scrape_range(start_id, end_id, target_count, save_path):
    """Scrape a range of article IDs and save results periodically."""
    articles = []
    scraped_urls = set()
    count = 0
    failed_count = 0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Starting scrape from ID {start_id} to {end_id}, target: {target_count} articles")

    for article_id in range(start_id, end_id, -1):
        if count >= target_count:
            break

        article_found = False
        for category in CATEGORIES:
            if count >= target_count:
                break

            url = f"https://news.milli.az/{category}/{article_id}.html"
            if url in scraped_urls:
                continue

            data = scrape_article(url)
            if data:
                articles.append(data)
                scraped_urls.add(url)
                count += 1
                article_found = True
                print(f"[SUCCESS] {count}/{target_count} → {data['category']}: {data['title'][:60]}... "
                      f"(Views: {data['views']})")
                break  # Move to next article ID

        if not article_found:
            failed_count += 1
            print(f"[SKIP] No article found for ID {article_id} (Failed: {failed_count})")

        # Sleep to be polite
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        # Save periodically
        if count % SAVE_INTERVAL == 0 and count > 0:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            print(f"[SAVE] Progress saved at {count} articles.")

        # Progress log
        if count % 10 == 0:
            print(f"Progress: {count} found, {failed_count} failed")

    # Final save
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"COMPLETED! Saved {len(articles)} articles to {save_path}")
    print(f"Success rate: {count}/{(start_id - end_id)} = {(count / (start_id - end_id) * 100):.1f}%")
    return articles


def test_scraper():
    """Test the scraper with a known working URL."""
    test_url = "https://news.milli.az/incedent/1302680.html"
    print("Testing scraper with known URL...")
    result = scrape_article(test_url)
    if result:
        print("TEST SUCCESSFUL!")
        print(f"Title: {result['title']}")
        print(f"Category: {result['category']}")
        print(f"Date: {result['date']}")
        print(f"Views: {result['views']}")
        print(f"Content preview: {result['content'][:200]}...")
        print(f"Tags: {result['tags']}")
    else:
        print("TEST FAILED!")
    return result


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    # Test scraper first
    test_result = test_scraper()

    if test_result:
        print("\n===================================")
        print(" SCRAPING EXTRA 15000 HUMAN ARTICLES")
        print("===================================\n")
        articles = scrape_range(
            start_id = 630000,
            end_id   = 500000,          # oldest in batch
            target_count=10200,
            save_path="data/raw/human.json"
        )
    else:
        print("Scraper test failed. Please check the website structure.")