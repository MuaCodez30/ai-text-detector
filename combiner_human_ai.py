import json

with open("data/processed/ai_clean.json", "r", encoding="utf-8") as f:
    data1 = json.load(f)

with open("data/processed/ai_clean_extra.json", "r", encoding="utf-8") as f:
    data2 = json.load(f)

combined = data1 + data2

with open("data/processed/ai_clean_combined.json", "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)

print("Done! Combined length:", len(combined))