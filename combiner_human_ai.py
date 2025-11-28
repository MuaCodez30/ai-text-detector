import json

with open("data/raw/human_combined1.json", "r", encoding="utf-8") as f:
    data1 = json.load(f)

with open("data/raw/human_test_new3.json", "r", encoding="utf-8") as f:
    data2 = json.load(f)

combined = data1 + data2

with open("data/raw/human_combined.json", "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)

print("Done! Combined length:", len(combined))